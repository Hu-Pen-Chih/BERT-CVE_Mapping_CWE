from docx import Document
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import warnings
import logging
import pandas as pd
import os
import pickle
import random
import numpy as np

# 設置隨機數種子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# 隱藏tokenizer截斷的warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")


def read_data(file_path, column_name):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    
    # 查看文件
    print(f"文件：{file_path}")
    print("列名：", df.columns)
    print("行數：", len(df))

    return df[column_name].dropna().tolist()  # 移除任何空值

# 文件路徑和欄位名稱
files_columns = {
    'All_CAPEC.csv': 'CAPEC-Description',
    'All_CWE.csv': 'CWE-Description',
    '2023_CVE_CWE.csv': 'CVE-Description',
    'cve_data_all_2000_to_2022.csv':'CVE-Description'
}

# 載入數據
texts = []
for file_path, column_name in files_columns.items():
    texts.extend(read_data(file_path, column_name))

# 確認讀取的文本數量
print(f"共載入 {len(texts)} 篇文本。")


# 15% 的 token 會被隨機選擇進行遮蔽。在這些被選中的 token 中，80% 會被替換為 [MASK] 標記，10% 會被替換為隨機的其他 token，而剩下的 10% 則保持不變
def create_masked_lm_inputs(texts, tokenizer, max_length=512, mask_prob=0.15, replace_mask_prob=0.8, random_replace_prob=0.10, keep_original_prob=0.10):
    pairs = []
    for i, text in enumerate(texts):
        inputs = tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        # 進行隨機遮蔽
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 替換為 [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, replace_mask_prob)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 隨機替換
        indices_random = torch.bernoulli(torch.full(labels.shape, random_replace_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
        valid_random_words = ~torch.tensor(tokenizer.get_special_tokens_mask(random_words.tolist(), already_has_special_tokens=True), dtype=torch.bool)
        input_ids[indices_random & valid_random_words] = random_words[indices_random & valid_random_words]

        pairs.append((input_ids, labels, attention_mask))

    return pairs


# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pairs = create_masked_lm_inputs(texts, tokenizer)
print(f"創建的MLM輸入數量: {len(pairs)}")


# 創建數據集
class CustomTextDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_ids, labels, attention_mask = self.pairs[idx]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

dataset = CustomTextDataset(pairs)

print("數據集創建完成。")

# 分割數據集為訓練集和驗證集
train_size = int(0.8 * len(pairs))
train_pairs = pairs[:train_size]
eval_pairs = pairs[train_size:]

train_dataset = CustomTextDataset(train_pairs)
eval_dataset = CustomTextDataset(eval_pairs)
print("數據集創建完成。")

batch_size = 16

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 設置訓練參數
num_epochs = 20
learning_rate = 2e-5
beta1 = 0.9  # AdamW 的 Beta1
beta2 = 0.999  # AdamW 的 Beta2
weight_decay = 0.01  # 權重衰減
total_steps = num_epochs * len(train_loader)
warmup_steps = total_steps // 10  # 在訓練初期逐漸增加學習率的步數
early_stopping_patience = 2  # 設置early stopping的耐心次數


# 初始化 BERT MLM 模型
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
model.train()
# 初始化最佳loss和early stopping
best_loss = float('inf')
patience_counter = 0

# 開始訓練並使用 tqdm 顯示訓練進度
print("訓練開始。")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for step, batch in enumerate(epoch_iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} 平均訓練損失: {avg_epoch_loss}")

    # 計算驗證損失
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            eval_loss += loss.item()
    
    avg_eval_loss = eval_loss / len(eval_loader)
    print(f"Epoch {epoch + 1} 平均驗證損失: {avg_eval_loss}")

    # 在每個時期結束時儲存模型
    save_directory = f'pretrain_model_storage/pre-train-cve-2e-5-all-0714/epoch_{epoch + 1}'
    model_path = os.path.join(save_directory, 'pytorch_model.bin')
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(save_directory)

    config_path = os.path.join(save_directory, 'config.json')
    model.config.to_json_file(config_path)
    print(f"模型權重和配置已保存到 {model_path} 和 {config_path}")

    # 檢查是否需要早停
    if avg_eval_loss < best_loss:
        best_loss = avg_eval_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"早停於Epoch {epoch + 1}")
        break

print("訓練完成。")
