import gdown
import pandas as pd
import random
import time
import warnings
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
import os

# 抑制特定警告
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")


cve_file_path = '2023_CVE_CWE.csv'

all_cwe_file_path = 'All_CWE.csv'


# 讀取 All_CWE.csv 資料
try:
    df_cwe = pd.read_csv(all_cwe_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_cwe = pd.read_csv(all_cwe_file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_cwe = pd.read_csv(all_cwe_file_path, encoding='latin1')

# 提取特徵並刪除不必要的欄位
selected_cwe_features = ['CWE-ID', 'CWE-Description']
df_all_cwe = df_cwe[selected_cwe_features]
df_all_cwe = df_all_cwe.drop_duplicates(subset=['CWE-ID'], keep='first')

print("CWE 資料處理完成")
print(df_all_cwe.head())
print("df_all_cwe 的行數:", df_all_cwe.shape[0])
print("df_all_cwe 的列數:", df_all_cwe.shape[1])
print(df_all_cwe.dtypes)

# 讀取 2023_CVE_CWE.csv 資料
try:
    df_selected = pd.read_csv(cve_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_selected = pd.read_csv(cve_file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_selected = pd.read_csv(cve_file_path, encoding='latin1')

# 檢查 2023_CVE_CWE.csv 文件的列名
print("2023_CVE_CWE.csv 文件的列名:")
# print("cve_data_2000_2022.csv 文件的列名:")
print(df_selected.columns)

# 確保所需的列存在
required_columns = ['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description']
missing_columns = [col for col in required_columns if col not in df_selected.columns]
if missing_columns:
    raise KeyError(f"缺少列: {missing_columns}")

print("df_selected 的行數:", df_selected.shape[0])
print("df_selected 的列數:", df_selected.shape[1])
print(df_selected.dtypes)

# 創建空的 DataFrame
df_mapping = pd.DataFrame(columns=['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description', 'P/N'])

# 獲取所有的 CWE-ID
all_cwe_ids = df_all_cwe['CWE-ID'].unique()
cve_to_cwe_dict = df_selected.set_index('CVE-ID')['CWE-ID'].to_dict()

# 使用列表收集所有的數據
rows = []
for cve_id, cve_description in zip(df_selected['CVE-ID'], df_selected['CVE-Description']):
    cwe_id = cve_to_cwe_dict[cve_id]
    cwe_description = df_selected[df_selected['CWE-ID'] == cwe_id]['CWE-Description'].iloc[0]
    rows.append({'CVE-ID': cve_id, 'CVE-Description': cve_description, 'CWE-ID': cwe_id, 'CWE-Description': cwe_description, 'P/N': 'P'})

    cwe_ids = list(all_cwe_ids)
    if cwe_id in cwe_ids:
        cwe_ids.remove(cwe_id)
    random.shuffle(cwe_ids)

    for random_cwe_id in cwe_ids[:1]:
        random_cwe_description = df_all_cwe[df_all_cwe['CWE-ID'] == random_cwe_id]['CWE-Description'].iloc[0]
        rows.append({'CVE-ID': cve_id, 'CVE-Description': cve_description, 'CWE-ID': random_cwe_id, 'CWE-Description': random_cwe_description, 'P/N': 'N'})

df_mapping = pd.DataFrame(rows)
print(df_mapping.head())

# 將 'P' 標籤轉換為 1，將 'N' 標籤轉換為 0
df_mapping['P/N'] = df_mapping['P/N'].replace({'P': 1, 'N': 0})

from sklearn.utils import resample
# 重複正面樣本以匹配負面樣本數量
positive_samples = df_mapping[df_mapping['P/N'] == 1]
negative_samples = df_mapping[df_mapping['P/N'] == 0]
# 計算需要重複的次數
repeat_times = len(negative_samples) // len(positive_samples)
# 重複正面樣本
resampled_positive_samples = resample(positive_samples, replace=True, n_samples=len(negative_samples))
# 合併正面和負面樣本
balanced_samples = pd.concat([resampled_positive_samples, negative_samples])
# 確認正面和負面樣本的數量
print("正面樣本數量:", len(balanced_samples[balanced_samples['P/N'] == 1]))
print("負面樣本數量:", len(balanced_samples[balanced_samples['P/N'] == 0]))

import torch # 引入PyTorch資料庫，用於深度學習
import torch.nn as nn # 引入PyTorch的神經網路Function
from transformers import BertModel, BertTokenizer, BertConfig # 引入transformers資料庫中的BertModel和BertTokenizer
from torch.utils.data import DataLoader, Dataset # 引入PyTorch的DataLoader和Dataset模組
from sklearn.model_selection import train_test_split # 用於拆分數據集

# 初始化 BERT Tokenizer 和 Model，使用預訓練的模型
model_path = '../pre-train/pretrain_model_storage/pre-train-cve-2e-5-2015-2023-0713/epoch_4/'
weights_path = model_path + 'pytorch_model.bin'  # 確保路徑正確
#tokenizer_path = 'expanded_tokenizer_cve'
tokenizer = BertTokenizer.from_pretrained(model_path)


# 將數據集拆分為訓練集和驗證集，測試集比例為10%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    balanced_samples[['CVE-Description', 'CWE-Description']], balanced_samples['P/N'], test_size=0.1)

# 顯示訓練集和驗證集的大小(包含標籤)
print("訓練集大小:", train_texts.shape, "訓練標籤大小:", train_labels.shape)
print("驗證集大小:", val_texts.shape, "驗證標籤大小:", val_labels.shape)

# 顯示訓練集和驗證集標籤PN數量，確認轉換Label有無成功
print("\n訓練集標籤PN:\n", train_labels.value_counts())
print("驗證集標籤PN:\n", val_labels.value_counts())

# 顯示幾個訓練和驗證集樣本查看，確保沒有轉換錯
print("\n訓練集樣本:\n", train_texts.head())
print("驗證集樣本:\n", val_texts.head())

# 把有關於tokenization轉換的warning關掉
import logging # 引入 logging 庫
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # 關閉transformers中的特定warning

# 定義文本預處理函數，將文本進行編碼
def encode_texts(tokenizer, texts):
    return tokenizer(texts['CVE-Description'].tolist(), texts['CWE-Description'].tolist(),
                     padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# 編碼訓練集和驗證集文本
train_encodings = encode_texts(tokenizer, train_texts)
val_encodings = encode_texts(tokenizer, val_texts)

# 輸出訓練和驗證編碼結果的詳細信息
print("訓練編碼的key：", train_encodings.keys())
print("驗證編碼的key：", val_encodings.keys())

# 檢查訓練input_ids和attention_mask的尺寸
# 通常torch.Size([180, 512])，前面數字是數量，後面是文本最大長度
print("訓練input_ids尺寸：", train_encodings['input_ids'].shape)
print("訓練attention_mask尺寸：", train_encodings['attention_mask'].shape)
print("驗證input_ids尺寸：", val_encodings['input_ids'].shape)
print("驗證attention_mask尺寸：", val_encodings['attention_mask'].shape)

config = BertConfig.from_pretrained(os.path.join(model_path, 'config.json'))
# 載入整個 BertForMaskedLM 模型

# 定義BERT模型
class CustomBERTModel(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(CustomBERTModel, self).__init__()
        self.bert = bert_model  # 使用預訓練的BertModel
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Average pooling
        avg_pooling_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        diff = torch.abs(cls_embeddings - avg_pooling_embeddings)
        mul = cls_embeddings * avg_pooling_embeddings
        
        combined_features = torch.cat((diff, mul), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

# 數據加載和模型訓練的準備
class SecurityDataset(Dataset):
    def __init__(self, encodings, labels): # 初始化數據集類，接收文本編碼和標籤作為參數
        self.encodings = encodings # 保存文本的編碼
        self.labels = labels # 保存對應的標籤

    def __getitem__(self, idx):  # 根據索引 idx 獲取一條數據，將文本編碼和標籤轉換為 PyTorch 的 tensor
        # item = {key: torch.tensor(val[idx])  for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # 修改此處
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self): # 獲取數據集長度
        return len(self.labels)


# 將訓練數據和標籤、驗證數據和標籤轉換為上面設定的SecurityDataset的樣式
train_dataset = SecurityDataset(train_encodings, train_labels.to_numpy())
val_dataset = SecurityDataset(val_encodings, val_labels.to_numpy())

# 確認訓練集和驗證集的樣本數量
print("訓練集樣本數量:", len(train_dataset))
print("驗證集樣本數量:", len(val_dataset))

# 使用 DataLoader 將測試和驗證集分批次加載，batch_size設定為16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 從 .bin 檔案加載預訓練的權重
pretrained_weights = torch.load(weights_path)

# 過濾出僅屬於 BertModel 的權重
filtered_weights = {k[5:]: v for k, v in pretrained_weights.items() if k.startswith("bert.")}

# 實例化 BertModel
bert_base_model = BertModel.from_pretrained(model_path)

# 實例化您的自定義模型並載入過濾後的權重
model = CustomBERTModel(bert_base_model)
model.bert.load_state_dict(filtered_weights, strict=False)

# 初始化 optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

import torch
import os
import matplotlib .pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")


def save_model(model, tokenizer, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, 'best_model_average_cvecwecapec_2e-5_2015-2023-0713.pt')
    tokenizer_path = os.path.join(save_path, 'tokenizer')
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"模型已保存到 {model_path}")
    print(f"Tokenizer已保存到 {tokenizer_path}")

# 統一保存路徑
model_save_path = 'train_model_save_bert'

# 確認保存路徑存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# 初始化最佳模型的損失為一個很高的數值
best_loss = float('inf')
patience_counter = 0  # 設置耐心次數
early_stopping_patience = 3  # 耐心次數設定為3


# 儲存每次 epoch 的評估模型指標
train_losses = []  # 訓練損失
eval_losses = []   # 驗證損失
accuracies = []    # 準確度
precisions = []    # 精確度
recalls = []       # 召回率
f1_scores = []     # F1分數

# 將模型移動到設備 (GPU 或 CPU)
model.to(device)
# 開始計時
start_time = time.time()

# 訓練模型
for epoch in range(10):  # 訓練過程 epoch 迭代 5次
    model.train()  # 設置模型為訓練模式
    total_train_loss = 0  # 剛開始訓練損失設定為 0

    for batch in tqdm(train_loader, desc=f"訓練階段 Epoch {epoch+1}/10"):
        input_ids = batch['input_ids'].to(device)  # 獲取輸入的 input_ids 並移動到設備
        attention_mask = batch['attention_mask'].to(device)  # 獲取輸入的 attention_mask 並移動到設備
        labels = batch['labels'].to(device)  # 獲取真實標籤並移動到設備

        optimizer.zero_grad()  # 清空前一次迭代的梯度
        outputs = model(input_ids, attention_mask)  # 透過 forward 前向傳播獲取模型輸出
        loss = nn.CrossEntropyLoss()(outputs, labels)  # 獲取輸出之後透過 CrossEntropy 計算損失
        loss.backward()  # 計算梯度
        optimizer.step()  # 更新模型參數

        total_train_loss += loss.item()  # 累加訓練損失

    average_train_loss = total_train_loss / len(train_loader)  # 計算平均訓練損失
    train_losses.append(average_train_loss)  # 紀錄訓練損失

    # 驗證階段
    model.eval()  # 設置模型為驗證模式
    total_eval_loss = 0  # 剛開始設定初始化驗證損失為 0
    all_preds = []  # 初始化所有預測值的列表
    all_labels = []  # 初始化所有真實標籤的列表

    # 禁用梯度計算並驗證數據集（節省內存和計算資源）
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"驗證階段 Epoch {epoch+1}/10"):
            input_ids = batch['input_ids'].to(device)  # 獲取輸入的 input_ids 並移動到設備
            attention_mask = batch['attention_mask'].to(device)  # 獲取輸入的 attention_mask 並移動到設備
            labels = batch['labels'].to(device)  # 獲取真實標籤並移動到設備

            outputs = model(input_ids, attention_mask)  # 通過模型進行 forward，獲取輸出 outputs
            loss = nn.CrossEntropyLoss()(outputs, labels)  # 計算當前 batch 的 CrossEntropy Loss
            total_eval_loss += loss.item()  # 將每個 batch 計算出來的 Loss 累加上去

            _, predicted = torch.max(outputs, dim=1)  # 使用 torch.max 獲取預測結果
            all_preds.extend(predicted.cpu().numpy())  # 將當前 batch 的預測結果放在 all_preds 中，最後驗證完才會輸出 print
            all_labels.extend(labels.cpu().numpy())  # 將當前 batch 的真實標籤放在 all_preds 中，最後驗證完才會輸出 print

    average_eval_loss = total_eval_loss / len(val_loader)  # 計算平均驗證損失
    eval_losses.append(average_eval_loss)  # 紀錄驗證損失

    # 計算並列印準確度、精確度、召回率和 F1 分數
    accuracy = accuracy_score(all_labels, all_preds)  # 計算 accuracy 準確度
    precision = precision_score(all_labels, all_preds, average='binary')  # 計算 precision 精確度
    recall = recall_score(all_labels, all_preds, average='binary')  # 計算 recall 召回率
    f1 = f1_score(all_labels, all_preds, average='binary')  # 計算 F1 分數

    # 紀錄度量用於繪圖
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # 輸出每次 epoch 的評估模型指標
    print(f"Epoch {epoch+1}, Average Training Loss: {average_train_loss:.4f}, Average Validation Loss: {average_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # 檢查是否需要早停
    if average_eval_loss < best_loss:
        best_loss = average_eval_loss
        patience_counter = 0  # 重置耐心次數
        save_model(model, tokenizer, model_save_path)  # 更新函數的調用
        print(f"模型和Tokenizer已保存到 {model_save_path}，當前最佳驗證損失為: {best_loss:.4f}")
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"早停於Epoch {epoch + 1}")
        break

# 訓練結束計時
end_time = time.time()
total_training_time = end_time - start_time
print(f"總訓練時間: {total_training_time:.2f} 秒")

# 用 matplotlib 繪圖 loss function，x 軸為 epoch 次數，y 軸為 0 到 1 數值
epochs = range(1, 11)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, eval_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 用 matplotlib 繪圖 Accuracy、Precision、Recall、F1-Score，x 軸為 epoch 次數，y 軸為 0 到 1 數值
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, label='Accuracy')
plt.plot(epochs, precisions, label='Precision')
plt.plot(epochs, recalls, label='Recall')
plt.plot(epochs, f1_scores, label='F1 Score')
plt.title('Accuracy, Precision, Recall, F1 Score')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()
plt.close()  # 關閉圖表，釋放資源

from sklearn.metrics import confusion_matrix # 從sklearn資料庫中引入confusion_matrix function，用於計算混淆矩陣
import seaborn as sns # 引入seaborn資料庫，用於繪製數據可視化圖表

# 計算混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
# 這裡的 all_labels 是真實標籤，all_preds 是模型預測的標籤

# 使用Seaborn繪製混淆矩陣
plt.figure(figsize=(10, 7))  # 創建一個繪圖對象，並設置圖像大小為 10x7
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # 使用熱力圖繪製混淆矩陣，annot=True 表示在每個方格顯示數值，fmt='d' 表示數值格式為整數，cmap='Blues' 表示顏色映射使用藍色調
plt.title('Confusion Matrix') # 圖像的標題
plt.xlabel('Predicted Label') # 設置 X 軸標籤為 "Predicted Label"
plt.ylabel('True Label') # 設置 Y 軸標籤為 "True Label"
plt.show()
plt.close()
