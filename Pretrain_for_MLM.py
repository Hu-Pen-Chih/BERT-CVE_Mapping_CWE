from docx import Document
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import warnings
import logging
import pandas as pd
import os
import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Suppress specific warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")

# File paths and column names
files_columns = {
    'All_CWE.csv': 'CWE-Description',
    #'Pretrain_data_200k.csv': 'CVE-Description',
    'Pretrain_data_100k.csv': 'CVE-Description'
}

# Hyperparameters
MAX_LENGTH = 512
MASK_PROB = 0.15
REPLACE_MASK_PROB = 0.8
RANDOM_REPLACE_PROB = 0.1
KEEP_ORIGINAL_PROB = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 5e-4
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3

# Read data function
def read_data(file_path, column_name):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    
    # Display file info
    print(f"File: {file_path}")
    print("Columns:", df.columns)
    print("Number of rows:", len(df))

    return df[column_name].dropna().tolist()  # Remove any NaN values

# Load data
texts = []
for file_path, column_name in files_columns.items():
    texts.extend(read_data(file_path, column_name))

# Confirm the number of loaded texts
print(f"Total loaded texts: {len(texts)}")

# Create masked LM inputs
def create_masked_lm_inputs(texts, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROB, replace_mask_prob=REPLACE_MASK_PROB, random_replace_prob=RANDOM_REPLACE_PROB, keep_original_prob=KEEP_ORIGINAL_PROB):
    pairs = []
    for text in texts:
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

        # Random masking
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)
        special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, replace_mask_prob)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Random replacement
        indices_random = torch.bernoulli(torch.full(labels.shape, random_replace_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
        valid_random_words = ~torch.tensor(tokenizer.get_special_tokens_mask(random_words.tolist(), already_has_special_tokens=True), dtype=torch.bool)
        input_ids[indices_random & valid_random_words] = random_words[indices_random & valid_random_words]

        pairs.append((input_ids, labels, attention_mask))

    return pairs

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pairs = create_masked_lm_inputs(texts, tokenizer)
print(f"Created MLM inputs: {len(pairs)}")

# Create dataset
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
print("Dataset created.")

# Split dataset into training and validation sets
train_size = int(0.8 * len(pairs))
train_pairs = pairs[:train_size]
eval_pairs = pairs[train_size:]

train_dataset = CustomTextDataset(train_pairs)
eval_dataset = CustomTextDataset(eval_pairs)
print("Train and validation datasets created.")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Set training parameters
total_steps = NUM_EPOCHS * len(train_loader)
warmup_steps = total_steps // 10  # Warm-up steps for learning rate

# Initialize BERT MLM model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
model.train()

# Initialize best loss and early stopping
best_loss = float('inf')
patience_counter = 0

# Start training
print("Training started.")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
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
    print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss}")

    # Evaluate
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
    print(f"Epoch {epoch + 1} Average Validation Loss: {avg_eval_loss}")

    # Save model at the end of each epoch
    save_directory = f'pretrain_model_storage/best_pre-train-cve-2e-5-all-0714/epoch_{epoch + 1}'
    model_path = os.path.join(save_directory, 'pytorch_model.bin')
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(save_directory)

    config_path = os.path.join(save_directory, 'config.json')
    model.config.to_json_file(config_path)
    print(f"Model weights and config saved to {model_path} and {config_path}")

    # Check for early stopping
    if avg_eval_loss < best_loss:
        best_loss = avg_eval_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at Epoch {epoch + 1}")
        break

print("Training completed.")
