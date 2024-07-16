import gdown
import pandas as pd
import random
import time
import warnings
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# File paths
cve_file_path = '2023_CVE_CWE.csv'
all_cwe_file_path = 'All_CWE.csv'
model_path = 'best_pre-model_CVECWECAPEC'
weights_path = os.path.join(model_path, 'pytorch_model.bin')
model_save_path = 'train_model_save_bert'

# Ensure the save path exists
os.makedirs(model_save_path, exist_ok=True)

# Hyperparameters
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
NUM_LABELS = 2
DROPOUT_RATE = 0.1
FREEZE_LAYERS = 6

# Define a text preprocessing function to encode the text
def encode_texts(tokenizer, texts):
    return tokenizer(texts['CVE-Description'].tolist(), texts['CWE-Description'].tolist(),
                     padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')

# Define the BERT model
class CustomBERTModel(nn.Module):
    def __init__(self, bert_model, num_labels=NUM_LABELS):
        super(CustomBERTModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        avg_pooling_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        diff = torch.abs(cls_embeddings - avg_pooling_embeddings)
        mul = cls_embeddings * avg_pooling_embeddings

        combined_features = torch.cat((diff, mul), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

    def freeze_layers(self, num_layers):
        for name, param in self.bert.named_parameters():
            if any(f"layer.{i}." in name for i in range(num_layers)):
                param.requires_grad = False
            else:
                param.requires_grad = True

# Dataset class
class SecurityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Save model function
def save_model(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'best_model.pt')
    tokenizer_path = os.path.join(save_path, 'tokenizer')
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

# Read All_CWE.csv data
try:
    df_cwe = pd.read_csv(all_cwe_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_cwe = pd.read_csv(all_cwe_file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_cwe = pd.read_csv(all_cwe_file_path, encoding='latin1')

# Extract features and remove unnecessary columns
selected_cwe_features = ['CWE-ID', 'CWE-Description']
df_all_cwe = df_cwe[selected_cwe_features].drop_duplicates(subset=['CWE-ID'], keep='first')

print("CWE data processing complete")
print(df_all_cwe.head())
print("Number of rows in df_all_cwe:", df_all_cwe.shape[0])
print("Number of columns in df_all_cwe:", df_all_cwe.shape[1])
print(df_all_cwe.dtypes)

# Read 2023_CVE_CWE.csv data
try:
    df_selected = pd.read_csv(cve_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_selected = pd.read_csv(cve_file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_selected = pd.read_csv(cve_file_path, encoding='latin1')

# Check column names in 2023_CVE_CWE.csv file
print("Column names in 2023_CVE_CWE.csv file:")
print(df_selected.columns)

# Ensure required columns are present
required_columns = ['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description']
missing_columns = [col for col in required_columns if col not in df_selected.columns]
if missing_columns:
    raise KeyError(f"Missing columns: {missing_columns}")

print("Number of rows in df_selected:", df_selected.shape[0])
print("Number of columns in df_selected:", df_selected.shape[1])
print(df_selected.dtypes)

# Create an empty DataFrame
df_mapping = pd.DataFrame(columns=['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Description', 'P/N'])

# Get all CWE-IDs
all_cwe_ids = df_all_cwe['CWE-ID'].unique()
cve_to_cwe_dict = df_selected.set_index('CVE-ID')['CWE-ID'].to_dict()

# Collect all data using a list
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

# Convert 'P' label to 1 and 'N' label to 0
df_mapping['P/N'] = df_mapping['P/N'].replace({'P': 1, 'N': 0})

# Upsample positive samples to match the number of negative samples
positive_samples = df_mapping[df_mapping['P/N'] == 1]
negative_samples = df_mapping[df_mapping['P/N'] == 0]
resampled_positive_samples = resample(positive_samples, replace=True, n_samples=len(negative_samples))
balanced_samples = pd.concat([resampled_positive_samples, negative_samples])

# Confirm the number of positive and negative samples
print("Number of positive samples:", len(balanced_samples[balanced_samples['P/N'] == 1]))
print("Number of negative samples:", len(balanced_samples[balanced_samples['P/N'] == 0]))

# Initialize BERT Tokenizer and Model using a pretrained model
tokenizer = BertTokenizer.from_pretrained(model_path)

# Split the dataset into training and validation sets with a test size of 10%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    balanced_samples[['CVE-Description', 'CWE-Description']], balanced_samples['P/N'], test_size=0.1)

# Display the size of the training and validation sets (including labels)
print("Training set size:", train_texts.shape, "Training labels size:", train_labels.shape)
print("Validation set size:", val_texts.shape, "Validation labels size:", val_labels.shape)

# Display the number of PN labels in the training and validation sets to confirm successful label conversion
print("\nTraining labels PN:\n", train_labels.value_counts())
print("Validation labels PN:\n", val_labels.value_counts())

# Display some samples from the training and validation sets to ensure there are no conversion errors
print("\nTraining set samples:\n", train_texts.head())
print("Validation set samples:\n", val_texts.head())

# Encode training and validation set texts
train_encodings = encode_texts(tokenizer, train_texts)
val_encodings = encode_texts(tokenizer, val_texts)

# Output detailed information about the training and validation encodings
print("Training encodings key:", train_encodings.keys())
print("Validation encodings key:", val_encodings.keys())

# Check the dimensions of training input_ids and attention_mask
print("Training input_ids dimensions:", train_encodings['input_ids'].shape)
print("Training attention_mask dimensions:", train_encodings['attention_mask'].shape)
print("Validation input_ids dimensions:", val_encodings['input_ids'].shape)
print("Validation attention_mask dimensions:", val_encodings['attention_mask'].shape)

config = BertConfig.from_pretrained(os.path.join(model_path, 'config.json'))

# Instantiate the BertModel
bert_base_model = BertModel.from_pretrained(model_path)

# Instantiate your custom model and load the filtered weights
model = CustomBERTModel(bert_base_model)
pretrained_weights = torch.load(weights_path)
filtered_weights = {k[5:]: v for k, v in pretrained_weights.items() if k.startswith("bert.")}
model.bert.load_state_dict(filtered_weights, strict=False)

# Freeze the first few layers
model.freeze_layers(FREEZE_LAYERS)

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the device (GPU or CPU)
model.to(device)

# Training
best_loss = float('inf')
patience_counter = 0
train_losses = []
eval_losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    train_losses.append(average_train_loss)

    model.eval()
    total_eval_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_eval_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_eval_loss = total_eval_loss / len(val_loader)
    eval_losses.append(average_eval_loss)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Epoch {epoch+1}, Average Training Loss: {average_train_loss:.4f}, Average Validation Loss: {average_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    if average_eval_loss < best_loss:
        best_loss = average_eval_loss
        patience_counter = 0
        save_model(model, tokenizer, model_save_path)
        print(f"Model and tokenizer saved to {model_save_path}, current best validation loss: {best_loss:.4f}")
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at Epoch {epoch + 1}")
        break

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Plot loss function with matplotlib
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, eval_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy, Precision, Recall, F1-Score with matplotlib
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
plt.close()

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.close()
