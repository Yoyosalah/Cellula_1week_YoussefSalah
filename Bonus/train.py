import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from preprocessor import Preprocessor
from model import get_model

BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
MAX_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# a pytorch wrapper for dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
filepath = r'/home/youssef-salah/Cellula/NLP/week 1/cellula toxic data.csv'
df = pd.read_csv(filepath)
df = df.drop_duplicates()

X = df['query']
y = df['Toxic Category']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))

# Preprocessing
preprocessor = Preprocessor(max_len=MAX_LEN)
preprocessor.fit_labels(y_train.tolist())

train_encodings = preprocessor.tokenize(X_train.tolist())
val_encodings   = preprocessor.tokenize(X_val.tolist())
test_encodings  = preprocessor.tokenize(X_test.tolist())

y_train_enc = preprocessor.encode_labels(y_train.tolist())
y_val_enc   = preprocessor.encode_labels(y_val.tolist())
y_test_enc  = preprocessor.encode_labels(y_test.tolist())

num_classes = preprocessor.num_labels()
print("Number of classes:", num_classes)

# Data loading
train_dataset = TextDataset(train_encodings, y_train_enc)
val_dataset   = TextDataset(val_encodings, y_val_enc)
test_dataset  = TextDataset(test_encodings, y_test_enc)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model
model = get_model(num_labels=num_classes,device=DEVICE)
optimizer = AdamW(model.parameters(),lr=LR)
class_weights = compute_class_weight(class_weight="balanced",classes=np.unique(y_train_enc),y=y_train_enc)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
train_loss = []
val_loss = []
val_f1_score = []

for epoch in range(EPOCHS):

    model.train()
    total_train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = criterion(outputs.logits, batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    # Evaluation on Validation data
    model.eval()
    total_val_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            loss = criterion(outputs.logits, batch["labels"])
            total_val_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss.append(avg_val_loss)

    val_f1 = f1_score(true_labels, preds, average="macro")
    val_f1_score.append(val_f1)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Macro F1: {val_f1:.4f}")

# Evaluation on Test data
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        predictions = torch.argmax(outputs.logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

macro_f1 = f1_score(true_labels, preds, average="macro")
accuracy = accuracy_score(true_labels, preds)

print("\nTest Accuracy:", accuracy)
print("Test Macro F1:", macro_f1)

print("\nClassification Report:\n")
print(classification_report(true_labels,preds,target_names=preprocessor.label_encoder.classes_))


# Base directories for saving
base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
models_dir = os.path.join(base_dir, "saved_model")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,annot=True,fmt="d",xticklabels=preprocessor.label_encoder.classes_, yticklabels=preprocessor.label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "confusion_matrix.png"))
plt.show()

# Training Plots
plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(images_dir, "loss_curve.png"))
plt.show()

plt.figure()
plt.plot(val_f1_score, label="Val Macro F1")
plt.legend()
plt.title("Validation F1 Curve")
plt.savefig(os.path.join(images_dir, "val_f1_curve.png"))
plt.show()

# Saving model and label encoder
model.save_pretrained(os.path.join(models_dir, "lora_model"))
torch.save(preprocessor.label_encoder, os.path.join(models_dir, "label_encoder.pt"))
preprocessor.tokenizer.save_pretrained(os.path.join(models_dir, "tokenizer"))

print("\nModel, tokenizer, and label encoder saved successfully.")