import os
import glob

# 1. Find the local CUDA directory installed by pip
cuda_paths = glob.glob(os.path.join(
    os.environ.get('VIRTUAL_ENV', '/home/youssef-salah/Cellula/NLP/nlp_venv'), 
    'lib/python*/site-packages/nvidia/cuda_nvcc'
))

# 2. Tell TensorFlow XLA exactly where it is
if cuda_paths:
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_paths[0]}"
    print(f"✅ Automatically linked XLA to: {cuda_paths[0]}")
else:
    print("⚠️ Could not find local CUDA path. XLA might fail.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocessing import Preprocessor
from model import build_model, compute_class_weights

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

filepath = r'/home/youssef-salah/Cellula/NLP/week 1/cellula toxic data.csv'
df = pd.read_csv(filepath)
df = df.drop_duplicates()

X = df['query']
y = df['Toxic Category']

# Split
X_train, X_tmp, y_train, y_tmp = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X,y,test_size=0.5,stratify=y,random_state=42)

print('Train size:', len(X_train))
print('Train size:', len(X_val))
print('Test size:', len(X_test))

# Preprocessing
preprocessor = Preprocessor()
X_train_seq, y_train_enc = preprocessor.fit_transform(X_train.tolist(), y_train.tolist())

X_val_seq = preprocessor.transform_texts(X_val.tolist())
y_val_enc = preprocessor.transform_labels(y_val.tolist())

X_test_seq = preprocessor.transform_texts(X_test.tolist())
y_test_enc = preprocessor.transform_labels(y_test.tolist())

vocab_size = len(preprocessor.tokenizer.word_index)
num_classes = len(np.unique(y_train_enc))
print(f"Vocab Size: {vocab_size} | Classes: {num_classes}")

# Model
model = build_model(
    vocab_size=vocab_size,
    embedding_dim=100,
    max_len=preprocessor.max_len,
    num_classes=num_classes,
    lstm_units=64,
    bi=True,
    dropout=0.3  
)

class_weights = compute_class_weights(y_train_enc)
print("Class weights:", class_weights)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode='min'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    )
]

# Train
tf.debugging.set_log_device_placement(True)

history = model.fit(
    X_train_seq,
    y_train_enc,
    validation_data=(X_val_seq, y_val_enc), 
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

y_pred_probs = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_probs, axis=1)


# Metrics
macro_f1 = f1_score(y_test_enc, y_pred, average="macro")

print("\nMacro F1 Score:", macro_f1)
print("\nClassification Report:\n")
print(classification_report(y_test_enc,y_pred,target_names=preprocessor.label_encoder.classes_))


# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=preprocessor.label_encoder.classes_,
    yticklabels=preprocessor.label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# Training Plots
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("loss_curve.png")
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("accuracy_curve.png")
plt.show()

# Saving Model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/bilstm_model.h5")
import pickle
with open("saved_model/tokenizer.pkl", "wb") as f:
    pickle.dump(preprocessor.tokenizer, f)

with open("saved_model/label_encoder.pkl", "wb") as f:
    pickle.dump(preprocessor.label_encoder, f)

print("\nModel and preprocessors saved successfully.")







