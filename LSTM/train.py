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
from model import build_model,build_dual_input_model, compute_class_weights

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

filepath = r'/home/youssef-salah/Cellula/NLP/week 1/cellula toxic data.csv'
df = pd.read_csv(filepath)
df = df.drop_duplicates()

X_query = df['query']
X_image = df['image descriptions']
y = df['Toxic Category']

# Split
X_q_train, X_q_tmp, X_i_train, X_i_tmp, y_train, y_tmp = train_test_split(X_query, X_image, y, test_size=0.2, stratify=y, random_state=42)
X_q_val, X_q_test, X_i_val, X_i_test, y_val, y_test = train_test_split(X_q_tmp, X_i_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

print('Train size:', len(X_q_train), len(X_i_train))
print('Train size:', len(X_q_val),len(X_i_val))
print('Test size:', len(X_q_test),len(X_i_test))

# Preprocessing
preprocessor = Preprocessor()
combined_train_texts = X_q_train.tolist() + X_i_train.tolist()

preprocessor.fit(combined_train_texts, y_train.tolist())

X_q_train_seq = preprocessor.transform_texts(X_q_train.tolist())
X_q_val_seq   = preprocessor.transform_texts(X_q_val.tolist())
X_q_test_seq  = preprocessor.transform_texts(X_q_test.tolist())

X_i_train_seq = preprocessor.transform_texts(X_i_train.tolist())
X_i_val_seq   = preprocessor.transform_texts(X_i_val.tolist())
X_i_test_seq  = preprocessor.transform_texts(X_i_test.tolist())

y_train_enc = preprocessor.transform_labels(y_train.tolist())
y_val_enc   = preprocessor.transform_labels(y_val.tolist())
y_test_enc  = preprocessor.transform_labels(y_test.tolist())

vocab_size = len(preprocessor.tokenizer.word_index)
num_classes = len(np.unique(y_train_enc))
print(f"Vocab Size: {vocab_size} | Classes: {num_classes}")

# Model
model = build_dual_input_model(
    vocab_size=vocab_size,
    embedding_dim=256,
    max_len_query=preprocessor.max_len,
    max_len_image=preprocessor.max_len,
    num_classes=num_classes,
    lstm_units=64,
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
    [X_q_train_seq, X_i_train_seq],
    y_train_enc,
    validation_data=([X_q_val_seq, X_i_val_seq], y_val_enc), 
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

y_pred_probs = model.predict([X_q_test_seq, X_i_test_seq])
y_pred = np.argmax(y_pred_probs, axis=1)


# Metrics
macro_f1 = f1_score(y_test_enc, y_pred, average="macro")

print("\nMacro F1 Score:", macro_f1)
print("\nClassification Report:\n")
print(classification_report(y_test_enc,y_pred,target_names=preprocessor.label_encoder.classes_))

base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
models_dir = os.path.join(base_dir, "saved_model")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

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

plt.savefig(os.path.join(images_dir, "confusion_matrix.png"))
plt.show()

# Training Plots
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
# Save to the images folder
plt.savefig(os.path.join(images_dir, "loss_curve.png"))
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
# Save to the images folder
plt.savefig(os.path.join(images_dir, "accuracy_curve.png"))
plt.show()

# Saving Model
# Save to the saved_model folder
model.save(os.path.join(models_dir, "bilstm_model.h5"))
import pickle
with open(os.path.join(models_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(preprocessor.tokenizer, f)

with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(preprocessor.label_encoder, f)

print(f"\nModel and preprocessors saved successfully in: {models_dir}")
print(f"Images saved successfully in: {images_dir}")