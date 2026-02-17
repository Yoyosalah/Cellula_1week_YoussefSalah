from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def build_model(vocab_size, embedding_dim, max_len, num_classes, lstm_units=64,bi = True,dropout=0.2):
    if bi:
        lstm = Bidirectional(LSTM(lstm_units,dropout= dropout))
    else:
        lstm = LSTM(lstm_units,dropout= dropout)

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim= vocab_size+1, output_dim= embedding_dim),
        Dropout(dropout),
        lstm,
        # Dense(32, activation='relu')
        Dense(num_classes,activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def compute_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

