from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate
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
        Dense(32, activation='relu'),
        Dense(num_classes,activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_dual_input_model(vocab_size, embedding_dim, max_len_query, max_len_image, num_classes, lstm_units=64,bi = True,dropout=0.2):

    query_input = Input(shape=(max_len_query,), name='query_input')
    image_input = Input(shape=(max_len_image,), name='image_input')

    shared_embedding = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim)
    
    emb_query = shared_embedding(query_input)
    emb_image = shared_embedding(image_input)
    emb_query = Dropout(dropout)(emb_query)
    emb_image = Dropout(dropout)(emb_image)

    lstm_query = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_activation='hard_sigmoid'))(emb_query)
    lstm_image = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_activation='hard_sigmoid'))(emb_image)
    merged = Concatenate()([lstm_query, lstm_image])

    x = Dense(32, activation='relu')(merged)
    x = Dropout(dropout)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[query_input, image_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def compute_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

