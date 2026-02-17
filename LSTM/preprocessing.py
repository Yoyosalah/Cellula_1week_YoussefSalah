import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import regex as re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, max_len=15):
        self.max_len = max_len
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.important_words = {
            'not', 'no', 'never', 'cannot', "can't", "won't", "doesn't", "isn't",
            'what', 'why', 'how', 'who', 'which', 'when', 'where', 'whom', 'whose',
            'any', 'all', 'some',
            'must', 'should', 'could', 'would', 'might'
        }
        

        self.stopwords = STOP_WORDS - self.important_words
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s!?]", "", text)

        doc = self.nlp(text)
        cleaned_tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in self.stopwords]

        return cleaned_tokens

    def fit(self, texts, labels):
        texts_cleaned = [" ".join(self.clean_text(t)) for t in texts]
        self.tokenizer.fit_on_texts(texts_cleaned)
        self.label_encoder.fit(labels)

    def transform_texts(self, texts):
        texts_cleaned = [" ".join(self.clean_text(t)) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(texts_cleaned)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post')

    def transform_labels(self, labels):
        return self.label_encoder.transform(labels)

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform_texts(texts), self.transform_labels(labels)
    
if __name__ == "__main__":
    train_queries = [
        "I love this product, it is amazing!",
        "You are an idiot and I hate you.",
        "How do I reset my password?",
        "This is a highly toxic and disgusting comment!!"
    ]
    train_labels = ["Neutral", "Toxic", "Neutral", "Toxic"]

    test_queries = [
        "That is a stupid idea.", 
        "Please help me with my account."
    ]

    preprocessor = Preprocessor(max_len=6)
    print("--- Training Data Phase ---")
    X_train, y_train = preprocessor.fit_transform(train_queries, train_labels)
    
    print("X_train (Padded Sequences):\n", X_train)
    print("y_train (Encoded Labels):", y_train)
    print("Vocabulary Index:", preprocessor.tokenizer.word_index)
    print("\n")

    # 4. Transform Test Data (Unseen)
    print("--- Test Data Phase (Unseen) ---")
    # Note: We ONLY use transform here, NOT fit
    X_test = preprocessor.transform_texts(test_queries)
    
    print("X_test (Should contain <OOV> for unknown words):\n", X_test)

    # 5. Inverse Check
    # Let's see what the numbers actually represent
    sample_seq = X_test[0]
    words = [preprocessor.tokenizer.index_word.get(i, '?') for i in sample_seq if i != 0]
    print(f"\nOriginal test string: '{test_queries[0]}'")
    print(f"Processed tokens: {words}")