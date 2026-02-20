import regex as re
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast

class Preprocessor:
    def __init__(self, max_len=128):
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.label_encoder = LabelEncoder()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, texts):
        texts = [self.clean_text(t) for t in texts]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

    def fit_labels(self, labels):
        self.label_encoder.fit(labels)

    def encode_labels(self, labels):
        return self.label_encoder.transform(labels)

    def num_labels(self):
        return len(self.label_encoder.classes_)