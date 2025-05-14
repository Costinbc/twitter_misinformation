import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
src_dir = os.path.join(base_dir, "src")
data_dir = os.path.join(base_dir, "data")
mide22_dir = os.path.join(data_dir, "MiDe22", "dataset", "EN")
folds_dir = os.path.join(mide22_dir, "folds")
checkpoint_dir = os.path.join(src_dir, "bertweet-results")
output_dir = os.path.join(base_dir, "models")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

training_args = TrainingArguments(
    output_dir=os.path.join(checkpoint_dir, "MiDe22"),
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

class Mide22Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+|url", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(mide22_dir, "mide22_en_misinfo_tweets.tsv"), encoding='utf-8', sep="\t")
    df['clean_text'] = df['text'].apply(clean_text)
    label_map = {"True": 0, "False": 1, "Other": 2}
    df['label_id'] = df['label'].map(label_map)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['clean_text'], df['label_id'], test_size=0.2, random_state=42, stratify=df['label_id']
    )

    config = AutoConfig.from_pretrained(os.path.join(output_dir, "bertweet-ANTiVax"), num_labels=3)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "bertweet-ANTiVax"), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir, "bertweet-ANTiVax"),
                                                               config=config, ignore_mismatched_sizes=True)

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

    train_dataset = Mide22Dataset(train_encodings, train_labels.tolist())
    val_dataset = Mide22Dataset(val_encodings, val_labels.tolist())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    checkpoints = [f.path for f in os.scandir(os.path.join(checkpoint_dir, "MiDe22")) if f.is_dir() and "checkpoint" in f.name]

    if checkpoints:
        print("Resuming from latest checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoints found. Starting fresh training...")
        trainer.train()

    model.save_pretrained(os.path.join(output_dir, "bertweet-MiDe22"))
    tokenizer.save_pretrained(os.path.join(output_dir, "bertweet-MiDe22"))

    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    cm = confusion_matrix(y_true, y_pred)

    labels = ['true', 'false', 'other']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix - MiDe22 BERTweet-Base")
    plt.savefig("MiDe22_confusion_matrix.png", dpi=300, bbox_inches='tight')