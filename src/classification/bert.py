"""Baseline transformer classifier: fine-tunes DistilBERT (distilbert-base-uncased, max 128 tokens). Compared with bert_large.py."""
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# check metrics while training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert_model(data_dir, model_dir):
    print("--- 1. Loading Data ---")
    # load data
    train_df = pd.read_pickle(os.path.join(data_dir, "processed/train_data.pkl"))
    val_df = pd.read_pickle(os.path.join(data_dir, "processed/val_data.pkl"))
    
    # BERT 
    print("Using 'text_bert' column (full text) for BERT model.")
    train_texts = train_df['text_bert'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['text_bert'].tolist()
    val_labels = val_df['label'].tolist()

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    print("\n--- 2. Tokenization ---")
    # load tokenizer for DistilBERT
    model_name = "distilbert-base-uncased" 
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    class GenderDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = GenderDataset(train_encodings, train_labels)
    val_dataset = GenderDataset(val_encodings, val_labels)

    print("\n--- 3. Initializing Model ---")
    # load DistilBERT with Classification Head
    # num_labels=2 - Hard AI / Soft AI
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device) 

    #parameters 
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "bert_results"), 
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    print("\n--- 4. Training ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics, 
    )

    trainer.train()

    print("\n--- 5. Saving Final Model ---")
    # save model
    save_path = os.path.join(model_dir, "bert_finetuned")
    
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved successfully to {save_path}")

def main():
  
  DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
  MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"
  train_bert_model(DATA_DIR, MODEL_DIR)
