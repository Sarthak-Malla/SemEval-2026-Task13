import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeBERTTrainer:
    def __init__(self, task_subset='A', max_length=512, model_name="microsoft/codebert-base", num_procs=8):
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.num_procs = num_procs
        
    def load_and_prepare_data(self):
        logger.info(f"Loading dataset subset {self.task_subset}...")
        
        try:
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)
            
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} training samples")
            
            df = train_data.to_pandas()
            
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Sample data:\n{df.head()}")
            
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")
            
            df = df.dropna(subset=['code', 'label'])
            
            df['label'] = df['label'].astype(int)
            self.num_labels = df['label'].nunique()
            
            logger.info(f"Number of unique labels: {self.num_labels}")
            logger.info(f"Label range: {df['label'].min()} to {df['label'].max()}")
            logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
            
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].reset_index(drop=True)
            val_df = df[train_size:].reset_index(drop=True)
            
            logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
            
            return train_df, val_df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def initialize_model_and_tokenizer(self):
        logger.info(f"Initializing {self.model_name} model and tokenizer...")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        logger.info(f"Model initialized with {self.num_labels} labels")
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def prepare_datasets(self, train_df, val_df):
        logger.info("Preparing datasets for training...")
        
        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])
        
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=self.num_procs
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=self.num_procs
        )
        
        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        return trainer
    
    def evaluate_model(self, trainer, val_dataset):
        logger.info("Evaluating model...")
        
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        logger.info("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        return predictions
    
    def run_full_pipeline(self, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        try:
            train_df, val_df = self.load_and_prepare_data()
            
            self.initialize_model_and_tokenizer()
            
            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
            
            trainer = self.train(
                train_dataset, val_dataset, 
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            self.evaluate_model(trainer, val_dataset)
            
            logger.info("Pipeline completed successfully!")
            return trainer
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train CodeBERT on SemEval-2026-Task13')
    parser.add_argument('--task', choices=['A', 'B', 'C'], default='A', help='Task subset to use')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = CodeBERTTrainer(
        task_subset=args.task,
        max_length=args.max_length
    )
    
    trainer.run_full_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
    
