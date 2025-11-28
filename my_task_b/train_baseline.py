import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    PreTrainedModel,
    PretrainedConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# configs for the trainer setup
@dataclass
class TrainingConfig:
    task_subset: str = 'B'
    output_dir: str = "./outputs"
    
    # wandb setup
    wandb_project: str = "who-wrote-this-code"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    model_name: str = "Salesforce/codet5p-220m"

    # settings to play around with
    max_length: int = 256
    batch_size: int = 32
    gradient_accumulation_steps: int = 2  # effective batch = 64
    epochs: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  

    num_labels: int = 11
    max_samples_per_class: int = 20_000 
    use_class_weights: bool = True

    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    early_stopping_patience: int = 5
    save_total_limit: int = 2

    seed: int = 42
    fp16: bool = True
    num_workers: int = 4
    num_procs: int = 8

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    masked = hidden * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    return summed / lengths


class CodeT5Config(PretrainedConfig):
    model_type = "codet5_classifier"
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m",
        num_labels: int = 11,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        # Convert tensor to list for JSON serialization
        if class_weights is not None and isinstance(class_weights, torch.Tensor):
            self.class_weights = class_weights.tolist()
        else:
            self.class_weights = class_weights

class CodeT5Classifier(PreTrainedModel):
    def __init__(self, config: CodeT5Config):
        super().__init__(config)
        self.config = config

        self.encoder = T5EncoderModel.from_pretrained(config.model_name)
        dim = self.encoder.config.d_model

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(dim),
            nn.Linear(dim, config.num_labels),
        )

        # register class weights as buffer if provided
        if config.class_weights is not None:
            # Ccnvert list back to tensor if needed
            if isinstance(config.class_weights, list):
                weights_tensor = torch.FloatTensor(config.class_weights)
            else:
                weights_tensor = config.class_weights
            self.register_buffer('class_weights', weights_tensor)
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss = F.cross_entropy(
                    logits, 
                    labels, 
                    weight=self.class_weights.to(logits.device)
                )
            else:
                loss = F.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class CodeOriginTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

        # initialize WandB
        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or f"codet5-task{cfg.task_subset}",
                config=asdict(cfg),
                reinit=True,
            )
            print("WandB initialized successfully")
        elif cfg.use_wandb and not WANDB_AVAILABLE:
            print("Warning: WandB requested but not available. Install with: pip install wandb")
            cfg.use_wandb = False

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = None
        self.trainer = None
        self.train_labels = None

    def load_data(self):
        """Load and balance dataset"""
        print(f"Loading dataset subset {self.cfg.task_subset}...")
        
        try:
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.cfg.task_subset)
            train_data = dataset['train']
            print(f"Loaded {len(train_data)} training samples")
            
            df = train_data.to_pandas()
            
            print(f"Dataset columns: {df.columns.tolist()}")
            
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")
            
            df = df.dropna(subset=['code', 'label'])
            df['label'] = df['label'].astype(int)
            
            print(f"\nOriginal label distribution:")
            print(df['label'].value_counts().sort_index())
            
            # balance the dataset by undersampling majority classes
            max_samples = self.cfg.max_samples_per_class
            df_balanced = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max_samples), random_state=self.cfg.seed)
            ).reset_index(drop=True)
            
            print(f"\nBalanced dataset from {len(df)} to {len(df_balanced)} samples")
            print(f"New label distribution:")
            print(df_balanced['label'].value_counts().sort_index())
            
            self.cfg.num_labels = df_balanced['label'].nunique()
            
            # Stratified train-val split
            train_df, val_df = train_test_split(
                df_balanced,
                test_size=0.2,
                stratify=df_balanced['label'],
                random_state=self.cfg.seed
            )
            
            print(f"\nTrain samples: {len(train_df)}, Val samples: {len(val_df)}")
            print(f"Train label distribution:")
            print(train_df['label'].value_counts().sort_index())
            
            # store labels for class weight calculation
            self.train_labels = train_df['label'].values
            
            # convert to HuggingFace datasets
            train = Dataset.from_pandas(train_df[['code', 'label']], preserve_index=False)
            val = Dataset.from_pandas(val_df[['code', 'label']], preserve_index=False)
            
            # log to WandB
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.config.update({
                    "original_samples": len(df),
                    "balanced_samples": len(df_balanced),
                    "train_samples": len(train_df),
                    "val_samples": len(val_df),
                })
            
            return train, val
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def preprocess_datasets(self, train, val):
        """Tokenize and prepare datasets"""
        print("Tokenizing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['code'],
                truncation=True,
                max_length=self.cfg.max_length,
            )
        
        train = train.map(
            tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=self.cfg.num_procs,
            desc="Tokenizing train"
        )
        val = val.map(
            tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=self.cfg.num_procs,
            desc="Tokenizing val"
        )
        
        train = train.rename_column('label', 'labels')
        val = val.rename_column('label', 'labels')
        
        print("Tokenization complete")
        return train, val

    def initialize_model(self):
        """Initialize model with optional class weights"""
        print(f"Initializing model...")
        
        class_weights = None
        if self.cfg.use_class_weights and self.train_labels is not None:
            print("Computing class weights...")
            weights = compute_class_weight(
                'balanced',
                classes=np.arange(self.cfg.num_labels),
                y=self.train_labels
            )
            class_weights = torch.FloatTensor(weights)
            print(f"Class weights: {class_weights.numpy()}")
            
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"class_weight_{i}": float(w) 
                    for i, w in enumerate(weights)
                })
        
        model_config = CodeT5Config(
            model_name=self.cfg.model_name,
            num_labels=self.cfg.num_labels,
            class_weights=class_weights,
        )
        
        self.model = CodeT5Classifier(model_config)
        print(f"Model initialized with {self.cfg.num_labels} labels")

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        acc = accuracy_score(labels, predictions)
        p, r, f, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )

        per_class_flat = {}
        for i in range(self.cfg.num_labels):
            per_class_flat[f"class_{i}_precision"] = float(p[i]) if i < len(p) else 0.0
            per_class_flat[f"class_{i}_recall"] = float(r[i]) if i < len(r) else 0.0
            per_class_flat[f"class_{i}_f1"] = float(f[i]) if i < len(f) else 0.0

        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
        }
        
        metrics.update(per_class_flat)

        return metrics

    def train(self, train_dataset, val_dataset):
        """Train the model using HuggingFace Trainer"""
        print("Starting training...")
        
        report_to = []
        if self.cfg.use_wandb and WANDB_AVAILABLE:
            report_to.append("wandb")
        
        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            logging_dir=os.path.join(self.cfg.output_dir, 'logs'),
            logging_steps=self.cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=self.cfg.eval_steps,
            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            learning_rate=self.cfg.lr,
            lr_scheduler_type="linear",
            save_total_limit=self.cfg.save_total_limit,
            fp16=self.cfg.fp16,
            dataloader_num_workers=self.cfg.num_workers,
            seed=self.cfg.seed,
            remove_unused_columns=False,
            report_to=report_to,
            run_name=self.cfg.wandb_run_name if self.cfg.use_wandb else None,
        )
        
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            ]
        )
        
        self.trainer.train()
        
        # saving model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        
        with open(os.path.join(self.cfg.output_dir, "training_config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        
        print(f"Training completed. Model saved to {self.cfg.output_dir}")
        
        return self.trainer

    def evaluate_model(self, val_dataset):
        """Evaluate model and save metrics"""
        print("Evaluating model...")
        
        predictions = self.trainer.predict(val_dataset)
        
        metrics = {
            'test_loss': float(predictions.metrics.get('test_loss', 0)),
            'accuracy': float(predictions.metrics.get('test_accuracy', 0)),
            'macro_f1': float(predictions.metrics.get('test_macro_f1', 0)),
            'macro_precision': float(predictions.metrics.get('test_macro_precision', 0)),
            'macro_recall': float(predictions.metrics.get('test_macro_recall', 0)),
        }
        
        per_class = {}
        for key, value in predictions.metrics.items():
            if key.startswith('test_class_'):
                per_class[key.replace('test_', '')] = value
        
        metrics['per_class'] = per_class
        
        with open(os.path.join(self.cfg.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nFinal Results:")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        if self.cfg.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "final/test_loss": metrics['test_loss'],
                "final/accuracy": metrics['accuracy'],
                "final/macro_f1": metrics['macro_f1'],
                "final/macro_precision": metrics['macro_precision'],
                "final/macro_recall": metrics['macro_recall'],
            })
        
        return predictions

    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        try:
            train, val = self.load_data()
            train, val = self.preprocess_datasets(train, val)
            self.initialize_model()
            self.train(train, val)
            self.evaluate_model(val)
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
            
            print("\nPipeline completed successfully!")
            return self.trainer
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.finish(exit_code=1)
            raise

def parse_args():
    p = argparse.ArgumentParser(
        description='Train CodeT5 Classifier with Optimizations and WandB'
    )
    p.add_argument("--task", choices=['A', 'B', 'C'], default='B', 
                   help='Task subset to use')
    p.add_argument("--output_dir", default="./outputs", 
                   help='Output directory')

    # train params
    p.add_argument("--epochs", type=int, default=10, 
                   help='Number of training epochs')
    p.add_argument("--batch_size", type=int, default=32, 
                   help='Batch size per device')
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help='Gradient accumulation steps')
    p.add_argument("--lr", type=float, default=5e-5, 
                   help='Learning rate')
    p.add_argument("--max_length", type=int, default=256, 
                   help='Maximum sequence length')
    
    p.add_argument("--max_samples_per_class", type=int, default=15_000,
                   help='Maximum samples per class for balancing')
    p.add_argument("--no_class_weights", action='store_true',
                   help='Disable class weights')
    
    p.add_argument("--eval_steps", type=int, default=200, 
                   help='Evaluation interval in steps')
    p.add_argument("--save_steps", type=int, default=200, 
                   help='Save checkpoint interval in steps')
    p.add_argument("--early_stopping_patience", type=int, default=5, 
                   help='Early stopping patience')
    
    p.add_argument("--wandb_project", default="who-wrote-this-code",
                   help='WandB project name')
    p.add_argument("--wandb_entity", default=None,
                   help='WandB entity (username or team)')
    p.add_argument("--wandb_run_name", default=None,
                   help='WandB run name')
    p.add_argument("--no_wandb", action='store_true',
                   help='Disable WandB logging')
    
    p.add_argument("--seed", type=int, default=42, 
                   help='Random seed')
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainingConfig(
        task_subset=args.task,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        max_length=args.max_length,
        max_samples_per_class=args.max_samples_per_class,
        use_class_weights=not args.no_class_weights,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_wandb=True,
        seed=args.seed,
    )

    print("="*60)
    print("Training Configuration:")
    print("="*60)
    for key, value in asdict(cfg).items():
        print(f"{key:30s}: {value}")
    print("="*60)

    trainer = CodeOriginTrainer(cfg)
    trainer.run_full_pipeline()