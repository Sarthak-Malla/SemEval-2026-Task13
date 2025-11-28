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
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warn("Warning: wandb not installed. Install with: pip install wandb")

# logger.info the GPU available
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(torch.cuda.get_device_properties(i))

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
    embed_batch_size: int = 128 # this is used for the undersampling (embedding)
    gradient_accumulation_steps: int = 2
    epochs: int = 5
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2
    
    # optimization strategies:
    # pooling strategy
    pooling_strategy: str = "mean"

    # focal loss function
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0

    num_labels: int = 11
    max_samples_per_class: int = 20_000 
    use_class_weights: bool = False

    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    early_stopping_patience: int = 5
    save_total_limit: int = 2

    seed: int = 42
    fp16: bool = True
    num_workers: int = 4
    num_procs: int = 16

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean_pool(hidden, mask):
    """Mean pooling over sequence dimension"""
    mask = mask.unsqueeze(-1)
    masked = hidden * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    return summed / lengths

class AttentionPooling(nn.Module):
    """Learnable attention-based pooling"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden, mask):
        # hidden: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len)
        
        # compute attention scores
        attn_scores = self.attention(hidden).squeeze(-1)  # (batch, seq_len)
        
        # mask out padding positions
        attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)
        
        # weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden).squeeze(1)  # (batch, hidden_dim)
        
        return pooled

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Weights for each class.
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) - logits
        # targets: (N) - class labels
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CodeT5Config(PretrainedConfig):
    model_type = "codet5_classifier"
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m",
        num_labels: int = 11,
        pooling_strategy: str = "mean",
        use_focal_loss: bool = False,
        focal_loss_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
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
        
        # initialize pooling layer
        self.pooling_strategy = config.pooling_strategy
        if self.pooling_strategy == "attention":
            self.attention_pool = AttentionPooling(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(dim),
            nn.Linear(dim, config.num_labels),
        )

        # register class weights as buffer if provided
        weights_tensor = None
        if config.class_weights is not None:
            # convert list back to tensor if needed
            if isinstance(config.class_weights, list):
                weights_tensor = torch.FloatTensor(config.class_weights)
            else:
                weights_tensor = config.class_weights
            self.register_buffer('class_weights', weights_tensor)
        else:
            self.class_weights = None
        
        self.loss_fct = None
        if config.use_focal_loss:
            self.loss_fct = FocalLoss(
                alpha=weights_tensor, # passing class weights as alpha
                gamma=config.focal_loss_gamma
            )

    def pool_hidden_states(self, hidden, mask):
        """Apply the configured pooling strategy"""
        if self.pooling_strategy == "mean":
            return mean_pool(hidden, mask)
        elif self.pooling_strategy == "attention":
            return self.attention_pool(hidden, mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

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
        
        # apply pooling strategy
        pooled = self.pool_hidden_states(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # switch between Focal Loss and Cross Entropy
            if self.config.use_focal_loss:
                # The FocalLoss class handles device placement automatically 
                # but we ensure weights are on the right device if not using buffer
                if self.loss_fct.alpha is not None and self.loss_fct.alpha.device != logits.device:
                     self.loss_fct.alpha = self.loss_fct.alpha.to(logits.device)
                
                loss = self.loss_fct(logits, labels)
            else:
                # standard Cross Entropy
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

        # this is a duplicate for the data preprocess to work
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

        global GLOBAL_TOKENIZER, GLOBAL_MAX_LENGTH
        GLOBAL_TOKENIZER = tokenizer
        GLOBAL_MAX_LENGTH = cfg.max_length

        # initialize WandB
        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or f"codet5-task{cfg.task_subset}-attention-focal",
                config=asdict(cfg),
                reinit=True,
            )
            logger.info("WandB initialized successfully")
        elif cfg.use_wandb and not WANDB_AVAILABLE:
            logger.warn("Warning: WandB requested but not available. Install with: pip install wandb")
            cfg.use_wandb = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize a basic version of the model early to use for embedding extraction
        # this will be re-initialized properly in initialize_model() later
        temp_config = CodeT5Config(model_name=self.cfg.model_name, pooling_strategy=self.cfg.pooling_strategy)
        self.model = CodeT5Classifier(temp_config)
        self.model.to(self.device)

        self.trainer = None
        self.train_labels = None
    
    def _select_diverse_indices(self, embeddings: np.ndarray, n: int) -> np.ndarray:
        """Greedy max-min (farthest-point) sampling over cosine distance."""
        N = embeddings.shape[0]
        if n >= N:
            return np.arange(N)

        emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        rng = np.random.default_rng(self.cfg.seed)
        # start from a random point
        first_idx = int(rng.integers(0, N))
        selected = [first_idx]

        # min_dist[i] = min distance from point i to any selected point
        sims = emb @ emb[first_idx]          # (N,)
        min_dist = 1.0 - sims                # cosine distance

        min_dist[first_idx] = -1.0
        for _ in range(1, n):
            # pick the point that is farthest from the current selected set
            next_idx = int(np.argmax(min_dist))
            selected.append(next_idx)

            sims = emb @ emb[next_idx]       # cosine sim to new point
            dist = 1.0 - sims
            min_dist = np.minimum(min_dist, dist)

            min_dist[next_idx] = -1.0

        return np.array(selected, dtype=int)
    
    def _compute_embeddings(self, codes: list, batch_size: int = 32) -> np.ndarray:
        """Generates pooled embeddings using the CodeT5Classifier's encoder."""
        self.model.eval() # set model to evaluation mode
        all_embeddings = []
        
        logger.info(f"Computing embeddings for {len(codes)} samples using {self.cfg.model_name}...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(codes), batch_size), desc="Embedding"):
                batch_codes = codes[i : i + batch_size]
                
                inputs = self.tokenizer(
                    batch_codes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.encoder(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                pooled_output = self.model.pool_hidden_states(
                    outputs.last_hidden_state, 
                    inputs['attention_mask']
                )
                
                all_embeddings.append(pooled_output.cpu().numpy())

                if i == 2:
                    breakpoint()

        torch.cuda.empty_cache()
        
        return np.concatenate(all_embeddings, axis=0)

    def load_data(self):
        """Load dataset, split first, then balance ONLY the training split."""
        logger.info(f"Loading dataset subset {self.cfg.task_subset}...")
        
        try:
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.cfg.task_subset)
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} training samples")

            df = train_data.to_pandas()
            logger.info(f"Dataset columns: {df.columns.tolist()}")

            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            df = df.dropna(subset=['code', 'label'])
            df['label'] = df['label'].astype(int)

            logger.info("\nOriginal label distribution (full data):")
            logger.info(df['label'].value_counts().sort_index())

            # split the data before balancing
            train_df, val_df = train_test_split(
                df,
                test_size=0.2,
                stratify=df['label'],
                random_state=self.cfg.seed,
            )

            logger.info(f"\nSplit sizes -> Train: {len(train_df)}, Val: {len(val_df)}")
            logger.info("Train label distribution (before balancing):")
            logger.info(train_df['label'].value_counts().sort_index())
            logger.info("Val label distribution (kept original):")
            logger.info(val_df['label'].value_counts().sort_index())

            # we only balance the training data
            label_counts = train_df['label'].value_counts()
            target_n = label_counts.min()  # smallest class size in TRAIN only
            logger.info(f"\nSmallest TRAIN class size: {target_n} (target per-class sample size)")
            logger.info(f"\nWe will use: {target_n} x 11 (target per-class sample size)")
            target_n *= 11

            balanced_parts = []

            for label, df_label in train_df.groupby('label'):
                n_samples = len(df_label)
                logger.info(f"\nProcessing label {label} in TRAIN: {n_samples} samples")

                if n_samples <= target_n:
                    logger.info(f"  Using all {n_samples} samples (no undersampling needed).")
                    balanced_parts.append(df_label)
                    continue

                codes = df_label['code'].tolist()
                logger.info(f"  Computing embeddings for {len(codes)} samples...")
                embeddings = self._compute_embeddings(
                    codes,
                    batch_size=getattr(self.cfg, "embed_batch_size", 128),
                )

                logger.info(f"  Selecting {target_n} diverse samples...")
                idxs = self._select_diverse_indices(embeddings, target_n)
                df_selected = df_label.iloc[idxs].reset_index(drop=True)
                balanced_parts.append(df_selected)

            train_balanced_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

            logger.info(
                f"\nBalanced TRAIN dataset from {len(train_df)} to {len(train_balanced_df)} samples"
            )
            logger.info("New TRAIN label distribution:")
            logger.info(train_balanced_df['label'].value_counts().sort_index())

            # store labels for class weight calculation (use BALANCED train)
            self.train_labels = train_balanced_df['label'].values

            train = Dataset.from_pandas(
                train_balanced_df[['code', 'label']],
                preserve_index=False,
            )
            val = Dataset.from_pandas(
                val_df[['code', 'label']],
                preserve_index=False,
            )

            self.cfg.num_labels = train_balanced_df['label'].nunique()

            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.config.update({
                    "original_samples": len(df),
                    "train_original": len(train_df),
                    "train_balanced": len(train_balanced_df),
                    "val_samples": len(val_df),
                    "target_per_class_train": int(target_n),
                })

            return train, val

        except Exception as e:
            logger.info(f"Error loading dataset: {e}")
            raise

    def preprocess_datasets(self, train, val):
        """Tokenize and prepare datasets"""
        logger.info("Tokenizing datasets...")
        
        def tokenize_function(examples):
            return GLOBAL_TOKENIZER(
                examples['code'],
                truncation=True,
                # max_length=self.cfg.max_length,
                max_length=GLOBAL_MAX_LENGTH,
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
        
        logger.info("Tokenization complete")
        return train, val

    def initialize_model(self):
        """Initialize model with optional class weights"""
        logger.info(f"Initializing model with {self.cfg.pooling_strategy} pooling...")
        
        class_weights = None
        if self.cfg.use_class_weights and self.train_labels is not None:
            logger.info("Computing class weights...")
            weights = compute_class_weight(
                'balanced',
                classes=np.arange(self.cfg.num_labels),
                y=self.train_labels
            )
            class_weights = torch.FloatTensor(weights)
            logger.info(f"Class weights: {class_weights.numpy()}")
            
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"class_weight_{i}": float(w) 
                    for i, w in enumerate(weights)
                })
        
        model_config = CodeT5Config(
            model_name=self.cfg.model_name,
            num_labels=self.cfg.num_labels,
            pooling_strategy=self.cfg.pooling_strategy,
            class_weights=class_weights,
            use_focal_loss=self.cfg.use_focal_loss,
            focal_loss_gamma=self.cfg.focal_loss_gamma
        )
        
        # re-initialize the model to ensure it has the final correct config
        self.model = CodeT5Classifier(model_config)
        self.model.to(self.device)
        logger.info(f"Model initialized with {self.cfg.num_labels} labels")

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
        logger.info("Starting training...")
        
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
        
        logger.info(f"Training completed. Model saved to {self.cfg.output_dir}")
        
        return self.trainer

    def evaluate_model(self, val_dataset):
        """Evaluate model and save metrics"""
        logger.info("Evaluating model...")
        
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
        
        logger.info(f"\nFinal Results:")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
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
            
            logger.info("\nPipeline completed successfully!")
            return self.trainer
            
        except Exception as e:
            logger.info(f"Error in pipeline: {e}")
            if self.cfg.use_wandb and WANDB_AVAILABLE:
                wandb.finish(exit_code=1)
            raise

def parse_args():
    p = argparse.ArgumentParser(
        description='Train CodeT5 Classifier with Configurable Pooling'
    )
    p.add_argument("--task", choices=['A', 'B', 'C'], default='B', 
                   help='Task subset to use')
    p.add_argument("--output_dir", default="./outputs", 
                   help='Output directory')

    # train params
    p.add_argument("--epochs", type=int, default=5, 
                   help='Number of training epochs')
    p.add_argument("--batch_size", type=int, default=32, 
                   help='Batch size per device')
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help='Gradient accumulation steps')
    p.add_argument("--lr", type=float, default=5e-5, 
                   help='Learning rate')
    p.add_argument("--max_length", type=int, default=256, 
                   help='Maximum sequence length')
    
    # pooling strategy argument
    p.add_argument("--pooling", choices=['mean', 'attention'], default='mean',
                   help='Pooling strategy: mean or attention')

    # focal loss args
    p.add_argument("--use_focal_loss", action='store_true',
                   help='Enable Focal Loss instead of Cross Entropy')
    p.add_argument("--focal_loss_gamma", type=float, default=2.0,
                   help='Gamma parameter for Focal Loss (focusing factor)')
    
    p.add_argument("--max_samples_per_class", type=int, default=20_000,
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
        pooling_strategy=args.pooling,
        use_focal_loss=args.use_focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
        max_samples_per_class=args.max_samples_per_class,

        use_class_weights=not args.no_class_weights,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )

    logger.info("="*60)
    logger.info("Training Configuration:")
    logger.info("="*60)
    for key, value in asdict(cfg).items():
        logger.info(f"{key:30s}: {value}")
    logger.info("="*60)

    trainer = CodeOriginTrainer(cfg)
    trainer.run_full_pipeline()