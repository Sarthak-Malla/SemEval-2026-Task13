#!/usr/bin/env python3
import os
import torch
import argparse
import logging
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from my_task_b.train import CodeT5Classifier, CodeT5Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codet5_predict")


def load_model_and_tokenizer(model_path, device):
    """
    Loads the trained CodeT5 classifier + tokenizer from disk.
    """
    logger.info(f"Loading model from: {model_path}")

    config = CodeT5Config.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = CodeT5Classifier.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()

    logger.info("Model & tokenizer loaded.")
    return model, tokenizer


def collate_fn(batch, tokenizer, max_length):
    """
    Tokenizes a batch from the streaming dataset.
    """
    codes = [item["code"] for item in batch]
    ids = [item["ID"] for item in batch]

    enc = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc["ids"] = ids
    return enc


@torch.no_grad()
def predict(model_path, parquet_path, output_path,
            max_length=256, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(model_path, device)

    dataset = load_dataset(
        "parquet",
        data_files=parquet_path,
        split="train",
        streaming=True
    )

    first_row = next(iter(dataset))
    if not {"ID", "code"}.issubset(first_row.keys()):
        raise ValueError("Parquet file must contain columns: 'ID', 'code'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
    )

    with open(output_path, "w") as f:
        f.write("ID,label\n")

        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for i, original_id in enumerate(batch["ids"]):
                f.write(f"{original_id},{preds[i]}\n")

    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with CodeT5 authorship classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Folder containing model + tokenizer")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to parquet file with columns: code, ID")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save predictions CSV")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")

    args = parser.parse_args()

    predict(
        args.model_path,
        args.parquet_path,
        args.output_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )
