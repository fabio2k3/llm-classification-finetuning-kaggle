"""
Fase 2 — Fine-tuning de DeBERTa-v3 para clasificación de preferencias.

IMPORTANTE antes de correr en Kaggle:
    Este script asume que los pesos del modelo (tokenizer + config + weights)
    ya están disponibles localmente en `model_path` (un Kaggle Dataset/Model
    que subiste tú, NO se descargan de HuggingFace en tiempo de ejecución).
    Ver scripts/download_model_weights.py para bajarlos en tu máquina local.

Uso:
    python -m src.models.train_deberta --n_folds_to_run 1   # prueba rápida, 1 fold
    python -m src.models.train_deberta --n_folds_to_run 5   # corrida completa
"""
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.data.load_data import load_config, load_train, load_test, make_target_label
from src.data.cv_split import add_fold_column
from src.models.preference_dataset import PreferenceDataset
from src.make_submission import build_submission


def train_one_fold(fold, train_feat, test_feat, cfg, model_cfg, device):
    print(f"\n{'='*20} FOLD {fold} {'='*20}")

    tr_df = train_feat[train_feat["fold"] != fold].reset_index(drop=True)
    val_df = train_feat[train_feat["fold"] == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["model_path"], num_labels=3
    ).to(device)

    train_ds = PreferenceDataset(tr_df, tokenizer, max_length=model_cfg["max_length"])
    val_ds = PreferenceDataset(val_df, tokenizer, max_length=model_cfg["max_length"])
    test_ds = PreferenceDataset(test_feat, tokenizer, max_length=model_cfg["max_length"],
                                 has_labels=False)

    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"] * 2, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=model_cfg["batch_size"] * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"])
    total_steps = len(train_loader) * model_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=model_cfg["fp16"])

    best_val_logloss = np.inf
    best_val_preds = None

    for epoch in range(model_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=model_cfg["fp16"]):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running_loss / (step + 1):.4f}")

        # Validación al final de cada época
        val_logloss, val_preds = evaluate(model, val_loader, val_df, device)
        print(f"Epoch {epoch} — val log loss: {val_logloss:.5f}")

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_val_preds = val_preds
            torch.save(model.state_dict(),
                       f"{cfg['paths']['checkpoints_dir']}/deberta_fold{fold}_best.pt")

    # Inferencia en test con el mejor checkpoint del fold
    model.load_state_dict(
        torch.load(f"{cfg['paths']['checkpoints_dir']}/deberta_fold{fold}_best.pt")
    )
    test_preds = predict(model, test_loader, device)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_logloss, best_val_preds, test_preds


@torch.no_grad()
def evaluate(model, loader, df, device):
    model.eval()
    all_probs = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            logits = model(**batch).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    logloss = log_loss(df["target"], all_probs, labels=[0, 1, 2])
    return logloss, all_probs


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            logits = model(**batch).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main(n_folds_to_run: int = 5, model_path: str = None):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("[aviso] No hay GPU disponible. Esto va a ser MUY lento o va a fallar por tiempo.")

    model_cfg = {
        "model_path": model_path or "microsoft/deberta-v3-base",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "epochs": 2,
        "fp16": device.type == "cuda",
    }

    train = load_train(cfg)
    test = load_test(cfg)
    target = make_target_label(train)
    train["target"] = target
    train = add_fold_column(train, cfg, target=target)

    oof_preds = np.zeros((len(train), 3))
    test_preds = np.zeros((len(test), 3))
    fold_scores = []

    for fold in range(min(n_folds_to_run, cfg["data"]["n_folds"])):
        val_logloss, val_preds, fold_test_preds = train_one_fold(
            fold, train, test, cfg, model_cfg, device
        )
        fold_scores.append(val_logloss)
        val_idx = train["fold"] == fold
        oof_preds[val_idx.values] = val_preds
        test_preds += fold_test_preds / n_folds_to_run

    print(f"\n{'='*50}")
    print(f"Fold scores: {fold_scores}")
    print(f"Promedio: {np.mean(fold_scores):.5f}")
    print(f"{'='*50}")

    oof_df = pd.DataFrame(oof_preds, columns=["oof_a", "oof_b", "oof_tie"])
    oof_df["id"] = train["id"].values
    oof_df["target"] = train["target"].values
    oof_df.to_csv(f"{cfg['paths']['outputs_dir']}/fase2_deberta_oof.csv", index=False)

    build_submission(test["id"], test_preds, cfg, filename="fase2_deberta_v1.csv")

    return fold_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds_to_run", type=int, default=5)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    main(n_folds_to_run=args.n_folds_to_run, model_path=args.model_path)
