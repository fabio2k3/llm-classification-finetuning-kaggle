"""
Dataset para fine-tuning de DeBERTa (Fase 2) y, más adelante, LLMs con LoRA (Fase 3).

Formato de entrada elegido:
    [CLS] prompt: {prompt} [SEP] response A: {response_a} [SEP] response B: {response_b} [SEP]

Por qué este formato y no 2 pasadas separadas (prompt+A, prompt+B) con un head comparador:
- Es más simple de implementar y de depurar en la Fase 2.
- Le da al modelo el contexto completo de AMBAS respuestas en una sola pasada,
  lo cual importa porque el juez humano las compara directamente entre sí
  (no las evalúa de forma aislada).
- Es el formato que reportan la mayoría de soluciones públicas de esta competición.

Truncado: si el texto combinado excede max_length, se trunca dando prioridad
a mantener el prompt completo (suele ser corto) y recortando las respuestas
por partes iguales, no solo la última — evita perder siempre el final de response_b.
"""
import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int = 512, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def _build_text(self, row) -> str:
        prompt = str(row["prompt"])[:2000]       # tope duro anti-outliers extremos
        resp_a = str(row["response_a"])[:4000]
        resp_b = str(row["response_b"])[:4000]
        return (
            f"prompt: {prompt}\n\n"
            f"response A: {resp_a}\n\n"
            f"response B: {resp_b}"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._build_text(row)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            item["labels"] = torch.tensor(row["target"], dtype=torch.long)

        return item
