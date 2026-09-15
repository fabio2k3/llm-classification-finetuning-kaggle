"""
Corre esto UNA VEZ en tu máquina local (con internet) para bajar los pesos
de DeBERTa-v3 y dejarlos listos para subir como Kaggle Dataset.

Uso:
    python scripts/download_model_weights.py --model microsoft/deberta-v3-base

Después de correrlo, vas a tener una carpeta `model_weights/deberta-v3-base/`
con todo lo necesario (tokenizer, config, pesos). Sube ESA carpeta completa
como un nuevo Kaggle Dataset (igual que hiciste con llm-clf-src), y en el
notebook de Kaggle usa esa ruta en vez del nombre de HuggingFace:

    model_path = find_src_dataset(hint="deberta-v3-weights")  # o el hint que uses
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download(model_name: str, output_dir: str):
    print(f"Descargando tokenizer de '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Descargando modelo de '{model_name}' (esto puede tardar varios minutos)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.save_pretrained(output_dir)

    print(f"\nListo. Pesos guardados en: {output_dir}")
    print("Siguiente paso: sube esta carpeta completa a kaggle.com/datasets "
          "(New Dataset -> arrastra la carpeta -> Create).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, default="model_weights/deberta-v3-base")
    args = parser.parse_args()
    download(args.model, args.output_dir)
