"""
Detecta si el código está corriendo dentro de un Kaggle Notebook o en local,
y ajusta las rutas de datos/output automáticamente.

Esto es lo que te evita tener que editar config.yaml cada vez que subes
un notebook a Kaggle: el mismo código corre en ambos lados sin cambios.
"""
import os


def is_kaggle_env() -> bool:
    """Kaggle setea esta variable de entorno en todos sus notebooks."""
    return os.path.exists("/kaggle/input")


def resolve_paths(config: dict, competition_slug: str = None) -> dict:
    """
    Sobreescribe config['paths'] según el entorno.
    Local: usa lo que ya está en config.yaml (data/, outputs/, etc.)
    Kaggle: usa /kaggle/input/<competition_slug>/ para datos
            y /kaggle/working/ para outputs (único dir con permiso de escritura).
    """
    config = dict(config)  # copia superficial, no mutar el original
    slug = competition_slug or config["project"]["competition_slug"]

    if is_kaggle_env():
        kaggle_input = f"/kaggle/input/{slug}"
        config["paths"] = {
            **config["paths"],
            "data_dir": kaggle_input,
            "train_csv": f"{kaggle_input}/train.csv",
            "test_csv": f"{kaggle_input}/test.csv",
            "sample_submission": f"{kaggle_input}/sample_submission.csv",
            "outputs_dir": "/kaggle/working",
            "checkpoints_dir": "/kaggle/working/checkpoints",
            "logs_dir": "/kaggle/working/logs",
        }
        os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
        os.makedirs("/kaggle/working/logs", exist_ok=True)
        print(f"[env] Corriendo en Kaggle. Datos desde: {kaggle_input}")
    else:
        print("[env] Corriendo en local. Usando rutas de config.yaml.")

    return config
