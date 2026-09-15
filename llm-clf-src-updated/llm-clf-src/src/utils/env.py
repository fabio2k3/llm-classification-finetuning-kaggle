"""
Detecta si el código está corriendo dentro de un Kaggle Notebook o en local,
y ajusta las rutas de datos/output automáticamente.

Esto es lo que te evita tener que editar config.yaml cada vez que subes
un notebook a Kaggle: el mismo código corre en ambos lados sin cambios.

Nota: Kaggle tiene dos layouts posibles para /kaggle/input:
  - Clásico:  /kaggle/input/<competition_slug>/
  - Nuevo:    /kaggle/input/competitions/<competition_slug>/
Esta función detecta cuál aplica automáticamente.
"""
import os


def is_kaggle_env() -> bool:
    """Kaggle setea esta variable de entorno en todos sus notebooks."""
    return os.path.exists("/kaggle/input")


def _find_competition_dir(slug: str) -> str:
    """Encuentra la carpeta real del dataset de la competición, sea cual sea el layout."""
    classic_path = f"/kaggle/input/{slug}"
    new_path = f"/kaggle/input/competitions/{slug}"

    if os.path.exists(classic_path):
        return classic_path
    elif os.path.exists(new_path):
        return new_path
    else:
        raise FileNotFoundError(
            f"No se encontró el dataset de la competición ni en '{classic_path}' "
            f"ni en '{new_path}'. Corre `os.listdir('/kaggle/input')` para ver "
            f"qué hay disponible y ajusta manualmente."
        )


def resolve_paths(config: dict, competition_slug: str = None) -> dict:
    """
    Sobreescribe config['paths'] según el entorno.
    Local: usa lo que ya está en config.yaml (data/, outputs/, etc.)
    Kaggle: detecta automáticamente el layout (clásico o nuevo) y ajusta
            las rutas a donde esté realmente el dataset.
    """
    config = dict(config)  # copia superficial, no mutar el original
    slug = competition_slug or config["project"]["competition_slug"]

    if is_kaggle_env():
        kaggle_input = _find_competition_dir(slug)
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
