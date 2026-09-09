"""
Toma predicciones de probabilidad (N, 3) en el orden [a, b, tie] y arma
el submission.csv en el formato exacto que pide sample_submission.csv.

Uso:
    from src.make_submission import build_submission
    build_submission(test_ids, probs, config, filename="baseline_v1.csv")
"""
import numpy as np
import pandas as pd


def build_submission(test_ids: pd.Series, probs: np.ndarray, config: dict,
                      filename: str = "submission.csv") -> pd.DataFrame:
    """
    test_ids: Serie con la columna 'id' de test.csv
    probs: array (N, 3) con columnas en orden [winner_model_a, winner_model_b, winner_tie]
    """
    assert probs.shape[1] == 3, "Se esperan 3 columnas de probabilidad (a, b, tie)"

    submission = pd.DataFrame({
        "id": test_ids,
        "winner_model_a": probs[:, 0],
        "winner_model_b": probs[:, 1],
        "winner_tie": probs[:, 2],
    })

    out_path = f"{config['paths']['outputs_dir']}/{filename}"
    submission.to_csv(out_path, index=False)
    print(f"Submission guardada en: {out_path}")
    print(submission.head())
    return submission
