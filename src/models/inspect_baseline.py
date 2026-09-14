"""
Snippet rápido de diagnóstico: feature importance del baseline.
Pégalo al final de train_baseline() en baseline_lgbm.py, justo antes del
"return model, overall_logloss", o córrelo aparte reentrenando un modelo simple.

Uso rápido (agrega esto dentro de src/models/baseline_lgbm.py, después del loop de folds):
"""
import matplotlib.pyplot as plt
import pandas as pd


def print_feature_importance(model, feature_cols, top_n: int = 15):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features por importancia (gain):")
    print(importance.head(top_n).to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"].head(top_n)[::-1],
              importance["importance"].head(top_n)[::-1])
    plt.title("Feature importance (gain) - Fase 1 baseline")
    plt.tight_layout()
    plt.savefig("outputs/fase1_feature_importance.png", dpi=120)
    print("\nGráfico guardado en outputs/fase1_feature_importance.png")

    return importance