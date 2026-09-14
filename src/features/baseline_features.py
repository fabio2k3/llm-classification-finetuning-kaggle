"""
Fase 1 — Baseline LightGBM.

Objetivo de esta fase: NO ganar la competición, sino:
  1. Validar que el pipeline completo (features -> train -> OOF -> submission) funciona.
  2. Tener un número de referencia (log loss) contra el que medir Fase 2 y 3.
  3. Detectar señal obvia (verbosity bias, etc.) antes de gastar cómputo en LLMs.

Uso:
    python -m src.models.baseline_lgbm
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from src.data.load_data import load_config, load_train, load_test, make_target_label
from src.data.cv_split import add_fold_column
from src.features.baseline_features import build_feature_matrix
from src.make_submission import build_submission
from src.models.inspect_baseline import print_feature_importance


def train_baseline():
    cfg = load_config()
    train = load_train(cfg)
    test = load_test(cfg)

    target = make_target_label(train)
    train["target"] = target

    # CV split (GroupKFold sobre prompt, definido en config.yaml)
    train = add_fold_column(train, cfg, target=target)

    # Features (fit del TF-IDF sobre train, se reusa el mismo vectorizer en test)
    print("Construyendo features de train...")
    train_feat, feature_cols, vectorizer = build_feature_matrix(train)

    print("Construyendo features de test...")
    test_feat, _, _ = build_feature_matrix(test, vectorizer=vectorizer)

    print(f"\n{len(feature_cols)} features usadas: {feature_cols}\n")

    oof_preds = np.zeros((len(train_feat), 3))
    test_preds = np.zeros((len(test_feat), 3))
    n_folds = cfg["data"]["n_folds"]

    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": cfg["project"]["seed"],
        "verbosity": -1,
    }

    for fold in range(n_folds):
        print(f"\n===== Fold {fold} =====")
        tr_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        X_tr, y_tr = train_feat.loc[tr_idx, feature_cols], train_feat.loc[tr_idx, "target"]
        X_val, y_val = train_feat.loc[val_idx, feature_cols], train_feat.loc[val_idx, "target"]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        oof_preds[val_idx.values] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(test_feat[feature_cols],
                                     num_iteration=model.best_iteration) / n_folds

        fold_logloss = log_loss(y_val, oof_preds[val_idx.values])
        print(f"Fold {fold} log loss: {fold_logloss:.5f}")

    overall_logloss = log_loss(train_feat["target"], oof_preds)
    print(f"\n{'='*50}")
    print(f"OOF LOG LOSS TOTAL: {overall_logloss:.5f}")
    print(f"{'='*50}\n")

    # Guardar OOF para análisis posterior (comparar contra Fase 2/3, hacer ensemble luego)
    oof_df = pd.DataFrame(oof_preds, columns=["oof_a", "oof_b", "oof_tie"])
    oof_df["id"] = train_feat["id"].values
    oof_df["target"] = train_feat["target"].values
    oof_df.to_csv(f"{cfg['paths']['outputs_dir']}/fase1_baseline_oof.csv", index=False)

    # Submission
    build_submission(test_feat["id"], test_preds, cfg, filename="fase1_baseline_lgbm_v1.csv")

    # Diagnóstico: qué features está usando el modelo (del último fold entrenado)
    print_feature_importance(model, feature_cols)

    return model, overall_logloss


if __name__ == "__main__":
    train_baseline()