"""
Feature engineering para el baseline clásico (Fase 1).

Estas features NO son solo "para pasar el rato": varias de ellas capturan
sesgos conocidos de este dataset (verbosity bias, formato) que después
te sirven también para interpretar qué está aprendiendo el LLM en Fase 3.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _word_count(text: str) -> int:
    return len(str(text).split())


def _code_block_count(text: str) -> int:
    return len(re.findall(r"```", str(text))) // 2


def _has_list(text: str) -> int:
    return int(bool(re.search(r"^\s*[-*\d]+[.\)]\s", str(text), re.MULTILINE)))


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["prompt", "response_a", "response_b"]:
        df[f"{col}_char_len"] = df[col].astype(str).str.len()
        df[f"{col}_word_len"] = df[col].astype(str).apply(_word_count)
        df[f"{col}_code_blocks"] = df[col].astype(str).apply(_code_block_count)
        df[f"{col}_has_list"] = df[col].astype(str).apply(_has_list)
        df[f"{col}_newline_count"] = df[col].astype(str).str.count("\n")

    # Diffs y ratios A vs B: esto es lo que más señal suele dar en el baseline,
    # porque el "verbosity bias" es real en preferencias humanas.
    df["char_len_diff"] = df["response_a_char_len"] - df["response_b_char_len"]
    df["word_len_diff"] = df["response_a_word_len"] - df["response_b_word_len"]
    df["char_len_ratio"] = (df["response_a_char_len"] + 1) / (df["response_b_char_len"] + 1)
    df["code_blocks_diff"] = df["response_a_code_blocks"] - df["response_b_code_blocks"]
    df["has_list_diff"] = df["response_a_has_list"] - df["response_b_has_list"]

    return df


def add_tfidf_similarity_features(df: pd.DataFrame, max_features: int = 5000) -> pd.DataFrame:
    """
    Similitud TF-IDF entre prompt<->response_a y prompt<->response_b.
    Fit sobre todo el corpus disponible en este split para no filtrar
    información de train a test más allá de lo normal en un vectorizador de texto.
    """
    df = df.copy()
    corpus = pd.concat([df["prompt"], df["response_a"], df["response_b"]]).astype(str)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  stop_words="english")
    vectorizer.fit(corpus)

    prompt_vecs = vectorizer.transform(df["prompt"].astype(str))
    resp_a_vecs = vectorizer.transform(df["response_a"].astype(str))
    resp_b_vecs = vectorizer.transform(df["response_b"].astype(str))

    sim_a = np.array([cosine_similarity(prompt_vecs[i], resp_a_vecs[i])[0, 0]
                       for i in range(df.shape[0])])
    sim_b = np.array([cosine_similarity(prompt_vecs[i], resp_b_vecs[i])[0, 0]
                       for i in range(df.shape[0])])

    df["tfidf_sim_prompt_a"] = sim_a
    df["tfidf_sim_prompt_b"] = sim_b
    df["tfidf_sim_diff"] = sim_a - sim_b

    return df, vectorizer


def build_feature_matrix(df: pd.DataFrame, vectorizer=None):
    """Pipeline completo de features para una fila train/test."""
    df = add_length_features(df)
    if vectorizer is None:
        df, vectorizer = add_tfidf_similarity_features(df)
    else:
        # reusar vectorizer ya fiteado (para test set)
        prompt_vecs = vectorizer.transform(df["prompt"].astype(str))
        resp_a_vecs = vectorizer.transform(df["response_a"].astype(str))
        resp_b_vecs = vectorizer.transform(df["response_b"].astype(str))
        sim_a = np.array([cosine_similarity(prompt_vecs[i], resp_a_vecs[i])[0, 0]
                           for i in range(df.shape[0])])
        sim_b = np.array([cosine_similarity(prompt_vecs[i], resp_b_vecs[i])[0, 0]
                           for i in range(df.shape[0])])
        df["tfidf_sim_prompt_a"] = sim_a
        df["tfidf_sim_prompt_b"] = sim_b
        df["tfidf_sim_diff"] = sim_a - sim_b

    excluded_cols = {
        "id", "fold", "target",
        "winner_model_a", "winner_model_b", "winner_tie",  # target leakage
    }
    feature_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64, int, float]
                     and c not in excluded_cols]
    return df, feature_cols, vectorizer
