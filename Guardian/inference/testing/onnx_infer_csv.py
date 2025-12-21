import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import onnxruntime as ort


def _norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The engineered CSV has a leading empty header cell which becomes "Unnamed: 0".
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.rename(columns={c: _norm_col(c) for c in df.columns})
    return df


def _derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "hour" not in df.columns:
        df["hour"] = ts.dt.hour.fillna(0).astype(np.int64)
    if "dow" not in df.columns:
        df["dow"] = ts.dt.dayofweek.fillna(0).astype(np.int64)
    if "is_weekend" not in df.columns:
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).fillna(False).astype(np.int64)
    return df


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    alt = name.replace("_", " ")
    for c in df.columns:
        if c.lower() == name.lower() or c.lower() == alt.lower():
            return c
    return None


def _to_input_array(series: pd.Series, ort_type: str) -> np.ndarray:
    t = (ort_type or "").lower()
    if "string" in t:
        vals = series.fillna("UNK").astype(str).to_numpy()
        return vals.reshape(-1, 1).astype(object)
    if "float16" in t:
        return series.astype(np.float16).to_numpy().reshape(-1, 1)
    if "float" in t:
        return series.astype(np.float32).to_numpy().reshape(-1, 1)
    if "int64" in t:
        return series.astype(np.int64).to_numpy().reshape(-1, 1)
    if "int32" in t:
        return series.astype(np.int32).to_numpy().reshape(-1, 1)
    # Safe fallback: treat as float32.
    return series.astype(np.float32).to_numpy().reshape(-1, 1)


def _build_feeds(df: pd.DataFrame, sess: ort.InferenceSession) -> dict[str, Any]:
    feeds: dict[str, Any] = {}
    for inp in sess.get_inputs():
        col = _find_column(df, inp.name)
        if col is None:
            raise KeyError(f"Missing required column for model input '{inp.name}'")
        feeds[inp.name] = _to_input_array(df[col], inp.type)
    return feeds


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ONNX model inference on engineered CSV rows.")
    ap.add_argument("--csv", default="datasets/engineered_testing_mini.csv")
    ap.add_argument("--model", default="~/Guardian/datasets/engineered_testing_mini.csv")
    ap.add_argument("--out", default="", help="Write CSV with predictions appended.")
    ap.add_argument("--rows", type=int, default=0, help="Limit number of rows (0 = all).")
    ap.add_argument("--derive-time", action="store_true", help="Derive hour/dow/is_weekend from Timestamp if missing.")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    df = _load_csv(args.csv)
    if args.derive_time:
        df = _derive_time_features(df)
    if args.rows and args.rows > 0:
        df = df.head(args.rows).copy()

    feeds = _build_feeds(df, sess)
    outs = sess.run(None, feeds)

    out_names = [o.name for o in sess.get_outputs()]
    out_map = {name: np.asarray(val) for name, val in zip(out_names, outs)}

    label = out_map.get("label")
    probs = out_map.get("probabilities")
    if label is None or probs is None:
        raise RuntimeError(f"Expected outputs 'label' and 'probabilities', got: {list(out_map.keys())}")

    label = label.reshape(-1)
    probs = probs.reshape(-1, 2)

    df_out = df.copy()
    df_out["pred_label"] = label.astype(np.int64)
    df_out["pred_prob_0"] = probs[:, 0].astype(np.float32)
    df_out["pred_prob_1"] = probs[:, 1].astype(np.float32)

    if args.out:
        df_out.to_csv(args.out, index=False)
        print(f"Wrote: {args.out} (rows={len(df_out)})")
    else:
        # Print a compact preview.
        print(df_out[["pred_label", "pred_prob_0", "pred_prob_1"]].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

