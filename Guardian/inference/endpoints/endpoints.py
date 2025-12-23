from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, PositiveInt

GUARDIAN_ROOT = Path(__file__).resolve().parents[2]  # .../Guardian
MODEL_PATH = GUARDIAN_ROOT / "models" / "model.onnx"
CSV_PATH = GUARDIAN_ROOT / "datasets" / "engineered_testing_mini.csv"

#---------------------------#
#--------- HELPERS ---------#
#---------------------------#


def _norm_col(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_").replace("__", "_")


def _load_engineered_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = []
    for c in df.columns:
        s = str(c).strip()
        if s == "" or s.lower().startswith("unnamed"):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.rename(columns={c: _norm_col(c) for c in df.columns})
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
    return series.astype(np.float32).to_numpy().reshape(-1, 1)


def _build_feeds(df: pd.DataFrame, sess: ort.InferenceSession) -> dict[str, Any]:
    feeds: dict[str, Any] = {}
    for inp in sess.get_inputs():
        col = _find_column(df, inp.name)
        if col is None:
            raise KeyError(f"Missing required column for model input '{inp.name}'")
        feeds[inp.name] = _to_input_array(df[col], inp.type)
    return feeds


def _run_onnx(sess: ort.InferenceSession, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feeds = _build_feeds(df, sess)
    outs = sess.run(None, feeds)

    out_names = [o.name for o in sess.get_outputs()]
    out_map = {name: np.asarray(val) for name, val in zip(out_names, outs)}

    label = out_map.get("label")
    probs = out_map.get("probabilities")
    if label is None or probs is None:
        raise RuntimeError(f"Expected outputs 'label' and 'probabilities', got: {list(out_map.keys())}")

    label = np.asarray(label).reshape(-1)
    probs = np.asarray(probs).reshape(-1, 2)
    return label, probs


def _timestamp_to_yyyymmdd(v: Any) -> int:
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid Timestamp value: {v!r}")
    return int(ts.strftime("%Y%m%d"))


def _sample_random_accounts(df: pd.DataFrame, n: int) -> pd.DataFrame:
    for col in ("From_Bank", "Account", "Timestamp"):
        if col not in df.columns:
            raise KeyError(f"Engineered dataset missing column '{col}' (available: {sorted(df.columns)})")

    accounts = df[["From_Bank", "Account"]].dropna().drop_duplicates(ignore_index=True)
    if n > len(accounts):
        raise ValueError(f"Requested {n} accounts, but only {len(accounts)} available in dataset")

    chosen = accounts.sample(n=n, replace=False)
    subset = df.merge(chosen, on=["From_Bank", "Account"], how="inner")
    subset = (
        subset.groupby(["From_Bank", "Account"], sort=False, group_keys=False)
        .sample(n=1)
        .reset_index(drop=True)
    )
    return subset


def _find_target_account(df: pd.DataFrame, bank_name: str, account_name: str) -> pd.DataFrame:
    for col in ("From_Bank", "Account", "Timestamp"):
        if col not in df.columns:
            raise KeyError(f"Engineered dataset missing column '{col}' (available: {sorted(df.columns)})")

    bank_name = str(bank_name).strip()
    account_name = str(account_name).strip()

    mask = df["From_Bank"].astype(str).str.strip().eq(bank_name) & df["Account"].astype(str).str.strip().eq(account_name)
    matches = df.loc[mask]

    if matches.empty:
        raise ValueError(f"No transactions found for From_Bank={bank_name!r}, Account={account_name!r}")

    return matches.reset_index(drop=True)


def _get_state(app: FastAPI) -> tuple[pd.DataFrame, ort.InferenceSession]:
    df = getattr(app.state, "df", None)
    sess = getattr(app.state, "ort_session", None)
    if df is None or sess is None:
        raise HTTPException(status_code=503, detail="Model or dataset not loaded yet")
    return df, sess


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.df = _load_engineered_csv(CSV_PATH)
        app.state.ort_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as e:
        raise RuntimeError(f"Startup load failed: {e}") from e
    yield


app = FastAPI(title="Guardian", version="1.0.0", lifespan=lifespan)

#--------------------------------#
#--------- BASE CLASSES ---------#
#--------------------------------#

# Requests
class BatchRandomInferenceRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    num_inferences: PositiveInt = Field(..., gt=0, le=10000, description="Number of random accounts to inference")

class TargetedAccountRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    bank_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank name") 
    account_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank account name") 

# Results 
class InferenceResult(BaseModel):
    prob: float = Field(..., ge=0.0, le=1.0, description="Probability of money laundering")
    prediction: bool
    bank_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank name") 
    account_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank account name") 
    timestamp: PositiveInt = Field(..., description="Timestamp in format YYYYMMDD")

class BatchRandomInferenceResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

class TargetedAccountResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

#------------------------------------------#
#--------- Logical Implementation ---------#
#------------------------------------------#


def _row_to_result(row: pd.Series, prob: float, threshold: float) -> InferenceResult:
    bank = str(row.get("From_Bank", "")).strip()
    account = str(row.get("Account", "")).strip()
    return InferenceResult(
        prob=float(prob),
        prediction=float(prob) >= float(threshold),
        bank_name=bank,
        account_name=account,
        timestamp=_timestamp_to_yyyymmdd(row.get("Timestamp")),
    )


def run_batch_random_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: BatchRandomInferenceRequest) -> BatchRandomInferenceResult:
    batch_df = _sample_random_accounts(df, int(req.num_inferences))
    _, probs = _run_onnx(sess, batch_df)
    prob1 = probs[:, 1].astype(np.float32)

    results = [_row_to_result(batch_df.iloc[i], float(prob1[i]), req.threshold) for i in range(len(batch_df))]
    num_flagged = sum(1 for r in results if r.prediction)
    return BatchRandomInferenceResult(results=results, num_flagged=int(num_flagged))

def run_targeted_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: TargetedAccountRequest) -> TargetedAccountResult:
    targeted_df = _find_target_account(df, req.bank_name, req.account_name)
    _, probs = _run_onnx(sess, targeted_df)
    prob1 = np.asarray(probs).reshape(-1, 2)[:, 1].astype(np.float32)
    
    results = [_row_to_result(targeted_df.iloc[i], float(prob1[i]), req.threshold) for i in range(len(targeted_df))]
    num_flagged = sum(1 for r in results if r.prediction)
    return BatchRandomInferenceResult(results=results, num_flagged=int(num_flagged))


@app.get("/health")
def health() -> dict[str, Any]:
    df, _ = _get_state(app)
    return {"status": "ok", "rows": int(len(df))}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/inference/batch-random", response_model=BatchRandomInferenceResult)
def batch_random(req: BatchRandomInferenceRequest) -> BatchRandomInferenceResult:
    df, sess = _get_state(app)
    try:
        return run_batch_random_inference(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
    
@app.post("/inference/target-account", response_model=TargetedAccountResult)
def batch_radom(req: TargetedAccountRequest) -> TargetedAccountResult:
    df, sess = _get_state(app)
    try:
        return run_targeted_inference(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
