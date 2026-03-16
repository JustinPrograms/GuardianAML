from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, PositiveInt

GUARDIAN_ROOT = Path(__file__).resolve().parents[2]  # .../Guardian
MODEL_PATH = GUARDIAN_ROOT / "models" / "small-model.onnx"
CSV_PATH = GUARDIAN_ROOT / "data" / "engineered_testing_mini.csv"
FRONTEND_PATH = GUARDIAN_ROOT / "frontend" / "index.html"

#---------------------------#
#--------- HELPERS ---------#
#---------------------------#


# Strips whitespace from column names 
def _norm_col(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_").replace("__", "_") # *note: needed while indexing rows


# Load csv into df 
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


# Return a column to work with
def _find_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    alt = name.replace("_", " ")
    for c in df.columns:
        if c.lower() == name.lower() or c.lower() == alt.lower():
            return c
    return None


# pandas series --> numpy array
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


def _timestamp_to_yyyymmddhhmmss(v: Any) -> int:
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid Timestamp value: {v!r}")
    return int(ts.strftime("%Y%m%d%H%M%S"))


# random account feature 
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

def _find_banks_transactions(df: pd.DataFrame, bank_name: str) -> pd.DataFrame:
    if "From_Bank" not in df.columns:
        raise KeyError(f"Engineered dataset missing column 'From_Bank' (available: {sorted(df.columns)})")

    bank_name = str(bank_name).strip()
    mask = df["From_Bank"].astype(str).str.strip().eq(bank_name)
    matches = df.loc[mask]

    if matches.empty:
        raise ValueError(f"No transactions found for From_Bank={bank_name!r}")

    return matches.reset_index(drop=True)


# n risky transactions by bank 
def run_bank_top_risky_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: BankTopRiskyRequest) -> BankTopRiskyResult:
    bank_df = _find_banks_transactions(df, req.bank_name)
    # run infrerence on bank_df
    _, probs = _run_onnx(sess, bank_df)
    prob1 = probs[:, 1].astype(np.float32)

    # Add probabilities to the dataframe
    bank_df["prob"] = prob1
    bank_df["flagged"] = bank_df["prob"] >= req.threshold

    # Compute global baseline risk and Bayesian parameters
    global_risk_rate = bank_df["prob"].mean()
    k = 20  # Bayesian prior strength
    alpha = global_risk_rate * k
    beta = (1 - global_risk_rate) * k

    # Group by account and aggregate results
    agg_df = (
        bank_df.groupby("Account")
        .agg(
            average_risk_prob=("prob", "mean"),
            sum_risk_prob=("prob", "sum"),
            num_flagged=("flagged", "sum"),
            num_transactions=("prob", "count"),
        )
        .reset_index()
    )

    # Compute smoothed risk
    agg_df["smoothed_risk"] = (agg_df["sum_risk_prob"] + alpha) / (agg_df["num_transactions"] + alpha + beta)

    # Sort by smoothed risk and take top N
    top_accounts = agg_df.sort_values("smoothed_risk", ascending=False).head(req.top_n)

    # Format results
    results = [
        RiskyAccountResult(
            bank_name=req.bank_name,
            account_name=row["Account"],
            average_risk_prob=row["average_risk_prob"],
            smoothed_risk=row["smoothed_risk"],
            num_flagged=int(row["num_flagged"]),
            # contextualizing for skewing in accounts with low amount of transactions
            num_transactions=int(row["num_transactions"]),
        )
        for _, row in top_accounts.iterrows()
    ]
    return BankTopRiskyResult(results=results)


# bank-wide risk feautre
def run_bank_summary_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: BankSummaryRequest) -> BankSummaryResult:
    bank_df = _find_banks_transactions(df, req.bank_name)
    _, probs = _run_onnx(sess, bank_df)
    prob1 = probs[:, 1].astype(np.float32)

    bank_df["prob"] = prob1
    bank_df["flagged"] = bank_df["prob"] >= req.threshold

    total_transactions = len(bank_df)
    num_flagged = int(bank_df["flagged"].sum())
    flagged_percentage = (num_flagged / total_transactions) * 100 if total_transactions > 0 else 0.0
    average_risk_prob = float(bank_df["prob"].mean())

    total_flagged_amount = 0.0
    amount_col = _find_column(bank_df, "Amount_Paid")
    if amount_col:
        flagged_amount = pd.to_numeric(bank_df.loc[bank_df["flagged"], amount_col], errors="coerce").sum()
        print(flagged_amount)
        if pd.isna(flagged_amount): flagged_amount = 0.0
        total_flagged_amount = float(flagged_amount)

    return BankSummaryResult(
        bank_name=req.bank_name,
        total_transactions=total_transactions,
        num_flagged=num_flagged,
        flagged_percentage=flagged_percentage,
        total_flagged_amount=total_flagged_amount, # look at issue here
        average_risk_prob=average_risk_prob,
    )


def _find_target_account_from(df: pd.DataFrame, bank_name: str, account_name: str) -> pd.DataFrame:
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


def _find_target_account_to(df: pd.DataFrame, bank_name: str, account_name: str) -> pd.DataFrame:
    for col in ("To_Bank", "To_Account", "Timestamp"):
        if col not in df.columns:
            raise KeyError(f"Engineered dataset missing column '{col}' (available: {sorted(df.columns)})")

    bank_name = str(bank_name).strip()
    account_name = str(account_name).strip()

    mask = df["To_Bank"].astype(str).str.strip().eq(bank_name) & df["To_Account"].astype(str).str.strip().eq(account_name)
    matches = df.loc[mask]

    if matches.empty:
        raise ValueError(f"No transactions found for To_Bank={bank_name!r}, To_Account={account_name!r}")

    return matches.reset_index(drop=True)


def _get_df(app: FastAPI) -> pd.DataFrame:
    df = getattr(app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")
    return df


def _get_session(app: FastAPI, model_name: str) -> ort.InferenceSession:
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail="Invalid model name")
    sessions: dict[str, ort.InferenceSession] = app.state.sessions
    if model_name not in sessions:
        model_path = GUARDIAN_ROOT / "models" / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name!r} not found")
        sessions[model_name] = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sessions[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.df = _load_engineered_csv(CSV_PATH)
        app.state.sessions = {}
        _get_session(app, MODEL_PATH.name)
    except Exception as e:
        raise RuntimeError(f"Startup load failed: {e}") from e
    yield


app = FastAPI(title="Guardian", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------------------------------#
#--------- BASE CLASSES ---------#
#--------------------------------#

# --------- Requests ---------
# Randomly check n accounts
class BatchRandomInferenceRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    num_inferences: PositiveInt = Field(..., gt=0, le=10000, description="Number of random accounts to inference")
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

# For specific account number under bank
class TargetedAccountRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    bank_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank name")
    account_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank account name")
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

# For specific account number under bank
class ToTargetedAccountRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    bank_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank name")
    account_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank account name")
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

# Bank-level
class BankSummaryRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    bank_name: str = Field(..., min_length=9, max_length=9, description="9 character bank name")
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

# Tranasction per-bank level
class BankTopRiskyRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0, description="Probaility threshold of money laundering transaction")
    bank_name: str = Field(..., min_length=9, max_length=9, description="9 character bank name")
    top_n : PositiveInt = Field(5, gt=0, le=100, description="Number of top risky accounts to return")
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

# --------- Results ---------
# Returned by eeach inference
class InferenceResult(BaseModel):
    prob: float = Field(..., ge=0.0, le=1.0, description="Probability of money laundering")
    prediction: bool
    bank_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank name") 
    account_name: str = Field(..., min_length=9, max_length=9,  description="9 character bank account name") 
    timestamp: PositiveInt = Field(..., description="Timestamp in format YYYYMMDDHHMMSS")

class BankSummaryResult(BaseModel):
    bank_name: str
    total_transactions: int
    num_flagged: int
    flagged_percentage: float = Field(..., ge=0.0, le=100.0)
    total_flagged_amount: float | None = Field(None, description="Total monetary value of flagged transactions (if amount data is available)")
    average_risk_prob: float = Field(..., ge=0.0, le=1.0)

class RiskyAccountResult(BaseModel):
    bank_name: str = Field(..., min_length=9, max_length=9, description="9 character bank name")
    account_name: str = Field(..., min_length=9, max_length=9, description="9 character bank account name")
    average_risk_prob: float = Field(..., ge=0.0, le=1.0, description="Average probability of money laundering for the account")
    smoothed_risk: float = Field(..., ge=0.0, le=1.0, description="Bayesian-smoothed risk score for ranking")
    num_flagged: int = Field(..., ge=0, description="Number of transactions flagged for this account")
    num_transactions: int = Field(..., ge=0, description="Total number of transactions for this account")

class BatchRandomInferenceResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

class BankTopRiskyResult(BaseModel):
    results: list[RiskyAccountResult]

class TargetedAccountResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

class ToTargetedAccountResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

#------------------------------------------#
#--------- Logical Implementation ---------#
#------------------------------------------#

def _row_to_result(row: pd.Series, prob: float, threshold: float, only_positive: bool, bank_col: str = "From_Bank", account_col: str = "Account") -> InferenceResult:
    bank = str(row.get(bank_col, "")).strip()
    account = str(row.get(account_col, "")).strip()

    # Only return if: only_positive = true && flagged || only_positive = false
    if only_positive and float(prob) < float(threshold):
        return None
    return InferenceResult(
        prob=float(prob),
        prediction=float(prob) >= float(threshold),
        bank_name=bank,
        account_name=account,
        timestamp=_timestamp_to_yyyymmddhhmmss(row.get("Timestamp")),
    )


def run_batch_random_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: BatchRandomInferenceRequest) -> BatchRandomInferenceResult:
    batch_df = _sample_random_accounts(df, int(req.num_inferences))
    _, probs = _run_onnx(sess, batch_df)
    prob1 = probs[:, 1].astype(np.float32)

    results = [
        r for i in range(len(batch_df))
        if (r := _row_to_result(batch_df.iloc[i], float(prob1[i]), req.threshold, req.only_positive)) is not None
    ]
    num_flagged = sum(1 for r in results if r.prediction)
    return BatchRandomInferenceResult(results=results, num_flagged=int(num_flagged))


def run_targeted_inference(df: pd.DataFrame, sess: ort.InferenceSession, req: TargetedAccountRequest, from_account: bool) -> TargetedAccountResult:
    if from_account:
        targeted_df = _find_target_account_from(df, req.bank_name, req.account_name)
        bank_col = "From_Bank"
        account_col = "Account"
    else:
        targeted_df = _find_target_account_to(df, req.bank_name, req.account_name)
        bank_col = "To_Bank"
        account_col = "To_Account"
    _, probs = _run_onnx(sess, targeted_df)
    prob1 = np.asarray(probs).reshape(-1, 2)[:, 1].astype(np.float32)
    
    results = [
        r for i in range(len(targeted_df))
        if (r := _row_to_result(targeted_df.iloc[i], float(prob1[i]), req.threshold, req.only_positive, bank_col=bank_col, account_col=account_col)) is not None
    ]

    num_flagged = sum(1 for r in results if r.prediction)
    return TargetedAccountResult(results=results, num_flagged=int(num_flagged))


def run_targeted_inference_combined(df: pd.DataFrame, sess: ort.InferenceSession, req: TargetedAccountRequest) -> TargetedAccountResult:
    results: list[InferenceResult] = []
    num_flagged = 0
    errors: list[str] = []

    for from_account in (True, False):
        try:
            res = run_targeted_inference(df, sess, req, from_account), 
            
        except ValueError as e:
            errors.append(str(e))
            continue
        results.extend(res.results)
        num_flagged += res.num_flagged

    if not results:
        raise ValueError("; ".join(errors) if errors else "No transactions found for target account")

    return TargetedAccountResult(results=results, num_flagged=int(num_flagged))


#------------------------------------------#
#------------- Full Endpoints -------------#
#------------------------------------------#

@app.get("/health")
def health() -> dict[str, Any]:
    df = _get_df(app)
    return {"status": "ok", "rows": int(len(df))}


@app.get("/models")
def list_models() -> dict[str, Any]:
    models_dir = GUARDIAN_ROOT / "models"
    files = sorted(f.name for f in models_dir.iterdir() if f.suffix == ".onnx")
    return {"models": files}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/index")


@app.get("/index", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_PATH)


@app.post("/inference/batch-random", response_model=BatchRandomInferenceResult)
def batch_random(req: BatchRandomInferenceRequest) -> BatchRandomInferenceResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_batch_random_inference(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account/from", response_model=TargetedAccountResult)
def target_account_from(req: TargetedAccountRequest) -> TargetedAccountResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference(df, sess, req, from_account=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account/to", response_model=TargetedAccountResult)
def target_account_to(req: TargetedAccountRequest) -> TargetedAccountResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference(df, sess, req, from_account=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account", response_model=TargetedAccountResult)
def target_account_combined(req: TargetedAccountRequest) -> TargetedAccountResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference_combined(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/bank/top-risky-accounts", response_model=BankTopRiskyResult)
def bank_top_risky_accounts(req: BankTopRiskyRequest) -> BankTopRiskyResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_bank_top_risky_inference(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/bank/summary", response_model=BankSummaryResult)
def bank_summary(req: BankSummaryRequest) -> BankSummaryResult:
    df = _get_df(app)
    sess = _get_session(app, req.model)
    try:
        return run_bank_summary_inference(df, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
