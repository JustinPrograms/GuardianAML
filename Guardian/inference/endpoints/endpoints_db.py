from __future__ import annotations

import sqlite3
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

GUARDIAN_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = GUARDIAN_ROOT / "models" / "standard-model.onnx" # FIXME: reference the directory not a specific model
DB_PATH = GUARDIAN_ROOT / "data" / "Transaction.db"
FRONTEND_PATH = GUARDIAN_ROOT / "frontend" / "index.html"

#---------------------------#
#--------- HELPERS ---------#
#---------------------------#

def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


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
        return series.fillna("UNK").astype(str).to_numpy().reshape(-1, 1).astype(object)
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
    out_map = {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), outs)}
    label = out_map.get("label")
    probs = out_map.get("probabilities")
    if label is None or probs is None:
        raise RuntimeError(f"Expected 'label' and 'probabilities', got: {list(out_map)}")
    return np.asarray(label).reshape(-1), np.asarray(probs).reshape(-1, 2)


def _timestamp_to_yyyymmddhhmmss(v: Any) -> int:
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid Timestamp value: {v!r}")
    return int(ts.strftime("%Y%m%d%H%M%S"))


#---------------------------------#
#--------- DB DATA LAYER ---------#
#---------------------------------#

def _sample_random_accounts_db(conn: sqlite3.Connection, n: int) -> pd.DataFrame:
    total = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    if n > total:
        raise ValueError(f"Requested {n} accounts, but only {total} available in dataset")

    sampled = pd.read_sql_query(
        "SELECT from_bank, account FROM accounts ORDER BY RANDOM() LIMIT ?",
        conn, params=(n,),
    )
    sampled.to_sql("_batch_sample", conn, if_exists="replace", index=False)

    df = pd.read_sql_query("""
        SELECT t.*
        FROM FullData t
        INNER JOIN (
            SELECT f.From_Bank, f.Account, MIN(f.id) AS min_id
            FROM FullData f
            INNER JOIN _batch_sample s
                ON f.From_Bank = s.from_bank
                AND f.Account = s.account
            GROUP BY f.From_Bank, f.Account
        ) sel ON t.id = sel.min_id
    """, conn)

    return df


def _find_bank_transactions_db(conn: sqlite3.Connection, bank_name: str) -> pd.DataFrame:
    bank_name = bank_name.strip()
    df = pd.read_sql_query(
        "SELECT * FROM FullData WHERE From_Bank = ?",
        conn, params=(bank_name,),
    )
    if df.empty:
        raise ValueError(f"No transactions found for From_Bank={bank_name!r}")
    return df


def _find_target_account_from_db(conn: sqlite3.Connection, bank_name: str, account_name: str) -> pd.DataFrame:
    bank_name = bank_name.strip()
    account_name = account_name.strip()
    df = pd.read_sql_query(
        "SELECT * FROM FullData WHERE From_Bank = ? AND Account = ?",
        conn, params=(bank_name, account_name),
    )
    if df.empty:
        raise ValueError(f"No transactions found for From_Bank={bank_name!r}, Account={account_name!r}")
    return df


def _find_target_account_to_db(conn: sqlite3.Connection, bank_name: str, account_name: str) -> pd.DataFrame:
    bank_name = bank_name.strip()
    account_name = account_name.strip()
    df = pd.read_sql_query(
        "SELECT * FROM FullData WHERE To_Bank = ? AND To_Account = ?",
        conn, params=(bank_name, account_name),
    )
    if df.empty:
        raise ValueError(f"No transactions found for To_Bank={bank_name!r}, To_Account={account_name!r}")
    return df


def _get_conn(app: FastAPI) -> sqlite3.Connection:
    conn = getattr(app.state, "db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not loaded yet")
    return conn


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
        app.state.db_conn = _open_db(DB_PATH)
        app.state.sessions = {}
        # Pre-load default model
        _get_session(app, MODEL_PATH.name)
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}") from e
    yield
    app.state.db_conn.close()


app = FastAPI(title="Guardian (DB)", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------------------------------#
#--------- BASE CLASSES ---------#
#--------------------------------#

class BatchRandomInferenceRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0)
    num_inferences: PositiveInt = Field(..., gt=0, le=10000)
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

class TargetedAccountRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0)
    bank_name: str = Field(..., min_length=9, max_length=9)
    account_name: str = Field(..., min_length=9, max_length=9)
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

class BankSummaryRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0)
    bank_name: str = Field(..., min_length=9, max_length=9)
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

class BankTopRiskyRequest(BaseModel):
    threshold: float = Field(..., gt=0, le=1.0)
    bank_name: str = Field(..., min_length=9, max_length=9)
    top_n: PositiveInt = Field(5, gt=0, le=100)
    only_positive: bool
    model: str = Field(MODEL_PATH.name)

class InferenceResult(BaseModel):
    prob: float = Field(..., ge=0.0, le=1.0)
    prediction: bool
    bank_name: str = Field(..., min_length=9, max_length=9)
    account_name: str = Field(..., min_length=9, max_length=9)
    timestamp: PositiveInt

class BatchRandomInferenceResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

class TargetedAccountResult(BaseModel):
    results: list[InferenceResult]
    num_flagged: int

class RiskyAccountResult(BaseModel):
    bank_name: str = Field(..., min_length=9, max_length=9)
    account_name: str = Field(..., min_length=9, max_length=9)
    average_risk_prob: float = Field(..., ge=0.0, le=1.0)
    smoothed_risk: float = Field(..., ge=0.0, le=1.0)
    num_flagged: int = Field(..., ge=0)
    num_transactions: int = Field(..., ge=0)

class BankTopRiskyResult(BaseModel):
    results: list[RiskyAccountResult]

class BankSummaryResult(BaseModel):
    bank_name: str
    total_transactions: int
    num_flagged: int
    flagged_percentage: float = Field(..., ge=0.0, le=100.0)
    total_flagged_amount: float | None = None
    average_risk_prob: float = Field(..., ge=0.0, le=1.0)

#------------------------------------------#
#--------- Logical Implementation ---------#
#------------------------------------------#

def _row_to_result(row: pd.Series, prob: float, threshold: float, only_positive: bool, bank_col: str = "From_Bank", account_col: str = "Account",) -> InferenceResult | None:
    if only_positive and float(prob) < float(threshold):
        return None
    bank = str(row.get(bank_col, "")).strip()
    account = str(row.get(account_col, "")).strip()
    return InferenceResult(
        prob=float(prob),
        prediction=float(prob) >= float(threshold),
        bank_name=bank,
        account_name=account,
        timestamp=_timestamp_to_yyyymmddhhmmss(row.get("Timestamp")),
    )


def run_batch_random_inference(conn: sqlite3.Connection, sess: ort.InferenceSession, req: BatchRandomInferenceRequest) -> BatchRandomInferenceResult:
    batch_df = _sample_random_accounts_db(conn, int(req.num_inferences))
    _, probs = _run_onnx(sess, batch_df)
    prob1 = probs[:, 1].astype(np.float32)
    results = [
        r for i in range(len(batch_df))
        if (r := _row_to_result(batch_df.iloc[i], float(prob1[i]), req.threshold, req.only_positive)) is not None
    ]
    return BatchRandomInferenceResult(results=results, num_flagged=sum(1 for r in results if r.prediction))


def run_targeted_inference(conn: sqlite3.Connection ,sess: ort.InferenceSession ,req: TargetedAccountRequest, from_account: bool,) -> TargetedAccountResult:
    if from_account:
        targeted_df = _find_target_account_from_db(conn, req.bank_name, req.account_name)
        bank_col, account_col = "From_Bank", "Account"
    else:
        targeted_df = _find_target_account_to_db(conn, req.bank_name, req.account_name)
        bank_col, account_col = "To_Bank", "To_Account"
    _, probs = _run_onnx(sess, targeted_df)
    prob1 = np.asarray(probs).reshape(-1, 2)[:, 1].astype(np.float32)
    results = [
        r for i in range(len(targeted_df))
        if (r := _row_to_result(
            targeted_df.iloc[i], float(prob1[i]), req.threshold, req.only_positive,
            bank_col=bank_col, account_col=account_col,
        )) is not None
    ]
    return TargetedAccountResult(results=results, num_flagged=sum(1 for r in results if r.prediction))


def run_targeted_inference_combined(conn: sqlite3.Connection, sess: ort.InferenceSession, req: TargetedAccountRequest) -> TargetedAccountResult:
    results: list[InferenceResult] = []
    errors: list[str] = []
    for from_account in (True, False):
        try:
            res = run_targeted_inference(conn, sess, req, from_account)
        except ValueError as e:
            errors.append(str(e))
            continue
        results.extend(res.results)
    if not results:
        raise ValueError("; ".join(errors) if errors else "No transactions found for target account")
    return TargetedAccountResult(results=results, num_flagged=sum(1 for r in results if r.prediction))


def run_bank_top_risky_inference(conn: sqlite3.Connection, sess: ort.InferenceSession, req: BankTopRiskyRequest) -> BankTopRiskyResult:
    bank_df = _find_bank_transactions_db(conn, req.bank_name)
    _, probs = _run_onnx(sess, bank_df)
    prob1 = probs[:, 1].astype(np.float32)
    bank_df = bank_df.copy()
    bank_df["prob"] = prob1
    bank_df["flagged"] = bank_df["prob"] >= req.threshold
    global_risk = bank_df["prob"].mean()
    k = 20.0
    alpha = global_risk * k
    beta = (1 - global_risk) * k
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
    agg_df["smoothed_risk"] = (agg_df["sum_risk_prob"] + alpha) / (agg_df["num_transactions"] + alpha + beta)
    top_accounts = agg_df.sort_values("smoothed_risk", ascending=False).head(req.top_n)
    results = [
        RiskyAccountResult(
            bank_name=req.bank_name,
            account_name=row["Account"],
            average_risk_prob=row["average_risk_prob"],
            smoothed_risk=row["smoothed_risk"],
            num_flagged=int(row["num_flagged"]),
            num_transactions=int(row["num_transactions"]),
        )
        for _, row in top_accounts.iterrows()
    ]
    return BankTopRiskyResult(results=results)


def run_bank_summary_inference(conn: sqlite3.Connection, sess: ort.InferenceSession, req: BankSummaryRequest) -> BankSummaryResult:
    bank_df = _find_bank_transactions_db(conn, req.bank_name)
    _, probs = _run_onnx(sess, bank_df)
    prob1 = probs[:, 1].astype(np.float32)
    bank_df = bank_df.copy()
    bank_df["prob"] = prob1
    bank_df["flagged"] = bank_df["prob"] >= req.threshold
    total_transactions = len(bank_df)
    num_flagged = int(bank_df["flagged"].sum())
    flagged_percentage = (num_flagged / total_transactions * 100) if total_transactions > 0 else 0.0
    average_risk_prob = float(bank_df["prob"].mean())
    total_flagged_amount = 0.0
    amount_col = _find_column(bank_df, "Amount_Paid")
    if amount_col:
        val = pd.to_numeric(bank_df.loc[bank_df["flagged"], amount_col], errors="coerce").sum()
        total_flagged_amount = float(val) if not pd.isna(val) else 0.0
    return BankSummaryResult(
        bank_name=req.bank_name,
        total_transactions=total_transactions,
        num_flagged=num_flagged,
        flagged_percentage=flagged_percentage,
        total_flagged_amount=total_flagged_amount,
        average_risk_prob=average_risk_prob,
    )


#------------------------------------------#
#------------- Full Endpoints -------------#
#------------------------------------------#

@app.get("/health")
def health() -> dict[str, Any]:
    conn = _get_conn(app)
    rows = conn.execute("SELECT COUNT(*) FROM FullData").fetchone()[0]
    return {"status": "ok", "rows": int(rows)}


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
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_batch_random_inference(conn, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account/from", response_model=TargetedAccountResult)
def target_account_from(req: TargetedAccountRequest) -> TargetedAccountResult:
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference(conn, sess, req, from_account=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account/to", response_model=TargetedAccountResult)
def target_account_to(req: TargetedAccountRequest) -> TargetedAccountResult:
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference(conn, sess, req, from_account=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/target-account", response_model=TargetedAccountResult)
def target_account_combined(req: TargetedAccountRequest) -> TargetedAccountResult:
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_targeted_inference_combined(conn, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/bank/top-risky-accounts", response_model=BankTopRiskyResult)
def bank_top_risky_accounts(req: BankTopRiskyRequest) -> BankTopRiskyResult:
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_bank_top_risky_inference(conn, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/bank/summary", response_model=BankSummaryResult)
def bank_summary(req: BankSummaryRequest) -> BankSummaryResult:
    conn = _get_conn(app)
    sess = _get_session(app, req.model)
    try:
        return run_bank_summary_inference(conn, sess, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
