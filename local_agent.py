#!/usr/bin/env python3
"""Local training agent for SigmaCloud AutoML.

Usage examples:
1) Pull job spec from backend and train/sync:
    python local_agent.py --connect <job_id> --api-url http://localhost:8000 --token "YOUR_FIREBASE_ID_TOKEN"

2) Train from exported spec JSON:
   python local_agent.py --spec-file ./job_spec.json --offline

3) Offline train, then later sync:
   python local_agent.py --spec-file ./job_spec.json --offline --sync-file ./pending_sync.json
    python local_agent.py --sync-file ./pending_sync.json --api-url http://localhost:8000 --token "YOUR_FIREBASE_ID_TOKEN"
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

torch = None
nn = None


def _ensure_torch() -> None:
    global torch, nn
    if torch is not None and nn is not None:
        return
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for local GPU execution. Install with: pip install torch"
        ) from exc


@dataclass
class AgentContext:
    logs: List[str]

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        self.logs.append(line)
        print(line)


def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_job_spec(api_url: str, job_id: str, token: Optional[str], ctx: AgentContext) -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/api/training/local-job-spec/{job_id}"
    ctx.log(f"Fetching local job spec: {url}")
    resp = requests.get(url, headers=_headers(token), timeout=60)
    if resp.status_code == 401:
        raise RuntimeError(
            "Unauthorized (401). Use a valid Firebase ID token, not the training job id. "
            "Pass it as: --token \"YOUR_FIREBASE_ID_TOKEN\""
        )
    resp.raise_for_status()
    return resp.json()


def load_spec(spec_file: Optional[str], connect_job: Optional[str], api_url: str, token: Optional[str], ctx: AgentContext) -> Dict[str, Any]:
    if spec_file:
        ctx.log(f"Loading spec from file: {spec_file}")
        with open(spec_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if connect_job:
        return fetch_job_spec(api_url, connect_job, token, ctx)
    raise ValueError("Provide either --spec-file or --connect <job_id>")


def load_dataset(path: str, ctx: AgentContext) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    ctx.log(f"Loading dataset: {path}")
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def create_torch_mlp(input_dim: int, output_dim: int, regression: bool):
    _ensure_torch()

    class TorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.regression = regression
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return TorchMLP()


def build_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col]).copy()
    X = pd.get_dummies(X, drop_first=True)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0)
    if X.shape[1] == 0:
        raise ValueError("No usable features after preprocessing")
    return X, y


def detect_task(y: pd.Series, declared: Optional[str]) -> str:
    if declared in {"classification", "regression"}:
        return declared
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    clean = y.dropna()
    unique = int(clean.nunique())
    unique_rate = float(unique / max(len(clean), 1))
    return "classification" if unique <= 20 and unique_rate <= 0.2 else "regression"


def train_local_torch(spec: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
    _ensure_torch()
    dataset_path = spec["dataset_path"]
    target_col = spec["target_column"]
    declared_task = spec.get("task_type")

    df = load_dataset(dataset_path, ctx)
    if target_col not in df.columns:
        raise ValueError(f"Target column not present: {target_col}")

    X_df, y_s = build_features(df, target_col)
    task_type = detect_task(y_s, declared_task)
    ctx.log(f"Detected task type: {task_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.log(f"GPU available: {torch.cuda.is_available()} | selected device: {device}")

    valid = y_s.notna()
    X_df = X_df.loc[valid]
    y_s = y_s.loc[valid]

    if task_type == "classification":
        labels = sorted(y_s.astype(str).unique().tolist())
        label_to_idx = {v: i for i, v in enumerate(labels)}
        y_arr = y_s.astype(str).map(label_to_idx).to_numpy(dtype=np.int64)
        stratify = y_arr if len(np.unique(y_arr)) > 1 else None
    else:
        labels = None
        y_arr = pd.to_numeric(y_s, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        stratify = None

    X_arr = X_df.to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=float(spec.get("hyperparameters", {}).get("test_size", 0.2)),
        random_state=42,
        stratify=stratify,
    )

    input_dim = X_train.shape[1]
    output_dim = 1 if task_type == "regression" else int(np.max(y_train)) + 1

    model = create_torch_mlp(input_dim=input_dim, output_dim=output_dim, regression=task_type == "regression").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    if task_type == "regression":
        yt = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    else:
        yt = torch.tensor(y_train, dtype=torch.long, device=device)

    epochs = 35
    batch_size = min(256, max(32, len(X_train) // 8))
    start = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(Xt.size(0), device=device)
        epoch_loss = 0.0
        for i in range(0, Xt.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if ep in {0, 9, 19, 34}:
            ctx.log(f"epoch={ep+1}/{epochs} loss={epoch_loss:.4f}")

    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_raw = model(Xte).detach().cpu().numpy()

    if task_type == "regression":
        y_pred = pred_raw.reshape(-1)
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        confusion = None
        roc_curve_data = None
    else:
        y_pred_idx = np.argmax(pred_raw, axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_idx)),
            "precision": float(precision_score(y_test, y_pred_idx, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_idx, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_idx, average="weighted", zero_division=0)),
        }
        n_classes = int(np.max(y_test)) + 1
        confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
        for a, p in zip(y_test.tolist(), y_pred_idx.tolist()):
            confusion[int(a)][int(p)] += 1
        roc_curve_data = None

    training_time = float(time.time() - start)
    ctx.log(f"Training finished in {training_time:.2f}s")

    output_dir = spec.get("output_dir") or os.path.abspath(f"./local_outputs/{spec['job_id']}")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{spec['job_id']}_local_torch_mlp.pt")
    preprocess_path = os.path.join(output_dir, f"{spec['job_id']}_feature_columns.joblib")
    torch.save(model.state_dict(), model_path)
    joblib.dump({"feature_columns": X_df.columns.tolist(), "labels": labels}, preprocess_path)
    ctx.log(f"Saved local model artifact: {model_path}")

    return {
        "job_id": spec["job_id"],
        "status": "completed",
        "task_type": task_type,
        "execution_env": {
            "device": device,
            "gpu_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
        },
        "logs": ctx.logs,
        "models": [
            {
                "model_name": "Local Torch MLP",
                "model_type": "local_torch_mlp",
                "task_type": task_type,
                "metrics": metrics,
                "feature_importance": None,
                "confusion_matrix": confusion,
                "roc_curve_data": roc_curve_data,
                "cv_scores": [],
                "training_time": training_time,
                "artifact_ref": {
                    "path": model_path,
                    "preprocess_path": preprocess_path,
                },
            }
        ],
    }


def sync_payload(api_url: str, payload: Dict[str, Any], token: Optional[str], ctx: AgentContext) -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/api/training/local-sync"
    ctx.log(f"Syncing results to backend: {url}")
    resp = requests.post(url, headers=_headers(token), data=json.dumps(payload), timeout=120)
    if resp.status_code == 401:
        raise RuntimeError(
            "Unauthorized (401) during sync. Use a valid Firebase ID token: --token \"YOUR_FIREBASE_ID_TOKEN\""
        )
    resp.raise_for_status()
    return resp.json()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SigmaCloud local training agent")
    p.add_argument("--connect", type=str, default=None, help="Job id to pull local spec from backend")
    p.add_argument("--spec-file", type=str, default=None, help="Path to local job spec JSON")
    p.add_argument("--api-url", type=str, default=os.getenv("SIGMACLOUD_API_URL"), help="Full backend URL, e.g. http://localhost:8000")
    p.add_argument("--api-host", type=str, default=os.getenv("SIGMACLOUD_API_HOST", "localhost"), help="Backend host when --api-url is not provided")
    p.add_argument("--api-port", type=int, default=int(os.getenv("SIGMACLOUD_API_PORT", os.getenv("SIGMACLOUD_PORT", "8000"))), help="Backend port when --api-url is not provided")
    p.add_argument("--token", type=str, default=os.getenv("SIGMACLOUD_TOKEN"), help="Firebase bearer token")
    p.add_argument("--offline", action="store_true", help="Do not sync immediately; save payload to file")
    p.add_argument("--sync-file", type=str, default=None, help="Existing payload file to sync OR output file for --offline")
    return p.parse_args()


def resolve_api_url(args: argparse.Namespace) -> str:
    if args.api_url:
        return args.api_url
    return f"http://{args.api_host}:{args.api_port}"


def main() -> None:
    args = parse_args()
    ctx = AgentContext(logs=[])
    api_url = resolve_api_url(args)

    # Sync-only mode for offline payloads.
    if args.sync_file and os.path.exists(args.sync_file) and not args.spec_file and not args.connect:
        ctx.log(f"Loading offline sync payload: {args.sync_file}")
        with open(args.sync_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        res = sync_payload(api_url, payload, args.token, ctx)
        ctx.log(f"Sync response: {res}")
        return

    spec = load_spec(args.spec_file, args.connect, api_url, args.token, ctx)
    payload = train_local_torch(spec, ctx)

    if args.offline:
        output = args.sync_file or os.path.abspath(f"./pending_sync_{payload['job_id']}.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        ctx.log(f"Offline mode enabled. Sync payload saved: {output}")
        return

    res = sync_payload(api_url, payload, args.token, ctx)
    ctx.log(f"Sync response: {res}")


if __name__ == "__main__":
    main()
