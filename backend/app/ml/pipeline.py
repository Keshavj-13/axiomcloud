"""
Axiom Cloud AI - AutoML Pipeline
Automatically detects task type, preprocesses data, trains multiple models,
evaluates performance, and saves results.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.base import clone

# Explainability
import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover
    LimeTabularExplainer = None

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, classification_report,
    precision_score, recall_score, balanced_accuracy_score,
    explained_variance_score, median_absolute_error
)

from app.core.config import settings

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """Production-grade AutoML pipeline that automatically:
    1. Detects problem type (classification/regression)
    2. Preprocesses data (imputation, encoding, scaling)
    3. Trains multiple models
    4. Evaluates with cross-validation
    5. Returns comprehensive metrics
    """

    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=settings.RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, verbose=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=settings.RANDOM_STATE),
    }

    REGRESSION_MODELS = {
        "Linear Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=settings.RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=settings.RANDOM_STATE, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=settings.RANDOM_STATE, verbose=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=settings.RANDOM_STATE),
    }

    def __init__(self, job_id: str, progress_callback=None):
        self.job_id = job_id
        self.progress_callback = progress_callback
        self.label_encoder = None
        self.feature_names = None
        self.tuning_meta: Dict[str, Any] = {}

    def _score_metric(self, task_type: str) -> str:
        return "accuracy" if task_type == "classification" else "r2"

    def tune_estimator(
        self,
        model_name: str,
        estimator,
        preprocessor: ColumnTransformer,
        X_train,
        y_train,
        task_type: str,
        cv_folds: int,
        tuning_trials: int,
        tuning_time_budget_sec: int,
    ):
        """Tune estimator hyperparameters with Optuna (if available)."""
        if optuna is None:
            return estimator, {"enabled": False, "reason": "optuna_not_installed"}

        score_metric = self._score_metric(task_type)

        def suggest(trial):
            if model_name == "Random Forest":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 18),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                }
            if model_name == "XGBoost":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 260),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
            if model_name == "LightGBM":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 260),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 96),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
            if model_name == "Gradient Boosting":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 60, 240),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }
            if model_name == "Logistic Regression":
                return {
                    "C": trial.suggest_float("C", 1e-3, 20.0, log=True),
                }
            if model_name == "Linear Regression":
                return {
                    "alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True),
                }
            return {}

        def objective(trial):
            params = suggest(trial)
            est = clone(estimator)
            if params:
                est.set_params(**params)

            step_name = "classifier" if task_type == "classification" else "regressor"
            candidate = Pipeline([
                ("preprocessor", preprocessor),
                (step_name, est),
            ])

            if task_type == "classification":
                cv = StratifiedKFold(n_splits=max(2, min(5, cv_folds)), shuffle=True, random_state=settings.RANDOM_STATE)
            else:
                cv = KFold(n_splits=max(2, min(5, cv_folds)), shuffle=True, random_state=settings.RANDOM_STATE)

            scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring=score_metric)
            return float(np.mean(scores))

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=tuning_trials, timeout=tuning_time_budget_sec)

            best_params = study.best_params or {}
            tuned = clone(estimator)
            if best_params:
                tuned.set_params(**best_params)

            return tuned, {
                "enabled": True,
                "score_metric": score_metric,
                "n_trials": len(study.trials),
                "best_value": float(study.best_value) if study.best_value is not None else None,
                "best_params": best_params,
            }
        except Exception as e:
            logger.warning("Tuning failed for %s: %s", model_name, e)
            return estimator, {"enabled": True, "failed": True, "reason": str(e)}

    def _update_progress(self, progress: int, message: str = ""):
        """Update training progress."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        logger.info(f"[{self.job_id}] Progress {progress}%: {message}")

    def detect_task_type(self, y: pd.Series) -> str:
        """
        Automatically detect if task is classification or regression.
        Heuristic: ≤ 20 unique values OR dtype is object/bool → classification
        """
        n_unique = y.nunique()
        dtype = y.dtype

        if dtype == "object" or dtype == "bool":
            return "classification"
        if n_unique <= 20 and n_unique / len(y) < 0.05:
            return "classification"
        return "regression"

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2
    ) -> Tuple[Any, Any, Any, Any, ColumnTransformer, List[str]]:
        """
        Full preprocessing pipeline:
        - Drop rows with missing targets
        - Handle numerical & categorical features
        - Apply imputation, encoding, and scaling
        - Split into train/test
        """
        # Drop target NaNs
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column].copy()

        # Identify column types
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Drop high-cardinality categoricals (> 50 unique values)
        cat_cols = [c for c in cat_cols if X[c].nunique() <= 50]

        self.feature_names = num_cols + cat_cols
        X = X[self.feature_names]

        # Build preprocessor
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ], remainder="drop")

        # Encode target for classification
        if y.dtype == "object" or y.dtype == "bool":
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y.astype(str)), name=target_column)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=settings.RANDOM_STATE,
            stratify=y if len(y.unique()) < 20 else None
        )

        return X_train, X_test, y_train, y_test, preprocessor, self.feature_names

    def get_feature_importances(self, model, preprocessor: ColumnTransformer, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importances from a fitted pipeline."""
        try:
            estimator = model.named_steps.get("classifier") or model.named_steps.get("regressor")
            if estimator is None:
                return {}

            fitted_preprocessor = model.named_steps.get("preprocessor")
            if fitted_preprocessor is None:
                return {}

            # Get feature names after one-hot encoding
            ohe_features = []
            for name, transformer, cols in fitted_preprocessor.transformers_:
                if name == "num":
                    ohe_features.extend(cols)
                elif name == "cat":
                    enc = transformer.named_steps.get("encoder")
                    if enc and hasattr(enc, "get_feature_names_out"):
                        ohe_features.extend(enc.get_feature_names_out(cols).tolist())

            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_).flatten()
            else:
                return {}

            # Aggregate OHE features back to original
            importance_dict = {}
            ohe_idx = 0
            for fname in feature_names:
                # Count OHE columns for this feature
                matching = [f for f in ohe_features if f == fname or f.startswith(f"{fname}_")]
                n = len(matching) if matching else 1
                if ohe_idx < len(importances):
                    importance_dict[fname] = float(np.sum(importances[ohe_idx:ohe_idx + n]))
                    ohe_idx += n

            # Normalize
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v / total for k, v in importance_dict.items()}

            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            return {}

    def get_shap_explanation(self, pipeline, X_sample, task_type: str = "classification", nsamples: int = 100):
        """Return SHAP values for a sample of the data."""
        try:
            estimator = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")
            if estimator is None:
                return {"error": "No estimator found in pipeline"}

            preprocessor = pipeline.named_steps["preprocessor"]
            X_proc = preprocessor.transform(X_sample)
            if hasattr(X_proc, "toarray"):
                X_proc = X_proc.toarray()

            feature_names = list(getattr(X_sample, "columns", []))
            if not feature_names:
                feature_names = [str(i) for i in range(X_proc.shape[1])]

            explainer = shap.Explainer(estimator, X_proc)
            shap_values = explainer(X_proc[:nsamples])

            values = np.array(shap_values.values)
            if values.ndim == 2:
                mean_abs = np.abs(values).mean(axis=0)
            elif values.ndim == 3:
                mean_abs = np.abs(values).mean(axis=(0, 2))
            else:
                mean_abs = np.array([])

            global_importance = []
            if mean_abs.size:
                for idx, val in enumerate(mean_abs.tolist()):
                    fname = feature_names[idx] if idx < len(feature_names) else str(idx)
                    global_importance.append({"feature": fname, "importance": float(val)})
                global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)[:20]

            base_values = getattr(shap_values, "base_values", None)
            if isinstance(base_values, np.ndarray):
                expected_value = base_values.tolist()
            else:
                expected_value = base_values

            return {
                "shap_values": values.tolist(),
                "expected_value": expected_value,
                "data": X_proc[:nsamples].tolist(),
                "feature_names": feature_names,
                "global_importance": global_importance,
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"error": str(e)}

    def get_lime_explanation(self, pipeline, X_sample, y_sample, task_type: str = "classification", nsamples: int = 5):
        """Return LIME explanations for a few samples."""
        if LimeTabularExplainer is None:
            return {"error": "LIME dependency is not installed. Install 'lime' package."}
        try:
            estimator = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")
            if estimator is None:
                return {"error": "No estimator found in pipeline"}

            preprocessor = pipeline.named_steps["preprocessor"]
            X_proc = preprocessor.transform(X_sample)
            if hasattr(X_proc, "toarray"):
                X_proc = X_proc.toarray()

            feature_names = list(getattr(X_sample, "columns", []))
            if not feature_names:
                feature_names = [str(i) for i in range(X_proc.shape[1])]

            class_names = None
            if task_type == "classification":
                class_names = [str(v) for v in sorted(pd.Series(y_sample).dropna().unique().tolist())]

            explainer = LimeTabularExplainer(
                X_proc,
                feature_names=feature_names,
                class_names=class_names,
                mode=task_type,
            )

            explanations = []
            for i in range(min(nsamples, X_proc.shape[0])):
                exp = explainer.explain_instance(
                    X_proc[i],
                    estimator.predict_proba if task_type == "classification" else estimator.predict,
                    num_features=min(10, X_proc.shape[1]),
                )
                explanations.append({
                    "instance": i,
                    "explanation": exp.as_list(),
                })

            return {"lime_explanations": explanations}
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return {"error": str(e)}

    def evaluate_classification(
        self, pipeline, X_test, y_test, X_train, y_train, cv_folds: int
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        }

        # ROC-AUC (binary and multiclass)
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            else:
                y_prob = pipeline.predict_proba(X_test)
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report (macro/weighted diagnostics)
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics["classification_report"] = report
        except Exception:
            metrics["classification_report"] = None

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=settings.RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=cv, scoring="accuracy")
        metrics["cv_scores"] = cv_scores.tolist()
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())
        metrics["per_fold"] = [
            {"fold": int(i + 1), "score": float(v)} for i, v in enumerate(cv_scores.tolist())
        ]

        return metrics

    def evaluate_regression(
        self, pipeline, X_test, y_test, X_train, y_train, cv_folds: int
    ) -> Dict[str, Any]:
        """Compute regression metrics."""
        y_pred = pipeline.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred)),
            "explained_variance": float(explained_variance_score(y_test, y_pred)),
            "median_ae": float(median_absolute_error(y_test, y_pred)),
            "residuals": {
                "predicted": y_pred.tolist()[:200],
                "actual": y_test.tolist()[:200],
            }
        }

        # Robust MAPE (avoid divide-by-zero explosions)
        try:
            epsilon = 1e-8
            denom = np.where(np.abs(np.array(y_test)) < epsilon, epsilon, np.abs(np.array(y_test)))
            mape = float(np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / denom)))
            metrics["mape"] = mape
        except Exception:
            metrics["mape"] = None

        # Cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=settings.RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=cv, scoring="r2")
        metrics["cv_scores"] = cv_scores.tolist()
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())
        metrics["per_fold"] = [
            {"fold": int(i + 1), "score": float(v)} for i, v in enumerate(cv_scores.tolist())
        ]

        return metrics

    def train_all_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        models_to_train: Optional[List[str]] = None,
        enable_tuning: bool = False,
        tuning_trials: int = 12,
        tuning_time_budget_sec: int = 180,
    ) -> Dict[str, Any]:
        """
        Main training entrypoint.
        Returns a dict with task_type and list of model results.
        """
        self._update_progress(5, "Preprocessing data...")

        X_train, X_test, y_train, y_test, preprocessor, feature_names = self.preprocess(
            df, target_column, test_size
        )

        # Auto-detect task type
        if task_type is None or task_type == "auto":
            task_type = self.detect_task_type(pd.concat([y_train, y_test]))
        logger.info(f"Task type detected: {task_type}")

        model_catalog = (
            self.CLASSIFICATION_MODELS if task_type == "classification"
            else self.REGRESSION_MODELS
        )

        if models_to_train:
            model_catalog = {k: v for k, v in model_catalog.items() if k in models_to_train}

        results = []
        n_models = len(model_catalog)

        for i, (model_name, estimator) in enumerate(model_catalog.items()):
            progress = 10 + int((i / n_models) * 80)
            self._update_progress(progress, f"Training {model_name}...")

            try:
                start = time.time()

                # Build full pipeline
                tuning_info = {"enabled": False}
                if enable_tuning:
                    tuned_estimator, tuning_info = self.tune_estimator(
                        model_name=model_name,
                        estimator=estimator,
                        preprocessor=preprocessor,
                        X_train=X_train,
                        y_train=y_train,
                        task_type=task_type,
                        cv_folds=cv_folds,
                        tuning_trials=tuning_trials,
                        tuning_time_budget_sec=tuning_time_budget_sec,
                    )
                    estimator = tuned_estimator

                step_name = "classifier" if task_type == "classification" else "regressor"
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    (step_name, estimator),
                ])

                pipeline.fit(X_train, y_train)
                training_time = time.time() - start

                # Evaluate
                if task_type == "classification":
                    metrics = self.evaluate_classification(pipeline, X_test, y_test, X_train, y_train, cv_folds)
                else:
                    metrics = self.evaluate_regression(pipeline, X_test, y_test, X_train, y_train, cv_folds)

                # Feature importances
                feature_importance = self.get_feature_importances(pipeline, preprocessor, feature_names)

                # Save model
                model_filename = f"{self.job_id}_{model_name.replace(' ', '_')}.joblib"
                model_path = os.path.join(settings.MODEL_STORAGE_PATH, model_filename)
                joblib.dump({
                    "pipeline": pipeline,
                    "feature_names": feature_names,
                    "label_encoder": self.label_encoder,
                    "task_type": task_type,
                    "model_name": model_name,
                }, model_path)

                results.append({
                    "model_name": model_name,
                    "model_type": type(estimator).__name__,
                    "task_type": task_type,
                    "file_path": model_path,
                    "metrics": metrics,
                    "tuning": tuning_info,
                    "feature_importance": feature_importance,
                    "training_time": training_time,
                    # Flatten key metrics
                    "accuracy": metrics.get("accuracy"),
                    "f1_score": metrics.get("f1_score"),
                    "roc_auc": metrics.get("roc_auc"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2_score": metrics.get("r2_score"),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                    "roc_curve_data": metrics.get("roc_curve"),
                    "cv_scores": metrics.get("cv_scores"),
                })

                logger.info(f"{model_name} trained in {training_time:.2f}s")

            except Exception as e:
                logger.exception(f"Failed to train {model_name}: {e}")
                results.append({
                    "model_name": model_name,
                    "model_type": type(estimator).__name__,
                    "task_type": task_type,
                    "error": str(e),
                    "training_time": 0,
                })

        self._update_progress(100, "Training complete!")

        return {
            "task_type": task_type,
            "models": results,
            "feature_names": feature_names,
        }
