# advanced_ml_predictor.py
# ------------------------------------------------------------
# ML avanzado para AtmosAtlas:
# - Regresión (mm de precipitación)
# - Clasificación (llueve > umbral), con calibración
# - Ensamble con pesos por error de validación
# - Conversión mm -> prob con calibración logística basada en ventana histórica
# - Métricas y guardado/carga
# ------------------------------------------------------------
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import warnings
import logging

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    brier_score_loss, roc_auc_score
)

# Opcionales (silenciados)
HAS_XGB = False
HAS_LGBM = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    pass

try:
    import lightgbm as lgb
    HAS_LGBM = True
    logging.getLogger("lightgbm").setLevel(logging.ERROR)  # reduce logs
except Exception:
    pass

# Filtra advertencias ruidosas
warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")
warnings.filterwarnings("ignore", message="^X does not have valid feature names.*")


def _safe_std(x: float, min_val: float = 1e-3) -> float:
    return float(max(min_val, x))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


class AdvancedClimatePredictor:
    """
    Predictor avanzado (regresión + clasificación) para precipitación.
    - target_reg: precipitación diaria (mm)
    - target_cls: 1 si precip > rain_threshold, 0 en caso contrario
    """

    def __init__(self, config: Optional[dict] = None, model_dir: str = "models"):
        self.config = config or self.get_default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # model store
        self.models_reg: Dict[str, object] = {}
        self.models_cls: Dict[str, object] = {}

        # scalers
        self.scalers: Dict[str, object] = {}
        # lista única de features compartida por reg y cls
        self.feature_names: List[str] | None = None

        self.ensemble_weights_reg: Dict[str, float] | None = None
        self.ensemble_weights_cls: Dict[str, float] | None = None

        # audit
        self.training_history = {
            "timestamp": None,
            "metrics_reg": {},
            "metrics_cls": {}
        }

    # ------------------- CONFIG -------------------
    @staticmethod
    def get_default_config() -> dict:
        cfg = {
            "regression": {
                "rf": dict(
                    n_estimators=300, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, n_jobs=-1, random_state=42
                ),
                "gbr": dict(
                    n_estimators=400, learning_rate=0.02, max_depth=4,
                    subsample=0.8, random_state=42
                ),
                "ridge": dict(alpha=1.0),  # Ridge no tiene random_state
                "elastic": dict(alpha=0.1, l1_ratio=0.5, random_state=42)
            },
            "classification": {
                "rf": dict(
                    n_estimators=400, max_depth=10, min_samples_split=4,
                    min_samples_leaf=2, n_jobs=-1, random_state=42
                ),
                "gbc": dict(
                    n_estimators=500, learning_rate=0.02, max_depth=3,
                    subsample=0.9, random_state=42
                ),
                "logreg": dict(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
            }
        }
        if HAS_XGB:
            cfg["regression"]["xgb"] = dict(
                n_estimators=700, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                verbosity=0
            )
            cfg["classification"]["xgb"] = dict(
                n_estimators=700, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                eval_metric="logloss", verbosity=0
            )
        if HAS_LGBM:
            # Solo parámetros nativos de LightGBM (evita sinónimos redundantes)
            lgb_common = dict(
                n_estimators=700, learning_rate=0.03, num_leaves=31,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_child_samples=20, random_state=42, n_jobs=-1, verbosity=-1
            )
            cfg["regression"]["lgbm"] = lgb_common.copy()
            cfg["classification"]["lgbm"] = lgb_common.copy()
        return cfg

    # ------------------- FEATURES -------------------
    def create_features(self, df: pd.DataFrame, target_col: str = "precip") -> pd.DataFrame:
        """
        Crea features temporales, lags, rolling stats, anomalías y señales simples.
        Usa solo información pasada (shift) para evitar leakage.
        """
        d = df.copy()
        assert "date" in d.columns, "El DataFrame debe tener columna 'date'"
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)

        # básicos
        d["year"] = d["date"].dt.year
        d["month"] = d["date"].dt.month
        d["doy"] = d["date"].dt.dayofyear
        d["week"] = d["date"].dt.isocalendar().week.astype(int)
        d["is_weekend"] = (d["date"].dt.dayofweek >= 5).astype(int)
        # cíclicos
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)

        # lags + rolling
        base_cols = [c for c in ["precip", "t2m", "rh2m", "wind", "ps", "rad"] if c in d.columns]
        for c in base_cols:
            for lag in [1, 3, 7, 14, 30]:
                d[f"{c}_lag_{lag}"] = d[c].shift(lag)
            for w in [7, 14, 30, 60]:
                roll = d[c].rolling(w, min_periods=2)
                d[f"{c}_roll_mean_{w}"] = roll.mean()
                d[f"{c}_roll_std_{w}"] = roll.std()
                d[f"{c}_roll_sum_{w}"] = roll.sum()
                d[f"{c}_roll_max_{w}"] = roll.max()
                d[f"{c}_roll_min_{w}"] = roll.min()

        # cambios
        for c in [c for c in ["t2m", "ps", "rh2m"] if c in d.columns]:
            d[f"{c}_d1"] = d[c].diff(1)
            d[f"{c}_d7"] = d[c].diff(7)
            d[f"{c}_d30"] = d[c].diff(30)

        # índices sencillos
        if "t2m" in d.columns and "rh2m" in d.columns:
            d["humidex"] = d["t2m"] + 0.5555 * (6.112 * 10 ** (7.5 * d["t2m"] / (237.7 + d["t2m"])) * d["rh2m"] / 100 - 10)
            d["th_index"] = d["t2m"] * (1 + d["rh2m"] / 100.0)

        # anomalías mensuales
        for c in [c for c in ["precip", "t2m", "rh2m"] if c in d.columns]:
            m_mean = d.groupby("month")[c].transform("mean")
            m_std = d.groupby("month")[c].transform("std")
            d[f"{c}_anom"] = (d[c] - m_mean) / (m_std + 1e-5)

        return d.copy()

    # ------------------- SPLIT -------------------
    @staticmethod
    def _time_split(
        df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split temporal simple: 70% train, 15% val (el 15% restante es test, no usado aquí).
        Se puede forzar un conjunto de columnas de features para garantizar consistencia.
        """
        dfc = df.dropna(subset=[target_col]).copy()
        dfc = dfc.sort_values("date")
        feats = feature_cols if feature_cols is not None else [c for c in dfc.columns if c not in ["date", target_col]]
        n = len(dfc)
        tr_end = int(n * 0.7)
        va_end = int(n * 0.85)
        X_train, y_train = dfc.iloc[:tr_end][feats], dfc.iloc[:tr_end][target_col]
        X_val, y_val = dfc.iloc[tr_end:va_end][feats], dfc.iloc[tr_end:va_end][target_col]
        return X_train, y_train, X_val, y_val

    # ------------------- TRAIN: REG -------------------
    def train_regression(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> Dict:
        # guardar lista unificada de columnas (primera vez)
        if self.feature_names is None:
            self.feature_names = Xtr.columns.tolist()

        # escalers exclusivos para regresión
        self.scalers["std_reg"] = StandardScaler().fit(Xtr)
        self.scalers["rob_reg"] = RobustScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(self.scalers["std_reg"].transform(Xtr), columns=Xtr.columns)
        Xva_s = pd.DataFrame(self.scalers["std_reg"].transform(Xva), columns=Xva.columns)
        Xtr_r = pd.DataFrame(self.scalers["rob_reg"].transform(Xtr), columns=Xtr.columns)
        Xva_r = pd.DataFrame(self.scalers["rob_reg"].transform(Xva), columns=Xva.columns)

        metrics = {}

        # RF
        self.models_reg["rf"] = RandomForestRegressor(**self.config["regression"]["rf"])
        self.models_reg["rf"].fit(Xtr_r, ytr)
        pred = self.models_reg["rf"].predict(Xva_r)
        metrics["rf"] = self._reg_metrics(yva, pred)

        # GBR
        self.models_reg["gbr"] = GradientBoostingRegressor(**self.config["regression"]["gbr"])
        self.models_reg["gbr"].fit(Xtr_r, ytr)
        pred = self.models_reg["gbr"].predict(Xva_r)
        metrics["gbr"] = self._reg_metrics(yva, pred)

        # Ridge
        self.models_reg["ridge"] = Ridge(**self.config["regression"]["ridge"])
        self.models_reg["ridge"].fit(Xtr_s, ytr)
        pred = self.models_reg["ridge"].predict(Xva_s)
        metrics["ridge"] = self._reg_metrics(yva, pred)

        # Elastic
        self.models_reg["elastic"] = ElasticNet(**self.config["regression"]["elastic"])
        self.models_reg["elastic"].fit(Xtr_s, ytr)
        pred = self.models_reg["elastic"].predict(Xva_s)
        metrics["elastic"] = self._reg_metrics(yva, pred)

        # XGB / LGBM (opcionales)
        if HAS_XGB:
            self.models_reg["xgb"] = xgb.XGBRegressor(**self.config["regression"]["xgb"])
            self.models_reg["xgb"].fit(Xtr_s, ytr, verbose=False)
            pred = self.models_reg["xgb"].predict(Xva_s)
            metrics["xgb"] = self._reg_metrics(yva, pred)

        if HAS_LGBM:
            self.models_reg["lgbm"] = lgb.LGBMRegressor(**self.config["regression"]["lgbm"])
            self.models_reg["lgbm"].fit(Xtr_s, ytr)
            pred = self.models_reg["lgbm"].predict(Xva_s)
            metrics["lgbm"] = self._reg_metrics(yva, pred)

        # Pesos inversos al MSE
        inv = {k: 1.0 / (m["mse"] + 1e-9) for k, m in metrics.items()}
        s = sum(inv.values())
        self.ensemble_weights_reg = {k: v / s for k, v in inv.items()}
        self.training_history["metrics_reg"] = metrics
        return metrics

    # ------------------- TRAIN: CLS -------------------
    def train_classification(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> Dict:
        # asegurar que usamos la MISMA lista de columnas que en regresión
        if self.feature_names is None:
            self.feature_names = Xtr.columns.tolist()
        else:
            # reordenar por si acaso
            Xtr = Xtr[self.feature_names]
            Xva = Xva[self.feature_names]

        # Scalers exclusivos para clasificación
        self.scalers["std_cls"] = StandardScaler().fit(Xtr)
        self.scalers["rob_cls"] = RobustScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(self.scalers["std_cls"].transform(Xtr), columns=Xtr.columns)
        Xva_s = pd.DataFrame(self.scalers["std_cls"].transform(Xva), columns=Xva.columns)
        Xtr_r = pd.DataFrame(self.scalers["rob_cls"].transform(Xtr), columns=Xtr.columns)
        Xva_r = pd.DataFrame(self.scalers["rob_cls"].transform(Xva), columns=Xva.columns)

        metrics = {}

        # RF
        rf = RandomForestClassifier(**self.config["classification"]["rf"])
        self.models_cls["rf"] = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
        self.models_cls["rf"].fit(Xtr_r, ytr)
        p = self.models_cls["rf"].predict_proba(Xva_r)[:, 1]
        metrics["rf"] = self._cls_metrics(yva, p)

        # GBC
        gbc = GradientBoostingClassifier(**self.config["classification"]["gbc"])
        self.models_cls["gbc"] = CalibratedClassifierCV(gbc, method="sigmoid", cv=3)
        self.models_cls["gbc"].fit(Xtr_r, ytr)
        p = self.models_cls["gbc"].predict_proba(Xva_r)[:, 1]
        metrics["gbc"] = self._cls_metrics(yva, p)

        # LogReg
        lr = LogisticRegression(**self.config["classification"]["logreg"])
        self.models_cls["logreg"] = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
        self.models_cls["logreg"].fit(Xtr_s, ytr)
        p = self.models_cls["logreg"].predict_proba(Xva_s)[:, 1]
        metrics["logreg"] = self._cls_metrics(yva, p)

        # opcionales
        if HAS_XGB:
            xgc = xgb.XGBClassifier(**self.config["classification"]["xgb"])
            self.models_cls["xgb"] = CalibratedClassifierCV(xgc, method="sigmoid", cv=3)
            self.models_cls["xgb"].fit(Xtr_s, ytr)
            p = self.models_cls["xgb"].predict_proba(Xva_s)[:, 1]
            metrics["xgb"] = self._cls_metrics(yva, p)
        if HAS_LGBM:
            lgbc = lgb.LGBMClassifier(**self.config["classification"]["lgbm"])
            self.models_cls["lgbm"] = CalibratedClassifierCV(lgbc, method="sigmoid", cv=3)
            self.models_cls["lgbm"].fit(Xtr_s, ytr)
            p = self.models_cls["lgbm"].predict_proba(Xva_s)[:, 1]
            metrics["lgbm"] = self._cls_metrics(yva, p)

        # Pesos inversos a Brier
        inv = {k: 1.0 / (m["brier"] + 1e-9) for k, m in metrics.items()}
        s = sum(inv.values())
        self.ensemble_weights_cls = {k: v / s for k, v in inv.items()}
        self.training_history["metrics_cls"] = metrics
        return metrics

    # ------------------- PREDICT -------------------
    def predict_regression(self, X: pd.DataFrame) -> Dict:
        """Predicción de mm con ensamble y banda por dispersión entre modelos."""
        X = X[self.feature_names]  # asegurar orden
        Xs = pd.DataFrame(self.scalers["std_reg"].transform(X), columns=X.columns)
        Xr = pd.DataFrame(self.scalers["rob_reg"].transform(X), columns=X.columns)

        preds = {}
        for name, mdl in self.models_reg.items():
            if name in {"ridge", "elastic", "xgb", "lgbm"}:
                preds[name] = float(mdl.predict(Xs)[0])
            else:
                preds[name] = float(mdl.predict(Xr)[0])

        weights = self.ensemble_weights_reg or {k: 1.0 / len(preds) for k in preds}
        yhat = sum(weights[k] * v for k, v in preds.items())
        std = float(np.std(list(preds.values())))
        return {
            "prediction_mm": yhat,
            "individual_predictions_mm": preds,
            "uncertainty_mm": std,
            "confidence_interval_mm": {
                "lower": max(0.0, yhat - 1.96 * std),
                "upper": yhat + 1.96 * std
            },
            "ensemble_weights_reg": weights
        }

    def predict_classification(self, X: pd.DataFrame) -> Dict:
        """Probabilidad de lluvia (clasificación) con ensamble calibrado."""
        X = X[self.feature_names]  # asegurar orden
        Xs = pd.DataFrame(self.scalers["std_cls"].transform(X), columns=X.columns)
        Xr = pd.DataFrame(self.scalers["rob_cls"].transform(X), columns=X.columns)

        probs = {}
        for name, mdl in self.models_cls.items():
            if name in {"logreg", "xgb", "lgbm"}:
                probs[name] = float(mdl.predict_proba(Xs)[0, 1])
            else:
                probs[name] = float(mdl.predict_proba(Xr)[0, 1])

        weights = self.ensemble_weights_cls or {k: 1.0 / len(probs) for k in probs}
        phat = sum(weights[k] * v for k, v in probs.items())
        std = float(np.std(list(probs.values())))
        return {
            "probability": phat,
            "individual_probabilities": probs,
            "uncertainty": std,
            "confidence_level": "high" if std < 0.1 else ("medium" if std < 0.25 else "low"),
            "ensemble_weights_cls": weights
        }

    @staticmethod
    def mm_to_prob_calibrated(mm: float, rain_threshold: float, hist_std: float) -> float:
        """Transfiere mm a probabilidad usando logística centrada en el umbral y escalada por la σ histórica."""
        scale = _safe_std(hist_std)
        return float(_sigmoid((mm - rain_threshold) / scale))

    # ------------------- MÉTRICAS -------------------
    @staticmethod
    def _reg_metrics(y_true, y_pred) -> Dict:
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    @staticmethod
    def _cls_metrics(y_true, p) -> Dict:
        # AUC puede fallar si y_true es constante; manejarlo
        try:
            auc = roc_auc_score(y_true, p)
        except Exception:
            auc = None
        return {
            "brier": brier_score_loss(y_true, p),
            "auc": auc
        }

    # ------------------- SERIALIZACIÓN -------------------
    def save(self, prefix: str = "climate_adv") -> str:
        out = self.model_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "models_reg.pkl", "wb") as f:
            pickle.dump(self.models_reg, f)
        with open(out / "models_cls.pkl", "wb") as f:
            pickle.dump(self.models_cls, f)
        with open(out / "scalers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)
        meta = {
            "feature_names": self.feature_names,
            "ensemble_weights_reg": self.ensemble_weights_reg,
            "ensemble_weights_cls": self.ensemble_weights_cls,
            "training_history": self.training_history
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        return str(out)

    def load(self, path: str) -> None:
        path = Path(path)
        with open(path / "models_reg.pkl", "rb") as f:
            self.models_reg = pickle.load(f)
        with open(path / "models_cls.pkl", "rb") as f:
            self.models_cls = pickle.load(f)
        with open(path / "scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
        with open(path / "metadata.json", "r") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        self.ensemble_weights_reg = meta["ensemble_weights_reg"]
        self.ensemble_weights_cls = meta["ensemble_weights_cls"]
        self.training_history = meta["training_history"]


# ------------------- INTEGRACIÓN CON ANALIZADOR -------------------

def integrate_with_analyzer(analyzer,
                            predictor: AdvancedClimatePredictor,
                            window_df: pd.DataFrame,
                            target_date: str,
                            rain_threshold: float,
                            hist_precip_std: Optional[float] = None) -> Dict:
    """
    Entrena (regresión+clasificación) y predice para la fecha objetivo usando la última
    fila de features históricas como proxy del estado reciente.
    """
    # 1) Features
    feats = predictor.create_features(window_df, target_col="precip").copy()

    # Etiqueta para clasificación
    feats["rained"] = (feats["precip"] > rain_threshold).astype(int)

    # Conjunto CONSISTENTE de features para ambas tareas
    feature_cols = [c for c in feats.columns if c not in ["date", "precip", "rained"]]

    # Limpiezas
    feats = feats.sort_values("date").reset_index(drop=True)
    feats_clean_reg = feats.dropna(subset=["precip"] + feature_cols)
    feats_clean_cls = feats.dropna(subset=["rained"] + feature_cols)

    if len(feats_clean_reg) < 60 or len(feats_clean_cls) < 60:
        return {"error": "Datos insuficientes para ML avanzado (<60 muestras con features válidas)"}

    # 2) Splits usando mismas columnas
    Xtr, ytr, Xva, yva = predictor._time_split(feats_clean_reg, "precip", feature_cols)
    Xtrc, ytrc, Xvac, yvac = predictor._time_split(feats_clean_cls, "rained", feature_cols)

    # Ajustar lista única
    predictor.feature_names = feature_cols

    # Entrenamientos
    metrics_reg = predictor.train_regression(Xtr, ytr, Xva, yva)
    metrics_cls = predictor.train_classification(Xtrc, ytrc, Xvac, yvac)

    # 4) Vector de predicción (última fila)
    last = feats.iloc[[-1]][feature_cols].copy()

    # 5) Predicciones
    reg = predictor.predict_regression(last)
    cls = predictor.predict_classification(last)

    # 6) Calibración mm->prob
    hist_std = float(hist_precip_std) if hist_precip_std is not None else float(window_df["precip"].std() or 1.0)
    prob_from_mm = predictor.mm_to_prob_calibrated(reg["prediction_mm"], rain_threshold, hist_std)

    # 7) Fusión prob final (50% cls + 50% mm calibrado)
    prob_final = float(0.5 * cls["probability"] + 0.5 * prob_from_mm)

    return {
        "advanced_ml": {
            "precipitation_mm": float(reg["prediction_mm"]),
            "confidence_interval": reg["confidence_interval_mm"],
            "uncertainty": reg["uncertainty_mm"],
            "rain_probability_from_mm": prob_from_mm,
            "rain_probability_from_classifier": float(cls["probability"]),
            "rain_probability_fused": prob_final,
            "will_rain": bool(prob_final > 0.5),
            "confidence_level": cls["confidence_level"],
            "individual_models": {
                **{f"reg_{k}": v for k, v in reg["individual_predictions_mm"].items()},
                **{f"cls_{k}": v for k, v in cls["individual_probabilities"].items()},
            },
        },
        "model_weights": {
            "regression": predictor.ensemble_weights_reg,
            "classification": predictor.ensemble_weights_cls
        },
        "training_metrics": {
            "regression": metrics_reg,
            "classification": metrics_cls
        }
    }
