# advanced_ml_predictor.py
# ------------------------------------------------------------
# ML avanzado para AtmosAtlas:
# - Regresión (mm de precipitación)
# - Clasificación (llueve > umbral), con calibración
# - Ensamble con pesos por error de validación
# - Conversión mm -> prob con calibración logística basada en ventana histórica
# - Métricas y guardado/carga
# - *** Rendimiento ***: generación de features sin fragmentación (concat masivo)
# - *** Robustez ***: sanitización y columnas únicas (sin duplicados)
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
from datetime import datetime

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
from sklearn.model_selection import TimeSeriesSplit

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


# ------------------- UTILIDADES -------------------
def _safe_std(x: float, min_val: float = 1e-3) -> float:
    return float(max(min_val, x))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def _dedupe_list_keep_order(items: List[str]) -> List[str]:
    """Elimina duplicados preservando el orden."""
    return list(dict.fromkeys(items))


def _sanitize_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    fill: float = 0.0,
) -> pd.DataFrame:
    """
    - Se usan solo columnas esperadas (si se proveen)
    - Todas las columnas -> numéricas
    - Sin NaNs (rellena con 'fill')
    - dtype float32
    - Sin columnas duplicadas
    """
    if feature_names is not None:
        # si hay duplicados en feature_names, los quitamos
        feature_names = _dedupe_list_keep_order(feature_names)
        df = df.loc[:, feature_names]

    # elimina duplicados si los hubiera
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # numéricas
    df = df.select_dtypes(include=[np.number]).copy()

    # relleno y casteo
    df = df.fillna(fill).astype(np.float32)

    # reinyecta columnas faltantes como 0 para respetar orden
    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        for c in missing:
            df[c] = np.float32(fill)
        df = df[feature_names]

    return df


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
                "ridge": dict(alpha=1.0, random_state=42),
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
                "logreg": dict(max_iter=2000, C=1.0, solver="lbfgs",
                               random_state=42, class_weight="balanced"),
                "cv_splits": 3
            }
        }
        if HAS_XGB:
            cfg["regression"]["xgb"] = dict(
                n_estimators=700, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                tree_method="hist",
                verbosity=0
            )
            cfg["classification"]["xgb"] = dict(
                n_estimators=700, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss", verbosity=0
            )
        if HAS_LGBM:
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
        *** Anti-fragmentación: construye todo en un diccionario y concatena una sola vez. ***
        Usa solo información pasada (shift) para evitar leakage.
        """
        d = df.copy()
        assert "date" in d.columns, "El DataFrame debe tener columna 'date'"
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)

        feats: dict[str, pd.Series] = {}

        # básicos (prefijo cal_ para no colisionar con posibles columnas existentes)
        year = d["date"].dt.year
        month = d["date"].dt.month
        doy = d["date"].dt.dayofyear
        week = d["date"].dt.isocalendar().week.astype(int)
        is_weekend = (d["date"].dt.dayofweek >= 5).astype(int)

        feats["cal_year"] = year
        feats["cal_month"] = month
        feats["cal_doy"] = doy
        feats["cal_week"] = week
        feats["cal_is_weekend"] = is_weekend

        # cíclicos (también con prefijo)
        feats["cal_month_sin"] = np.sin(2 * np.pi * month / 12)
        feats["cal_month_cos"] = np.cos(2 * np.pi * month / 12)
        feats["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feats["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # lags + rolling
        base_cols = [c for c in ["precip", "t2m", "rh2m", "wind", "ps", "rad"] if c in d.columns]
        lag_list = [1, 3, 7, 14, 30]
        win_list = [7, 14, 30, 60]

        for c in base_cols:
            s = pd.to_numeric(d[c], errors="coerce")
            # lags
            for lag in lag_list:
                feats[f"{c}_lag_{lag}"] = s.shift(lag)
            # rolling
            for w in win_list:
                roll = s.rolling(w, min_periods=2)
                feats[f"{c}_roll_mean_{w}"] = roll.mean()
                feats[f"{c}_roll_std_{w}"]  = roll.std()
                feats[f"{c}_roll_sum_{w}"]  = roll.sum()
                feats[f"{c}_roll_max_{w}"]  = roll.max()
                feats[f"{c}_roll_min_{w}"]  = roll.min()

        # cambios (diff)
        for c in [c for c in ["t2m", "ps", "rh2m"] if c in d.columns]:
            s = pd.to_numeric(d[c], errors="coerce")
            feats[f"{c}_d1"]  = s.diff(1)
            feats[f"{c}_d7"]  = s.diff(7)
            feats[f"{c}_d30"] = s.diff(30)

        # índices sencillos
        if "t2m" in d.columns and "rh2m" in d.columns:
            t2m = pd.to_numeric(d["t2m"], errors="coerce")
            rh  = pd.to_numeric(d["rh2m"], errors="coerce")
            feats["humidex"]   = t2m + 0.5555 * (6.112 * 10 ** (7.5 * t2m / (237.7 + t2m)) * rh / 100 - 10)
            feats["th_index"]  = t2m * (1 + rh / 100.0)

        # anomalías mensuales (z-score por mes)
        for c in [c for c in ["precip", "t2m", "rh2m"] if c in d.columns]:
            s = pd.to_numeric(d[c], errors="coerce")
            m_mean = s.groupby(month).transform("mean")
            m_std  = s.groupby(month).transform("std")
            feats[f"{c}_anom"] = (s - m_mean) / (m_std.replace(0, np.nan) + 1e-5)

        # concat masivo y eliminación de columnas duplicadas
        feat_df = pd.DataFrame(feats, index=d.index)
        out = pd.concat([d, feat_df], axis=1, copy=False)
        out = out.loc[:, ~out.columns.duplicated()].copy()
        return out

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
        feats = _dedupe_list_keep_order(feats)
        n = len(dfc)
        tr_end = int(n * 0.7)
        va_end = int(n * 0.85)
        X_train, y_train = dfc.iloc[:tr_end][feats], dfc.iloc[:tr_end][target_col]
        X_val, y_val     = dfc.iloc[tr_end:va_end][feats], dfc.iloc[tr_end:va_end][target_col]
        return X_train, y_train, X_val, y_val

    # ------------------- TRAIN: REG -------------------
    def train_regression(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> Dict:
        if self.feature_names is None:
            self.feature_names = _dedupe_list_keep_order(Xtr.columns.tolist())

        # Sanitización consistente
        Xtr = _sanitize_features(Xtr, self.feature_names, fill=0.0)
        Xva = _sanitize_features(Xva, self.feature_names, fill=0.0)
        ytr = pd.to_numeric(ytr, errors="coerce").fillna(0.0).astype(np.float32)
        yva = pd.to_numeric(yva, errors="coerce").fillna(0.0).astype(np.float32)

        # Scalers para regresión
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
            self.models_reg["xgb"].fit(Xtr_s.values, ytr.values, verbose=False)
            pred = self.models_reg["xgb"].predict(Xva_s.values)
            metrics["xgb"] = self._reg_metrics(yva, pred)

        if HAS_LGBM:
            self.models_reg["lgbm"] = lgb.LGBMRegressor(**self.config["regression"]["lgbm"])
            self.models_reg["lgbm"].fit(Xtr_s, ytr)
            pred = self.models_reg["lgbm"].predict(Xva_s)
            metrics["lgbm"] = self._reg_metrics(yva, pred)

        # Pesos inversos al MSE
        inv = {k: 1.0 / (m["mse"] + 1e-9) for k, m in metrics.items()}
        s = sum(inv.values()) or 1.0
        self.ensemble_weights_reg = {k: v / s for k, v in inv.items()}
        self.training_history["metrics_reg"] = metrics
        self.training_history["timestamp"] = datetime.utcnow().isoformat()
        return metrics

    # ------------------- TRAIN: CLS -------------------
    def train_classification(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> Dict:
        if self.feature_names is None:
            self.feature_names = _dedupe_list_keep_order(Xtr.columns.tolist())
        else:
            self.feature_names = _dedupe_list_keep_order(self.feature_names)

        # Sanitización
        Xtr = _sanitize_features(Xtr, self.feature_names, fill=0.0)
        Xva = _sanitize_features(Xva, self.feature_names, fill=0.0)
        ytr = pd.to_numeric(ytr, errors="coerce").fillna(0.0).astype(np.float32)
        yva = pd.to_numeric(yva, errors="coerce").fillna(0.0).astype(np.float32)

        # Scalers
        self.scalers["std_cls"] = StandardScaler().fit(Xtr)
        self.scalers["rob_cls"] = RobustScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(self.scalers["std_cls"].transform(Xtr), columns=Xtr.columns)
        Xva_s = pd.DataFrame(self.scalers["std_cls"].transform(Xva), columns=Xva.columns)
        Xtr_r = pd.DataFrame(self.scalers["rob_cls"].transform(Xtr), columns=Xtr.columns)
        Xva_r = pd.DataFrame(self.scalers["rob_cls"].transform(Xva), columns=Xva.columns)

        # CV temporal para calibración
        n_splits = max(2, int(self.config["classification"].get("cv_splits", 3)))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics = {}

        # RF
        rf = RandomForestClassifier(**self.config["classification"]["rf"])
        self.models_cls["rf"] = CalibratedClassifierCV(rf, method="sigmoid", cv=tscv)
        self.models_cls["rf"].fit(Xtr_r, ytr)
        p = self.models_cls["rf"].predict_proba(Xva_r)[:, 1]
        metrics["rf"] = self._cls_metrics(yva, p)

        # GBC
        gbc = GradientBoostingClassifier(**self.config["classification"]["gbc"])
        self.models_cls["gbc"] = CalibratedClassifierCV(gbc, method="sigmoid", cv=tscv)
        self.models_cls["gbc"].fit(Xtr_r, ytr)
        p = self.models_cls["gbc"].predict_proba(Xva_r)[:, 1]
        metrics["gbc"] = self._cls_metrics(yva, p)

        # LogReg
        lr = LogisticRegression(**self.config["classification"]["logreg"])
        self.models_cls["logreg"] = CalibratedClassifierCV(lr, method="sigmoid", cv=tscv)
        self.models_cls["logreg"].fit(Xtr_s, ytr)
        p = self.models_cls["logreg"].predict_proba(Xva_s)[:, 1]
        metrics["logreg"] = self._cls_metrics(yva, p)

        # opcionales
        if HAS_XGB:
            xgc = xgb.XGBClassifier(**self.config["classification"]["xgb"])
            self.models_cls["xgb"] = CalibratedClassifierCV(xgc, method="sigmoid", cv=tscv)
            self.models_cls["xgb"].fit(Xtr_s.values, ytr.values)
            p = self.models_cls["xgb"].predict_proba(Xva_s.values)[:, 1]
            metrics["xgb"] = self._cls_metrics(yva, p)

        if HAS_LGBM:
            lgbc = lgb.LGBMClassifier(**self.config["classification"]["lgbm"])
            self.models_cls["lgbm"] = CalibratedClassifierCV(lgbc, method="sigmoid", cv=tscv)
            self.models_cls["lgbm"].fit(Xtr_s, ytr)
            p = self.models_cls["lgbm"].predict_proba(Xva_s)[:, 1]
            metrics["lgbm"] = self._cls_metrics(yva, p)

        # Pesos inversos a Brier
        inv = {k: 1.0 / (m["brier"] + 1e-9) for k, m in metrics.items()}
        s = sum(inv.values()) or 1.0
        self.ensemble_weights_cls = {k: v / s for k, v in inv.items()}
        self.training_history["metrics_cls"] = metrics
        self.training_history["timestamp"] = datetime.utcnow().isoformat()
        return metrics

    # ------------------- PREDICT -------------------
    def predict_regression(self, X: pd.DataFrame) -> Dict:
        """Predicción de mm con ensamble y banda por dispersión entre modelos."""
        X = _sanitize_features(X, self.feature_names, fill=0.0)
        Xs = pd.DataFrame(self.scalers["std_reg"].transform(X), columns=X.columns)
        Xr = pd.DataFrame(self.scalers["rob_reg"].transform(X), columns=X.columns)

        preds = {}
        for name, mdl in self.models_reg.items():
            if name in {"ridge", "elastic", "xgb", "lgbm"}:
                data = Xs.values if name == "xgb" else Xs
                preds[name] = float(mdl.predict(data)[0])
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
        X = _sanitize_features(X, self.feature_names, fill=0.0)
        Xs = pd.DataFrame(self.scalers["std_cls"].transform(X), columns=X.columns)
        Xr = pd.DataFrame(self.scalers["rob_cls"].transform(X), columns=X.columns)

        probs = {}
        for name, mdl in self.models_cls.items():
            if name in {"logreg", "xgb", "lgbm"}:
                data = Xs.values if name == "xgb" else Xs
                probs[name] = float(mdl.predict_proba(data)[0, 1])
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

    # Conjunto CONSISTENTE de features para ambas tareas (sin duplicados)
    feature_cols = [c for c in feats.columns if c not in ["date", "precip", "rained"]]
    feature_cols = _dedupe_list_keep_order(feature_cols)

    # Limpiezas/sorting y eliminación de duplicados por si acaso
    feats = feats.sort_values("date").reset_index(drop=True)
    feats = feats.loc[:, ~feats.columns.duplicated()].copy()

    feats_clean_reg = feats.dropna(subset=["precip"] + feature_cols)
    feats_clean_cls = feats.dropna(subset=["rained"] + feature_cols)

    if len(feats_clean_reg) < 60 or len(feats_clean_cls) < 60:
        return {"error": "Datos insuficientes para ML avanzado (<60 muestras con features válidas)"}

    # 2) Splits usando mismas columnas
    Xtr, ytr, Xva, yva     = predictor._time_split(feats_clean_reg, "precip", feature_cols)
    Xtrc, ytrc, Xvac, yvac = predictor._time_split(feats_clean_cls, "rained", feature_cols)

    # Ajustar lista única (sin duplicados)
    predictor.feature_names = _dedupe_list_keep_order(feature_cols)

    # Entrenamientos
    metrics_reg = predictor.train_regression(Xtr, ytr, Xva, yva)
    metrics_cls = predictor.train_classification(Xtrc, ytrc, Xvac, yvac)

    # 4) Vector de predicción (última fila)
    last = feats.iloc[[-1]][predictor.feature_names].copy()

    # 5) Predicciones
    reg = predictor.predict_regression(last)
    cls = predictor.predict_classification(last)

    # 6) Calibración mm->prob
    hist_std = float(hist_precip_std) if hist_precip_std is not None else float((window_df["precip"].std() or 1.0))
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
