# analizador_integrado.py
# ------------------------------------------------------------
# Orquestador de análisis:
# - Descarga (POWER) o ingestión multi-fuente (EnhancedDataFetcher)
# - Estadísticas históricas (tu analizador base)
# - ML básico (LogReg + RF)
# - ML avanzado (regresión mm + clasificación) con calibración
# - Consenso ponderado y EXPLICABILIDAD (climatología, últimos N días, drivers del modelo)
# - Reporte final con cobertura de datos (rango histórico)
# ------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

from analizador import ClimateAnalyzer  # baseline (frecuencias + ML básico)
from advanced_ml_predictor import AdvancedClimatePredictor, integrate_with_analyzer
from enhanced_data_fetcher import EnhancedDataFetcher, DataQualityAnalyzer


def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "NA"
    if isinstance(v, float):
        return f"{v:.2f}{unit}"
    return f"{v}{unit}"


class IntegratedClimateAnalyzer:
    def __init__(self):
        self.basic = ClimateAnalyzer()
        self.adv = AdvancedClimatePredictor()
        self.results: Dict = {}

    def analyze(
        self,
        lat: float, lon: float,
        target_date: str,
        start_date: str = "1990-01-01",
        end_date: str = "2024-12-31",
        window_days: int = 7,
        rain_threshold: float = 0.5,
        use_multisource: bool = False,
        explain_recent_days: int = 14
    ) -> Dict:
        """
        Pipeline maestro:
        - Datos (POWER o Ensamble multi-fuente)
        - Ventana histórica para target_date
        - Estadísticas + ML básico
        - ML avanzado + calibración
        - Recomendación final + Explicabilidad
        """
        # 1) Datos
        if use_multisource:
            fetcher = EnhancedDataFetcher(verbose=False)
            df = fetcher.fetch_ensemble_data(lat, lon, start_date, end_date)
            quality = DataQualityAnalyzer.assess(df, provenance=fetcher.last_provenance)
            if df is None or df.empty:
                # degradación elegante a POWER-only
                self.basic.download_data(lat, lon, start_date, end_date)
            else:
                self.basic.df = df.copy()
            self.results["data_quality"] = quality
        else:
            self.basic.download_data(lat, lon, start_date, end_date)

        # 2) Ventana histórica (climatología entorno del DOY)
        win = self.basic.get_historical_window(target_date, window_days)

        # 3) Estadísticas históricas
        stats = self.basic.calculate_probabilities(win, rain_threshold=rain_threshold)
        self.results["statistics"] = stats
        self.results["target_date"] = target_date
        self.results["location"] = {"lat": lat, "lon": lon}
        self.results["config"] = {
            "rain_threshold": rain_threshold,
            "window_days": window_days,
            "explain_recent_days": explain_recent_days,
            "use_multisource": bool(use_multisource)
        }

        # 4) ML básico
        ml_basic = self.basic.train_ml_models(win, rain_threshold)
        if ml_basic:
            pred_basic = self.basic.predict_for_date(target_date, win, ml_basic)
            self.results["ml_basic"] = {"prediction": pred_basic, "metrics": ml_basic["metrics"]}

        # 5) ML avanzado (reg + cls + calibración mm->prob)
        adv = integrate_with_analyzer(
            self.basic, self.adv, win, target_date, rain_threshold,
            hist_precip_std=stats.get("precip_std", 1.0)
        )
        self.results["ml_advanced"] = adv

        # 6) Explicabilidad
        self.results["explainability"] = self._build_explainability(win, recent_days=explain_recent_days)

        # 7) Recomendación final (consenso)
        self._final_consensus()

        # 8) Cobertura de datos (rango histórico disponible)
        cov = self._coverage_summary()
        self.results["data_coverage"] = cov

        return self.results

    # ------------------ EXPLICABILIDAD ------------------
    def _summary_block(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {}
        out = {}
        def agg(col, fn):
            return float(fn(df[col].dropna())) if col in df.columns and df[col].notna().any() else None

        out["precip_sum_mm"]     = agg("precip", np.nansum)
        out["precip_mean_mm"]    = agg("precip", np.nanmean)
        out["t2m_mean_c"]        = agg("t2m", np.nanmean)
        out["rh2m_mean_pct"]     = agg("rh2m", np.nanmean)
        out["wind_mean_ms"]      = agg("wind", np.nanmean)
        out["rad_mean_kwh_m2d"]  = agg("rad", np.nanmean)
        out["ps_mean_kpa"]       = agg("ps", np.nanmean)
        out["n_days"]            = int(len(df))
        return out

    def _top_model_features(self, feats_df: pd.DataFrame, last_vec: pd.Series, top_k: int = 5) -> List[Dict]:
        """Extrae las features más influyentes agregando importancias de modelos de cls + reg."""
        importances = pd.Series(0.0, index=feats_df.columns, dtype=float)
        # Clasificación: estimadores dentro de CalibratedClassifierCV
        for name, mdl in self.adv.models_cls.items():
            if hasattr(mdl, "calibrated_classifiers_"):
                for cc in mdl.calibrated_classifiers_:
                    est = getattr(cc, "estimator", None)
                    if est is None:
                        continue
                    if hasattr(est, "feature_importances_"):
                        importances += pd.Series(est.feature_importances_, index=feats_df.columns)
                    elif hasattr(est, "coef_"):
                        importances += pd.Series(np.abs(est.coef_).ravel(), index=feats_df.columns)
        # Regresión
        for name, mdl in self.adv.models_reg.items():
            if hasattr(mdl, "feature_importances_"):
                importances += pd.Series(mdl.feature_importances_, index=feats_df.columns)
            elif hasattr(mdl, "coef_"):
                importances += pd.Series(np.abs(mdl.coef_).ravel(), index=feats_df.columns)

        if (importances == 0).all():
            return []

        imp_norm = (importances / (importances.sum() + 1e-9)).sort_values(ascending=False)
        top = imp_norm.head(top_k).index.tolist()

        # z-score de esas features respecto al conjunto (dirección del driver)
        zscores = {}
        for f in top:
            mu = float(np.nanmean(feats_df[f])) if feats_df[f].notna().any() else 0.0
            sd = float(np.nanstd(feats_df[f])) if feats_df[f].notna().any() else 1.0
            z = (float(last_vec[f]) - mu) / (sd + 1e-9)
            zscores[f] = z

        return [
            {"feature": f, "importance": float(imp_norm[f]), "zscore_last": float(zscores[f]), "last_value": float(last_vec[f])}
            for f in top
        ]

    def _build_explainability(self, window_df: pd.DataFrame, recent_days: int = 14) -> Dict:
        """Construye narrativa: climatología (±window), últimos N días, y drivers del modelo."""
        df_all = self.basic.df.sort_values("date").reset_index(drop=True)

        # Últimos N días del dataset disponible
        if len(df_all) > 0 and "date" in df_all.columns:
            last_date = df_all["date"].max()
            cutoff = last_date - pd.Timedelta(days=recent_days - 1)
            recent_df = df_all[df_all["date"] >= cutoff].copy()
        else:
            recent_df = pd.DataFrame()

        # Resúmenes
        recent_summary = self._summary_block(recent_df)
        climatology_summary = self._summary_block(window_df)

        # Heurísticas para narrativa (usando recent_summary si existe, si no climatology)
        S = recent_summary if recent_summary else climatology_summary
        rt = self.results.get("config", {}).get("rain_threshold", 0.5)
        reasons = []

        # Precipitación
        if S.get("precip_sum_mm") is not None:
            if S["precip_sum_mm"] < rt:
                reasons.append(f"Poca lluvia reciente: Σ{recent_days}d = {_fmt(S['precip_sum_mm'],' mm')}")
            else:
                reasons.append(f"Acumulado reciente de lluvia moderado/alto: Σ{recent_days}d = {_fmt(S['precip_sum_mm'],' mm')}")

        # Humedad
        if S.get("rh2m_mean_pct") is not None:
            if S["rh2m_mean_pct"] < 60:
                reasons.append(f"Humedad media baja ({_fmt(S['rh2m_mean_pct'],'%')})")
            elif S["rh2m_mean_pct"] > 80:
                reasons.append(f"Humedad media alta ({_fmt(S['rh2m_mean_pct'],'%')})")

        # Radiación
        if S.get("rad_mean_kwh_m2d") is not None:
            if S["rad_mean_kwh_m2d"] >= 5.0:
                reasons.append(f"Radiación solar elevada ({_fmt(S['rad_mean_kwh_m2d'])} kWh/m²·día) → menor nubosidad")
            elif S["rad_mean_kwh_m2d"] <= 3.0:
                reasons.append(f"Radiación solar baja ({_fmt(S['rad_mean_kwh_m2d'])} kWh/m²·día) → mayor nubosidad")

        # Presión
        if S.get("ps_mean_kpa") is not None:
            if S["ps_mean_kpa"] >= 101.5:
                reasons.append(f"Presión media alta ({_fmt(S['ps_mean_kpa'],' kPa')}) → condiciones estables")
            elif S["ps_mean_kpa"] <= 100.5:
                reasons.append(f"Presión media baja ({_fmt(S['ps_mean_kpa'],' kPa')}) → inestabilidad")

        # Viento
        if S.get("wind_mean_ms") is not None:
            if S["wind_mean_ms"] < 2.0:
                reasons.append(f"Viento débil ({_fmt(S['wind_mean_ms'],' m/s')}) → poca advección")
            elif S["wind_mean_ms"] > 8.0:
                reasons.append(f"Viento fuerte ({_fmt(S['wind_mean_ms'],' m/s')})")

        # Drivers del modelo (top features) usando el predictor ya entrenado
        feats = self.adv.create_features(window_df)
        feat_cols = [c for c in feats.columns if c not in ["date", "precip"]]
        if not feats.empty and len(feat_cols) > 0:
            feats = feats.dropna(subset=feat_cols)
            if not feats.empty:
                last_vec = feats.iloc[-1][feat_cols]
                top_feats = self._top_model_features(feats[feat_cols], last_vec)
            else:
                top_feats = []
        else:
            top_feats = []

        # Texto de explicación
        explanation_text = " · ".join(reasons) if reasons else "Sin señales recientes claras; se usa climatología y modelos."

        return {
            "recent_days_used": recent_days,
            "recent_summary": recent_summary,
            "climatology_summary": climatology_summary,
            "drivers": reasons,
            "model_top_features": top_feats,
            "explanation_text": explanation_text
        }

    # ------------------ COBERTURA ------------------
    def _coverage_summary(self) -> Dict:
        df = getattr(self.basic, "df", pd.DataFrame())
        if df is None or df.empty or "date" not in df.columns:
            return {}
        dmin = pd.to_datetime(df["date"]).min()
        dmax = pd.to_datetime(df["date"]).max()
        ndays = int((dmax.normalize() - dmin.normalize()).days) + 1
        nyears = (dmax.year - dmin.year) + 1
        return {
            "start": dmin.strftime("%Y-%m-%d"),
            "end": dmax.strftime("%Y-%m-%d"),
            "n_days": ndays,
            "n_years": nyears
        }

    # ------------------ CONSENSO ------------------
    def _final_consensus(self):
        s = float(self.results["statistics"].get("rain_probability", 0.0))               # histórica
        b = float(self.results.get("ml_basic", {}).get("prediction", {}).get("ensemble_prob", 0.0))  # ML básico
        a = float(self.results.get("ml_advanced", {}).get("advanced_ml", {}).get("rain_probability_fused", 0.0))  # ML avanzado

        probs = [s, b, a]
        weights = [0.25, 0.25, 0.50]  # más peso al avanzado
        wprob = float(sum(p*w for p, w in zip(probs, weights)))
        votes = sum(1 for p in probs if p > 0.5)
        consensus = votes / len(probs)

        decision = "LLOVERÁ" if wprob > 0.5 else "NO LLOVERÁ"
        expl = self.results.get("explainability", {}).get("explanation_text", "")

        self.results["final_recommendation"] = {
            "will_rain": bool(wprob > 0.5),
            "probability": wprob,
            "consensus": float(consensus),
            "confidence": "high" if consensus >= 0.8 or consensus <= 0.2 else "medium",
            "individual_predictions": {"statistical": s, "ml_basic": b, "ml_advanced": a},
            "reasoning": f"{decision} porque {expl}" if expl else decision
        }


def _cli():
    ap = argparse.ArgumentParser(description="Analizador integrado AtmosAtlas (robusto con explicabilidad)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--target_date", required=True)
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--window_days", type=int, default=7)
    ap.add_argument("--rain_threshold", type=float, default=0.5)
    ap.add_argument("--multisource", action="store_true", help="POWER + IMERG/MODIS si hay credenciales")
    ap.add_argument("--explain_recent_days", type=int, default=14, help="Días recientes para resumen explicativo")
    ap.add_argument("--output", help="Guardar resultados JSON")
    args = ap.parse_args()

    analyzer = IntegratedClimateAnalyzer()
    res = analyzer.analyze(args.lat, args.lon, args.target_date,
                           start_date=args.start, end_date=args.end,
                           window_days=args.window_days, rain_threshold=args.rain_threshold,
                           use_multisource=args.multisource,
                           explain_recent_days=args.explain_recent_days)

    final = res["final_recommendation"]
    stat = res["statistics"].get("rain_probability", 0.0)
    basic = res.get("ml_basic", {}).get("prediction", {}).get("ensemble_prob", 0.0)
    adv = res.get("ml_advanced", {}).get("advanced_ml", {}).get("rain_probability_fused", 0.0)
    expl = res.get("explainability", {})
    rsum = expl.get("recent_summary", {})
    cov = res.get("data_coverage", {})
    dq = res.get("data_quality", {})

    print("\n" + "="*72)
    print(f"🌍 ({args.lat:.4f}, {args.lon:.4f})  📅 {args.target_date}")
    if cov:
        print(f"🗂  Cobertura de datos: {cov.get('start','NA')} → {cov.get('end','NA')} "
              f"({cov.get('n_years','NA')} años, {cov.get('n_days','NA')} días)")
    if dq:
        rel = dq.get("reliability", {}).get("overall_score", None)
        if rel is not None:
            print(f"🧪 Calidad de ensamble (estimada): score={rel:.2f}")

    print(f"Prob lluvia (histórica / ML básico / ML avanzado): {stat:.2f} / {basic:.2f} / {adv:.2f}")
    print("— Resumen últimos", args.explain_recent_days, "días (dataset disponible): "
          f"Σprecip={_fmt(rsum.get('precip_sum_mm'),' mm')}, "
          f"RH={_fmt(rsum.get('rh2m_mean_pct'),'%')}, "
          f"Rad={_fmt(rsum.get('rad_mean_kwh_m2d'))} kWh/m²·d, "
          f"Ps={_fmt(rsum.get('ps_mean_kpa'),' kPa')}, "
          f"Viento={_fmt(rsum.get('wind_mean_ms'),' m/s')}.")
    print("— Motivos:", expl.get("explanation_text", "NA"))
    print(f"🎯 Consenso: {'LLOVERÁ' if final['will_rain'] else 'NO LLOVERÁ'} "
          f"({final['probability']*100:.1f}% | conf: {final['confidence']})")
    print("="*72 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("💾 Guardado:", args.output)


if __name__ == "__main__":
    _cli()
