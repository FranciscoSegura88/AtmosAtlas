#!/usr/bin/env python3
"""
predict_rain_for_date_power.py (versión corregida)

- Solicita UN solo rango a la API POWER (reduce errores 500).
- No pide años futuros (clampa con el año actual).
- Maneja errores y rellena/filtra apropiadamente.
- Entrena LogisticRegression + RandomForest si hay suficientes datos,
  si no usa heurística (frecuencia histórica).
"""
import argparse
import datetime
import numpy as np
import pandas as pd
from power_timeseries import fetch_power_point
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def build_history_power_batch(target_date, lat, lon, years_back=10, power_vars=None, verbose=True):
    """
    Solicita UN rango a POWER desde (target_year - years_back) hasta min(target_year-1,current_year)
    y extrae el mismo dia-mes para cada año.
    Retorna DataFrame con columnas: date, year, precip, t2m, rh2m, wind, rad
    """
    if power_vars is None:
        power_vars = ["PRECTOTCORR","T2M","RH2M","WS10M","ALLSKY_SFC_SW_DWN"]

    T = datetime.date.fromisoformat(target_date)
    current_year = datetime.date.today().year
    start_year = T.year - years_back
    end_year = min(T.year - 1, current_year)  # no pedimos años en el futuro

    if end_year < start_year:
        raise ValueError("No hay años históricos válidos en el rango solicitado (quizá pediste años futuros).")

    # Rango entero: desde el primer dia del primer año hasta el ultimo dia del ultimo año
    start_date_str = datetime.date(start_year, 1, 1).isoformat()
    end_date_str = datetime.date(end_year, 12, 31).isoformat()

    if verbose:
        print(f"[INFO] Solicitando POWER una vez: {start_date_str} -> {end_date_str} para punto {lat},{lon}")
    try:
        # fetch full range (POWER accepts start=end as YYYYMMDD without dashes in helper)
        full_df = fetch_power_point(lat, lon, start_date_str, end_date_str, parameters=power_vars)
    except Exception as e:
        # Si falla la llamada global, intentar fallback por años individuales pero limitado
        if verbose:
            print("[WARN] Llamada a POWER para rango falló:", e)
            print("[WARN] Intentando llamadas por año (fallback, menos eficiente).")
        rows = []
        for y in range(start_year, end_year + 1):
            d = datetime.date(y, T.month, T.day).isoformat()
            try:
                df_single = fetch_power_point(lat, lon, d, d, parameters=power_vars)
                if df_single.empty:
                    rows.append((d, y, None, None, None, None, None))
                else:
                    r = df_single.iloc[0].to_dict()
                    rows.append((d, y, r.get("PRECTOTCORR", None), r.get("T2M", None),
                                 r.get("RH2M", None), r.get("WS10M", None), r.get("ALLSKY_SFC_SW_DWN", None)))
            except Exception as e2:
                if verbose: print("  fallo año", y, ":", e2)
                rows.append((d, y, None, None, None, None, None))
        df_hist = pd.DataFrame(rows, columns=["date","year","precip","t2m","rh2m","wind","rad"])
        return df_hist

    # full_df tiene filas por cada día del rango; extraer solo las fechas con mismo mes/día
    out_rows = []
    for y in range(start_year, end_year + 1):
        dt = datetime.date(y, T.month, T.day)
        key = dt.isoformat()
        # en full_df la columna "date" es string 'YYYY-MM-DD'
        row_df = full_df[full_df["date"] == key]
        if row_df.empty:
            out_rows.append({"date": key, "year": y, "precip": None, "t2m": None, "rh2m": None, "wind": None, "rad": None})
        else:
            r = row_df.iloc[0].to_dict()
            out_rows.append({
                "date": key,
                "year": y,
                "precip": r.get("PRECTOTCORR", None),
                "t2m": r.get("T2M", None),
                "rh2m": r.get("RH2M", None),
                "wind": r.get("WS10M", None),
                "rad": r.get("ALLSKY_SFC_SW_DWN", None)
            })
    df_hist = pd.DataFrame(out_rows)
    return df_hist

def train_models(df, umbral=0.5, verbose=True, min_train_rows=4):
    """Prepara features y entrena modelos. Devuelve dict de modelos o None si pocos datos."""
    df2 = df.copy()
    df2["llovio"] = df2["precip"].apply(lambda x: 1 if (x is not None and not pd.isna(x) and float(x) > umbral) else 0)
    features = ["t2m","rh2m","wind","rad"]

    # Contar filas con precip disponible (no-NaN)
    n_precip = df2["precip"].dropna().shape[0]
    if verbose:
        print(f"[INFO] años con precip disponible: {n_precip} de {len(df2)}")

    # Si tenemos menos de min_train_rows con precip, no entrenamos (poco señal)
    if n_precip < min_train_rows:
        if verbose:
            print(f"[WARN] Datos insuficientes para entrenar (necesarios >= {min_train_rows} años con precip).")
        return None, df2, features

    # Rellenar features con mediana (si todas NaN, mediana será NaN -> rellenamos con 0)
    for f in features:
        med = df2[f].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        df2[f] = df2[f].fillna(med)

    X = df2[features].values
    y = df2["llovio"].values.astype(int)

    # Si la variable objetivo no tiene variedad (todos 0 o todos 1), evitar entrenar modelos complejos
    if len(np.unique(y)) == 1:
        if verbose:
            print("[WARN] Variable objetivo sin diversidad (todo 0 o todo 1). No se entrenará modelo ML.")
        return None, df2, features

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)

    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X, y)

    if verbose:
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:,1]
        print("Train acc (logreg):", accuracy_score(y, preds))
        try:
            print("Train AUC (logreg):", roc_auc_score(y, probs))
        except Exception:
            print("Train AUC (logreg): NA")

    return {"logreg": pipe, "rf": rf}, df2, features

def predict(models, features, df_hist, umbral=0.5):
    # target features: median of history
    row = {f: float(df_hist[f].median(skipna=True)) if f in df_hist.columns else 0.0 for f in features}
    if models is None:
        hist_freq = df_hist["precip"].apply(lambda x: 1 if (x is not None and not pd.isna(x) and float(x) > umbral) else 0).mean()
        return {"method":"frequency","prob":float(hist_freq)}
    import numpy as np
    x = np.array([row[f] for f in features]).reshape(1,-1)
    p1 = float(models["logreg"].predict_proba(x)[:,1][0])
    p2 = float(models["rf"].predict_proba(x)[:,1][0])
    return {"method":"model","logreg_prob":p1,"rf_prob":p2,"prob":(p1+p2)/2.0}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--years_back", type=int, default=10)
    p.add_argument("--umbral", type=float, default=0.5)
    args = p.parse_args()

    print("Construyendo histórico para", args.date, "en", args.lat, args.lon)
    df = build_history_power_batch(args.date, args.lat, args.lon, years_back=args.years_back, verbose=True)
    print("Histórico (primeras filas):")
    print(df.head(20))

    models, df_clean, features = train_models(df, umbral=args.umbral)
    res = predict(models, features, df_clean, umbral=args.umbral)
    if res["method"] == "frequency":
        print(f"Predicción por frecuencia: P(llover) = {res['prob']:.2f}")
    else:
        print("Predicción (logreg, rf, avg):", res["logreg_prob"], res["rf_prob"], res["prob"])

    df.to_csv("history_used_power.csv", index=False)
    print("Histórico guardado en history_used_power.csv")

if __name__ == "__main__":
    main()
