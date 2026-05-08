"""
API REST FastAPI pour le Dashboard Climatique (Phase 6).
Expose les donnees, statistiques, tests, predictions et rapports via HTTP.
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import io
import json
from datetime import datetime

from data_processor import load_dataset, validate_data, get_basic_stats, filter_by_region, filter_by_saison
from statistical_analysis import test_normalite_shapiro, test_homogeneite_variance, resume_statistique_complet
from predictive_models import preparer_donnees, entrainer_modele, comparer_modeles, predire_valeur
from report_generator import export_excel_complet, generate_html_report

app = FastAPI(
    title="Dashboard Climatique API",
    description="API REST pour acceder aux donnees climatiques, analyses statistiques et predictions",
    version="1.6.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Chargement du dataset au demarrage
try:
    df = load_dataset()
    numeric_cols = ['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']
    categorical_cols = ['region', 'saison']
except Exception as e:
    df = None
    print(f"Erreur chargement dataset : {e}")


def _check_data():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset non disponible")


# ============================================================================
# MODELES PYDANTIC
# ============================================================================

class DataResponse(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[str]
    data: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    variable: str
    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skewness: float
    kurtosis: float


class CorrelationResponse(BaseModel):
    variables: List[str]
    matrix: Dict[str, Dict[str, float]]


class PredictionRequest(BaseModel):
    target: str = Field(..., description="Variable cible a predire")
    features: List[str] = Field(..., description="Variables predictives")
    model_type: str = Field(default="random_forest", description="Type de modele : linear, random_forest, gradient_boosting, xgboost")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Hyperparametres du modele")


class PredictInputRequest(BaseModel):
    model_type: str = Field(default="random_forest", description="Type de modele")
    target: str = Field(..., description="Variable cible")
    features: List[str] = Field(..., description="Variables predictives")
    input_values: Dict[str, Any] = Field(..., description="Valeurs pour la prediction")


class PredictionResponse(BaseModel):
    target: str
    prediction: float
    model_type: str
    timestamp: str


class ComparisonResponse(BaseModel):
    results: List[Dict[str, Any]]
    best_model: Optional[str]
    best_r2: Optional[float]


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    n_rows: Optional[int]
    n_cols: Optional[int]
    timestamp: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
def root():
    return {
        "message": "Dashboard Climatique API",
        "version": "1.6.0",
        "phases": "1-6 completes",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        dataset_loaded=df is not None,
        n_rows=len(df) if df is not None else None,
        n_cols=len(df.columns) if df is not None else None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/data", response_model=DataResponse)
def get_data(
    region: Optional[str] = Query(None, description="Filtrer par region (Nord, Sud, Est, Ouest, Centre)"),
    saison: Optional[str] = Query(None, description="Filtrer par saison (saison_des_pluies, saison_sèche)"),
    limit: int = Query(100, ge=1, le=501, description="Nombre maximum de lignes")
):
    _check_data()
    d = df.copy()
    if region:
        d = d[d['region'] == region]
    if saison:
        d = d[d['saison'] == saison]
    d = d.head(limit)
    return DataResponse(
        n_rows=len(d),
        n_cols=len(d.columns),
        columns=list(d.columns),
        data=d.replace({np.nan: None}).to_dict(orient='records')
    )


@app.get("/data/variables")
def get_variables():
    _check_data()
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "all": list(df.columns)
    }


@app.get("/stats/descriptive", response_model=List[StatsResponse])
def get_descriptive_stats(
    variable: Optional[str] = Query(None, description="Variable specifique, ou toutes si non precise")
):
    _check_data()
    vars_to_compute = [variable] if variable and variable in numeric_cols else numeric_cols
    results = []
    for var in vars_to_compute:
        if var not in df.columns:
            continue
        stats = resume_statistique_complet(df, var)
        results.append(StatsResponse(
            variable=var,
            count=int(stats['count']),
            mean=round(stats['mean'], 4),
            std=round(stats['std'], 4),
            min=round(stats['min'], 4),
            q25=round(stats['quartiles']['q1'], 4),
            median=round(stats['median'], 4),
            q75=round(stats['quartiles']['q3'], 4),
            max=round(stats['max'], 4),
            skewness=round(stats['skewness'], 4),
            kurtosis=round(stats['kurtosis'], 4)
        ))
    return results


@app.get("/stats/correlation", response_model=CorrelationResponse)
def get_correlation():
    _check_data()
    corr = df[numeric_cols].corr().round(4)
    return CorrelationResponse(
        variables=numeric_cols,
        matrix=corr.to_dict()
    )


@app.get("/stats/by-region")
def get_stats_by_region():
    _check_data()
    agg = df.groupby('region')[numeric_cols].agg(['mean', 'std', 'count']).round(4)
    # Aplatir les multi-index
    result = {}
    for region in agg.index:
        result[region] = {}
        for col in numeric_cols:
            result[region][col] = {
                "mean": float(agg.loc[region, (col, 'mean')]),
                "std": float(agg.loc[region, (col, 'std')]),
                "count": int(agg.loc[region, (col, 'count')])
            }
    return result


@app.get("/stats/by-saison")
def get_stats_by_saison():
    _check_data()
    agg = df.groupby('saison')[numeric_cols].agg(['mean', 'std', 'count']).round(4)
    result = {}
    for saison in agg.index:
        result[saison] = {}
        for col in numeric_cols:
            result[saison][col] = {
                "mean": float(agg.loc[saison, (col, 'mean')]),
                "std": float(agg.loc[saison, (col, 'std')]),
                "count": int(agg.loc[saison, (col, 'count')])
            }
    return result


@app.get("/tests/normality/{variable}")
def test_normality(variable: str):
    _check_data()
    if variable not in numeric_cols:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' non numerique ou inexistante")
    result = test_normalite_shapiro(df, variable)
    return result


@app.get("/tests/homogeneity/{variable}")
def test_homogeneity(
    variable: str,
    group: str = Query(default="region", description="Variable de groupement : region ou saison")
):
    _check_data()
    if variable not in numeric_cols:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' non numerique")
    if group not in categorical_cols:
        raise HTTPException(status_code=400, detail=f"Groupe '{group}' invalide")
    result = test_homogeneite_variance(df, variable, group)
    return result


@app.post("/models/train")
def train_model(req: PredictionRequest):
    _check_data()
    valid_models = ["linear", "random_forest", "gradient_boosting", "xgboost"]
    if req.model_type not in valid_models:
        raise HTTPException(status_code=400, detail=f"Modele invalide. Choix : {valid_models}")
    
    try:
        data = preparer_donnees(df, req.target, req.features)
        result = entrainer_modele(data, model_type=req.model_type, params=req.params)
        return {
            "model_type": req.model_type,
            "target": req.target,
            "features": req.features,
            "metrics": result['metrics'],
            "feature_importance": result['feature_importance']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/compare")
def compare_models(
    target: str = Query(..., description="Variable cible"),
    features: List[str] = Query(..., description="Variables predictives (liste)")
):
    _check_data()
    try:
        compare_df = comparer_modeles(df, target, features)
        results = compare_df.replace({np.nan: None}).to_dict(orient='records')
        best = None
        best_r2 = None
        if 'R2 Test' in compare_df.columns and not compare_df['R2 Test'].isna().all():
            idx = compare_df['R2 Test'].idxmax()
            best = compare_df.loc[idx, 'Modele']
            best_r2 = float(compare_df.loc[idx, 'R2 Test'])
        return ComparisonResponse(results=results, best_model=best, best_r2=best_r2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictInputRequest):
    _check_data()
    try:
        data = preparer_donnees(df, req.target, req.features)
        result = entrainer_modele(data, model_type=req.model_type)
        prediction = predire_valeur(result, req.input_values)
        return PredictionResponse(
            target=req.target,
            prediction=round(prediction, 4),
            model_type=req.model_type,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/excel")
def get_excel_report():
    _check_data()
    try:
        excel_buffer = export_excel_complet(df)
        return StreamingResponse(
            io.BytesIO(excel_buffer.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=rapport_climatique.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/html", response_class=HTMLResponse)
def get_html_report():
    _check_data()
    try:
        html = generate_html_report(df)
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
