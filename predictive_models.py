"""
Module de modeles predictifs avances pour le Dashboard Climatique.
Random Forest, Gradient Boosting, XGBoost (optionnel) avec comparaison et predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# XGBoost optionnel (peut ne pas etre installe sur Windows)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def preparer_donnees(df, target_col, features_cols=None, test_size=0.2, random_state=42):
    """
    Prepare les donnees pour l'entrainement : encodage, split train/test.
    """
    df_model = df.copy()
    
    # Selection des features par defaut
    if features_cols is None:
        features_cols = [c for c in df.columns if c != target_col]
    
    # Encodage des variables categorielles
    encoders = {}
    for col in features_cols:
        if df_model[col].dtype == 'object' or df_model[col].dtype.name == 'category':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            encoders[col] = le
    
    X = df_model[features_cols]
    y = df_model[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'features': features_cols,
        'encoders': encoders,
        'scaler': None
    }


def entrainer_modele(data, model_type='random_forest', params=None):
    """
    Entraine un modele predictif et retourne les resultats.
    """
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    if params is None:
        params = {}
    
    # Initialisation du modele
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost n'est pas installe. Utilisez 'pip install xgboost'.")
        model = XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42,
            verbosity=0
        )
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Modele inconnu : {model_type}")
    
    # Entrainement
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metriques
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    # Feature importance (si disponible)
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(data['features'], model.feature_importances_))
    elif hasattr(model, 'coef_'):
        importance = dict(zip(data['features'], np.abs(model.coef_)))
    
    return {
        'model': model,
        'model_type': model_type,
        'metrics': metrics,
        'feature_importance': importance,
        'predictions': {
            'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
            'y_pred_test': y_pred_test
        },
        'data': data
    }


def comparer_modeles(df, target_col, features_cols=None):
    """
    Entraine et compare plusieurs modeles sur les memes donnees.
    Retourne un DataFrame de comparaison.
    """
    data = preparer_donnees(df, target_col, features_cols)
    
    modeles = ['linear', 'random_forest', 'gradient_boosting']
    if XGBOOST_AVAILABLE:
        modeles.append('xgboost')
    
    results = []
    for mtype in modeles:
        try:
            res = entrainer_modele(data, model_type=mtype)
            results.append({
                'Modele': mtype.replace('_', ' ').title(),
                'R2 Train': round(res['metrics']['train_r2'], 4),
                'R2 Test': round(res['metrics']['test_r2'], 4),
                'RMSE Test': round(res['metrics']['test_rmse'], 4),
                'MAE Test': round(res['metrics']['test_mae'], 4),
            })
        except Exception as e:
            results.append({
                'Modele': mtype.replace('_', ' ').title(),
                'R2 Train': None,
                'R2 Test': None,
                'RMSE Test': None,
                'MAE Test': None,
                'Erreur': str(e)
            })
    
    return pd.DataFrame(results)


def predire_valeur(model_result, input_values):
    """
    Fait une prediction pour une nouvelle observation.
    input_values : dict {feature_name: value}
    """
    model = model_result['model']
    data = model_result['data']
    features = data['features']
    encoders = data['encoders']
    
    # Encoder les valeurs categorielles si necessaire
    X_input = []
    for f in features:
        val = input_values.get(f, 0)
        if f in encoders:
            # Si la valeur n'a pas ete vue pendant l'entrainement, on utilise 0
            try:
                val = encoders[f].transform([str(val)])[0]
            except ValueError:
                val = 0
        X_input.append(val)
    
    X_input = np.array(X_input).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    
    return prediction


def interpreter_performance(metrics):
    """
    Genere une interpretation textuelle des performances du modele.
    """
    r2 = metrics['test_r2']
    rmse = metrics['test_rmse']
    mae = metrics['test_mae']
    
    parts = []
    
    if r2 > 0.8:
        parts.append(f"Excellent pouvoir predictif (R² = {r2:.3f}). Le modele explique plus de 80% de la variance.")
    elif r2 > 0.5:
        parts.append(f"Bon pouvoir predictif (R² = {r2:.3f}). Le modele capture une part significative de la variance.")
    elif r2 > 0.2:
        parts.append(f"Pouvoir predictif modere (R² = {r2:.3f}). Des facteurs non modelises influencent la variable cible.")
    else:
        parts.append(f"Faible pouvoir predictif (R² = {r2:.3f}). La relation entre les variables est faible ou non lineaire.")
    
    parts.append(f"Erreur moyenne absolue (MAE) : {mae:.3f} — en moyenne, les predictions s'ecartent de {mae:.3f} unites de la valeur reelle.")
    parts.append(f"RMSE : {rmse:.3f} — penalise fortement les grosses erreurs.")
    
    # Comparaison train vs test (overfitting)
    train_r2 = metrics['train_r2']
    if train_r2 - r2 > 0.15:
        parts.append("⚠️ Ecart significatif entre R² train et test : risque de surapprentissage (overfitting).")
    elif train_r2 - r2 > 0.05:
        parts.append("Leger surapprentissage detecte.")
    else:
        parts.append("✅ Bonne generalisation : les performances train et test sont similaires.")
    
    return "\n\n".join(parts)
