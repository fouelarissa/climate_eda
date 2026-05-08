"""
Module d'analyses statistiques avancees pour le Dashboard Climatique.

Fournit des tests statistiques, regressions, et modeles predictifs
bases sur scipy et scikit-learn (optionnel).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, f_oneway, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def test_normalite_shapiro(df, variable):
    """
    Test de normalite de Shapiro-Wilk.
    H0 : les donnees suivent une loi normale.
    """
    data = df[variable].dropna()
    stat, p_value = shapiro(data)
    
    return {
        "variable": variable,
        "test": "Shapiro-Wilk",
        "statistique": round(stat, 4),
        "p_value": round(p_value, 6),
        "normal": p_value > 0.05,
        "interpretation": (
            "Distribution normale (p > 0.05)" if p_value > 0.05
            else "Distribution NON normale (p <= 0.05)"
        ),
        "n": len(data)
    }


def test_homogeneite_variance(df, variable, groupe_col):
    """
    Test de Levene pour l'homogeneite des variances.
    H0 : les variances sont egales entre les groupes.
    """
    groups = [group[variable].values for name, group in df.groupby(groupe_col)]
    stat, p_value = levene(*groups)
    
    return {
        "variable": variable,
        "groupe": groupe_col,
        "test": "Levene",
        "statistique": round(stat, 4),
        "p_value": round(p_value, 6),
        "homogene": p_value > 0.05,
        "interpretation": (
            "Variances homogenes (p > 0.05)" if p_value > 0.05
            else "Variances NON homogenes (p <= 0.05)"
        )
    }


def ttest_independant(df, variable, groupe_col, groupe1, groupe2):
    """
    T-test de Student pour 2 groupes independants.
    H0 : les moyennes des deux groupes sont egales.
    """
    data1 = df[df[groupe_col] == groupe1][variable].dropna()
    data2 = df[df[groupe_col] == groupe2][variable].dropna()
    
    stat, p_value = ttest_ind(data1, data2)
    
    return {
        "variable": variable,
        "groupe": groupe_col,
        "comparaison": f"{groupe1} vs {groupe2}",
        "test": "t-test independant",
        "statistique": round(stat, 4),
        "p_value": round(p_value, 6),
        "significatif": p_value < 0.05,
        "interpretation": (
            f"Difference SIGNIFICATIVE entre {groupe1} et {groupe2} (p < 0.05)"
            if p_value < 0.05
            else f"Pas de difference significative (p >= 0.05)"
        ),
        "moyenne_g1": round(data1.mean(), 2),
        "moyenne_g2": round(data2.mean(), 2),
        "n_g1": len(data1),
        "n_g2": len(data2)
    }


def anova_un_facteur(df, variable, groupe_col):
    """
    ANOVA a un facteur pour comparer les moyennes de plusieurs groupes.
    H0 : toutes les moyennes sont egales.
    """
    groups = [group[variable].values for name, group in df.groupby(groupe_col)]
    stat, p_value = f_oneway(*groups)
    
    group_stats = df.groupby(groupe_col)[variable].agg(['mean', 'std', 'count']).round(2)
    
    return {
        "variable": variable,
        "groupe": groupe_col,
        "test": "ANOVA (1 facteur)",
        "statistique_F": round(stat, 4),
        "p_value": round(p_value, 6),
        "significatif": p_value < 0.05,
        "interpretation": (
            "Au moins un groupe differe significativement (p < 0.05)"
            if p_value < 0.05
            else "Pas de difference significative entre les groupes"
        ),
        "statistiques_groupes": group_stats.to_dict()
    }


def chi2_independance(df, col1, col2):
    """
    Test du Chi2 d'independance pour 2 variables qualitatives.
    H0 : les deux variables sont independantes.
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        "variables": f"{col1} vs {col2}",
        "test": "Chi2 d'independance",
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "ddl": dof,
        "independantes": p_value > 0.05,
        "interpretation": (
            "Variables DEPENDANTES (association significative, p < 0.05)"
            if p_value < 0.05
            else "Variables INDEPENDANTES (p >= 0.05)"
        ),
        "tableau_contingence": contingency.to_dict()
    }


def regression_lineaire_simple(df, x_var, y_var):
    """
    Regression lineaire simple : y = a*x + b.
    """
    X = df[[x_var]].values
    y = df[y_var].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Test de significativite de la pente (approximation)
    n = len(y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Correlation et p-value
    corr, p_value = stats.pearsonr(df[x_var], df[y_var])
    
    return {
        "equation": f"{y_var} = {slope:.4f} * {x_var} + {intercept:.4f}",
        "pente": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r2, 4),
        "r": round(corr, 4),
        "rmse": round(rmse, 4),
        "p_value": round(p_value, 6),
        "significatif": p_value < 0.05,
        "interpretation": (
            f"Relation significative (r={corr:.2f}, p<0.05). {r2*100:.1f}% de la variance expliquee."
            if p_value < 0.05
            else f"Relation non significative (p={p_value:.3f})"
        ),
        "n": n
    }


def regression_multiple(df, y_var, x_vars):
    """
    Regression lineaire multiple.
    """
    # Encodage des variables categorielles
    df_encoded = df.copy()
    encoders = {}
    
    for col in x_vars:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    X = df_encoded[x_vars].values
    y = df_encoded[y_var].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Coefficients
    coeffs = dict(zip(x_vars, [round(c, 4) for c in model.coef_]))
    
    return {
        "equation": f"{y_var} = " + " + ".join([f"{c:.4f}*{v}" for v, c in zip(x_vars, model.coef_)]) + f" + {model.intercept_:.4f}",
        "coefficients": coeffs,
        "intercept": round(model.intercept_, 4),
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "variables": x_vars,
        "cible": y_var,
        "interpretation": f"Modele explique {r2*100:.1f}% de la variance de {y_var}. Erreur moyenne : {rmse:.2f} unites."
    }


def resume_statistique_complet(df, variable):
    """
    Resume statistique complet d'une variable avec tests.
    """
    data = df[variable].dropna()
    
    return {
        "variable": variable,
        "n": len(data),
        "moyenne": round(data.mean(), 2),
        "mediane": round(data.median(), 2),
        "ecart_type": round(data.std(), 2),
        "variance": round(data.var(), 2),
        "min": round(data.min(), 2),
        "max": round(data.max(), 2),
        "etendue": round(data.max() - data.min(), 2),
        "q1": round(data.quantile(0.25), 2),
        "q3": round(data.quantile(0.75), 2),
        "iqr": round(data.quantile(0.75) - data.quantile(0.25), 2),
        "skewness": round(data.skew(), 2),
        "kurtosis": round(data.kurtosis(), 2),
        "cv": round((data.std() / data.mean() * 100), 2) if data.mean() != 0 else None
    }


def detecter_outliers_iqr(df, variable):
    """
    Detection des outliers par la methode IQR.
    """
    data = df[variable]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(data < lower_bound) | (data > upper_bound)]
    
    return {
        "variable": variable,
        "q1": round(Q1, 2),
        "q3": round(Q3, 2),
        "iqr": round(IQR, 2),
        "borne_inf": round(lower_bound, 2),
        "borne_sup": round(upper_bound, 2),
        "n_outliers": len(outliers),
        "pct_outliers": round(len(outliers) / len(df) * 100, 2),
        "outliers": outliers[[variable, 'region', 'saison']].to_dict('records') if len(outliers) > 0 else []
    }
