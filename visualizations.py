"""
Module de visualisation pour le Dashboard Climatique.

Ce module contient toutes les fonctions de generation de graphiques
utilisant Plotly et Seaborn pour creer des visualisations interactives
et statistiques des donnees climatiques.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration globale des couleurs
PALETTE_REGION = {
    "Nord": "#1f77b4",
    "Sud": "#ff7f0e",
    "Est": "#2ca02c",
    "Ouest": "#d62728",
    "Centre": "#9467bd"
}

PALETTE_SAISON = {
    "saison_des_pluies": "#17becf",
    "saison_sèche": "#e377c2"
}


def plot_distribution_histograms(df):
    """
    Cree une grille de 4 histogrammes pour chaque variable numerique.
    
    Les variables visualisees sont :
    - temperature_moyenne
    - humidite
    - precipitations
    - vitesse_vent
    
    Args:
        df (pd.DataFrame): DataFrame contenant les donnees climatiques
        
    Returns:
        plotly.graph_objects.Figure: Figure avec 4 sous-graphiques
    """
    variables = [
        ("temperature_moyenne", "Temperature Moyenne (°C)", "#1f77b4"),
        ("humidite", "Humidite (%)", "#2ca02c"),
        ("precipitations", "Precipitations (mm)", "#17becf"),
        ("vitesse_vent", "Vitesse du Vent (km/h)", "#ff7f0e")
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[v[1] for v in variables],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (col, title, color), (row, col_idx) in zip(variables, positions):
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=title,
                marker_color=color,
                opacity=0.75,
                nbinsx=30,
                showlegend=False
            ),
            row=row, col=col_idx
        )
        
        # Ajout de la ligne de densite (KDE approxime)
        fig.add_trace(
            go.Scatter(
                x=np.sort(df[col]),
                y=np.zeros_like(np.sort(df[col])),
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title_text="Distributions des Variables Climatiques",
        title_font_size=18,
        height=600,
        template="plotly_white"
    )
    
    return fig


def plot_boxplot_by_region(df, variable):
    """
    Cree un boxplot d'une variable groupee par region.
    
    Le boxplot montre :
    - La mediane (ligne au centre)
    - Les quartiles (Q1, Q3)
    - Les valeurs aberrantes (points extremes)
    
    Args:
        df (pd.DataFrame): DataFrame source
        variable (str): Nom de la colonne a visualiser
        
    Returns:
        plotly.graph_objects.Figure: Boxplot interactif
    """
    fig = px.box(
        df,
        x="region",
        y=variable,
        color="region",
        color_discrete_map=PALETTE_REGION,
        title=f"Distribution de {variable} par Region",
        labels={
            "region": "Region",
            variable: variable.replace("_", " ").title()
        },
        template="plotly_white"
    )
    
    fig.update_layout(
        showlegend=False,
        height=450,
        title_font_size=16
    )
    
    return fig


def plot_boxplot_by_saison(df, variable):
    """
    Cree un boxplot d'une variable groupee par saison.
    
    Args:
        df (pd.DataFrame): DataFrame source
        variable (str): Nom de la colonne a visualiser
        
    Returns:
        plotly.graph_objects.Figure: Boxplot interactif
    """
    fig = px.box(
        df,
        x="saison",
        y=variable,
        color="saison",
        color_discrete_map=PALETTE_SAISON,
        title=f"Distribution de {variable} par Saison",
        labels={
            "saison": "Saison",
            variable: variable.replace("_", " ").title()
        },
        template="plotly_white"
    )
    
    fig.update_layout(
        showlegend=False,
        height=450,
        title_font_size=16
    )
    
    return fig


def plot_scatter_temperature_humidite(df):
    """
    Cree un nuage de points interactif : Temperature vs Humidite.
    
    Les points sont colories par region et leur taille est proportionnelle
    aux precipitations (normalisees pour eviter les tailles nulles),
    permettant de visualiser 4 dimensions simultanement.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot interactif
    """
    df_plot = df.copy()
    # Normalisation pour garantir des tailles positives (> 0)
    prec_min = df_plot["precipitations"].min()
    prec_max = df_plot["precipitations"].max()
    if prec_max != prec_min:
        df_plot["taille_precip"] = (
            (df_plot["precipitations"] - prec_min) / (prec_max - prec_min) * 45 + 5
        )
    else:
        df_plot["taille_precip"] = 20
    
    fig = px.scatter(
        df_plot,
        x="humidite",
        y="temperature_moyenne",
        color="region",
        size="taille_precip",
        facet_col="saison",
        color_discrete_map=PALETTE_REGION,
        title="Temperature vs Humidite (taille ~ Precipitations normalisees)",
        labels={
            "humidite": "Humidite (%)",
            "temperature_moyenne": "Temperature Moyenne (°C)",
            "taille_precip": "Taille (mm normalise)"
        },
        template="plotly_white",
        hover_data=["precipitations", "vitesse_vent"]
    )
    
    fig.update_layout(
        height=500,
        title_font_size=16
    )
    
    return fig


def plot_bar_comparison_by_region(df):
    """
    Cree un graphique en barres comparees des moyennes par region.
    
    Visualise simultanement les 4 variables numeriques moyennees
    pour chaque region, permettant une comparaison rapide.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Bar chart groupe
    """
    variables = ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"]
    
    # Normalisation pour comparer sur la meme echelle (0-100)
    df_norm = df.copy()
    for var in variables:
        min_val = df[var].min()
        max_val = df[var].max()
        if max_val != min_val:
            df_norm[var] = ((df[var] - min_val) / (max_val - min_val)) * 100
    
    grouped = df_norm.groupby("region")[variables].mean().reset_index()
    
    fig = go.Figure()
    
    colors = ["#1f77b4", "#2ca02c", "#17becf", "#ff7f0e"]
    
    for var, color in zip(variables, colors):
        fig.add_trace(go.Bar(
            name=var.replace("_", " ").title(),
            x=grouped["region"],
            y=grouped[var],
            marker_color=color
        ))
    
    fig.update_layout(
        title="Comparaison Normalisee des Variables par Region (0-100)",
        xaxis_title="Region",
        yaxis_title="Valeur Normalisee",
        barmode="group",
        template="plotly_white",
        height=500,
        title_font_size=16,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_correlation_heatmap(df):
    """
    Cree une heatmap de correlation entre les variables numeriques.
    
    Utilise le coefficient de correlation de Pearson pour mesurer
    la force et la direction des relations lineaires.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Heatmap interactive
    """
    numeric_cols = ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"]
    corr_matrix = df[numeric_cols].corr()
    
    # Renommage des labels pour l'affichage
    labels = {
        "temperature_moyenne": "Temperature",
        "humidite": "Humidite",
        "precipitations": "Precipitations",
        "vitesse_vent": "Vent"
    }
    
    corr_matrix.rename(index=labels, columns=labels, inplace=True)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="equal",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matrice de Correlation des Variables Climatiques",
        template="plotly_white"
    )
    
    fig.update_traces(
        texttemplate="%{z:.2f}",
        textfont={"size": 14}
    )
    
    fig.update_layout(
        height=500,
        width=600,
        title_font_size=16
    )
    
    return fig


def plot_violin_by_saison(df, variable):
    """
    Cree un violin plot pour visualiser la distribution par saison.
    
    Le violin plot combine un boxplot avec une estimation de densite,
    montrant la forme complete de la distribution.
    
    Args:
        df (pd.DataFrame): DataFrame source
        variable (str): Nom de la colonne a visualiser
        
    Returns:
        plotly.graph_objects.Figure: Violin plot interactif
    """
    fig = px.violin(
        df,
        y=variable,
        x="saison",
        color="saison",
        color_discrete_map=PALETTE_SAISON,
        box=True,
        points="all",
        title=f"Distribution de {variable} par Saison",
        labels={
            "saison": "Saison",
            variable: variable.replace("_", " ").title()
        },
        template="plotly_white"
    )
    
    fig.update_layout(
        showlegend=False,
        height=450,
        title_font_size=16
    )
    
    return fig


def plot_mean_by_region_saison(df):
    """
    Cree un heatmap des moyennes par region et saison.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Heatmap avec barres
    """
    grouped = df.groupby(["region", "saison"])[
        ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"]
    ].mean().reset_index()
    
    fig = px.density_heatmap(
        grouped,
        x="region",
        y="saison",
        z="temperature_moyenne",
        title="Temperature Moyenne par Region et Saison",
        labels={
            "region": "Region",
            "saison": "Saison",
            "temperature_moyenne": "Temp. Moyenne (°C)"
        },
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    
    fig.update_layout(
        height=450,
        title_font_size=16
    )
    
    return fig


def plot_pairplot_streamlit(df):
    """
    Cree une matrice de scatter plots pour toutes les paires de variables.
    
    Version optimisee pour Streamlit utilisant Plotly au lieu de Seaborn
    pour conserver l'interactivite.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Scatter matrix
    """
    numeric_cols = ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"]
    
    fig = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color="region",
        color_discrete_map=PALETTE_REGION,
        title="Matrice de Correlation (Scatter Matrix)",
        template="plotly_white",
        opacity=0.6
    )
    
    fig.update_layout(
        height=700,
        title_font_size=16,
        dragmode='select'
    )
    
    # Ajustement des tailles de police pour les axes
    fig.update_traces(diagonal_visible=False)
    
    return fig


def plot_barplot_qualitative(df, colonne):
    """
    Cree un diagramme en barres des effectifs pour une variable qualitative.
    
    Args:
        df (pd.DataFrame): DataFrame source
        colonne (str): Nom de la colonne qualitative ('region' ou 'saison')
        
    Returns:
        plotly.graph_objects.Figure: Barplot interactif
    """
    counts = df[colonne].value_counts().reset_index()
    counts.columns = [colonne, "effectif"]
    
    palette = PALETTE_REGION if colonne == "region" else PALETTE_SAISON
    
    fig = px.bar(
        counts,
        x=colonne,
        y="effectif",
        color=colonne,
        color_discrete_map=palette,
        title=f"Repartition des effectifs par {colonne.replace('_', ' ').title()}",
        labels={"effectif": "Nombre d'observations"},
        template="plotly_white",
        text="effectif"
    )
    
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=450,
        title_font_size=16,
        xaxis_title=colonne.replace("_", " ").title(),
        yaxis_title="Nombre d'observations"
    )
    
    return fig


def plot_pie_qualitative(df, colonne):
    """
    Cree un camembert (pie chart) des proportions pour une variable qualitative.
    
    Args:
        df (pd.DataFrame): DataFrame source
        colonne (str): Nom de la colonne qualitative ('region' ou 'saison')
        
    Returns:
        plotly.graph_objects.Figure: Pie chart interactif
    """
    counts = df[colonne].value_counts().reset_index()
    counts.columns = [colonne, "effectif"]
    
    palette = PALETTE_REGION if colonne == "region" else PALETTE_SAISON
    
    fig = px.pie(
        counts,
        names=colonne,
        values="effectif",
        color=colonne,
        color_discrete_map=palette,
        title=f"Proportions par {colonne.replace('_', ' ').title()}",
        template="plotly_white",
        hole=0.35  # Donut chart pour un rendu moderne
    )
    
    fig.update_traces(
        textinfo="percent+label",
        pull=[0.02] * len(counts),
        marker=dict(line=dict(color="white", width=2))
    )
    
    fig.update_layout(
        height=450,
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig


def plot_stacked_bar_region_saison(df):
    """
    Cree un diagramme en barres empile croisant region et saison.
    
    Montre la repartition des saisons au sein de chaque region.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Barplot empile interactif
    """
    crosstab = pd.crosstab(df["region"], df["saison"]).reset_index()
    crosstab_melted = crosstab.melt(id_vars="region", var_name="saison", value_name="effectif")
    
    fig = px.bar(
        crosstab_melted,
        x="region",
        y="effectif",
        color="saison",
        color_discrete_map=PALETTE_SAISON,
        title="Repartition Region x Saison (barres empilees)",
        labels={"effectif": "Nombre d'observations", "region": "Region"},
        template="plotly_white",
        text="effectif"
    )
    
    fig.update_traces(textposition="inside", textfont=dict(color="white", size=12))
    fig.update_layout(
        height=450,
        title_font_size=16,
        legend=dict(title="Saison", orientation="h", yanchor="bottom", y=-0.15),
        barmode="stack"
    )
    
    return fig


def plot_grouped_bar_region_saison(df):
    """
    Cree un diagramme en barres groupe croisant region et saison.
    
    Alternative au stacked bar pour comparer les effectifs cote a cote.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Barplot groupe interactif
    """
    crosstab = pd.crosstab(df["region"], df["saison"]).reset_index()
    crosstab_melted = crosstab.melt(id_vars="region", var_name="saison", value_name="effectif")
    
    fig = px.bar(
        crosstab_melted,
        x="region",
        y="effectif",
        color="saison",
        color_discrete_map=PALETTE_SAISON,
        title="Repartition Region x Saison (barres groupees)",
        labels={"effectif": "Nombre d'observations", "region": "Region"},
        template="plotly_white",
        barmode="group",
        text="effectif"
    )
    
    fig.update_traces(textposition="outside", textfont=dict(size=11))
    fig.update_layout(
        height=450,
        title_font_size=16,
        legend=dict(title="Saison", orientation="h", yanchor="bottom", y=-0.15)
    )
    
    return fig


def plot_precipitation_wind_bubble(df):
    """
    Cree un graphique a bulles : Precipitations vs Vent.
    
    La couleur represente la region et la taille la temperature
    (normalisee pour eviter les valeurs negatives).
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        plotly.graph_objects.Figure: Bubble chart
    """
    df_plot = df.copy()
    # Normalisation Min-Max + offset pour garantir des tailles positives
    temp_min = df_plot["temperature_moyenne"].min()
    temp_max = df_plot["temperature_moyenne"].max()
    if temp_max != temp_min:
        df_plot["taille_bulle"] = (
            (df_plot["temperature_moyenne"] - temp_min) / (temp_max - temp_min) * 50 + 5
        )
    else:
        df_plot["taille_bulle"] = 20
    
    fig = px.scatter(
        df_plot,
        x="vitesse_vent",
        y="precipitations",
        size="taille_bulle",
        color="region",
        color_discrete_map=PALETTE_REGION,
        facet_col="saison",
        title="Precipitations vs Vent (taille ~ Temperature normalisee)",
        labels={
            "vitesse_vent": "Vitesse du Vent (km/h)",
            "precipitations": "Precipitations (mm)",
            "taille_bulle": "Taille (°C normalise)"
        },
        template="plotly_white",
        hover_data=["temperature_moyenne", "humidite"]
    )
    
    fig.update_layout(
        height=500,
        title_font_size=16
    )
    
    return fig
