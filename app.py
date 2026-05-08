import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_processor import (
    load_dataset, validate_data, get_basic_stats, 
    get_data_info, filter_by_region, filter_by_saison,
    group_by_region_saison
)
from visualizations import (
    plot_distribution_histograms,
    plot_boxplot_by_region,
    plot_boxplot_by_saison,
    plot_scatter_temperature_humidite,
    plot_bar_comparison_by_region,
    plot_correlation_heatmap,
    plot_violin_by_saison,
    plot_mean_by_region_saison,
    plot_pairplot_streamlit,
    plot_precipitation_wind_bubble,
    plot_barplot_qualitative,
    plot_pie_qualitative,
    plot_stacked_bar_region_saison,
    plot_grouped_bar_region_saison
)
from auto_interpretation import (
    interpret_distribution,
    interpret_boxplot_by_group,
    interpret_scatter_temp_humidite,
    interpret_correlation_matrix,
    interpret_region_comparison,
    interpret_barplot_qualitative,
    interpret_crosstab_region_saison,
    interpret_precipitations_vent,
    interpret_violin_by_saison,
    interpret_heatmap_region_saison
)
from statistical_analysis import (
    test_normalite_shapiro,
    test_homogeneite_variance,
    ttest_independant,
    anova_un_facteur,
    chi2_independance,
    regression_lineaire_simple,
    regression_multiple,
    resume_statistique_complet,
    detecter_outliers_iqr
)
from report_generator import (
    export_excel_complet,
    generate_pdf_report,
    generate_html_report,
    fig_to_image
)
from predictive_models import (
    preparer_donnees,
    entrainer_modele,
    comparer_modeles,
    predire_valeur,
    interpreter_performance,
    XGBOOST_AVAILABLE
)

st.set_page_config(
    page_title="Dashboard Climatique",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric label {
        color: #31333F !important;
        font-weight: 600 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    .graph-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)


# Dictionnaire de mapping mots-cles -> graphiques pertinents
GRAPHES_PAR_VARIABLE = {
    "temperature": {
        "titre": "Analyses de la Temperature",
        "icon": "🌡️",
        "graphes": [
            {
                "nom": "Distribution de la temperature",
                "fonction": lambda df: plot_distribution_histograms(df),
                "interpretation": "Forme de la distribution — centree = normale, asymetrique = biais, multi-pics = sous-groupes.",
                "relevant": True
            },
            {
                "nom": "Boxplot Temperature par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "temperature_moyenne"),
                "interpretation": "Mediane (ligne), dispersion (boite Q1-Q3), et valeurs aberrantes (points).",
                "relevant": True
            },
            {
                "nom": "Boxplot Temperature par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "temperature_moyenne"),
                "interpretation": "Comparaison saisonniere — hauteur de boite = variabilite, decalage mediane = difference.",
                "relevant": True
            },
            {
                "nom": "Violin Temperature par Saison",
                "fonction": lambda df: plot_violin_by_saison(df, "temperature_moyenne"),
                "interpretation": "Boxplot + densite + points — forme complete de la distribution par saison.",
                "relevant": True
            },
            {
                "nom": "Heatmap Temperature par Region et Saison",
                "fonction": lambda df: plot_mean_by_region_saison(df),
                "interpretation": "Couleurs chaudes = temperatures elevees, froides = temperatures basses — patterns geographiques immediats.",
                "relevant": True
            },
            {
                "nom": "Scatter Temperature vs Humidite",
                "fonction": lambda df: plot_scatter_temperature_humidite(df),
                "interpretation": "5D en un graphique — humidite (X), temperature (Y), region (couleur), pluie (taille), vent (survol).",
                "relevant": True
            },
            {
                "nom": "Matrice de correlation",
                "fonction": lambda df: plot_correlation_heatmap(df),
                "interpretation": "+1 = lien positif fort, -1 = lien negatif fort, 0 = independance.",
                "relevant": False
            }
        ]
    },
    "humidite": {
        "titre": "Analyses de l'Humidite",
        "icon": "💧",
        "graphes": [
            {
                "nom": "Distribution de l'humidite",
                "fonction": lambda df: plot_distribution_histograms(df),
                "interpretation": "Forme de la distribution — centree = normale, asymetrique = biais.",
                "relevant": True
            },
            {
                "nom": "Boxplot Humidite par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "humidite"),
                "interpretation": "Mediane, dispersion et valeurs aberrantes par region.",
                "relevant": True
            },
            {
                "nom": "Boxplot Humidite par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "humidite"),
                "interpretation": "Comparaison saisonniere de l'humidite relative.",
                "relevant": True
            },
            {
                "nom": "Violin Humidite par Saison",
                "fonction": lambda df: plot_violin_by_saison(df, "humidite"),
                "interpretation": "Forme complete de la distribution de l'humidite par saison.",
                "relevant": True
            },
            {
                "nom": "Scatter Temperature vs Humidite",
                "fonction": lambda df: plot_scatter_temperature_humidite(df),
                "interpretation": "Correlation entre humidite et temperature, couleur = region.",
                "relevant": True
            },
            {
                "nom": "Matrice de correlation",
                "fonction": lambda df: plot_correlation_heatmap(df),
                "interpretation": "Correlation entre humidite et les autres variables.",
                "relevant": True
            }
        ]
    },
    "precipitations": {
        "titre": "Analyses des Precipitations",
        "icon": "🌧️",
        "graphes": [
            {
                "nom": "Distribution des precipitations",
                "fonction": lambda df: plot_distribution_histograms(df),
                "interpretation": "Forme de la distribution des precipitations en mm.",
                "relevant": True
            },
            {
                "nom": "Boxplot Precipitations par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "precipitations"),
                "interpretation": "Comparaison des precipitations medianes par region.",
                "relevant": True
            },
            {
                "nom": "Boxplot Precipitations par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "precipitations"),
                "interpretation": "Differences saisonnieres des precipitations attendues.",
                "relevant": True
            },
            {
                "nom": "Violin Precipitations par Saison",
                "fonction": lambda df: plot_violin_by_saison(df, "precipitations"),
                "interpretation": "Forme de la distribution saisonniere des precipitations.",
                "relevant": True
            },
            {
                "nom": "Precipitations vs Vent",
                "fonction": lambda df: plot_precipitation_wind_bubble(df),
                "interpretation": "Relation vent-precipitations, taille = temperature normalisee.",
                "relevant": True
            },
            {
                "nom": "Matrice de correlation",
                "fonction": lambda df: plot_correlation_heatmap(df),
                "interpretation": "Liens entre precipitations et autres variables climatiques.",
                "relevant": True
            }
        ]
    },
    "vent": {
        "titre": "Analyses du Vent",
        "icon": "💨",
        "graphes": [
            {
                "nom": "Distribution de la vitesse du vent",
                "fonction": lambda df: plot_distribution_histograms(df),
                "interpretation": "Forme de la distribution des vitesses de vent.",
                "relevant": True
            },
            {
                "nom": "Boxplot Vent par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "vitesse_vent"),
                "interpretation": "Mediane et variabilite du vent par region.",
                "relevant": True
            },
            {
                "nom": "Boxplot Vent par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "vitesse_vent"),
                "interpretation": "Differences saisonnieres de la vitesse du vent.",
                "relevant": True
            },
            {
                "nom": "Violin Vent par Saison",
                "fonction": lambda df: plot_violin_by_saison(df, "vitesse_vent"),
                "interpretation": "Distribution complete des vitesses par saison.",
                "relevant": True
            },
            {
                "nom": "Precipitations vs Vent",
                "fonction": lambda df: plot_precipitation_wind_bubble(df),
                "interpretation": "Vent vs precipitations avec temperature comme taille des bulles.",
                "relevant": True
            },
            {
                "nom": "Matrice de correlation",
                "fonction": lambda df: plot_correlation_heatmap(df),
                "interpretation": "Correlation entre vent et autres parametres climatiques.",
                "relevant": True
            }
        ]
    },
    "region": {
        "titre": "Analyses par Region",
        "icon": "🗺️",
        "graphes": [
            {
                "nom": "Effectifs par Region (Barplot)",
                "fonction": lambda df: plot_barplot_qualitative(df, "region"),
                "interpretation": "Equilibre geographique du dataset — barres egales = bonne representativite.",
                "relevant": True
            },
            {
                "nom": "Proportions par Region (Pie chart)",
                "fonction": lambda df: plot_pie_qualitative(df, "region"),
                "interpretation": "Part de chaque region dans l'ensemble — survolez pour les effectifs exacts.",
                "relevant": True
            },
            {
                "nom": "Boxplot Temperature par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "temperature_moyenne"),
                "interpretation": "Comparaison des temperatures medianes entre les 5 regions.",
                "relevant": True
            },
            {
                "nom": "Boxplot Humidite par Region",
                "fonction": lambda df: plot_boxplot_by_region(df, "humidite"),
                "interpretation": "Niveaux d'humidite relatifs par zone geographique.",
                "relevant": True
            },
            {
                "nom": "Comparaison Normalisee par Region",
                "fonction": lambda df: plot_bar_comparison_by_region(df),
                "interpretation": "4 variables normalisees (0-100) — identifie les regions extremes.",
                "relevant": True
            },
            {
                "nom": "Croisement Region x Saison (Empile)",
                "fonction": lambda df: plot_stacked_bar_region_saison(df),
                "interpretation": "Hauteur totale = effectif region, couleurs = proportion saisonniere.",
                "relevant": True
            },
            {
                "nom": "Croisement Region x Saison (Groupe)",
                "fonction": lambda df: plot_grouped_bar_region_saison(df),
                "interpretation": "Comparaison cote-a-cote des effectifs saisonniers par region.",
                "relevant": True
            }
        ]
    },
    "saison": {
        "titre": "Analyses par Saison",
        "icon": "📅",
        "graphes": [
            {
                "nom": "Effectifs par Saison (Barplot)",
                "fonction": lambda df: plot_barplot_qualitative(df, "saison"),
                "interpretation": "Equilibre saisonnier — 50/50 ideal, desequilibre = biais de collecte.",
                "relevant": True
            },
            {
                "nom": "Proportions par Saison (Pie chart)",
                "fonction": lambda df: plot_pie_qualitative(df, "saison"),
                "interpretation": "Proportion des saisons — equilibre pour des comparaisons fiables.",
                "relevant": True
            },
            {
                "nom": "Boxplot Temperature par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "temperature_moyenne"),
                "interpretation": "Difference de temperature mediane entre saison des pluies et saison seche.",
                "relevant": True
            },
            {
                "nom": "Boxplot Humidite par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "humidite"),
                "interpretation": "Contraste d'humidite entre les deux periodes climatiques.",
                "relevant": True
            },
            {
                "nom": "Boxplot Precipitations par Saison",
                "fonction": lambda df: plot_boxplot_by_saison(df, "precipitations"),
                "interpretation": "Comparaison attendue : saison des pluies >> saison seche.",
                "relevant": True
            },
            {
                "nom": "Violin Temperature par Saison",
                "fonction": lambda df: plot_violin_by_saison(df, "temperature_moyenne"),
                "interpretation": "Forme complete de la distribution thermique saisonniere.",
                "relevant": True
            },
            {
                "nom": "Croisement Region x Saison",
                "fonction": lambda df: plot_stacked_bar_region_saison(df),
                "interpretation": "Repartition saisonniere au sein de chaque region geographique.",
                "relevant": True
            }
        ]
    },
    "global": {
        "titre": "Vue d'ensemble du Dataset",
        "icon": "📊",
        "graphes": [
            {
                "nom": "Distributions des 4 variables",
                "fonction": lambda df: plot_distribution_histograms(df),
                "interpretation": "Apercu simultane des distributions de toutes les variables quantitatives.",
                "relevant": True
            },
            {
                "nom": "Matrice de correlation",
                "fonction": lambda df: plot_correlation_heatmap(df),
                "interpretation": "Liens entre variables : +1 fort positif, -1 fort negatif, 0 independance.",
                "relevant": True
            },
            {
                "nom": "Matrice de scatter plots",
                "fonction": lambda df: plot_pairplot_streamlit(df),
                "interpretation": "Nuage allonge = forte correlation, disperse = faible correlation.",
                "relevant": True
            },
            {
                "nom": "Comparaison par Region (barres normalisees)",
                "fonction": lambda df: plot_bar_comparison_by_region(df),
                "interpretation": "4 variables sur echelle commune (0-100) pour comparer les regions.",
                "relevant": True
            }
        ]
    }
}


def init_app():
    """
    Initialise l'application en chargeant et validant les donnees.
    
    Cette fonction est appelee au demarrage et met en cache les donnees
    pour optimiser les performances.
    """
    st.session_state['data_loaded'] = True


@st.cache_data
def get_cached_data():
    """
    Charge les donnees avec mise en cache Streamlit.
    
    La mise en cache evite de recharger le CSV a chaque interaction,
    ameliorant ainsi les performances de l'application.
    
    Returns:
        pd.DataFrame: DataFrame mis en cache
    """
    return load_dataset()


def main():
    """Fonction principale de l'application Streamlit."""
    
    st.title("🌦️ Dashboard Climatique")
    st.subheader("Analyse et visualisation des donnees climatiques")
    
    st.markdown("---")
    
    try:
        # Chargement des donnees
        with st.spinner("Chargement des donnees..."):
            df = get_cached_data()
        
        st.success(f"Donnees chargees avec succes : {len(df)} enregistrements")
        
        # Validation des donnees
        validation = validate_data(df)
        
        if validation['alertes']:
            with st.expander("📋 Rapport de validation"):
                if validation['valeurs_manquantes']:
                    st.warning("⚠️ Valeurs manquantes detectees")
                for alerte in validation['alertes']:
                    st.info(alerte)
        
        # Informations generales
        st.header("📊 Informations Generales")
        
        info = get_data_info(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre d'enregistrements", info['nombre_lignes'])
        with col2:
            st.metric("Nombre de variables", info['nombre_colonnes'])
        with col3:
            st.metric("Regions", len(info['regions_uniques']))
        with col4:
            st.metric("Saisons", len(info['saisons_uniques']))
        
        st.markdown("---")
        
        # Apercu des donnees
        st.header("📋 Apercu des Donnees")
        
        tab1, tab2, tab3 = st.tabs(["Donnees brutes", "Statistiques", "Apercu par region/saison"])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            stats = get_basic_stats(df)
            st.dataframe(stats, use_container_width=True)
        
        with tab3:
            grouped = group_by_region_saison(df)
            st.dataframe(grouped, use_container_width=True)
        
        st.markdown("---")
        
        # Filtres interactifs
        st.header("🔍 Filtres Interactifs")
        
        col_filtre1, col_filtre2 = st.columns(2)
        
        with col_filtre1:
            region_selectionnee = st.selectbox(
                "Selectionner une region",
                options=["Toutes"] + info['regions_uniques']
            )
        
        with col_filtre2:
            saison_selectionnee = st.selectbox(
                "Selectionner une saison",
                options=["Toutes"] + info['saisons_uniques']
            )
        
        # Application des filtres
        df_filtre = df.copy()
        
        if region_selectionnee != "Toutes":
            df_filtre = filter_by_region(df_filtre, region_selectionnee)
        
        if saison_selectionnee != "Toutes":
            df_filtre = filter_by_saison(df_filtre, saison_selectionnee)
        
        st.write(f"Resultats filtres : **{len(df_filtre)}** enregistrements")
        
        if len(df_filtre) > 0:
            st.dataframe(df_filtre, use_container_width=True)
        else:
            st.warning("Aucun resultat pour ces filtres")
        
        st.markdown("---")
        
        # ============================================
        # SECTION VISUALISATIONS AVEC RECHERCHE
        # ============================================
        st.markdown("---")
        st.header("📈 Recherche et Visualisations")

        # Barre de recherche dans la sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Recherche par Variable")
        st.sidebar.markdown("Mots-cles : temperature, humidite, precipitations, vent, region, saison, global")
        recherche_input = st.sidebar.text_input(
            "Rechercher une variable",
            value="",
            placeholder="Ex: temperature, humidite, vent..."
        )
        recherche = recherche_input.strip().lower()

        def trouver_variable(recherche):
            if not recherche:
                return None
            mapping = {
                "temperature": ["temperature", "temp", "chaleur", "degre", "thermo", "°c", "celsius"],
                "humidite": ["humidite", "hum", "hydrometrie", "humid", "eau"],
                "precipitations": ["precipitations", "precip", "pluie", "pluviometrie", "mm", "pluviosite"],
                "vent": ["vent", "vitesse_vent", "wind", "air", "rafale", "kmh"],
                "region": ["region", "geo", "geographique", "zone", "nord", "sud", "est", "ouest", "centre", "carte"],
                "saison": ["saison", "periode", "pluie", "seche", "hiver", "ete", "annuel"],
                "global": ["global", "tout", "ensemble", "overview", "resume", "general", "dashboard"]
            }
            for variable, mots_cles in mapping.items():
                if any(mot in recherche for mot in mots_cles):
                    return variable
            for variable, mots_cles in mapping.items():
                if len(recherche) >= 3 and any(mot.startswith(recherche[:3]) for mot in mots_cles):
                    return variable
            return None

        variable_trouvee = trouver_variable(recherche)
        
        # Fonction d'interpretation automatique
        def get_auto_interpretation(nom_graphe, df):
            nom = nom_graphe.lower()
            try:
                if "distribution" in nom or "histogram" in nom:
                    if "temperature" in nom or "temp" in nom:
                        return interpret_distribution(df, "temperature_moyenne")
                    elif "humidite" in nom or "humid" in nom:
                        return interpret_distribution(df, "humidite")
                    elif "precipitation" in nom or "pluie" in nom:
                        return interpret_distribution(df, "precipitations")
                    elif "vent" in nom or "vitesse" in nom:
                        return interpret_distribution(df, "vitesse_vent")
                    else:
                        return "Distribution des 4 variables climatiques. Chaque histogramme montre la frequence des valeurs observees."
                elif "boxplot" in nom and "region" in nom:
                    if "temperature" in nom:
                        return interpret_boxplot_by_group(df, "temperature_moyenne", "region")
                    elif "humidite" in nom:
                        return interpret_boxplot_by_group(df, "humidite", "region")
                    elif "precipitation" in nom:
                        return interpret_boxplot_by_group(df, "precipitations", "region")
                    elif "vent" in nom:
                        return interpret_boxplot_by_group(df, "vitesse_vent", "region")
                elif "boxplot" in nom and "saison" in nom:
                    if "temperature" in nom:
                        return interpret_boxplot_by_group(df, "temperature_moyenne", "saison")
                    elif "humidite" in nom:
                        return interpret_boxplot_by_group(df, "humidite", "saison")
                    elif "precipitation" in nom:
                        return interpret_boxplot_by_group(df, "precipitations", "saison")
                    elif "vent" in nom:
                        return interpret_boxplot_by_group(df, "vitesse_vent", "saison")
                elif "scatter" in nom and "temperature" in nom and "humidite" in nom:
                    return interpret_scatter_temp_humidite(df)
                elif "correlation" in nom and "matrice" not in nom:
                    return interpret_correlation_matrix(df)
                elif "comparaison" in nom and "region" in nom:
                    return interpret_region_comparison(df)
                elif "barplot" in nom and "region" in nom:
                    return interpret_barplot_qualitative(df, "region")
                elif "barplot" in nom and "saison" in nom:
                    return interpret_barplot_qualitative(df, "saison")
                elif "pie" in nom and "region" in nom:
                    return interpret_barplot_qualitative(df, "region")
                elif "pie" in nom and "saison" in nom:
                    return interpret_barplot_qualitative(df, "saison")
                elif "croisement" in nom or "empile" in nom or "groupe" in nom:
                    return interpret_crosstab_region_saison(df)
                elif "precipitation" in nom and "vent" in nom:
                    return interpret_precipitations_vent(df)
                elif "violin" in nom:
                    if "temperature" in nom:
                        return interpret_violin_by_saison(df, "temperature_moyenne")
                    elif "humidite" in nom:
                        return interpret_violin_by_saison(df, "humidite")
                    elif "precipitation" in nom:
                        return interpret_violin_by_saison(df, "precipitations")
                    elif "vent" in nom:
                        return interpret_violin_by_saison(df, "vitesse_vent")
                elif "heatmap" in nom or "temperature par region" in nom:
                    return interpret_heatmap_region_saison(df)
                elif "scatter matrix" in nom or "matrice de scatter" in nom:
                    return interpret_correlation_matrix(df)
                else:
                    return "Analyse automatique non disponible pour ce type de graphique."
            except Exception as e:
                return f"Analyse automatique indisponible : {e}"
        
        # Affichage conditionnel selon la recherche
        if variable_trouvee:
            config = GRAPHES_PAR_VARIABLE[variable_trouvee]
            st.subheader(f"{config['icon']} {config['titre']}")
            
            graphes_pertinents = [g for g in config["graphes"] if g.get("relevant", True)]
            
            if not graphes_pertinents:
                st.warning("Aucun graphique specifique pour cette variable.")
            
            for i, graphe_config in enumerate(graphes_pertinents):
                with st.container():
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    st.markdown(f"**📊 {graphe_config['nom']}**  <span style='color:#888'>#{i+1}/{len(graphes_pertinents)}</span>", unsafe_allow_html=True)
                    
                    try:
                        fig = graphe_config["fonction"](df)
                        st.plotly_chart(fig, use_container_width=True, key=f"g_{variable_trouvee}_{i}")
                        
                        with st.expander("📖 Interpretation automatique", expanded=False):
                            interpretation = get_auto_interpretation(graphe_config['nom'], df)
                            st.info(interpretation)
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
            
            st.success(f"✅ {len(graphes_pertinents)} graphique(s) pour '{recherche}'")
            
        elif recherche:
            st.warning(f"❓ Variable '{recherche}' non reconnue")
            st.info("Essayez : temperature, humidite, precipitations, vent, region, saison, global")
            
            st.subheader("📋 Ou selectionnez une variable")
            var_choisie = st.selectbox(
                "",
                options=["", "temperature", "humidite", "precipitations", "vent", "region", "saison", "global"],
                format_func=lambda x: {
                    "": "-- Choisir --",
                    "temperature": "🌡️ Temperature",
                    "humidite": "💧 Humidite",
                    "precipitations": "🌧️ Precipitations",
                    "vent": "💨 Vent",
                    "region": "🗺️ Region",
                    "saison": "📅 Saison",
                    "global": "📊 Vue globale"
                }.get(x, x)
            )
            if var_choisie:
                config = GRAPHES_PAR_VARIABLE[var_choisie]
                st.subheader(f"{config['icon']} {config['titre']}")
                
                graphes_pertinents = [g for g in config["graphes"] if g.get("relevant", True)]
                for i, graphe_config in enumerate(graphes_pertinents):
                    with st.container():
                        st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                        st.markdown(f"**📊 {graphe_config['nom']}**  <span style='color:#888'>#{i+1}/{len(graphes_pertinents)}</span>", unsafe_allow_html=True)
                        try:
                            fig = graphe_config["fonction"](df)
                            st.plotly_chart(fig, use_container_width=True, key=f"g_alt_{var_choisie}_{i}")
                            with st.expander("📖 Interpretation automatique", expanded=False):
                                interpretation = get_auto_interpretation(graphe_config['nom'], df)
                                st.info(interpretation)
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
        
        else:
            # Aucune recherche : message d'accueil
            st.info("🔍 Tapez une variable dans la sidebar (ex: 'temperature') ou selectionnez ci-dessus.")
        
        # ============================================
        # SECTION ANALYSES STATISTIQUES - PHASE 3
        # ============================================
        st.markdown("---")
        st.header("📐 Analyses Statistiques Avancees")
        
        # Onglets pour organiser les analyses
        tab_stats, tab_tests, tab_reg, tab_outliers = st.tabs([
            "📊 Resume Statistique", 
            "🔬 Tests Statistiques", 
            "📈 Regressions", 
            "⚠️ Detection d'Outliers"
        ])
        
        with tab_stats:
            st.subheader("Resumes descriptifs complets")
            var_stats = st.selectbox(
                "Variable a analyser",
                options=["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"],
                format_func=lambda x: {
                    "temperature_moyenne": "🌡️ Temperature (°C)",
                    "humidite": "💧 Humidite (%)",
                    "precipitations": "🌧️ Precipitations (mm)",
                    "vitesse_vent": "💨 Vitesse du vent (km/h)"
                }.get(x, x),
                key="stats_var"
            )
            
            stats_result = resume_statistique_complet(df, var_stats)
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Moyenne", f"{stats_result['moyenne']}")
                st.metric("Mediane", f"{stats_result['mediane']}")
            with col_s2:
                st.metric("Ecart-type", f"{stats_result['ecart_type']}")
                st.metric("CV (%)", f"{stats_result['cv']}" if stats_result['cv'] else "N/A")
            with col_s3:
                st.metric("Min", f"{stats_result['min']}")
                st.metric("Max", f"{stats_result['max']}")
            with col_s4:
                st.metric("Q1", f"{stats_result['q1']}")
                st.metric("Q3", f"{stats_result['q3']}")
            
            with st.expander("📋 Details complets"):
                st.json(stats_result)
        
        with tab_tests:
            st.subheader("Tests d'hypotheses")
            
            test_type = st.selectbox(
                "Choisir un test",
                options=[
                    "normalite_shapiro",
                    "homogeneite_levene",
                    "ttest_independant",
                    "anova",
                    "chi2"
                ],
                format_func=lambda x: {
                    "normalite_shapiro": "🎯 Test de normalite (Shapiro-Wilk)",
                    "homogeneite_levene": "⚖️ Test d'homogeneite des variances (Levene)",
                    "ttest_independant": "📏 T-test de Student (2 groupes)",
                    "anova": "📊 ANOVA (plusieurs groupes)",
                    "chi2": "🔗 Test du Chi2 (independance)"
                }.get(x, x)
            )
            
            if test_type == "normalite_shapiro":
                var_test = st.selectbox("Variable", ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"], key="test_var_shapiro")
                result = test_normalite_shapiro(df, var_test)
                st.write(f"**Variable** : {result['variable']}")
                st.write(f"**Statistique W** : {result['statistique']}")
                st.write(f"**p-value** : {result['p_value']}")
                if result['normal']:
                    st.success(f"✅ {result['interpretation']}")
                else:
                    st.warning(f"⚠️ {result['interpretation']}")
                    st.info("Consequence : les tests parametriques (t-test, ANOVA) peuvent etre moins fiables.")
                
            elif test_type == "homogeneite_levene":
                var_test = st.selectbox("Variable", ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"], key="test_var_levene")
                groupe_test = st.selectbox("Groupe", ["region", "saison"], key="test_group_levene")
                result = test_homogeneite_variance(df, var_test, groupe_test)
                st.write(f"**Variable** : {result['variable']} | **Groupe** : {result['groupe']}")
                st.write(f"**Statistique W** : {result['statistique']}")
                st.write(f"**p-value** : {result['p_value']}")
                if result['homogene']:
                    st.success(f"✅ {result['interpretation']}")
                else:
                    st.warning(f"⚠️ {result['interpretation']}")
                    st.info("Consequence : utiliser un t-test de Welch (variances inegales) plutot que Student classique.")
                
            elif test_type == "ttest_independant":
                var_test = st.selectbox("Variable", ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"], key="test_var_ttest")
                groupe_test = st.selectbox("Groupe", ["region", "saison"], key="test_group_ttest")
                
                groupes = df[groupe_test].unique().tolist()
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    g1 = st.selectbox("Groupe 1", groupes, index=0, key="ttest_g1")
                with col_g2:
                    g2 = st.selectbox("Groupe 2", [g for g in groupes if g != g1], index=0, key="ttest_g2")
                
                result = ttest_independant(df, var_test, groupe_test, g1, g2)
                st.write(f"**Comparaison** : {result['comparaison']} | **Variable** : {result['variable']}")
                st.write(f"**Moyenne {g1}** : {result['moyenne_g1']} (n={result['n_g1']})")
                st.write(f"**Moyenne {g2}** : {result['moyenne_g2']} (n={result['n_g2']})")
                st.write(f"**Statistique t** : {result['statistique']}")
                st.write(f"**p-value** : {result['p_value']}")
                
                if result['significatif']:
                    st.success(f"✅ {result['interpretation']}")
                    diff = abs(result['moyenne_g1'] - result['moyenne_g2'])
                    st.info(f"Ecart absolu : {diff:.2f} unites.")
                else:
                    st.info(f"ℹ️ {result['interpretation']}")
                
            elif test_type == "anova":
                var_test = st.selectbox("Variable", ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"], key="test_var_anova")
                groupe_test = st.selectbox("Groupe", ["region", "saison"], key="test_group_anova")
                result = anova_un_facteur(df, var_test, groupe_test)
                
                st.write(f"**Variable** : {result['variable']} | **Groupe** : {result['groupe']}")
                st.write(f"**F-statistique** : {result['statistique_F']}")
                st.write(f"**p-value** : {result['p_value']}")
                
                if result['significatif']:
                    st.success(f"✅ {result['interpretation']}")
                    st.info("Post-hoc necessaire : un test de Tukey HSD permettrait d'identifier quels groupes different.")
                else:
                    st.info(f"ℹ️ {result['interpretation']}")
                
                with st.expander("📋 Statistiques par groupe"):
                    st.json(result['statistiques_groupes'])
                
            elif test_type == "chi2":
                col1_test = st.selectbox("Variable 1", ["region", "saison"], index=0, key="chi2_v1")
                col2_test = st.selectbox("Variable 2", ["saison", "region"], index=1, key="chi2_v2")
                result = chi2_independance(df, col1_test, col2_test)
                
                st.write(f"**Variables** : {result['variables']}")
                st.write(f"**Chi2** : {result['chi2']}")
                st.write(f"**p-value** : {result['p_value']}")
                st.write(f"**DDL** : {result['ddl']}")
                
                if result['independantes']:
                    st.success(f"✅ {result['interpretation']}")
                else:
                    st.warning(f"⚠️ {result['interpretation']}")
                    st.info("Les proportions des categories different selon les groupes.")
                
                with st.expander("📋 Tableau de contingence"):
                    st.json(result['tableau_contingence'])
        
        with tab_reg:
            st.subheader("Regressions et Modeles Predictifs")
            
            reg_type = st.selectbox(
                "Type de regression",
                options=["simple", "multiple"],
                format_func=lambda x: {
                    "simple": "📉 Regression lineaire simple (2 variables)",
                    "multiple": "📊 Regression multiple (plusieurs variables)"
                }.get(x, x)
            )
            
            if reg_type == "simple":
                col_x, col_y = st.columns(2)
                with col_x:
                    x_var = st.selectbox("Variable X (predictive)", ["humidite", "precipitations", "vitesse_vent"], key="reg_simple_x")
                with col_y:
                    y_var = st.selectbox("Variable Y (cible)", ["temperature_moyenne", "humidite", "precipitations"], key="reg_simple_y")
                
                result = regression_lineaire_simple(df, x_var, y_var)
                
                st.write(f"**Equation** : `{result['equation']}`")
                st.write(f"**Coefficient de determination R²** : {result['r2']}")
                st.write(f"**Correlation r** : {result['r']}")
                st.write(f"**RMSE** : {result['rmse']}")
                st.write(f"**p-value** : {result['p_value']}")
                
                if result['significatif']:
                    st.success(f"✅ {result['interpretation']}")
                    direction = "augmente" if result['pente'] > 0 else "diminue"
                    st.info(f"Quand {x_var} augmente de 1 unite, {y_var} {direction} de {abs(result['pente']):.4f} unites en moyenne.")
                else:
                    st.info(f"ℹ️ {result['interpretation']}")
                    st.warning("Le modele n'a pas de pouvoir predictif significatif.")
                
                # Graphique de regression
                import plotly.express as px
                fig_reg = px.scatter(df, x=x_var, y=y_var, trendline="ols",
                                     title=f"Regression : {y_var} ~ {x_var}")
                st.plotly_chart(fig_reg, use_container_width=True)
                
            else:
                y_var = st.selectbox("Variable cible (Y)", ["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"], key="reg_multi_y")
                x_vars = st.multiselect(
                    "Variables predictives (X)",
                    options=["humidite", "precipitations", "vitesse_vent", "region", "saison"],
                    default=["humidite", "precipitations", "region"]
                )
                
                if len(x_vars) >= 1:
                    result = regression_multiple(df, y_var, x_vars)
                    st.write(f"**Equation** : `{result['equation']}`")
                    st.write(f"**R²** : {result['r2']} ({result['r2']*100:.1f}% de variance expliquee)")
                    st.write(f"**RMSE** : {result['rmse']}")
                    
                    st.subheader("Coefficients")
                    coeffs_df = pd.DataFrame([
                        {"Variable": k, "Coefficient": v}
                        for k, v in result['coefficients'].items()
                    ])
                    st.dataframe(coeffs_df, use_container_width=True)
                    
                    if result['r2'] > 0.5:
                        st.success(f"✅ {result['interpretation']}")
                    elif result['r2'] > 0.2:
                        st.info(f"ℹ️ {result['interpretation']}")
                    else:
                        st.warning(f"⚠️ {result['interpretation']} — modele peu performant.")
                else:
                    st.warning("Selectionnez au moins une variable predictive.")
        
        with tab_outliers:
            st.subheader("Detection des Valeurs Aberrantes (Methode IQR)")
            var_out = st.selectbox(
                "Variable a analyser",
                options=["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"],
                format_func=lambda x: {
                    "temperature_moyenne": "🌡️ Temperature (°C)",
                    "humidite": "💧 Humidite (%)",
                    "precipitations": "🌧️ Precipitations (mm)",
                    "vitesse_vent": "💨 Vitesse du vent (km/h)"
                }.get(x, x),
                key="outliers_var"
            )
            
            result = detecter_outliers_iqr(df, var_out)
            
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                st.metric("Q1", result['q1'])
                st.metric("Q3", result['q3'])
            with col_o2:
                st.metric("IQR", result['iqr'])
                st.metric("Borne inf.", result['borne_inf'])
            with col_o3:
                st.metric("Borne sup.", result['borne_sup'])
                st.metric("Outliers", f"{result['n_outliers']} ({result['pct_outliers']}%)")
            
            if result['n_outliers'] > 0:
                st.warning(f"⚠️ {result['n_outliers']} valeur(s) aberrante(s) detectee(s)")
                with st.expander("📋 Details des outliers"):
                    outliers_df = pd.DataFrame(result['outliers'])
                    st.dataframe(outliers_df, use_container_width=True)
            else:
                st.success("✅ Aucune valeur aberrante detectee selon la methode IQR.")
            
            # Visualisation
            fig_out = plot_boxplot_by_region(df, var_out) if st.checkbox("Afficher le boxplot") else None
            if fig_out:
                st.plotly_chart(fig_out, use_container_width=True)
        
        # ============================================
        # SECTION EXPORT DE RAPPORTS - PHASE 4
        # ============================================
        st.markdown("---")
        st.header("📤 Export de Rapports")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.subheader("📊 Excel")
            st.markdown("Export multi-feuilles : donnees, stats, correlations, aggregations")
            
            if st.button("Generer le rapport Excel", key="btn_excel"):
                with st.spinner("Generation du fichier Excel..."):
                    try:
                        excel_buffer = export_excel_complet(df)
                        st.download_button(
                            label="Telecharger rapport.xlsx",
                            data=excel_buffer,
                            file_name="rapport_climatique.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_excel"
                        )
                        st.success("✅ Rapport Excel genere avec succes!")
                    except Exception as e:
                        st.error(f"Erreur Excel : {e}")
        
        with col_exp2:
            st.subheader("📄 PDF")
            st.markdown("Rapport complet avec statistiques, tableaux et graphiques")
            
            inclure_graphs = st.checkbox("Inclure les graphiques (necessite kaleido)", value=False, key="chk_pdf_graphs")
            
            if st.button("Generer le rapport PDF", key="btn_pdf"):
                with st.spinner("Generation du PDF..."):
                    try:
                        figures_dict = None
                        if inclure_graphs:
                            # Generer quelques graphiques pour le PDF
                            figures_dict = {
                                "Distributions": plot_distribution_histograms(df),
                                "Correlation": plot_correlation_heatmap(df),
                                "Comparaison Region": plot_bar_comparison_by_region(df)
                            }
                        
                        pdf_buffer = generate_pdf_report(df, figures_dict=figures_dict)
                        st.download_button(
                            label="Telecharger rapport.pdf",
                            data=pdf_buffer,
                            file_name="rapport_climatique.pdf",
                            mime="application/pdf",
                            key="dl_pdf"
                        )
                        st.success("✅ Rapport PDF genere avec succes!")
                    except Exception as e:
                        st.error(f"Erreur PDF : {e}")
                        st.info("Conseil : si kaleido n'est pas installe, decochez 'Inclure les graphiques'")
        
        with col_exp3:
            st.subheader("🌐 HTML")
            st.markdown("Rapport web complet avec tableaux interactifs")
            
            if st.button("Generer le rapport HTML", key="btn_html"):
                with st.spinner("Generation du HTML..."):
                    try:
                        html_content = generate_html_report(df)
                        st.download_button(
                            label="Telecharger rapport.html",
                            data=html_content,
                            file_name="rapport_climatique.html",
                            mime="text/html",
                            key="dl_html"
                        )
                        st.success("✅ Rapport HTML genere avec succes!")
                    except Exception as e:
                        st.error(f"Erreur HTML : {e}")
        
        # ============================================
        # SECTION MODELES PREDICTIFS - PHASE 5
        # ============================================
        st.markdown("---")
        st.header("🤖 Phase 5 : Modeles Predictifs Avances")
        
        tab_compare, tab_train, tab_predict = st.tabs([
            "📊 Comparaison des modeles",
            "🔧 Entrainement personnalise",
            "🔮 Prediction interactive"
        ])
        
        with tab_compare:
            st.subheader("Comparaison automatique de 3 a 4 modeles")
            st.markdown("Entrainement simultane de : **Regression Lineaire**, **Random Forest**, **Gradient Boosting**" + 
                       (" et **XGBoost**" if XGBOOST_AVAILABLE else " (XGBoost non installe)") + 
                       " sur les memes donnees.")
            
            col_target, col_feats = st.columns(2)
            with col_target:
                target_var = st.selectbox(
                    "Variable cible a predire",
                    options=["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"],
                    format_func=lambda x: {"temperature_moyenne": "🌡️ Temperature", "humidite": "💧 Humidite",
                                          "precipitations": "🌧️ Precipitations", "vitesse_vent": "💨 Vent"}.get(x, x),
                    key="pred_target_compare"
                )
            with col_feats:
                all_features = [c for c in df.columns if c != target_var]
                selected_features = st.multiselect(
                    "Variables predictives",
                    options=all_features,
                    default=[c for c in all_features if c in ["humidite", "precipitations", "vitesse_vent", "region", "saison"]],
                    key="pred_feats_compare"
                )
            
            if st.button("Lancer la comparaison", key="btn_compare"):
                if len(selected_features) < 1:
                    st.warning("Selectionnez au moins une variable predictive.")
                else:
                    with st.spinner("Entrainement des modeles en cours..."):
                        try:
                            compare_df = comparer_modeles(df, target_var, selected_features)
                            st.dataframe(compare_df, use_container_width=True)
                            
                            # Mettre en evidence le meilleur
                            if 'R2 Test' in compare_df.columns:
                                best_idx = compare_df['R2 Test'].idxmax()
                                best_model = compare_df.loc[best_idx, 'Modele']
                                best_r2 = compare_df.loc[best_idx, 'R2 Test']
                                st.success(f"🏆 Meilleur modele : **{best_model}** (R² test = {best_r2})")
                            
                            # Graphique de comparaison
                            fig_compare = px.bar(
                                compare_df, x='Modele', y='R2 Test',
                                title="R² Test par modele",
                                color='R2 Test', color_continuous_scale='RdYlGn',
                                range_y=[0, 1]
                            )
                            st.plotly_chart(fig_compare, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur comparaison : {e}")
        
        with tab_train:
            st.subheader("Entrainement et analyse d'un modele")
            
            col1, col2 = st.columns(2)
            with col1:
                target_var_train = st.selectbox(
                    "Variable cible",
                    options=["temperature_moyenne", "humidite", "precipitations", "vitesse_vent"],
                    key="pred_target_train"
                )
                model_choice = st.selectbox(
                    "Modele",
                    options=["linear", "random_forest", "gradient_boosting", "xgboost"],
                    format_func=lambda x: {"linear": "Regression Lineaire", "random_forest": "Random Forest",
                                          "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost"}.get(x, x),
                    key="pred_model_choice"
                )
            with col2:
                all_f_train = [c for c in df.columns if c != target_var_train]
                feats_train = st.multiselect(
                    "Variables predictives",
                    options=all_f_train,
                    default=[c for c in all_f_train if c in ["humidite", "precipitations", "region", "saison"]],
                    key="pred_feats_train"
                )
                n_estimators = st.slider("Nombre d'arbres", 10, 500, 100, key="pred_n_estimators")
                max_depth = st.slider("Profondeur max", 1, 20, 5, key="pred_max_depth")
            
            if st.button("Entrainer le modele", key="btn_train"):
                if len(feats_train) < 1:
                    st.warning("Selectionnez au moins une variable predictive.")
                elif model_choice == "xgboost" and not XGBOOST_AVAILABLE:
                    st.error("XGBoost n'est pas installe. Executez 'pip install xgboost'.")
                else:
                    with st.spinner("Entrainement..."):
                        try:
                            data = preparer_donnees(df, target_var_train, feats_train)
                            result = entrainer_modele(data, model_type=model_choice,
                                                      params={'n_estimators': n_estimators, 'max_depth': max_depth})
                            
                            # Metriques
                            m = result['metrics']
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("R² Train", f"{m['train_r2']:.3f}")
                            c2.metric("R² Test", f"{m['test_r2']:.3f}")
                            c3.metric("RMSE Test", f"{m['test_rmse']:.3f}")
                            c4.metric("MAE Test", f"{m['test_mae']:.3f}")
                            
                            # Interpretation
                            st.markdown("**Interpretation des performances**")
                            st.info(interpreter_performance(m))
                            
                            # Feature importance
                            if result['feature_importance']:
                                st.subheader("Importance des variables")
                                imp_df = pd.DataFrame([
                                    {"Variable": k, "Importance": round(v, 4)}
                                    for k, v in sorted(result['feature_importance'].items(),
                                                        key=lambda x: x[1], reverse=True)
                                ])
                                st.dataframe(imp_df, use_container_width=True)
                                
                                fig_imp = px.bar(imp_df, x='Variable', y='Importance',
                                                title="Importance des variables predictives",
                                                color='Importance', color_continuous_scale='Blues')
                                st.plotly_chart(fig_imp, use_container_width=True)
                            
                            # Scatter reel vs predit
                            st.subheader("Valeurs reelles vs Predit")
                            pred_df = pd.DataFrame({
                                'Reel': result['predictions']['y_test'],
                                'Predit': result['predictions']['y_pred_test']
                            })
                            fig_pred = px.scatter(pred_df, x='Reel', y='Predit',
                                                  title="Prediction vs Valeurs reelles",
                                                  labels={'Reel': 'Valeur reelle', 'Predit': 'Valeur predite'})
                            # Ligne y=x
                            min_val = min(pred_df['Reel'].min(), pred_df['Predit'].min())
                            max_val = max(pred_df['Reel'].max(), pred_df['Predit'].max())
                            fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                              line=dict(dash="dash", color="red"))
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Stocker en session state pour la prediction
                            st.session_state['last_model'] = result
                            st.session_state['last_target'] = target_var_train
                            st.session_state['last_features'] = feats_train
                            
                        except Exception as e:
                            st.error(f"Erreur entrainement : {e}")
        
        with tab_predict:
            st.subheader("Prediction interactive")
            st.markdown("Entrainez d'abord un modele dans l'onglet **Entrainement personnalise**, puis revenez ici pour predire.")
            
            if 'last_model' not in st.session_state:
                st.info("Aucun modele entraine. Allez dans l'onglet 'Entrainement personnalise'.")
            else:
                model_result = st.session_state['last_model']
                target_name = st.session_state.get('last_target', 'cible')
                features = st.session_state.get('last_features', [])
                
                st.success(f"Modele disponible pour predire **{target_name}**")
                
                input_values = {}
                cols = st.columns(min(len(features), 4))
                for i, feat in enumerate(features):
                    with cols[i % len(cols)]:
                        if df[feat].dtype == 'object' or df[feat].dtype.name == 'category':
                            input_values[feat] = st.selectbox(feat, df[feat].unique(), key=f"pred_input_{feat}")
                        else:
                            min_val = float(df[feat].min())
                            max_val = float(df[feat].max())
                            mean_val = float(df[feat].mean())
                            input_values[feat] = st.number_input(
                                feat, min_value=min_val, max_value=max_val,
                                value=mean_val, key=f"pred_input_{feat}"
                            )
                
                if st.button("Predire", key="btn_predict"):
                    try:
                        prediction = predire_valeur(model_result, input_values)
                        st.balloons()
                        st.metric(f"{target_name} predite", f"{prediction:.3f}")
                        
                        # Contexte
                        target_mean = df[target_name].mean()
                        target_std = df[target_name].std()
                        if abs(prediction - target_mean) < target_std:
                            st.info("La valeur predite est proche de la moyenne historique.")
                        elif prediction > target_mean + target_std:
                            st.warning("Valeur elevee par rapport a la moyenne historique.")
                        elif prediction < target_mean - target_std:
                            st.warning("Valeur basse par rapport a la moyenne historique.")
                    except Exception as e:
                        st.error(f"Erreur prediction : {e}")
        
        # Pied de page
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Dashboard Climatique - Application Streamlit | Phase 5 : Modeles Predictifs"
            "</div>", 
            unsafe_allow_html=True
        )
        
    except FileNotFoundError:
        st.error("❌ Fichier dataset_climat.csv non trouve")
        st.info("Veuillez verifier que le fichier dataset_climat.csv est present dans le repertoire")
    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")


if __name__ == "__main__":
    main()
