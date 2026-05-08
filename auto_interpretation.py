"""
Module d'interpretation automatique basee sur les statistiques reelles du dataset.
Genere des textes d'analyse contextualises pour chaque type de graphique.
"""

import pandas as pd
import numpy as np


def interpret_distribution(df, variable):
    """Interprete la distribution d'une variable quantitative."""
    data = df[variable].dropna()
    mean = data.mean()
    median = data.median()
    std = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # Skewness approxime
    skew = (mean - median) / std if std > 0 else 0
    
    # Test de platitude (coefficient de variation)
    cv = (std / mean * 100) if mean != 0 else 0
    
    texts = []
    
    # Forme de la distribution
    if abs(skew) < 0.2:
        texts.append(f"Distribution **symetrique** (moyenne ≈ mediane : {mean:.1f} vs {median:.1f}).")
    elif skew > 0.3:
        texts.append(f"Distribution **asymetrique a droite** (queue vers les hautes valeurs). Moyenne ({mean:.1f}) > Mediane ({median:.1f}).")
    elif skew < -0.3:
        texts.append(f"Distribution **asymetrique a gauche** (queue vers les basses valeurs). Moyenne ({mean:.1f}) < Mediane ({median:.1f}).")
    
    # Dispersion
    if cv < 15:
        texts.append(f"Faible variabilite (CV = {cv:.1f}%) — les valeurs sont concentrees autour de la moyenne.")
    elif cv < 30:
        texts.append(f"Variabilite moderée (CV = {cv:.1f}%).")
    else:
        texts.append(f"**Forte variabilite** (CV = {cv:.1f}%) — les donnees sont tres dispersees.")
    
    # Etendue
    texts.append(f"Etendue : [{min_val:.1f} ; {max_val:.1f}], soit une amplitude de {max_val - min_val:.1f} unites.")
    
    return " ".join(texts)


def interpret_boxplot_by_group(df, variable, group_col):
    """Interprete un boxplot par groupe (region ou saison)."""
    grouped = df.groupby(group_col)[variable].agg(['median', 'std', 'min', 'max']).round(2)
    
    # Groupe avec la mediane la plus haute/basse
    max_group = grouped['median'].idxmax()
    min_group = grouped['median'].idxmin()
    max_med = grouped.loc[max_group, 'median']
    min_med = grouped.loc[min_group, 'median']
    
    # Groupe avec la plus forte dispersion
    max_std_group = grouped['std'].idxmax()
    max_std = grouped.loc[max_std_group, 'std']
    
    texts = [
        f"**{max_group}** presente la valeur mediane la plus **elevee** ({max_med}),",
        f"tandis que **{min_group}** a la mediane la plus **basse** ({min_med}).",
        f"Ecart entre ces deux groupes : {max_med - min_med:.1f} unites.",
    ]
    
    if max_std > grouped['std'].median() * 1.5:
        texts.append(f"**{max_std_group}** montre la plus forte variabilite (std = {max_std:.1f}), indiquant des conditions tres heterogenes.")
    
    return " ".join(texts)


def interpret_scatter_temp_humidite(df):
    """Interprete le scatter temperature vs humidite."""
    corr = df['temperature_moyenne'].corr(df['humidite'])
    
    # Saison des pluies vs saison seche
    pluies = df[df['saison'] == 'saison_des_pluies']
    seche = df[df['saison'] == 'saison_sèche']
    
    texts = []
    
    if abs(corr) > 0.5:
        direction = "positive" if corr > 0 else "negative"
        texts.append(f"**Correlation {direction} forte** entre temperature et humidite (r = {corr:.2f}).")
    elif abs(corr) > 0.2:
        direction = "positive" if corr > 0 else "negative"
        texts.append(f"Correlation {direction} moderée (r = {corr:.2f}).")
    else:
        texts.append(f"**Faible correlation** entre temperature et humidite (r = {corr:.2f}) — ces variables evoluent de maniere relativement independante.")
    
    # Analyse saisonniere
    if len(pluies) > 0 and len(seche) > 0:
        temp_diff = pluies['temperature_moyenne'].mean() - seche['temperature_moyenne'].mean()
        hum_diff = pluies['humidite'].mean() - seche['humidite'].mean()
        
        texts.append(
            f"Saison des pluies : temp moy = {pluies['temperature_moyenne'].mean():.1f}°C, "
            f"humidite = {pluies['humidite'].mean():.1f}%"
        )
        texts.append(
            f"Saison seche : temp moy = {seche['temperature_moyenne'].mean():.1f}°C, "
            f"humidite = {seche['humidite'].mean():.1f}%"
        )
        
        if abs(temp_diff) > 2:
            texts.append(f"Ecart thermique saisonnier significatif : {abs(temp_diff):.1f}°C.")
    
    return " ".join(texts)


def interpret_correlation_matrix(df):
    """Interprete la matrice de correlation."""
    numeric_cols = ['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']
    corr = df[numeric_cols].corr()
    
    # Trouver les correlations les plus fortes (hors diagonale)
    corr_flat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_stacked = corr_flat.stack().reset_index()
    corr_stacked.columns = ['Var1', 'Var2', 'Correlation']
    corr_stacked['AbsCorr'] = corr_stacked['Correlation'].abs()
    corr_stacked = corr_stacked.sort_values('AbsCorr', ascending=False)
    
    texts = []
    
    # Correlation la plus forte
    if len(corr_stacked) > 0:
        top = corr_stacked.iloc[0]
        v1 = top['Var1'].replace('_', ' ').title()
        v2 = top['Var2'].replace('_', ' ').title()
        r = top['Correlation']
        direction = "positive" if r > 0 else "negative"
        force = "forte" if abs(r) > 0.6 else "moderée" if abs(r) > 0.3 else "faible"
        
        texts.append(
            f"**Lien le plus marque** : {v1} ↔ {v2} (r = {r:.2f}) — correlation {direction} {force}."
        )
    
    # Nombre de correlations significatives
    sig_count = (corr_stacked['AbsCorr'] > 0.3).sum()
    texts.append(f"{sig_count} paire(s) sur {len(corr_stacked)} montrent une correlation significative (|r| > 0.3).")
    
    # Correlation temperature-humidite specifique
    temp_hum = df['temperature_moyenne'].corr(df['humidite'])
    if abs(temp_hum) > 0.3:
        texts.append(
            f"Temperature et humidite sont liees (r = {temp_hum:.2f}), "
            f"suggerant un climat ou ces parametres coevoluent."
        )
    else:
        texts.append(
            f"Temperature et humidite sont quasiment independantes (r = {temp_hum:.2f}), "
            f"ce qui est atypique pour un climat tropical classique."
        )
    
    return " ".join(texts)


def interpret_region_comparison(df):
    """Interprete la comparaison normalisee par region."""
    grouped = df.groupby('region')[['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']].mean()
    
    # Region la plus chaude / plus humide / plus pluvieuse / plus venteuse
    hottest = grouped['temperature_moyenne'].idxmax()
    coldest = grouped['temperature_moyenne'].idxmin()
    wettest = grouped['humidite'].idxmax()
    driest = grouped['humidite'].idxmin()
    rainiest = grouped['precipitations'].idxmax()
    windiest = grouped['vitesse_vent'].idxmax()
    
    texts = [
        f"**{hottest}** est la region la plus **chaude** (temp moy = {grouped.loc[hottest, 'temperature_moyenne']:.1f}°C),",
        f"**{coldest}** la plus **froide** ({grouped.loc[coldest, 'temperature_moyenne']:.1f}°C).",
        f"**{wettest}** enregistre l'**humidite** la plus elevee ({grouped.loc[wettest, 'humidite']:.1f}%),",
        f"**{rainiest}** les **precipitations** les plus importantes ({grouped.loc[rainiest, 'precipitations']:.1f} mm).",
        f"**{windiest}** est la region la plus **venteuse** ({grouped.loc[windiest, 'vitesse_vent']:.1f} km/h).",
    ]
    
    return " ".join(texts)


def interpret_barplot_qualitative(df, colonne):
    """Interprete un barplot de variable qualitative."""
    counts = df[colonne].value_counts()
    total = len(df)
    
    max_cat = counts.idxmax()
    min_cat = counts.idxmin()
    max_pct = counts.max() / total * 100
    min_pct = counts.min() / total * 100
    
    texts = [
        f"**{max_cat}** est la categorie la plus representee ({counts.max()} obs, {max_pct:.1f}%).",
        f"**{min_cat}** est la moins representee ({counts.min()} obs, {min_pct:.1f}%).",
    ]
    
    # Test d'equilibre
    n_cats = len(counts)
    expected = total / n_cats
    deviation = max(abs(counts - expected)) / expected * 100
    
    if deviation < 10:
        texts.append(f"Repartition **equilibree** (ecart max = {deviation:.1f}% vs repartition uniforme).")
    else:
        texts.append(f"Repartition **desequilibree** (ecart max = {deviation:.1f}% vs repartition uniforme).")
    
    return " ".join(texts)


def interpret_crosstab_region_saison(df, mode="stacked"):
    """Interprete le croisement region x saison."""
    crosstab = pd.crosstab(df['region'], df['saison'])
    
    # Region avec le plus grand desequilibre saisonnier
    ratios = crosstab.div(crosstab.sum(axis=1), axis=0)
    max_ratio_diff = (ratios.max(axis=1) - ratios.min(axis=1)).idxmax()
    max_diff_val = (ratios.max(axis=1) - ratios.min(axis=1)).max()
    
    texts = [
        f"**{max_ratio_diff}** montre le plus fort desequilibre saisonnier "
        f"(ecart de proportions = {max_diff_val:.1%}).",
    ]
    
    # Region la plus equilibree
    min_diff_region = (ratios.max(axis=1) - ratios.min(axis=1)).idxmin()
    min_diff_val = (ratios.max(axis=1) - ratios.min(axis=1)).min()
    texts.append(
        f"**{min_diff_region}** est la plus equilibree saisonnierement "
        f"(ecart = {min_diff_val:.1%})."
    )
    
    return " ".join(texts)


def interpret_precipitations_vent(df):
    """Interprete le graphique precipitations vs vent."""
    corr = df['precipitations'].corr(df['vitesse_vent'])
    
    texts = []
    
    if abs(corr) > 0.5:
        texts.append(f"**Correlation significative** pluie-vent (r = {corr:.2f}) — les episodes pluvieux sont associes a des vents forts.")
    elif abs(corr) > 0.2:
        texts.append(f"Correlation pluie-vent moderée (r = {corr:.2f}).")
    else:
        texts.append(f"**Independance** entre precipitations et vent (r = {corr:.2f}) — la pluie ne depend pas significativement de la vitesse du vent.")
    
    # Zones extremes
    high_rain_low_wind = df[(df['precipitations'] > df['precipitations'].quantile(0.8)) & 
                             (df['vitesse_vent'] < df['vitesse_vent'].quantile(0.3))]
    
    if len(high_rain_low_wind) > 0:
        texts.append(f"{len(high_rain_low_wind)} episodes de **fortes pluies sans vent fort** detectes — possiblement des orages statiques.")
    
    return " ".join(texts)


def interpret_violin_by_saison(df, variable):
    """Interprete un violin plot par saison."""
    pluies = df[df['saison'] == 'saison_des_pluies'][variable]
    seche = df[df['saison'] == 'saison_sèche'][variable]
    
    if len(pluies) == 0 or len(seche) == 0:
        return "Donnees insuffisantes pour comparer les saisons."
    
    median_diff = pluies.median() - seche.median()
    mean_diff = pluies.mean() - seche.mean()
    
    texts = [
        f"Saison des pluies : mediane = {pluies.median():.1f}, moyenne = {pluies.mean():.1f}.",
        f"Saison seche : mediane = {seche.median():.1f}, moyenne = {seche.mean():.1f}.",
    ]
    
    if abs(median_diff) > (pluies.std() + seche.std()) / 4:
        texts.append(
            f"**Difference significative** entre saisons : ecart median de {abs(median_diff):.1f} unites."
        )
    else:
        texts.append(f"Difference saisonniere **faible** : ecart median de {abs(median_diff):.1f} unites.")
    
    # Forme des distributions
    pluies_iqr = pluies.quantile(0.75) - pluies.quantile(0.25)
    seche_iqr = seche.quantile(0.75) - seche.quantile(0.25)
    
    if pluies_iqr > seche_iqr * 1.3:
        texts.append(f"La saison des pluies montre plus de variabilite (IQR = {pluies_iqr:.1f} vs {seche_iqr:.1f} en saison seche).")
    elif seche_iqr > pluies_iqr * 1.3:
        texts.append(f"La saison seche est plus variable (IQR = {seche_iqr:.1f} vs {pluies_iqr:.1f} en saison des pluies).")
    
    return " ".join(texts)


def interpret_heatmap_region_saison(df):
    """Interprete la heatmap temperature par region et saison."""
    grouped = df.groupby(['region', 'saison'])['temperature_moyenne'].mean().unstack()
    
    # Region la plus chaude/froide par saison
    if 'saison_des_pluies' in grouped.columns:
        hottest_pluies = grouped['saison_des_pluies'].idxmax()
        coldest_pluies = grouped['saison_des_pluies'].idxmin()
        texts = [
            f"Saison des pluies : **{hottest_pluies}** la plus chaude ({grouped.loc[hottest_pluies, 'saison_des_pluies']:.1f}°C),",
            f"**{coldest_pluies}** la plus froide ({grouped.loc[coldest_pluies, 'saison_des_pluies']:.1f}°C).",
        ]
    
    if 'saison_sèche' in grouped.columns:
        hottest_seche = grouped['saison_sèche'].idxmax()
        coldest_seche = grouped['saison_sèche'].idxmin()
        texts += [
            f"Saison seche : **{hottest_seche}** la plus chaude ({grouped.loc[hottest_seche, 'saison_sèche']:.1f}°C),",
            f"**{coldest_seche}** la plus froide ({grouped.loc[coldest_seche, 'saison_sèche']:.1f}°C).",
        ]
    
    # Amplitude thermique par region
    if grouped.shape[1] == 2:
        amplitudes = (grouped.iloc[:, 0] - grouped.iloc[:, 1]).abs()
        max_amp_region = amplitudes.idxmax()
        texts.append(
            f"**{max_amp_region}** subit l'amplitude thermique saisonniere la plus forte ({amplitudes.max():.1f}°C d'ecart)."
        )
    
    return " ".join(texts)
