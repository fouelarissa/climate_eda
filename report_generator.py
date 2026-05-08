"""
Module de generation de rapports PDF et Excel pour le Dashboard Climatique.
Exporte les donnees, statistiques, tests et graphiques dans des formats telechargeables.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime
import plotly.io as pio


def export_excel_complet(df, stats=None, corr=None, tests=None):
    """
    Genere un fichier Excel multi-feuilles avec les donnees et analyses.
    Retourne un buffer binaire (BytesIO).
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille 1 : Donnees brutes
        df.to_excel(writer, sheet_name='Donnees brutes', index=False)
        
        # Feuille 2 : Statistiques descriptives
        desc = df.describe().T
        desc['skewness'] = df.skew(numeric_only=True)
        desc['kurtosis'] = df.kurtosis(numeric_only=True)
        desc.to_excel(writer, sheet_name='Statistiques')
        
        # Feuille 3 : Correlation
        numeric_cols = ['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']
        corr_matrix = df[numeric_cols].corr()
        corr_matrix.to_excel(writer, sheet_name='Correlations')
        
        # Feuille 4 : Agregation par region
        if 'region' in df.columns:
            agg_region = df.groupby('region')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'count'])
            agg_region.to_excel(writer, sheet_name='Par Region')
        
        # Feuille 5 : Agregation par saison
        if 'saison' in df.columns:
            agg_saison = df.groupby('saison')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'count'])
            agg_saison.to_excel(writer, sheet_name='Par Saison')
        
        # Feuille 6 : Resultats de tests (si fournis)
        if tests:
            tests_df = pd.DataFrame(tests)
            tests_df.to_excel(writer, sheet_name='Tests statistiques', index=False)
    
    output.seek(0)
    return output


def fig_to_image(fig, width=800, height=500, scale=2):
    """
    Convertit une figure Plotly en image PNG (bytes).
    Necessite kaleido.
    """
    try:
        img_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=scale)
        return img_bytes
    except Exception as e:
        print(f"Erreur conversion image Plotly : {e}")
        return None


def generate_pdf_report(df, figures_dict=None):
    """
    Genere un rapport PDF avec matplotlib PdfPages (support UTF-8 natif).
    figures_dict : dict {titre: fig_plotly}
    Retourne un BytesIO.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    
    output = io.BytesIO()
    numeric_cols = ['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']
    n_rows, n_cols = df.shape
    
    with PdfPages(output) as pdf:
        # === PAGE 1 : TITRE ET RESUME ===
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.5, 0.92, 'Dashboard Climatique - Rapport d\'Analyse', 
                 ha='center', fontsize=18, fontweight='bold', color='#1f77b4')
        fig.text(0.5, 0.88, f'Genere le {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                 ha='center', fontsize=10, color='gray')
        
        y_pos = 0.80
        fig.text(0.1, y_pos, 'Resume du Dataset', fontsize=14, fontweight='bold', color='#ff7f0e')
        y_pos -= 0.05
        
        resume_lines = [
            f'Observations : {n_rows}',
            f'Variables : {n_cols}',
            f'Numeriques : {", ".join(numeric_cols)}',
            f'Qualitatives : region, saison',
            '',
            'Ce rapport presente une analyse complete des donnees climatiques.',
        ]
        for line in resume_lines:
            fig.text(0.1, y_pos, line, fontsize=10)
            y_pos -= 0.03
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # === PAGE 2 : STATISTIQUES DESCRIPTIVES ===
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Statistiques Descriptives', ha='center', fontsize=14, 
                fontweight='bold', color='#ff7f0e', transform=ax.transAxes)
        
        desc = df[numeric_cols].describe().T.round(2)
        table_data = [[var, int(row['count']), row['mean'], row['std'], 
                       row['min'], row['25%'], row['50%'], row['75%'], row['max']]
                      for var, row in desc.iterrows()]
        headers = ['Variable', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        
        table = ax.table(cellText=[headers] + table_data,
                        loc='center', cellLoc='center',
                        colWidths=[0.15, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Colorer l'en-tete
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#1f77b4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # === PAGE 3 : CORRELATIONS ===
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Matrice de Correlation (Pearson)', ha='center', fontsize=14,
                fontweight='bold', color='#ff7f0e', transform=ax.transAxes)
        
        corr = df[numeric_cols].corr().round(3)
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, corr.values[i, j], ha='center', va='center', 
                       color='white' if abs(corr.values[i, j]) > 0.5 else 'black',
                       fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.6)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # === PAGE 4 : PAR REGION ===
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Moyennes par Region', ha='center', fontsize=14,
                fontweight='bold', color='#ff7f0e', transform=ax.transAxes)
        
        agg_region = df.groupby('region')[numeric_cols].mean().round(2)
        headers = ['Region'] + numeric_cols
        rows = [[r] + [agg_region.loc[r, c] for c in numeric_cols] for r in agg_region.index]
        
        table = ax.table(cellText=[headers] + rows, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#1f77b4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # === PAGE 5 : PAR SAISON ===
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Moyennes par Saison', ha='center', fontsize=14,
                fontweight='bold', color='#ff7f0e', transform=ax.transAxes)
        
        agg_saison = df.groupby('saison')[numeric_cols].mean().round(2)
        headers = ['Saison'] + numeric_cols
        rows = [[s] + [agg_saison.loc[s, c] for c in numeric_cols] for s in agg_saison.index]
        
        table = ax.table(cellText=[headers] + rows, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#1f77b4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # === PAGES GRAPHIQUES ===
        if figures_dict:
            for title, fig_plotly in figures_dict.items():
                img_bytes = fig_to_image(fig_plotly)
                if img_bytes:
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis('off')
                    ax.text(0.5, 0.97, title, ha='center', fontsize=14,
                            fontweight='bold', color='#ff7f0e', transform=ax.transAxes)
                    ax.imshow(img, aspect='auto')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        # === PAGE FINALE ===
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.92, 'Conclusion', ha='center', fontsize=14, fontweight='bold', color='#ff7f0e')
        
        y_pos = 0.85
        conclusion_lines = [
            'Ce rapport a ete genere automatiquement par le Dashboard Climatique.',
            '',
            'Les analyses presentees couvrent :',
            '- Statistiques descriptives',
            '- Correlations entre variables',
            '- Comparaisons par region et saison',
            '- Visualisations graphiques',
            '',
            'Pour des analyses avancees, consultez l\'interface interactive.'
        ]
        for line in conclusion_lines:
            fig.text(0.1, y_pos, line, fontsize=10)
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    output.seek(0)
    return output


def generate_html_report(df):
    """
    Genere un rapport HTML complet telechargeable.
    Retourne une string HTML.
    """
    n_rows, n_cols = df.shape
    numeric_cols = ['temperature_moyenne', 'humidite', 'precipitations', 'vitesse_vent']
    
    desc = df[numeric_cols].describe().T.round(2).to_html(classes='table', border=0)
    corr = df[numeric_cols].corr().round(3).to_html(classes='table', border=0)
    agg_region = df.groupby('region')[numeric_cols].mean().round(2).to_html(classes='table', border=0)
    agg_saison = df.groupby('saison')[numeric_cols].mean().round(2).to_html(classes='table', border=0)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport Climatique</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
            h2 {{ color: #ff7f0e; margin-top: 30px; }}
            .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            .table th {{ background: #1f77b4; color: white; padding: 10px; text-align: left; }}
            .table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            .table tr:nth-child(even) {{ background: #f8f9fa; }}
            .footer {{ margin-top: 40px; font-size: 0.9em; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>Dashboard Climatique - Rapport d'Analyse</h1>
        <p><strong>Genere le :</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        
        <div class="summary">
            <h2>Resume du Dataset</h2>
            <ul>
                <li><strong>Observations :</strong> {n_rows}</li>
                <li><strong>Variables :</strong> {n_cols}</li>
                <li><strong>Numeriques :</strong> {', '.join(numeric_cols)}</li>
                <li><strong>Qualitatives :</strong> region, saison</li>
            </ul>
        </div>
        
        <h2>1. Statistiques Descriptives</h2>
        {desc}
        
        <h2>2. Matrice de Correlation</h2>
        {corr}
        
        <h2>3. Moyennes par Region</h2>
        {agg_region}
        
        <h2>4. Moyennes par Saison</h2>
        {agg_saison}
        
        <div class="footer">
            <p>Rapport genere automatiquement par le Dashboard Climatique Streamlit</p>
        </div>
    </body>
    </html>
    """
    return html
