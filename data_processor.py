import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "dataset_climat.csv"


def load_dataset():
    """
    Charge le dataset climatique depuis le fichier CSV.
    
    Returns:
        pd.DataFrame: DataFrame contenant les donnees climatiques
        
    Raises:
        FileNotFoundError: Si le fichier dataset_climat.csv n'est pas trouve
        pd.errors.EmptyDataError: Si le fichier est vide
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset charge avec succes : {len(df)} enregistrements")
        return df
    except FileNotFoundError:
        logger.error(f"Fichier non trouve : {DATASET_PATH}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        raise


def validate_data(df):
    """
    Valide la qualite et l'integrite des donnees.
    
    Cette fonction verifie :
    - La presence de toutes les colonnes attendues
    - L'absence de valeurs manquantes
    - Les types de donnees corrects
    - La presence de valeurs aberrantes potentielles
    
    Args:
        df (pd.DataFrame): DataFrame a valider
        
    Returns:
        dict: Dictionnaire contenant les resultats de validation
    """
    expected_columns = [
        'humidite', 'precipitations', 'vitesse_vent', 
        'region', 'saison', 'temperature_moyenne'
    ]
    
    validation_results = {
        'colonnes_attendues': True,
        'valeurs_manquantes': False,
        'types_corrects': True,
        'alertes': []
    }
    
    # Verification des colonnes
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        validation_results['colonnes_attendues'] = False
        validation_results['alertes'].append(f"Colonnes manquantes : {missing_cols}")
    
    # Verification des valeurs manquantes
    if df.isnull().sum().sum() > 0:
        validation_results['valeurs_manquantes'] = True
        validation_results['alertes'].append("Valeurs manquantes detectees")
    
    # Verification des types numeriques pour les colonnes quantitatives
    numeric_cols = ['humidite', 'precipitations', 'vitesse_vent', 'temperature_moyenne']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation_results['types_corrects'] = False
            validation_results['alertes'].append(f"Colonne {col} n'est pas numerique")
    
    # Detection des valeurs aberrantes (methode IQR)
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            if len(outliers) > 0:
                validation_results['alertes'].append(
                    f"{len(outliers)} valeurs aberrantes detectees dans {col}"
                )
    
    return validation_results


def get_basic_stats(df):
    """
    Calcule les statistiques descriptives de base.
    
    Args:
        df (pd.DataFrame): DataFrame des donnees climatiques
        
    Returns:
        pd.DataFrame: Statistiques descriptives (count, mean, std, min, max, quartiles)
    """
    return df.describe()


def get_data_info(df):
    """
    Retourne les informations generales sur le dataset.
    
    Args:
        df (pd.DataFrame): DataFrame des donnees climatiques
        
    Returns:
        dict: Informations sur le dataset (nombre de lignes, colonnes, types, etc.)
    """
    return {
        'nombre_lignes': len(df),
        'nombre_colonnes': len(df.columns),
        'colonnes': list(df.columns),
        'types_donnees': df.dtypes.to_dict(),
        'regions_uniques': df['region'].unique().tolist() if 'region' in df.columns else [],
        'saisons_uniques': df['saison'].unique().tolist() if 'saison' in df.columns else []
    }


def filter_by_region(df, region):
    """
    Filtre les donnees par region.
    
    Args:
        df (pd.DataFrame): DataFrame source
        region (str): Nom de la region a filtrer
        
    Returns:
        pd.DataFrame: Donnees filtrees par region
    """
    return df[df['region'] == region].copy()


def filter_by_saison(df, saison):
    """
    Filtre les donnees par saison.
    
    Args:
        df (pd.DataFrame): DataFrame source
        saison (str): Nom de la saison a filtrer
        
    Returns:
        pd.DataFrame: Donnees filtrees par saison
    """
    return df[df['saison'] == saison].copy()


def group_by_region_saison(df):
    """
    Agrege les donnees par region et saison avec les moyennes.
    
    Args:
        df (pd.DataFrame): DataFrame source
        
    Returns:
        pd.DataFrame: Moyennes des variables par region et saison
    """
    return df.groupby(['region', 'saison']).agg({
        'temperature_moyenne': 'mean',
        'humidite': 'mean',
        'precipitations': 'mean',
        'vitesse_vent': 'mean'
    }).round(2).reset_index()
