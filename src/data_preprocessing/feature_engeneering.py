import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Engeneering:
    def __init__(self, chemin: str = None, df_input: pd.DataFrame = None, delimiteur: str = ','):
        """
        Initialise la classe Engeneering.

        Args:
            chemin (str): Chemin vers le fichier CSV.
            df_input (pd.DataFrame, optional): DataFrame à utiliser directement.
            delimiteur (str, optional): Délimiteur utilisé dans le fichier CSV. Par défaut ','.
        """
        self.df = None  # Initialisation par défaut
        print("n")
        if chemin is not None and not isinstance(df_input, pd.DataFrame):
            try:
                logging.info("Initialisation de la classe Engeneering avec un fichier CSV.")
                if not isinstance(chemin, str) or not chemin.endswith('.csv'):
                    raise ValueError("Le chemin doit être une chaîne de caractères pointant vers un fichier CSV.")
                self.df = pd.read_csv(chemin, delimiter=delimiteur)
                logging.info(f"Dataset chargé avec succès depuis {chemin}. Dimensions: {self.df.shape}")
            except FileNotFoundError:
                logging.error(f"Le fichier spécifié n'a pas été trouvé: {chemin}")
                raise
            except pd.errors.EmptyDataError:
                logging.error(f"Le fichier est vide ou corrompu: {chemin}")
                raise
            except Exception as e:
                logging.error(f"Erreur lors du chargement du fichier: {e}")
                raise
        elif chemin is None and isinstance(df_input, pd.DataFrame):
            try:
                logging.info("Initialisation de la classe Engeneering avec un DataFrame.")
                if df_input.empty:
                    raise ValueError("Le DataFrame fourni est vide.")
                self.df = df_input
                logging.info(f"Dataset chargé avec succès. Dimensions: {self.df.shape}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement du DataFrame: {e}")
                raise
        else:
            raise ValueError("Les paramètres fournis sont invalides. Fournissez soit un chemin vers un fichier CSV, soit un DataFrame valide.")



    def extract_time_features(self, time_column: str = 'heure') -> None:
        """
        Extrait des caractéristiques temporelles à partir de la colonne horaire.

        Args:
            time_column (str): Nom de la colonne contenant l'heure.
        """
        logging.info("Extraction des caractéristiques temporelles.")
        if time_column not in self.df.columns:
            raise ValueError(f"La colonne '{time_column}' n'existe pas dans le DataFrame.")
        
        self.df['heure'] = pd.to_numeric(self.df[time_column], errors='coerce')
        self.df['periode_jour'] = pd.cut(
            self.df['heure'],
            bins=[0, 6, 12, 18, 24],
            labels=['nuit', 'matin', 'après-midi', 'soir'],
            right=False
        )
        logging.info("Caractéristiques temporelles extraites avec succès.")

    def create_speed_features(self) -> None:
        """
        Crée des nouvelles variables à partir des colonnes de vitesse.
        """
        logging.info("Création des nouvelles variables de vitesse.")
        if not {'AvgSp', 'MedSp', 'P5sp', 'P95sp', 'SdSp', 'HvgSp'}.issubset(self.df.columns):
            raise ValueError("Certaines colonnes nécessaires pour les calculs de vitesse sont manquantes.")
        
        self.df['ecart_moyenne_mediane'] = self.df['AvgSp'] - self.df['MedSp']
        self.df['amplitude_vitesse'] = self.df['P95sp'] - self.df['P5sp']
        self.df['rapport_moyenne_harmonique'] = self.df['AvgSp'] / self.df['HvgSp']
        self.df['ecart_type_normalise'] = self.df['SdSp'] / self.df['AvgSp']
        logging.info("Variables de vitesse créées avec succès.")

    def create_ratios(self) -> None:
        """
        Crée des ratios spécifiques pour capturer des relations entre les variables.
        """
        logging.info("Création des ratios spécifiques.")
        if not {'AvgSp', 'MedSp', 'Hits', 'AvgTt'}.issubset(self.df.columns):
            raise ValueError("Certaines colonnes nécessaires pour les calculs de ratios sont manquantes.")
        
        self.df['ratio_vitesse_moyenne_mediane'] = self.df['AvgSp'] / self.df['MedSp']
        self.df['ratio_hits_avgtt'] = self.df['Hits'] / self.df['AvgTt']
        logging.info("Ratios créés avec succès.")

    def get_df(self):
        return self.df
    
    def save_to_csv(self, output_path: str) -> None:
        """
        Sauvegarde le DataFrame transformé dans un fichier CSV.

        Args:
            output_path (str): Chemin du fichier de sortie.
        """
        logging.info(f"Sauvegarde du DataFrame transformé dans {output_path}.")
        self.df.to_csv(output_path, index=False)
        logging.info("Fichier sauvegardé avec succès.")