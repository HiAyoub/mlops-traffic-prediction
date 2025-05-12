import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  StandardScaler, LabelEncoder
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessing:
    def __init__(self, chemin: str, delimiteur: str = None):
        """
        Initialise la classe Preprocessing.

        Args:
            chemin (str): Chemin vers le fichier CSV.
            delimiteur (str, optional): Délimiteur utilisé dans le fichier CSV. Par défaut None.
        """
        try:
            logging.info("Initialisation de la classe Preprocessing.")
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
        self.X = None
        self.encoders = {}
        self.scaler = StandardScaler()


    

    def show_stats(self) -> None:
        """
        Affiche un résumé des statistiques du DataFrame :
        - Dimensions du dataset
        - Types des colonnes
        - Nombre de valeurs manquantes par colonne (en pourcentage)
        - Statistiques descriptives pour les colonnes numériques
        - Fréquences des valeurs uniques pour les colonnes catégorielles
        - Nombre de valeurs uniques par colonne
        """
        logging.info("Affichage des statistiques du DataFrame.")
        print(f"Dimensions du dataset: {self.df.shape[0]} lignes et {self.df.shape[1]} colonnes")
        logging.info(f"Dimensions du dataset: {self.df.shape[0]} lignes et {self.df.shape[1]} colonnes")
        
        # Types de colonnes
        print("\nTypes de données par colonne :")
        print(self.df.dtypes)
        logging.info("Types de données par colonne affichés.")
        
        # Colonnes numériques, catégorielles, et chaînes
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"\nColonnes numériques : {len(numeric_cols)}")
        print(f"Colonnes catégorielles : {len(categorical_cols)}")
        
        # Valeurs manquantes
        print("\nValeurs manquantes par colonne (%):")
        missing_values = self.df.isnull().sum() / len(self.df) * 100

        
        print(missing_values.sort_values(ascending=False))
        logging.info("Valeurs manquantes par colonne affichées.")
        
        # Statistiques descriptives pour les colonnes numériques
        print("\nStatistiques descriptives pour les colonnes numériques :")
        print(self.df[numeric_cols].describe())
        
        # Statistiques pour les colonnes catégorielles
        print("\nFréquences des valeurs uniques pour les colonnes catégorielles :")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
        
        # Valeurs uniques
        print("\nNombre de valeurs uniques par colonne :")
        print(self.df.nunique())

    def drop_columns_with_all_missing(self) -> None:
        """
        Supprime les colonnes ayant 100% de valeurs manquantes.
        """
        logging.info("Suppression des colonnes avec 100% de valeurs manquantes.")
        self.df = self.df.dropna(axis=1, how='all')
        print("Colonnes avec 100% de valeurs manquantes supprimées.")
        logging.info("Colonnes supprimées avec succès.")

    def drop_colonne(self, nom_col: str = '') -> None:
        """
        Supprime une ou plusieurs colonnes spécifiées du DataFrame.

        Args:
            nom_col (str): Nom(s) de la ou des colonnes à supprimer, séparés par des virgules si plusieurs.

        Raises:
            ValueError: Si les colonnes spécifiées n'existent pas ou si l'entrée est invalide.
        """
        try:
            logging.info(f"Tentative de suppression des colonnes: {nom_col}")
            
            # Vérification de l'entrée
            if not isinstance(nom_col, str) or not nom_col.strip():
                raise ValueError("Le paramètre 'nom_col' doit être une chaîne de caractères non vide.")
            
            # Séparer les noms de colonnes par des virgules
            columns_to_drop = [col.strip() for col in nom_col.split(',') if col.strip()]
            
            if not columns_to_drop:
                raise ValueError("Aucune colonne valide spécifiée pour suppression.")
            
            # Vérifier si les colonnes existent dans le DataFrame
            missing_columns = [col for col in columns_to_drop if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Les colonnes suivantes n'existent pas dans le DataFrame: {missing_columns}")
            
            # Suppression des colonnes
            self.df.drop(columns=columns_to_drop, inplace=True)
            logging.info(f"Colonnes supprimées avec succès: {columns_to_drop}")
            print(f"Colonnes supprimées: {columns_to_drop}")
        
        except ValueError as ve:
            logging.error(f"Erreur de validation: {ve}")
            raise
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la suppression des colonnes: {e}")
            raise

    def find_outliers(self, dfX: pd.DataFrame = None) -> tuple[dict, dict]:
        """
        Identifie les outliers dans les colonnes numériques en utilisant la méthode IQR.

        Args:
            dfX (pd.DataFrame, optional): DataFrame à analyser. Si None, utilise self.df.

        Returns:
            tuple[dict, dict]: 
                - Dictionnaire contenant les indices des outliers pour chaque colonne.
                - Dictionnaire contenant les pourcentages d'outliers pour chaque colonne.
        """
        logging.info("Recherche des outliers dans les colonnes numériques.")
        if dfX is not None and not dfX.empty :
            df = dfX

        else :
            df = self.df

        outliers_dict = {}
        outlier_percentages = {}

        # Parcourir chaque colonne numérique
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            
            # Méthode IQR pour les distributions non normales
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # Sauvegarder les indices des outliers pour cette colonne
            outliers_dict[col] = outliers.index.tolist()
            
            # Calcul du pourcentage des outliers
            outlier_percentage = len(outliers) / len(df[col]) * 100
            outlier_percentages[col] = outlier_percentage

        logging.info(f"Outliers détectés dans les colonnes: {list(outliers_dict.keys())}")
        return outliers_dict, outlier_percentages
    
    def display_outliers_percentage(self, colonne: str = None) -> None:
        """
        Affiche les pourcentages d'outliers pour une colonne spécifique ou pour toutes les colonnes.

        Args:
            colonne (str, optional): Nom de la colonne à analyser. Si None, analyse toutes les colonnes.
        """
        logging.info("Affichage des pourcentages d'outliers.")
        outliers_dict, outlier_percentages = self.find_outliers()

        # Si colonne est spécifiée, affiche uniquement pour cette colonne
        if colonne:
            if colonne in outlier_percentages:
                print(f"{colonne}: {outlier_percentages[colonne]:.2f}%")
                logging.info(f"Affichage des outliers pour la colonne: {colonne}")
                # Boxplot pour les distributions non normales
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=self.df[colonne])
                plt.title(f"Boxplot de {colonne}")
                plt.show()
            else:
                print(f"Colonne {colonne} non trouvée.")
        
        # Si colonne est None, affiche les pourcentages pour toutes les colonnes
        else:
            print("Pourcentage des outliers par colonne :")
            logging.info("Affichage des outliers pour toutes les colonnes.")
            for col, percentage in outlier_percentages.items():
                print(f"{col}: {percentage:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=self.df[col])
                plt.title(f"Boxplot de {col}")
                plt.show()

    def split_X_Y(self, save: bool = False) -> list[pd.DataFrame]:
        """
        Divise les données en variables X (indépendantes) et Y (dépendantes).

        Args:
            save (bool, optional): Si True, sauvegarde les DataFrames dans des fichiers CSV. Par défaut False.

        Returns:
            list[pd.DataFrame]: Liste de DataFrames contenant X et une seule colonne Y.
        """
        try:
            logging.info("Division des données en X et Y.")
            if self.df is None or self.df.empty:
                raise ValueError("Le DataFrame est vide ou non initialisé.")
            # Définir les colonnes X (de 'periode' à 'debit_24', inclut 'année')
            X_columns = ['Id','AvgTt','MedTt','ratio','AvgSp','HvgSp','MedSp','SdSp','Hits',
                         'P5sp','P10sp','P15sp','P20sp','P25sp','P30sp','P35sp','P40sp','P45sp',
                         'P50sp','P55sp','P60sp','P65sp','P70sp','P75sp','P80sp','P85sp','P90sp',
                         'P95sp','heure','capteur','zone']

            self.X = self.df[X_columns]  # Sélectionner X à partir des colonnes définies

            # Liste pour stocker les DataFrames avec X et un seul Y
            dfs_with_y = []

            # Parcourir toutes les colonnes restantes comme Y
            Y_columns = [col for col in self.df.columns if col not in X_columns]

            # Pour chaque colonne Y, créer un DataFrame avec X et une seule colonne Y
            for y_col in Y_columns:
                temp_df = self.X.copy()
                temp_df[y_col] = self.df[y_col]  # Ajouter la colonne Y à X
                dfs_with_y.append(temp_df)  # Ajouter le DataFrame à la liste

                # Sauvegarder le DataFrame dans un fichier si save_path est spécifié
                if save:
                    try:
                        file_path = f"output/{y_col}.csv"
                        temp_df.to_csv(file_path, index=False)
                        logging.info(f"Le DataFrame pour {y_col} a été sauvegardé sous {file_path}")
                    except Exception as e:
                        logging.error(f"Erreur lors de la sauvegarde du fichier {file_path}: {e}")
                        raise

            if save:
                logging.info(f"Les DataFrames avec X et Y ont été sauvegardés dans le dossier 'output'.")
            return dfs_with_y
        except Exception as e:
            logging.error(f"Erreur lors de la division des données en X et Y: {e}")
            raise

    def process_X(self, method: str = 'median') -> None:
        """
        Remplace les outliers détectés dans X par une valeur spécifiée.

        Args:
            method (str, optional): Méthode de remplacement ('median', 'mean', 'zero', 'interpolate'). Par défaut 'median'.
        """
        try:
            logging.info(f"Traitement des outliers dans X en utilisant la méthode: {method}.")
            if self.X is None or self.X.empty:
                raise ValueError("X n'a pas été initialisé. Exécutez split_X_Y() d'abord.")
            if method not in ['median', 'mean', 'zero', 'interpolate']:
                raise ValueError("La méthode spécifiée est invalide. Choisissez parmi 'median', 'mean', 'zero', 'interpolate'.")
            else:
                df= self.X
            outliers_dict, _ = self.find_outliers(df)

            for col, outlier_indices in outliers_dict.items():
                if method == "median":
                    replacement_value = df[col].median()
                elif method == "mean":
                    replacement_value = df[col].mean()
                elif method == "zero":
                    replacement_value = 0
                elif method == "interpolate":
                    df[col] = df[col].interpolate()  
                    continue

                df.loc[outlier_indices, col] = replacement_value
            logging.info("Traitement des outliers terminé.")
        except Exception as e:
            logging.error(f"Erreur lors du traitement des outliers: {e}")
            raise

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Sépare les données en ensembles d'entraînement et de test.

        Args:
            df (pd.DataFrame): DataFrame contenant les données.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                - X_train: Variables indépendantes pour l'entraînement.
                - X_test: Variables indépendantes pour le test.
                - y_train: Variable dépendante pour l'entraînement.
                - y_test: Variable dépendante pour le test.
        """
        logging.info("Division des données en ensembles d'entraînement et de test.")
        # Sélectionner X (toutes les colonnes sauf la cible)
        X = self.X
        # Sélectionner y (la colonne restante)
        y = df.drop(columns=X.columns)  # 'y' est la colonne restante

        # Créer un masque pour les valeurs manquantes de y
        mask_missing_y = y.isnull()

        # Séparer les données où y n'est pas manquant (train)
        X_train = X[~mask_missing_y.values.flatten()]  # Données d'entraînement où y n'est pas manquant
        y_train = y[~mask_missing_y.values.flatten()]  # Valeurs complètes de y pour l'entraînement

        # Séparer les données où y est manquant (test)
        X_test = X[mask_missing_y.values.flatten()]    # Données où y est manquant pour le test
        y_test = y[mask_missing_y.values.flatten()]    # y_test, qui contient uniquement les NaN

        # Vérification
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logging.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        # Vérifie s'il y a des NaN dans X_train ou y_train
        nan_in_X_train = X_train.isnull().any().any()  # Vérifie si X_train contient des NaN
        nan_in_y_train = y_train.isnull().any()  # Vérifie si y_train contient des NaN

        print("X_train contient des NaN:", nan_in_X_train)
        print("y_train contient des NaN:", nan_in_y_train)

        # Vérifie s'il y a des valeurs non NaN dans y_test
        not_nan_in_y_test = y_test.notnull().any()  # Vérifie si y_test ne contient pas de NaN

        print("y_test contient des valeurs non NaN:", not_nan_in_y_test)

        return X_train, X_test, y_train, y_test

    def identify_columns(self, X: pd.DataFrame) -> tuple[list, list]:
        """
        Identifie les colonnes catégorielles et numériques.

        Args:
            X (pd.DataFrame): DataFrame à analyser.

        Returns:
            tuple[list, list]: 
                - Liste des colonnes catégorielles.
                - Liste des colonnes numériques.
        """
        logging.info("Identification des colonnes catégorielles et numériques.")
        cat_columns = X.select_dtypes(include=['object']).columns
        num_columns = X.select_dtypes(exclude=['object']).columns

        logging.info(f"Colonnes catégorielles: {list(cat_columns)}, Colonnes numériques: {list(num_columns)}")
        return cat_columns, num_columns

    def encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame, cat_columns: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode les colonnes catégorielles en utilisant LabelEncoder.

        Args:
            X_train (pd.DataFrame): Données d'entraînement.
            X_test (pd.DataFrame): Données de test.
            cat_columns (list): Liste des colonnes catégorielles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - X_train encodé.
                - X_test encodé.
        """
        try:
            logging.info("Encodage des colonnes catégorielles.")
            if not isinstance(cat_columns, list) or not all(isinstance(col, str) for col in cat_columns):
                raise ValueError("cat_columns doit être une liste de chaînes de caractères.")
            for col in cat_columns:
                encoder = LabelEncoder()
                X_train = X_train.copy()  # Évite SettingWithCopyWarning
                X_test = X_test.copy()

                X_train.loc[:, col] = encoder.fit_transform(X_train[col].astype(str))
                X_test.loc[:, col] = encoder.transform(X_test[col].astype(str))

                self.encoders[col] = encoder  # Stocke l'encodeur
            logging.info("Encodage des colonnes catégorielles terminé.")
            return X_train, X_test
        except Exception as e:
            logging.error(f"Erreur lors de l'encodage des colonnes catégorielles: {e}")
            raise

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, num_columns: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalise les colonnes numériques en utilisant StandardScaler.

        Args:
            X_train (pd.DataFrame): Données d'entraînement.
            X_test (pd.DataFrame): Données de test.
            num_columns (list): Liste des colonnes numériques.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - X_train normalisé.
                - X_test normalisé.
        """
        try:
            logging.info("Normalisation des colonnes numériques.")
            if not isinstance(num_columns, list) or not all(isinstance(col, str) for col in num_columns):
                raise ValueError("num_columns doit être une liste de chaînes de caractères.")
            self.scaler = StandardScaler()
            X_train = X_train.copy()
            X_test = X_test.copy()

            X_train.loc[:, num_columns] = self.scaler.fit_transform(X_train[num_columns])
            X_test.loc[:, num_columns] = self.scaler.transform(X_test[num_columns])

            logging.info("Normalisation des colonnes numériques terminée.")
            return X_train, X_test
        except Exception as e:
            logging.error(f"Erreur lors de la normalisation des colonnes numériques: {e}")
            raise

    def process_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Traite les colonnes catégorielles et numériques (encodage et normalisation).

        Args:
            X_train (pd.DataFrame): Données d'entraînement.
            X_test (pd.DataFrame): Données de test.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - X_train traité.
                - X_test traité.
        """
        logging.info("Traitement des colonnes catégorielles et numériques.")
        cat_columns = X_train.select_dtypes(include=['object']).columns.tolist()
        num_columns = X_train.select_dtypes(include(['int64', 'float64'])).columns.tolist()

        X_train, X_test = self.encode_categorical(X_train, X_test, cat_columns)
        X_train, X_test = self.scale_features(X_train, X_test, num_columns)

        logging.info("Traitement des colonnes terminé.")
        return X_train, X_test
    
    def inverse_transform(self, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inverse les transformations appliquées sur les données.

        Args:
            X_train_scaled (pd.DataFrame): Données d'entraînement transformées.
            X_test_scaled (pd.DataFrame): Données de test transformées.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - X_train avec les valeurs d'origine.
                - X_test avec les valeurs d'origine.
        """
        try:
            logging.info("Inversion des transformations pour retrouver les valeurs d'origine.")
            if X_train_scaled.empty or X_test_scaled.empty:
                raise ValueError("Les DataFrames fournis sont vides.")
            X_train = X_train_scaled.copy()
            X_test = X_test_scaled.copy()

            # Inverse l'encodage des colonnes catégorielles et convertit en `str`
            for col, encoder in self.encoders.items():
                X_train.loc[:, col] = encoder.inverse_transform(X_train[col].astype(int))
                X_test.loc[:, col] = encoder.inverse_transform(X_test[col].astype(int))

            # Inverse la normalisation et convertit les entiers en `int`
            num_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
            X_train.loc[:, num_columns] = self.scaler.inverse_transform(X_train[num_columns])
            X_test.loc[:, num_columns]  = self.scaler.inverse_transform(X_test[num_columns])

            # Convertir les colonnes entières en int
            for col in num_columns:
                if X_train[col].dtype == 'float64' :
                    X_train.loc[:, col] = X_train[col].astype(int)
                    X_test.loc[:, col] = X_test[col].astype(int)

            logging.info("Inversion des transformations terminée.")
            return X_train, X_test
        except Exception as e:
            logging.error(f"Erreur lors de l'inversion des transformations: {e}")
            raise

    def save_to_csv(self, output_path: str,df:pd.DataFrame = None) -> None:
        """
        Sauvegarde le DataFrame transformé dans un fichier CSV.

        Args:
            output_path (str): Chemin du fichier de sortie.
        """
        dataf = df if df else self.df
        logging.info(f"Sauvegarde du DataFrame transformé dans {output_path}.")
        dataf.to_csv(output_path, index=False)
        logging.info("Fichier sauvegardé avec succès.")