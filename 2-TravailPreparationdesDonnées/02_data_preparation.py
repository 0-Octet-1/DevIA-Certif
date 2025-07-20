#!/usr/bin/env python3
# preparation des donnees acceslibre pour machine learning
# suit la logique de l'exploration streamlit validee

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

# fonctions reprises de l'exploration streamlit pour coherence
def load_dataset():
    """charge le dataset acceslibre avec gestion d'erreurs"""
    file_path = os.path.join('..', 'SourceData', 'acceslibre-with-web-url.csv')
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        print(f"dataset charge : {len(df):,} lignes, {len(df.columns)} colonnes")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        print(f"dataset charge avec encodage latin-1 : {len(df):,} lignes")
        return df
    except Exception as e:
        print(f"erreur chargement dataset : {e}")
        return None

def analyze_completeness(df):
    """detecte et analyse les variables d'accessibilite (repris de l'exploration)"""
    # mots-cles pour detection automatique des variables d'accessibilite
    keywords_pmr = ['pmr', 'handicap', 'accessib']
    keywords_access = ['rampe', 'ascenseur', 'marche', 'largeur', 'toilette']
    
    # detection des colonnes d'accessibilite
    colonnes_accessibilite = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords_pmr + keywords_access):
            colonnes_accessibilite.append(col)
    
    # calcul de completude pour chaque variable detectee
    completude_data = []
    
    for col in colonnes_accessibilite:
        total = len(df)
        manquantes = df[col].isnull().sum()
        renseignees = total - manquantes
        
        completude_data.append({
            'variable': col,
            'valeurs_renseignees': renseignees,
            'manquantes': manquantes,
            'completude_pct': (renseignees / total) * 100,
            'manquantes_pct': (manquantes / total) * 100
        })
    
    completude_df = pd.DataFrame(completude_data)
    completude_df = completude_df.sort_values('completude_pct', ascending=False)
    
    return completude_df

def categorize_columns(df):
    """organise les colonnes par domaine metier (repris de l'exploration)"""
    categories = {
        'identification': ['nom', 'name', 'siret', 'uuid', 'id'],
        'localisation': ['adresse', 'commune', 'code_postal', 'latitude', 'longitude', 'coordonnees'],
        'contact': ['telephone', 'email', 'site_web', 'web_url'],
        'accessibilite': [],  # sera rempli automatiquement
        'services': ['activite', 'horaires', 'equipement'],
        'technique': ['date_creation', 'date_maj', 'source']
    }
    
    # detection automatique des variables d'accessibilite
    keywords_pmr = ['pmr', 'handicap', 'accessib', 'rampe', 'ascenseur', 'marche', 'largeur', 'toilette']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords_pmr):
            categories['accessibilite'].append(col)
    
    # organisation des colonnes par categorie
    category_stats = {}
    
    for category, keywords in categories.items():
        if category == 'accessibilite':
            # deja rempli par detection automatique
            columns = categories['accessibilite']
        else:
            # recherche par mots-cles
            columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    columns.append(col)
        
        # calcul de completude pour les principales colonnes
        completeness = []
        for col in columns[:5]:  # top 5 par categorie
            if col in df.columns:
                count = df[col].notna().sum()
                pct = (count / len(df)) * 100
                completeness.append((col, count, pct))
        
        category_stats[category] = {
            'count': len(columns),
            'columns': columns,
            'completeness': completeness
        }
    
    return category_stats

def get_ml_variables(df):
    """identifie les variables pertinentes pour machine learning (repris de l'exploration)"""
    numeric_vars = []
    categorical_vars = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # variables numeriques avec assez de donnees
            count = df[col].notna().sum()
            if count > 1000:  # seuil minimum
                numeric_vars.append({
                    'column': col,
                    'count': count,
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                })
        
        elif df[col].dtype == 'object':
            # variables categorielles avec nombre raisonnable de categories
            count = df[col].notna().sum()
            unique = df[col].nunique()
            
            if count > 1000 and 2 <= unique <= 50:
                top_values = df[col].value_counts().head(3).to_dict()
                categorical_vars.append({
                    'column': col,
                    'count': count,
                    'unique': unique,
                    'top_values': top_values
                })
    
    # tri par nombre de valeurs renseignees
    numeric_vars = sorted(numeric_vars, key=lambda x: x['count'], reverse=True)
    categorical_vars = sorted(categorical_vars, key=lambda x: x['count'], reverse=True)
    
    return numeric_vars, categorical_vars

def clean_data_types(df):
    """Nettoie les types de données pour éviter les erreurs de types mixtes"""
    print("\nNettoyage des types de données...")
    
    df_clean = df.copy()
    
    # Identifier les colonnes avec des types mixtes potentiels
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Convertir tout en string pour éviter les types mixtes
            df_clean[col] = df_clean[col].astype(str)
            # Remplacer les valeurs 'nan' par des NaN réels
            df_clean[col] = df_clean[col].replace('nan', np.nan)
    
    print(f"Types de données nettoyés pour {len(df_clean.columns)} colonnes")
    return df_clean

def prepare_data_for_ml():
    """prepare les donnees en suivant la logique de l'exploration validee"""
    print("preparation des donnees acceslibre pour machine learning")
    print("suit la logique de l'exploration streamlit validee")
    print()
    
    # chargement du dataset complet
    df = load_dataset()
    if df is None:
        return None
    
    print(f"dataset initial: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    # analyse de completude (comme dans l'exploration)
    print("\nanalyse de completude des variables d'accessibilite...")
    completude_df = analyze_completeness(df)
    print(f"variables d'accessibilite detectees: {len(completude_df)}")
    
    # affichage des principales variables detectees
    print("\ntop 10 variables d'accessibilite par completude:")
    for _, row in completude_df.head(10).iterrows():
        print(f"  {row['variable']}: {row['completude_pct']:.1f}% complete")
    
    # categorisation des colonnes (comme dans l'exploration)
    print("\ncategorisation des colonnes par domaine metier...")
    category_stats = categorize_columns(df)
    for category, stats in category_stats.items():
        print(f"  {category}: {stats['count']} colonnes")
    
    # identification des variables ml (comme dans l'exploration)
    print("\nidentification des variables pour machine learning...")
    numeric_vars, categorical_vars = get_ml_variables(df)
    print(f"  variables numeriques: {len(numeric_vars)}")
    print(f"  variables categorielles: {len(categorical_vars)}")
    
    # selection des colonnes basee sur les insights specifiques de l'exploration
    print("\nselection intelligente des variables basee sur l'exploration eda...")
    colonnes_selectionnees = []
    
    # 1. variables cles identifiees dans l'exploration streamlit
    variables_cles_exploration = [
        # variable cible principale (83% manquante mais essentielle)
        'entree_pmr',
        
        # variables avec insights avances (analyse marches/largeurs)
        'entree_marches',           # analyse marches: 0, 1-3, 4-10, >10
        'entree_largeur_mini',      # analyse largeurs: conforme pmr >=90cm
        
        # variables d'accessibilite avec meilleure completude
        'stationnement_ext_pmr',    # 53.9% complete
        'stationnement_pmr',        # 49.3% complete
        'entree_marches_rampe',     # 11.5% complete
        'entree_ascenseur',         # 6.9% complete
        
        # variables d'identification pour contexte
        'name', 'activite', 'commune',
        
        # variables numeriques importantes detectees
        'cheminement_ext_nombre_marches',
        'accueil_cheminement_nombre_marches',
        'accueil_chambre_nombre_accessibles'
    ]
    
    # ajout des variables cles si elles existent
    for var in variables_cles_exploration:
        if var in df.columns:
            colonnes_selectionnees.append(var)
    
    print(f"  variables cles exploration: {len([v for v in variables_cles_exploration if v in df.columns])}")
    
    # 2. variables d'accessibilite avec completude significative (>5%)
    # seuil abaisse car l'exploration montre l'importance meme avec faible completude
    variables_accessibilite_significatives = completude_df[completude_df['completude_pct'] > 5]['variable'].tolist()
    colonnes_selectionnees.extend(variables_accessibilite_significatives)
    print(f"  variables accessibilite >5%: {len(variables_accessibilite_significatives)}")
    
    # 3. variables numeriques avec donnees suffisantes (seuil reduit)
    # seuil reduit de 1000 a 500 pour ne pas perdre d'informations importantes
    variables_numeriques_pertinentes = [var['column'] for var in numeric_vars if var['count'] > 500][:15]
    colonnes_selectionnees.extend(variables_numeriques_pertinentes)
    print(f"  variables numeriques >500 valeurs: {len(variables_numeriques_pertinentes)}")
    
    # 4. variables categorielles avec diversite raisonnable
    # criteres assouplis pour inclure plus de contexte
    variables_categorielles_pertinentes = []
    for var in categorical_vars:
        if var['count'] > 500 and 2 <= var['unique'] <= 100:  # seuil elargi
            variables_categorielles_pertinentes.append(var['column'])
    
    colonnes_selectionnees.extend(variables_categorielles_pertinentes[:20])  # top 20
    print(f"  variables categorielles pertinentes: {len(variables_categorielles_pertinentes[:20])}")
    
    # suppression des doublons et selection
    colonnes_selectionnees = list(set(colonnes_selectionnees))
    colonnes_existantes = [col for col in colonnes_selectionnees if col in df.columns]
    
    print(f"\nselection finale: {len(colonnes_existantes)} colonnes pertinentes")
    
    # documentation des choix de selection
    print("\njustification de la selection (basee sur l'exploration eda):")
    print("  - variables cles avec insights avances (marches, largeurs)")
    print("  - seuil accessibilite abaisse a 5% (vs 10% initial)")
    print("  - seuil numerique abaisse a 500 valeurs (vs 1000 initial)")
    print("  - seuil categoriel elargi a 100 categories max (vs 50 initial)")
    print("  - priorite aux variables analysees dans l'exploration streamlit")
    
    # reduction du dataset aux colonnes selectionnees
    df_selected = df[colonnes_existantes].copy()
    print(f"\ndataset reduit: {len(df_selected):,} lignes, {len(df_selected.columns)} colonnes")
    
    # analyse de la reduction
    reduction_pct = (1 - len(df_selected.columns) / len(df.columns)) * 100
    print(f"reduction: {len(df.columns)} -> {len(df_selected.columns)} colonnes ({reduction_pct:.1f}% reduction)")
    
    return df_selected, completude_df, category_stats, numeric_vars, categorical_vars
    
    # Créer le dataset de travail
    df_pmr = df[colonnes_existantes].copy()
    
    # Afficher les valeurs manquantes AVANT imputation
    display_missing_values(df_pmr, "AVANT imputation - Valeurs manquantes par colonne")
    
    return df_pmr, colonnes_existantes

def display_missing_values(df, title="Valeurs manquantes par colonne"):
    """Affiche les statistiques des valeurs manquantes"""
    print(f"\n{title}:")
    missing_info = df.isnull().sum()
    missing_pct = (missing_info / len(df)) * 100
    
    for col in df.columns:
        if missing_info[col] > 0 or col in ['name', 'activite', 'entree_pmr', 'entree_marches', 'accueil_cheminement_rampe', 'stationnement_pmr']:
            print(f"{col:35} : {missing_info[col]:6,} ({missing_pct[col]:5.1f}%)")

def impute_public_places(df):
    """Impute entree_pmr=True pour les lieux publics obligatoirement accessibles
    et pour tous les établissements avec stationnement PMR"""
    
    print("\nImputation des lieux publics et établissements avec stationnement PMR")
    
    # Liste des lieux publics par nom
    lieux_publics_nom = [
        'mairie', 'la poste', 'poste', 'gendarmerie',
        'école', 'ecole', 'bibliothèque', 'bibliotheque',
        'médiathèque', 'mediatheque', 'toilettes publiques',
        'pharmacie', 'hôpital', 'hopital', 'centre médical',
        'centre medical', 'clinique'
    ]
    
    # Liste des activités publiques
    activites_publiques = [
        'Mairie', 'Bureau de poste', 'Gendarmerie',
        'École primaire', 'Bibliothèque médiathèque',
        'Toilettes publiques', 'Pharmacie', 'Hôpital',
        'Centre médical', 'Clinique'
    ]
    
    # Compteur des modifications
    modifications = 0
    
    # RÈGLE 1: Imputation basée sur le nom
    for lieu in lieux_publics_nom:
        mask_nom = (df['name'].str.lower().str.contains(lieu, na=False)) & (df['entree_pmr'].isna())
        nb_modif_nom = mask_nom.sum()
        if nb_modif_nom > 0:
            df.loc[mask_nom, 'entree_pmr'] = True
            modifications += nb_modif_nom
            print(f"  Nom '{lieu}': {nb_modif_nom} modifications")
    
    # RÈGLE 2: Imputation basée sur l'activité
    for activite in activites_publiques:
        mask_activite = (df['activite'] == activite) & (df['entree_pmr'].isna())
        nb_modif_activite = mask_activite.sum()
        if nb_modif_activite > 0:
            df.loc[mask_activite, 'entree_pmr'] = True
            modifications += nb_modif_activite
            print(f"  Activité '{activite}': {nb_modif_activite} modifications")
    
    # RÈGLE 3: Si stationnement PMR présent → établissement accessible
    if 'stationnement_pmr' in df.columns:
        mask_parking = (df['stationnement_pmr'] == 1) & (df['entree_pmr'].isna())
        nb_modif_parking = mask_parking.sum()
        if nb_modif_parking > 0:
            df.loc[mask_parking, 'entree_pmr'] = True
            modifications += nb_modif_parking
            print(f"  Stationnement PMR: {nb_modif_parking} modifications")
    
    print(f"\nTotal imputation métier: {modifications} valeurs")
    
    # Afficher les valeurs manquantes APRES imputation métier
    display_missing_values(df, "APRES imputation métier - Valeurs manquantes par colonne")
    
    return df

# def simple_mode_imputation(df):
#     """
#     MÉTHODE TESTÉE - Imputation par mode global
#     Remplacée par logistic_regression_imputation() pour de meilleurs résultats
#     Conservée pour traçabilité et comparaison
#     """
#     
#     print("\nImputation par mode global (focus sur entree_pmr)")
#     print("Principe: remplacer les valeurs manquantes par la valeur la plus fréquente")
#     
#     df_mode = df.copy()
#     col_target = 'entree_pmr'
#     
#     # Vérifier si la colonne existe
#     if col_target not in df_mode.columns:
#         print(f"Erreur: La colonne '{col_target}' n'existe pas dans le DataFrame")
#         print(f"Colonnes disponibles: {list(df_mode.columns)}")
#         return df_mode
#     
#     # Nettoyer entree_pmr : convertir les valeurs textuelles en NaN
#     df_mode[col_target] = df_mode[col_target].replace(['aucune', 'non renseigné', ''], np.nan)
#     
#     # Sauvegarder l'état avant imputation
#     missing_before = df_mode[col_target].isna().sum()
#     
#     # Calculer le mode global
#     if missing_before > 0:
#         global_mode = df_mode[col_target].mode()
#         if len(global_mode) > 0:
#             mode_value = global_mode.iloc[0]
#             print(f"Mode global détecté: {mode_value}")
#             
#             # Appliquer l'imputation
#             df_mode[col_target] = df_mode[col_target].fillna(mode_value)
#             
#             # Afficher les résultats
#             missing_after = df_mode[col_target].isna().sum()
#             corrections = missing_before - missing_after
#             print(f"\nRésultats de l'imputation par mode:")
#             print(f"  {col_target}: {corrections:,} valeurs imputées")
#             print(f"  Valeurs manquantes restantes: {missing_after}")
#         else:
#             print("Aucun mode détectable, pas d'imputation possible")
#     else:
#         print("Aucune valeur manquante à imputer")
#     
#     # Afficher les valeurs manquantes APRES imputation
#     display_missing_values(df_mode, "APRES imputation par mode - Valeurs manquantes par colonne")
#     
#     return df_mode

def logistic_regression_imputation(df):
    """Applique une imputation par régression logistique sur entree_pmr"""
    
    print("\nImputation par régression logistique (focus sur entree_pmr)")
    print("Principe: utiliser les autres variables PMR pour prédire entree_pmr")
    
    df_logistic = df.copy()
    col_target = 'entree_pmr'
    
    # Vérifier si la colonne existerhoooo
    if col_target not in df_logistic.columns:
        print(f"Erreur: La colonne '{col_target}' n'existe pas dans le DataFrame")
        return df_logistic
    
    # Nettoyer entree_pmr : convertir les valeurs textuelles en NaN
    df_logistic[col_target] = df_logistic[col_target].replace(['aucune', 'non renseigné', ''], np.nan)
    
    # Sauvegarder l'état avant imputation
    missing_before = df_logistic[col_target].isna().sum()
    print(f"Valeurs manquantes avant imputation: {missing_before:,}")
    
    if missing_before == 0:
        print("Aucune valeur manquante à imputer")
        return df_logistic
    
    # Préparer les variables prédictives
    predictive_cols = ['entree_marches', 'accueil_cheminement_rampe', 'stationnement_pmr']
    available_cols = [col for col in predictive_cols if col in df_logistic.columns]
    
    print(f"Variables prédictives disponibles: {available_cols}")
    
    if len(available_cols) == 0:
        print("Aucune variable prédictive disponible, fallback sur mode global")
        mode_value = df_logistic[col_target].mode()
        if len(mode_value) > 0:
            df_logistic[col_target] = df_logistic[col_target].fillna(mode_value.iloc[0])
        return df_logistic
    
    # Créer le dataset pour l'entraînement (lignes avec entree_pmr connu)
    mask_complete = df_logistic[col_target].notna()
    df_train = df_logistic[mask_complete].copy()
    df_predict = df_logistic[~mask_complete].copy()
    
    print(f"Données d'entraînement: {len(df_train):,} lignes")
    print(f"Données à prédire: {len(df_predict):,} lignes")
    
    if len(df_train) < 10:
        print("Pas assez de données d'entraînement, fallback sur mode global")
        mode_value = df_logistic[col_target].mode()
        if len(mode_value) > 0:
            df_logistic[col_target] = df_logistic[col_target].fillna(mode_value.iloc[0])
        return df_logistic
    
    # Préparer les features pour la régression logistique
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    
    # Encoder les variables catégorielles et gérer les valeurs manquantes
    X_train = df_train[available_cols].copy()
    X_predict = df_predict[available_cols].copy()
    
    encoders = {}
    for col in available_cols:
        if df_logistic[col].dtype == 'object' or df_logistic[col].dtype == 'bool':  # Variable catégorielle ou booléenne
            # Convertir d'abord tout en string pour éviter les types mixtes
            X_train[col] = X_train[col].astype(str)
            X_predict[col] = X_predict[col].astype(str)
            # Remplacer les valeurs manquantes par 'Unknown'
            X_train[col] = X_train[col].replace(['nan', 'None'], 'Unknown')
            X_predict[col] = X_predict[col].replace(['nan', 'None'], 'Unknown')
            
            # Encoder
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            
            # Pour les données à prédire, gérer les nouvelles catégories
            X_predict_encoded = []
            for val in X_predict[col]:
                if val in le.classes_:
                    X_predict_encoded.append(le.transform([val])[0])
                else:
                    X_predict_encoded.append(-1)  # Nouvelle catégorie
            X_predict[col] = X_predict_encoded
            encoders[col] = le
        else:  # Variable numérique
            # Remplacer les valeurs manquantes par la médiane
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_predict[col] = X_predict[col].fillna(median_val)
    
    # Préparer la variable cible
    y_train = df_train[col_target].astype(bool).astype(int)
    
    # Entraîner le modèle de régression logistique
    print("\nEntraînement du modèle de régression logistique...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Évaluer le modèle sur les données d'entraînement
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Précision sur données d'entraînement: {train_accuracy:.3f}")
    
    # Afficher l'importance des variables
    print("\nImportance des variables (coefficients):")
    for i, col in enumerate(available_cols):
        coef = model.coef_[0][i]
        print(f"  {col}: {coef:.3f}")
    
    # Prédire les valeurs manquantes
    if len(df_predict) > 0:
        y_pred = model.predict(X_predict)
        y_pred_proba = model.predict_proba(X_predict)[:, 1]  # Probabilité de True
        
        # Appliquer les prédictions
        df_logistic.loc[~mask_complete, col_target] = y_pred.astype(bool)
        
        # Statistiques des prédictions
        pred_true = np.sum(y_pred)
        pred_false = len(y_pred) - pred_true
        print(f"\nPrédictions:")
        print(f"  True (accessible): {pred_true:,} ({pred_true/len(y_pred)*100:.1f}%)")
        print(f"  False (non accessible): {pred_false:,} ({pred_false/len(y_pred)*100:.1f}%)")
        print(f"  Probabilité moyenne: {np.mean(y_pred_proba):.3f}")
    
    # Afficher les résultats
    missing_after = df_logistic[col_target].isna().sum()
    corrections = missing_before - missing_after
    print(f"\nRésultats de l'imputation par régression logistique:")
    print(f"  {col_target}: {corrections:,} valeurs imputées")
    print(f"  Valeurs manquantes restantes: {missing_after}")
    
    # Afficher les valeurs manquantes APRES imputation
    display_missing_values(df_logistic, "APRES imputation par régression logistique - Valeurs manquantes par colonne")
    
    return df_logistic

def create_accessibility_score(df):
    """Crée un score d'accessibilité basé sur les critères PMR"""
    
    print("\nCreation du score d'accessibilite")
    
    
    df_score = df.copy()
    
    # Initialiser le score
    df_score['score_accessibilite'] = 0
    
    # Critères positifs (ajoutent des points)
    criteres_positifs = {
        'stationnement_pmr': 2,
        'stationnement_ext_pmr': 2,
        'entree_pmr': 3,
        'cheminement_ext_terrain_stable': 1,
    }
    
    # Critères négatifs (retirent des points)
    criteres_negatifs = {
        'entree_marches': -2,
        'cheminement_ext_nombre_marches': -1,  # Par marche
        'accueil_cheminement_nombre_marches': -1,  # Par marche
    }
    
    # Appliquer les critères positifs
    for critere, points in criteres_positifs.items():
        if critere in df_score.columns:
            df_score.loc[df_score[critere] == True, 'score_accessibilite'] += points
    
    # Appliquer les critères négatifs
    for critere, points in criteres_negatifs.items():
        if critere in df_score.columns:
            if critere in ['cheminement_ext_nombre_marches', 'accueil_cheminement_nombre_marches']:
                # Pour le nombre de marches, multiplier par le nombre
                df_score.loc[df_score[critere].notna(), 'score_accessibilite'] += (
                    df_score[critere].fillna(0) * points
                )
            else:
                df_score.loc[df_score[critere] == True, 'score_accessibilite'] += points
    
    # Créer les catégories d'accessibilité
    def categoriser_accessibilite(score):
        if score >= 4:
            return 2  # Facilement accessible
        elif score >= 1:
            return 1  # Modérément accessible
        else:
            return 0  # Difficilement accessible
    
    df_score['accessibilite_categorie'] = df_score['score_accessibilite'].apply(categoriser_accessibilite)
    
    # Statistiques
    print("Distribution des scores:")
    print(df_score['score_accessibilite'].describe())
    
    print("\nDistribution des catégories:")
    categories = ['Difficilement', 'Modérément', 'Facilement']
    for i, cat in enumerate(categories):
        count = (df_score['accessibilite_categorie'] == i).sum()
        pct = count / len(df_score) * 100
        print(f"{cat:15} accessible: {count:6,} ({pct:5.1f}%)")
    
    return df_score

def prepare_features(df):
    """Prépare les features pour le machine learning"""
    
    print("\nPreparation des features")
    print("-" * 30)
    
    df_features = df.copy()
    label_encoders = {}
    
    # ÉTAPE 1: Nettoyer TOUTES les colonnes d'abord
    print("Nettoyage des types de données...")
    for col in df_features.columns:
        if col not in ['score_accessibilite', 'accessibilite_categorie']:
            # Identifier le type réel de la colonne
            sample_values = df_features[col].dropna().head(100)
            
            if len(sample_values) > 0:
                # Vérifier si c'est vraiment booléen
                unique_vals = set(sample_values.astype(str).str.lower())
                if unique_vals.issubset({'true', 'false', 'nan', 'none'}):
                    # Colonne booléenne
                    df_features[col] = df_features[col].astype(str).str.lower()
                    df_features[col] = df_features[col].replace({'true': 1, 'false': 0, 'nan': 0, 'none': 0})
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0).astype(int)
                    print(f"Booléen: {col} -> converti en 0/1")
                    
                elif df_features[col].dtype == 'object':
                    # Colonne catégorielle
                    df_features[col] = df_features[col].astype(str)
                    df_features[col] = df_features[col].replace(['nan', 'None', 'NaN', ''], 'Unknown')
                    
                    if df_features[col].nunique() > 1:
                        try:
                            le = LabelEncoder()
                            df_features[col + '_encoded'] = le.fit_transform(df_features[col])
                            label_encoders[col] = le
                            print(f"Catégoriel: {col} -> {col}_encoded")
                        except Exception as e:
                            print(f"Erreur encodage {col}: {str(e)} - Ignoré")
                    else:
                        print(f"Ignoré: {col} (une seule valeur)")
                        
                elif pd.api.types.is_numeric_dtype(df_features[col]):
                    # Colonne numérique
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    df_features[col] = df_features[col].fillna(df_features[col].median())
                    print(f"Numérique: {col} -> valeurs manquantes remplacées")
    
    print(f"Features préparées: {len(df_features.columns)} colonnes")
    
    return df_features, label_encoders

# def balance_classes_with_smote(X, y):
#     """Rééquilibre les classes avec SMOTE pour gérer le déséquilibre 16:1"""
#     
#     print("\n Rééquilibrage des classes avec SMOTE")
#     
#     # Vérifier la distribution avant SMOTE
#     unique, counts = np.unique(y, return_counts=True)
#     print(f"Distribution AVANT SMOTE:")
#     for val, count in zip(unique, counts):
#         print(f"  Classe {val}: {count:,} exemples ({count/len(y)*100:.1f}%)")
#     
#     # Appliquer SMOTE pour rééquilibrer
#     # Stratégie: augmenter la classe minoritaire pour avoir un ratio 1:2 (plus réaliste que 1:1)
#     smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 50% de la classe majoritaire
#     
#     try:
#         X_balanced, y_balanced = smote.fit_resample(X, y)
#         
#         # Vérifier la distribution après SMOTE
#         unique, counts = np.unique(y_balanced, return_counts=True)
#         print(f"Distribution APRES SMOTE:")
#         for val, count in zip(unique, counts):
#             print(f"  Classe {val}: {count:,} exemples ({count/len(y_balanced)*100:.1f}%)")
#         
#         print(f"Dataset rééquilibré: {len(X_balanced):,} lignes (vs {len(X):,} originales)")
#         
#         return X_balanced, y_balanced
#         
#     except Exception as e:
#         print(f"Erreur SMOTE: {e}")
#         print("Utilisation des données originales sans rééquilibrage")
#         return X, y

def compare_imputation_methods(df, target_col='entree_pmr'):
    """Compare différentes méthodes d'imputation pour montrer le parcours d'apprentissage"""
    
    print("\n=== COMPARAISON DES MÉTHODES D'IMPUTATION ===")
    print("Objectif pédagogique: montrer pourquoi j'ai choisi la régression logistique")
    
    # Préparer les données pour comparaison
    df_test = df[df[target_col].notna()].copy()  # Données complètes pour test
    
    if len(df_test) < 1000:
        print("Pas assez de données complètes pour comparaison fiable")
        return
    
    # Créer des valeurs manquantes artificielles pour tester
    test_size = min(1000, len(df_test))
    df_sample = df_test.sample(n=test_size, random_state=42).copy()
    
    # Sauvegarder les vraies valeurs
    true_values = df_sample[target_col].copy()
    
    # Créer des valeurs manquantes artificielles (30%)
    missing_indices = df_sample.sample(frac=0.3, random_state=42).index
    df_sample.loc[missing_indices, target_col] = np.nan
    
    print(f"Test sur {test_size} échantillons avec {len(missing_indices)} valeurs à prédire")
    
    # MÉTHODE 1: Imputation simple (mode)
    print("\n1. MÉTHODE BASIQUE - Mode/Médiane")
    mode_value = df_test[target_col].mode()[0]
    df_simple = df_sample.copy()
    df_simple[target_col].fillna(mode_value, inplace=True)
    
    # Calculer précision méthode simple
    predicted_simple = df_simple.loc[missing_indices, target_col]
    accuracy_simple = (predicted_simple == true_values.loc[missing_indices]).mean()
    print(f"   Stratégie: Remplacer par la valeur la plus fréquente ({mode_value})")
    print(f"   Précision: {accuracy_simple:.3f} ({accuracy_simple*100:.1f}%)")
    print(f"   Avantage: Simple et rapide")
    print(f"   Inconvénient: Ne tient pas compte du contexte")
    
    # MÉTHODE 2: Régression logistique (notre choix)
    print("\n2. MÉTHODE AVANCÉE - Régression Logistique")
    
    # Variables disponibles pour prédiction
    feature_cols = ['entree_marches', 'accueil_cheminement_rampe', 'stationnement_pmr']
    available_features = [col for col in feature_cols if col in df_sample.columns]
    
    if len(available_features) >= 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        # Préparer les données d'entraînement
        train_mask = df_sample[target_col].notna()
        X_train = df_sample.loc[train_mask, available_features].copy()
        y_train = df_sample.loc[train_mask, target_col].copy()
        
        # Nettoyer y_train pour s'assurer qu'il contient des classes valides
        y_train = y_train.astype(str).str.lower()
        y_train = y_train.replace({'true': 1, 'false': 0, 'nan': 0, 'none': 0})
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        
        # Encoder les variables si nécessaire
        encoders = {}
        for col in available_features:
            if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
                # Convertir d'abord tout en string pour éviter les types mixtes
                X_train[col] = X_train[col].astype(str)
                X_train[col] = X_train[col].replace(['nan', 'None'], 'Missing')
                encoders[col] = LabelEncoder()
                X_train[col] = encoders[col].fit_transform(X_train[col])
        
        # Remplir les valeurs manquantes pour l'entraînement
        X_train = X_train.fillna(X_train.median())
        
        # Entraîner le modèle
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Prédire sur les valeurs manquantes
        X_pred = df_sample.loc[missing_indices, available_features].copy()
        for col in available_features:
            if col in encoders:
                # Convertir en string et gérer les valeurs manquantes
                X_pred[col] = X_pred[col].astype(str)
                X_pred[col] = X_pred[col].replace(['nan', 'None'], 'Missing')
                # Transformer avec l'encodeur déjà entraîné
                try:
                    X_pred[col] = encoders[col].transform(X_pred[col])
                except ValueError:
                    # Si nouvelle catégorie, utiliser la première classe connue
                    X_pred[col] = X_pred[col].replace(X_pred[col].unique(), encoders[col].classes_[0])
                    X_pred[col] = encoders[col].transform(X_pred[col])
        X_pred = X_pred.fillna(X_pred.median())
        
        predicted_advanced = model.predict(X_pred)
        accuracy_advanced = (predicted_advanced == true_values.loc[missing_indices]).mean()
        
        print(f"   Stratégie: Utiliser {len(available_features)} variables prédictives")
        print(f"   Variables: {', '.join(available_features)}")
        print(f"   Précision: {accuracy_advanced:.3f} ({accuracy_advanced*100:.1f}%)")
        print(f"   Avantage: Tient compte du contexte et des corrélations")
        print(f"   Inconvénient: Plus complexe à implémenter")
        
        # CONCLUSION
        print("\n3. CONCLUSION DE LA COMPARAISON")
        improvement = accuracy_advanced - accuracy_simple
        print(f"   Amélioration: +{improvement:.3f} ({improvement*100:.1f} points de pourcentage)")
        
        if improvement > 0.05:
            print(f"   ✅ DÉCISION: Régression logistique justifiée par gain significatif")
        else:
            print(f"   ⚠️  DÉCISION: Gain marginal, mais approche plus rigoureuse")
            
    else:
        print("   ⚠️ Pas assez de variables disponibles pour comparaison complète")
    
    print("\n" + "="*60)
    print("APPRENTISSAGE: Cette comparaison m'a aidé à choisir la meilleure méthode")
    print("="*60)

def create_validation_plots(df_before, df_after, target_col='entree_pmr'):
    """Crée des graphiques de validation avant/après preprocessing"""
    
    print("\n Création graphiques de validation")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Comparaison valeurs manquantes
    missing_before = df_before.isnull().sum().sort_values(ascending=False).head(10)
    missing_after = df_after.isnull().sum().sort_values(ascending=False).head(10)
    
    ax1.bar(range(len(missing_before)), missing_before.values, alpha=0.7, label='Avant', color='lightcoral')
    ax1.set_title('Valeurs manquantes - Comparaison')
    ax1.set_ylabel('Nombre de valeurs manquantes')
    ax1.set_xticks(range(len(missing_before)))
    ax1.set_xticklabels([col[:15] + '...' if len(col) > 15 else col for col in missing_before.index], rotation=45)
    ax1.legend()
    
    # 2. Distribution variable cible avant/après
    if target_col in df_before.columns and target_col in df_after.columns:
        # Avant
        target_before = df_before[target_col].value_counts(dropna=False)
        labels_before = ['Manquant', 'Non accessible', 'Accessible']
        sizes_before = [df_before[target_col].isnull().sum(), 
                       (df_before[target_col] == False).sum(),
                       (df_before[target_col] == True).sum()]
        
        ax2.pie(sizes_before, labels=labels_before, autopct='%1.1f%%', 
               colors=['lightgray', 'lightcoral', 'lightgreen'], startangle=90)
        ax2.set_title('Variable cible - AVANT imputation')
        
        # Après
        target_after = df_after[target_col].value_counts(dropna=False)
        sizes_after = [df_after[target_col].isnull().sum(), 
                      (df_after[target_col] == False).sum(),
                      (df_after[target_col] == True).sum()]
        
        ax3.pie(sizes_after, labels=labels_before, autopct='%1.1f%%', 
               colors=['lightgray', 'lightcoral', 'lightgreen'], startangle=90)
        ax3.set_title('Variable cible - APRES imputation')
    
    # 3. Complétude globale
    completude_before = (df_before.notna().sum() / len(df_before) * 100).mean()
    completude_after = (df_after.notna().sum() / len(df_after) * 100).mean()
    
    ax4.bar(['Avant preprocessing', 'Après preprocessing'], 
           [completude_before, completude_after], 
           color=['lightcoral', 'lightgreen'])
    ax4.set_title('Complétude moyenne du dataset')
    ax4.set_ylabel('Pourcentage de complétude')
    ax4.set_ylim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate([completude_before, completude_after]):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('validation_preprocessing.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graphique de validation sauvegardé")

def save_prepared_data(df, filename="data_prepared.csv"):
    """Sauvegarde les données préparées"""
    
    # Sauvegarder dans le répertoire courant
    filepath = filename
    df.to_csv(filepath, index=False)
    print(f"\nDonnees sauvegardees: {filepath}")
    
    return filepath

def simple_imputation_accessibility(df):
    """
    Applique une imputation simple et rapide sur les variables clés d'accessibilité
    Médiane pour numériques, mode pour binaires
    """
    print("\nImputation simple des variables d'accessibilité")
    
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # Variables clés d'accessibilité à imputer
    accessibility_vars = [
        'entree_largeur_mini',      # Largeur entrée (numérique)
        'entree_marches',           # Nombre de marches (numérique)
        'sanitaires_presence',      # Sanitaires présents (binaire)
        'stationnement_pmr',        # Stationnement PMR (binaire)
        'transport_station_presence', # Transport accessible (binaire)
        'entree_plain_pied',        # Entrée plain-pied (binaire)
        'cheminement_ext_presence', # Cheminement extérieur (binaire)
        'stationnement_ext_pmr'     # Stationnement ext PMR (binaire)
    ]
    
    # Filtrer les variables qui existent dans le dataset
    vars_to_impute = [var for var in accessibility_vars if var in df.columns]
    print(f"Variables à imputer: {len(vars_to_impute)}")
    
    if len(vars_to_impute) == 0:
        print("Aucune variable d'accessibilité trouvée pour l'imputation")
        return df
    
    # Créer une copie pour l'imputation
    df_imputed = df.copy()
    
    # Variables binaires et numériques
    binary_vars = ['sanitaires_presence', 'stationnement_pmr', 'transport_station_presence', 
                   'entree_plain_pied', 'cheminement_ext_presence', 'stationnement_ext_pmr']
    numeric_vars = ['entree_largeur_mini', 'entree_marches']
    
    # Convertir les variables en numériques
    print("Conversion des variables en numériques...")
    for var in vars_to_impute:
        if var in df_imputed.columns:
            if var in binary_vars:
                # Convertir True/False en 1/0
                df_imputed[var] = df_imputed[var].replace({'True': 1, 'False': 0, True: 1, False: 0})
                df_imputed[var] = pd.to_numeric(df_imputed[var], errors='coerce')
            else:
                # Variables numériques
                df_imputed[var] = pd.to_numeric(df_imputed[var], errors='coerce')
    
    # Afficher l'état avant imputation
    print("\nValeurs manquantes AVANT imputation:")
    for var in vars_to_impute:
        if var in df_imputed.columns:
            missing_count = df_imputed[var].isna().sum()
            missing_pct = (missing_count / len(df_imputed)) * 100
            print(f"  {var}: {missing_count:,} ({missing_pct:.1f}%)")
    
    # Imputation des variables numériques (médiane)
    print("\nImputation des variables numériques (médiane)...")
    numeric_imputer = SimpleImputer(strategy='median')
    
    for var in numeric_vars:
        if var in df_imputed.columns:
            original_missing = df_imputed[var].isna().sum()
            if original_missing > 0:
                df_imputed[var] = numeric_imputer.fit_transform(df_imputed[[var]]).ravel()
                print(f"  {var}: {original_missing:,} valeurs imputées")
    
    # Imputation des variables binaires (mode)
    print("\nImputation des variables binaires (mode)...")
    binary_imputer = SimpleImputer(strategy='most_frequent')
    
    for var in binary_vars:
        if var in df_imputed.columns:
            original_missing = df_imputed[var].isna().sum()
            if original_missing > 0:
                df_imputed[var] = binary_imputer.fit_transform(df_imputed[[var]]).ravel()
                df_imputed[var] = df_imputed[var].astype(int)  # Forcer en entier
                print(f"  {var}: {original_missing:,} valeurs imputées")
    
    # Afficher l'état après imputation
    print("\nValeurs manquantes APRÈS imputation:")
    total_imputed = 0
    for var in vars_to_impute:
        if var in df_imputed.columns:
            missing_count = df_imputed[var].isna().sum()
            missing_pct = (missing_count / len(df_imputed)) * 100
            print(f"  {var}: {missing_count:,} ({missing_pct:.1f}%)")
            if missing_count == 0:
                total_imputed += 1
    
    print(f"\nImputation simple terminée avec succès sur {total_imputed}/{len(vars_to_impute)} variables")
    
    return df_imputed

def main():
    """fonction principale de preparation des donnees - suit la logique de l'exploration"""
    
    
    print("preparation des donnees acceslibre pour machine learning")
    print("suit la logique de l'exploration streamlit validee")
    
    # 1. preparation initiale basee sur l'exploration
    print("\n1. preparation initiale basee sur l'exploration")
    
    
    result = prepare_data_for_ml()
    if result is None:
        print("erreur lors de la preparation des donnees")
        return None
    
    df_selected, completude_df, category_stats, numeric_vars, categorical_vars = result
    
    # 2. nettoyage des types de donnees
    print("\n2. nettoyage des types de donnees")
    
    
    df_clean = clean_data_types(df_selected)
    
    # 3. imputation des lieux publics
    print("\n3. imputation des lieux publics")
    
    # appliquer l'imputation métier pour les lieux publics
    df_clean = impute_public_places(df_clean)
    
    # 4. imputation simple pour les variables d'accessibilité
    print("\n4. imputation simple pour les variables d'accessibilité")
    
    df_clean = simple_imputation_accessibility(df_clean)
    
    # 5. analyse des valeurs manquantes après imputation
    print("\n5. analyse des valeurs manquantes après imputation")
    
    
    missing_analysis = df_clean.isnull().sum()
    missing_pct = (missing_analysis / len(df_clean)) * 100
    
    print("variables avec le plus de valeurs manquantes:")
    missing_summary = pd.DataFrame({
        'variable': missing_analysis.index,
        'manquantes': missing_analysis.values,
        'pourcentage': missing_pct.values
    }).sort_values('pourcentage', ascending=False)
    
    for _, row in missing_summary.head(10).iterrows():
        if row['pourcentage'] > 0:
            print(f"  {row['variable']}: {row['manquantes']:,} ({row['pourcentage']:.1f}%)")
    
    # 6. preparation des features pour ml
    print("\n6. preparation des features pour machine learning")
    
    
    # identification de la variable cible principale
    target_candidates = ['entree_pmr', 'accessibilite_pmr', 'pmr']
    target_col = None
    
    for candidate in target_candidates:
        if candidate in df_clean.columns:
            target_col = candidate
            break
    
    if target_col:
        print(f"variable cible identifiee: {target_col}")
        target_missing = df_clean[target_col].isnull().sum()
        target_pct = (target_missing / len(df_clean)) * 100
        print(f"valeurs manquantes cible: {target_missing:,} ({target_pct:.1f}%)")
    else:
        print("aucune variable cible claire identifiee")
    
    # 7. preparation finale et sauvegarde
    print("\n7. preparation finale et sauvegarde")
    
    
    # sauvegarde du dataset prepare
    output_file = "data_prepared_EXPLORATION.csv"
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    print(f"dataset prepare sauvegarde: {output_file}")
    print(f"taille finale: {len(df_clean):,} lignes, {len(df_clean.columns)} colonnes")
    
    # resume des resultats
    print("\n" + "="*80)
    print("resume de la preparation")
    print("="*80)
    print(f"variables d'accessibilite detectees: {len(completude_df)}")
    print(f"variables numeriques ml: {len(numeric_vars)}")
    print(f"variables categorielles ml: {len(categorical_vars)}")
    print(f"colonnes finales selectionnees: {len(df_clean.columns)}")
    if target_col:
        print(f"variable cible: {target_col} ({target_pct:.1f}% manquante)")
    
    print("\npreparation terminee avec succes")
    print("dataset pret pour modelisation machine learning")
    
    return df_clean, completude_df, category_stats

if __name__ == "__main__":
    print("lancement de la preparation des donnees...")
    result = main()
    if result:
        print("\npreparation reussie !")
    else:
        print("\nerreur lors de la preparation")
    
   
