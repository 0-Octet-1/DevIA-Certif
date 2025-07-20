#!/usr/bin/env python3
"""
Objectif: Entraîner et comparer plusieurs algorithmes de ML pour prédire l'accessibilité PMR
des établissements à partir des données AccesLibre préparées.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from datetime import datetime



# Algorithmes de ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Outils de preprocessing et évaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Gestion du déséquilibre
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings('ignore')

def load_prepared_data(mode="echantillon"):
    """Charge les données préparées depuis le script 02"""
    
    print(f"\n Chargement des données préparées - Mode {mode.upper()}")
        
    # Chemin vers les données préparées (dans le répertoire racine)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join('..', '2-TravailPreparationdesDonnées', 'data_prepared_EXPLORATION.csv')

    try:
        df = pd.read_csv(file_path)
        print(f"Dataset chargé avec succès: {len(df):,} lignes, {len(df.columns)} colonnes")
        
        # Vérifier la colonne cible essentielle
        if 'entree_pmr' not in df.columns:
            print("Colonne cible 'entree_pmr' manquante")
            print("Vérifiez que le script 02_data_preparation.py a été exécuté")
            return None
            
        # Statistiques rapides
        print(f"\nAperçu des données:")
        print(f"- Variable cible binaire (entree_pmr): {df['entree_pmr'].value_counts().to_dict()}")
        print(f"- Colonnes disponibles: {len(df.columns)} variables")
        
        return df
        
    except FileNotFoundError:
        print(f"Fichier non trouvé: {filepath}")
        print("\nSolution: Exécutez d'abord le script 02_data_preparation.py")
        print("Commande: python ../2-TravailPreparationdesDonnees/02_data_preparation.py")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement: {str(e)}")
        return None

def prepare_features_for_ml(df, target_type="binaire"):
    """Prépare les features pour le machine learning à partir des données préparées"""
    
    print(f"\n Préparation des features pour ML - Cible {target_type}")
    
    
    # Colonnes à exclure (variables cibles et data leakage détecté)
    exclude_cols = [
        'entree_pmr',                   # Variable cible binaire
        'accessibilite_categorie',      # Data leakage: corrélation 0.79
        'score_accessibilite',          # Data leakage: corrélation 0.75
        'entree_vitree',                # Data leakage: 48.6% importance suspecte
        'sanitaires_adaptes',           # Data leakage: 8.9% importance - trop corrélé
        'entree_dispositif_appel',      # Data leakage: variable dérivée de l'accessibilité
        'entree_reperage',              # Data leakage: indicateur d'accessibilité
        'entree_vitree_vitrophanie',    # Data leakage: dérivée de entree_vitree
        # Variables géographiques/identification (pas pertinentes pour accessibilité)
        'latitude',                     # Position GPS non pertinente
        'longitude',                    # Position GPS non pertinente
        'name',                         # Nom établissement non pertinent
        'postal_code',                  # Code postal non pertinent
        'commune'                       # Commune non pertinente
    ]
    
    # Sélectionner les features (toutes sauf les cibles)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Features disponibles: {len(feature_cols)} colonnes")
    print(f"Premières features: {feature_cols[:10]}")
    
    # Préparer X (features)
    X = df[feature_cols].copy()
    
    # Encoder les colonnes catégorielles restantes
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':  # Colonne catégorielle
            print(f"Encodage de la colonne catégorielle: {col}")
            # Remplacer les valeurs manquantes
            X[col] = X[col].fillna('Unknown')
            # Encoder
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col].astype(str))
    
    # Gérer les valeurs manquantes restantes pour les colonnes numériques
    if X.isnull().sum().sum() > 0:
        print(f"valeurs manquantes détectées: {X.isnull().sum().sum()}")
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(0)
    
    # Préparer y (variable cible) - GARDER SEULEMENT LES VALEURS EXPLICITES
    if target_type == "binaire":
        # Filtrer pour garder seulement les True/False explicites
        mask_explicit = df['entree_pmr'].notna()
        df_filtered = df[mask_explicit].copy()
        X = X[mask_explicit].copy()
        
        y = df_filtered['entree_pmr'].copy()
        # Encoder en 0/1 pour le ML
        y = y.astype(bool).astype(int)  # True->1, False->0
        target_name = 'entree_pmr'
        
        print(f"Données filtrées: {len(df_filtered):,} lignes avec cible explicite")
        print(f"Variable cible binaire encodée: {y.value_counts().to_dict()}")
    else:
        y = df['accessibilite_categorie'].copy()
        target_name = 'accessibilite_categorie'
        print(f"Variable cible catégorielle: {y.value_counts().to_dict()}")
    
    print(f"\nFormes finales: X={X.shape}, y={y.shape}")
    print(f"Types de données X: {X.dtypes.value_counts().to_dict()}")
    
    return X, y, feature_cols, target_name

def compare_algorithms_pedagogique(X, y, target_name, mode="complet"):
    """Compare différents algorithmes ML - Approche pédagogique d'un étudiant"""
    
    print("\ncomparaison d'algorithmes")
    print("objectif: tester plusieurs algorithmes pour choisir le meilleur")
    
    # Division train/test stratifiée
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDivision des données:")
    print(f"- Entraînement: {X_train.shape[0]:,} échantillons")
    print(f"- Test: {X_test.shape[0]:,} échantillons")
    print(f"- Features: {X_train.shape[1]} variables")
    
    # Normalisation pour certains algorithmes
    # IMPORTANT: Tous les algorithmes n'ont pas besoin de normalisation !
    # - Tree-based (Random Forest, XGBoost) : robustes aux échelles différentes
    # - Linear-based (Logistic Regression, SVM) : sensibles aux échelles
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Définir les algorithmes à tester avec pondération des classes
    algorithms = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'scaled': True,
            'description': 'A voir si c est aussi simple'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
            'scaled': False,
            'description': ' plus robuste'
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=100),
            'scaled': False,
            'description': 'Plus performant'
        }
    }
    
    # SVM uniquement sur échantillon (trop lent sur dataset complet)
    if mode.lower() == "echantillon":
        algorithms['SVM'] = {
            'model': SVC(random_state=42, probability=True),
            'scaled': True,
            'description': 'Puissant pour classification complexe (échantillon uniquement)'
        }
        print("SVM inclus (mode échantillon)")
    else:
        print("SVM exclu (trop lent sur dataset complet - >10min pour performance similaire)")
    
    results = {}
    
    print("\ntests des algorithmes:")
   
    
    print(f"\nEntraînement de {len(algorithms)} modèles...")
    
    for i, (name, config) in enumerate(algorithms.items(), 1):
        print(f"\n[{i}/{len(algorithms)}] {name}: {config['description']}")
        
        try:
            # Choisir les données (normalisées ou non)
            # Adaptation intelligente selon l'algorithme :
            if config['scaled']:
                X_tr, X_te = X_train_scaled, X_test_scaled  # Pour LogReg, SVM
                print("   → Données normalisées (algorithme sensible aux échelles)")
            else:
                X_tr, X_te = X_train, X_test  # Pour RF, XGBoost
                print("   → Données brutes (algorithme robuste aux échelles)")
            
            # Entraînement
            print("Entraînement en cours...")
            model = config['model']
            model.fit(X_tr, y_train)
            
            # Prédictions
            y_pred = model.predict(X_te)
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Validation croisée
            print("Validation croisée (3 verifications  pour réduire le surapprentissage...")
            cv_scores = cross_val_score(model, X_tr, y_train, cv=3, scoring='f1_weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'scaled': config['scaled']
            }
            
            print(f"  accuracy: {accuracy:.3f}")
            print(f"  f1-score: {f1:.3f}")
            print(f"  cv f1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            
        except Exception as e:
            print(f"  erreur: {str(e)}")
            results[name] = None
    
    return results, X_test, y_test, scaler

def analyze_results_and_select_best(results):
    """Analyse les résultats et sélectionne le meilleur modèle - Approche étudiante"""
    
    print("\nanalyse des resultats")
    
    
    # Filtrer les résultats valides
    valid_results = {name: res for name, res in results.items() if res is not None}
    
    if not valid_results:
        print("aucun modele n'a pu etre entraine avec succes")
        return None, None
    
    # Créer un tableau de comparaison
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1_score'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'CV F1 Mean': res['cv_mean'],
            'CV F1 Std': res['cv_std']
        }
        for name, res in valid_results.items()
    }).T
    
    print("\ntableau comparatif:")
    print(comparison_df.round(3))
    
    # Sélection du meilleur modèle (critère principal: F1-score)
    best_name = comparison_df['F1-Score'].idxmax()
    best_model_info = valid_results[best_name]
    
    print(f"\nmeilleur modele: {best_name}")
    print(f"   f1-score: {best_model_info['f1_score']:.3f}")
    print(f"   accuracy: {best_model_info['accuracy']:.3f}")
    print(f"   cv f1: {best_model_info['cv_mean']:.3f} (±{best_model_info['cv_std']:.3f})")
    
    # Justification du choix
    print(f"\njustification du choix:")
    if best_name == 'Logistic Regression':
        print("   modele simple et interpretable")
        print("   rapide a entrainer et predire")
        print("   bon pour comprendre l'impact des variables")
    elif best_name == 'Random Forest':
        print("   robuste aux outliers et valeurs manquantes")
        print("   gere bien les interactions entre variables")
        print("   fournit l'importance des features")
    elif best_name == 'XGBoost':
        print("   tres performant sur de nombreux datasets")
        print("   gere bien le desequilibre des classes")
        print("   optimise pour la performance")
    elif best_name == 'SVM':
        print("   efficace pour les donnees de haute dimension")
        print("   robuste au surapprentissage")
        print("   fonctionne bien avec normalisation")
    
    return best_name, best_model_info

def create_detailed_evaluation(best_name, best_model_info, X_test, y_test, feature_cols):
    """Évaluation détaillée du meilleur modèle"""
    
    print(f"\nevaluation detaillee - {best_name}")
        
    model = best_model_info['model']
    
    # Prédictions
    if best_model_info['scaled']:
        # Le modèle a besoin de données normalisées
        scaler = StandardScaler()
        X_test_processed = scaler.fit_transform(X_test)
    else:
        X_test_processed = X_test
    
    y_pred = model.predict(X_test_processed)
    
    # Rapport de classification détaillé
    print("\nrapport de classification:")
    target_names = ['Non Accessible', 'Accessible'] if len(np.unique(y_test)) == 2 else None
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nmatrice de confusion:")
    print(cm)
    
    # Analyse des erreurs
    print(f"\nanalyse des erreurs:")
    total_errors = len(y_test) - accuracy_score(y_test, y_pred, normalize=False)
    print(f"   Erreurs totales: {total_errors}/{len(y_test)} ({total_errors/len(y_test)*100:.1f}%)")
    
    # Importance des features (si disponible)
    if hasattr(model, 'feature_importances_'):
        print(f"\nimportance des variables:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("   Top 10 variables les plus importantes:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature'][:30]:30} : {row['importance']:.3f}")
    
    elif hasattr(model, 'coef_'):
        print(f"\ncoefficients du modele:")
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("   Top 10 variables les plus influentes:")
        for i, (_, row) in enumerate(coef_df.head(10).iterrows()):
            direction = "+" if row['coefficient'] > 0 else "-"
            print(f"   {i+1:2d}. {row['feature'][:25]:25} : {row['coefficient']:+.3f} {direction}")

    return y_pred

def apply_smote_if_needed(X, y, apply_smote=False):
    """Applique SMOTE pour rééquilibrer les classes si nécessaire"""

    print(f"\ngestion du desequilibre des classes")
    
    # Analyser la distribution actuelle
    class_counts = Counter(y)
    print(f"Distribution actuelle: {dict(class_counts)}")

    # Calculer le ratio de déséquilibre
    minority_class = min(class_counts.values())
    majority_class = max(class_counts.values())
    imbalance_ratio = majority_class / minority_class

    print(f"Ratio de déséquilibre: {imbalance_ratio:.1f}:1")

    if not apply_smote:
        print("smote desactive - conservation du desequilibre naturel")
        print("   Raison: Préserver la distribution réelle des données")
        return X, y

    if imbalance_ratio > 3:
        print(f"application de smote (ratio > 3:1)")

        try:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            new_class_counts = Counter(y_balanced)
            print(f"Nouvelle distribution: {dict(new_class_counts)}")
            print(f"Nouvelles données: {len(X_balanced):,} échantillons (+{len(X_balanced)-len(X):,})")

            return X_balanced, y_balanced

        except Exception as e:
            print(f"erreur smote: {str(e)}")
            print("   Utilisation des données originales")
            return X, y
    else:
        print(f"desequilibre acceptable (ratio < 3:1)")
        return X, y

def create_learning_summary(best_name, best_model_info, mode):
    """Crée un résumé pédagogique de l'apprentissage"""

    print(f"\nresume pedagogique - mode {mode.upper()}")
    
    print(f"meilleur algorithme: {best_name}")
    print(f"   performance: f1-score = {best_model_info['f1_score']:.3f}")
    print(f"   stabilite: cv = {best_model_info['cv_mean']:.3f} (±{best_model_info['cv_std']:.3f})")

    print(f"\napprentissages cles:")
    print(f"   1. comparaison de 4 algorithmes differents")
    print(f"   2. importance de la validation croisee")
    print(f"   3. gestion du desequilibre des classes")
    print(f"   4. normalisation selon l'algorithme")

 

def save_best_model(best_name, best_model_info, feature_cols, mode):
    """Sauvegarde le meilleur modèle avec ses métadonnées"""
    
    import pickle
    from datetime import datetime
    
    model_filename = "Model_bloc3.pkl"
    
    # Supprimer l'ancien modèle s'il existe
    if os.path.exists(model_filename):
        try:
            os.remove(model_filename)
            print(f"ancien modèle supprimé: {model_filename}")
        except Exception as e:
            print(f"erreur suppression ancien modèle: {str(e)}")
    
    # Préparer les métadonnées du modèle
    model_data = {
        'model': best_model_info['model'],
        'model_name': best_name,
        'performance': {
            'f1_score': best_model_info['f1_score'],
            'accuracy': best_model_info['accuracy'],
            'precision': best_model_info['precision'],
            'recall': best_model_info['recall'],
            'cv_mean': best_model_info['cv_mean'],
            'cv_std': best_model_info['cv_std']
        },
        'features': feature_cols,
        'scaled': best_model_info['scaled'],
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'data_leakage_corrected': True,
        'excluded_features': ['accessibilite_categorie', 'score_accessibilite', 'stationnement_pmr']
    }
    
    # Sauvegarder le modèle
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nmodele sauvegarde: {model_filename}")
        print(f"   algorithme: {best_name}")
        print(f"   f1-score: {best_model_info['f1_score']:.3f}")
        print(f"   features: {len(feature_cols)} variables")
        print(f"   correction data leakage: oui")
        
        return model_filename
        
    except Exception as e:
        print(f"erreur sauvegarde modele: {str(e)}")
        return None

def plot_learning_curve(model, X, y, title, cv=3):
    """trace la courbe d'apprentissage pour evaluer le surapprentissage"""
    print(f"\ngeneration courbe d'apprentissage: {title}")
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("taille de l'echantillon d'entrainement")
    plt.ylabel("score f1")
    
    # tailles d'echantillons pour la courbe
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, 
            scoring='f1', random_state=42, n_jobs=-1
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid(True, alpha=0.3)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="entrainement")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="validation")
        plt.legend(loc="best")
        
        # sauvegarder le graphique
        filename = f'learning_curve_{title.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   courbe sauvegardee: {filename}")
        plt.show()
        
        return train_scores_mean, test_scores_mean
        
    except Exception as e:
        print(f"erreur generation courbe d'apprentissage: {str(e)}")
        plt.close()
        return None, None

def plot_feature_importance(model, feature_names, title="importance des variables", top_n=15):
    """visualise l'importance des features pour les modeles tree-based"""
    print(f"\ngeneration graphique importance: {title}")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"{title} - top {top_n}")
        plt.barh(range(top_n), importances[indices][::-1])
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel("importance")
        plt.grid(True, alpha=0.3)
        
        # sauvegarder le graphique
        filename = f'feature_importance_{title.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   graphique sauvegarde: {filename}")
        plt.show()
        
        return importances[indices], [feature_names[i] for i in indices]
    else:
        print("modele ne supporte pas l'importance des features")
        return None, None

def plot_xgboost_loss(X_train, y_train, X_test, y_test):
    """trace la courbe de loss pour xgboost pendant l'entrainement"""
    print(f"\ngeneration courbe de loss xgboost")
    
    # créer le modèle avec eval_set pour tracer le loss
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # entraîner avec suivi du loss
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # récupérer les résultats d'évaluation
    results = model.evals_result()
    
    # tracer les courbes de loss
    plt.figure(figsize=(10, 6))
    plt.title("Évolution du Loss - XGBoost")
    plt.xlabel("Itération (n_estimators)")
    plt.ylabel("Log Loss")
    
    epochs = range(len(results['validation_0']['logloss']))
    plt.plot(epochs, results['validation_0']['logloss'], 'r-', label='Train Loss')
    plt.plot(epochs, results['validation_1']['logloss'], 'g-', label='Test Loss')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # sauvegarder
    filename = 'xgboost_loss_curve.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   courbe loss sauvegardee: {filename}")
    plt.show()
    
    return model, results

def main(mode="echantillon", apply_smote=False, target_type="binaire"):
    """Fonction principale - Pipeline complet de modeling"""

    print(f"\nscript d'entrainement ml - acceslibre")
    print(f"mode: {mode.upper()} | smote: {'on' if apply_smote else 'off'} | cible: {target_type}")
    
    # Étape 1: Chargement des données préparées
    print("\n[1/6] Chargement des données")
    df = load_prepared_data(mode)
    if df is None:
        print("echec du chargement des donnees")
        return

    # Étape 2: Préparation des features
    print("\n[2/6] Préparation des features")
    X, y, feature_cols, target_name = prepare_features_for_ml(df, target_type)
    
    # Sauvegarde des données préparées complètes pour le Deep Learning (dossier 5)
    print("\n[2.5/6] Sauvegarde données préparées complètes")
    df_complet = X.copy()
    df_complet[target_name] = y
    
    # Chemin de sauvegarde dans le répertoire racine
    csv_path = "../data_prepared_COMPLET.csv"
    df_complet.to_csv(csv_path, index=False)
    print(f"Données complètes sauvegardées: {csv_path}")
    print(f"   {len(df_complet):,} lignes, {len(df_complet.columns)} colonnes")
    print(f"   Features: {len(feature_cols)}, Cible: {target_name}")

    # Étape 3: Gestion du déséquilibre (optionnel)
    print("\n[3/6] Gestion du déséquilibre")
    X, y = apply_smote_if_needed(X, y, apply_smote)

    # Étape 4: Comparaison des algorithmes
    print("\n[4/6] Entraînement des modèles")
    results, X_test, y_test, scaler = compare_algorithms_pedagogique(X, y, target_name, mode)

    # Étape 5: Sélection du meilleur modèle
    print("\n[5/6] Analyse des résultats")
    best_name, best_model_info = analyze_results_and_select_best(results)

    if best_name is None:
        print("aucun modele valide trouve")
        return

    # Étape 6: Évaluation détaillée
    print("\n[6/6] Évaluation détaillée")
    y_pred = create_detailed_evaluation(best_name, best_model_info, X_test, y_test, feature_cols)

    # Sauvegarde du meilleur modèle
    model_file = save_best_model(best_name, best_model_info, feature_cols, mode)

    # Résumé pédagogique
    create_learning_summary(best_name, best_model_info, mode)

    # Tracer la courbe d'apprentissage
    plot_learning_curve(best_model_info['model'], X, y, f"Courbe d'apprentissage - {best_name}")

    # Tracer l'importance des features
    plot_feature_importance(best_model_info['model'], feature_cols, f"Importance des variables - {best_name}")

    # Tracer la courbe de loss pour XGBoost
    print(f"\ngeneration courbe de loss xgboost (pour demonstration)")
    plot_xgboost_loss(X, y, X_test, y_test)

    print(f"\n PIPELINE ML TERMINÉ")
    print(f"   Modèle sauvegardé: {model_file}")
    print(f"   Meilleur algorithme: {best_name}")
    print(f"   Performance: F1 = {best_model_info['f1_score']:.3f}")

if __name__ == "__main__":
    # Configuration par défaut
    MODE = "complet"  # ou "echantillon"
    APPLY_SMOTE = False   # Désactivé - utilisation de class_weight='balanced' à la place
    TARGET_TYPE = "binaire"  # ou "categorielle"

    print(f"\n Config:")
    print(f"   Mode: {MODE}")
    print(f"   SMOTE: {'Activé' if APPLY_SMOTE else 'Désactivé'}")
    print(f"   Type de cible: {TARGET_TYPE}")

    main(mode=MODE, apply_smote=APPLY_SMOTE, target_type=TARGET_TYPE)
