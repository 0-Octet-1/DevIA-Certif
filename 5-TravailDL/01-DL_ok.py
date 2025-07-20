#!/usr/bin/env python3
"""
bloc 5 - deep learning fonctionnel
solution mlp avec scikit-learn
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """charge les donnees preparees"""
    print("\nchargement des donnees")
    
    file_path = "../data_prepared_COMPLET.csv"
    
    if not os.path.exists(file_path):
        print(f"fichier non trouve: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"dataset charge: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    return df

def create_deep_learning_models():
    """cree differentes architectures de reseaux de neurones"""
    
    models = {
        'MLP_Simple': {
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'description': 'mlp simple: 64 -> 32 -> 1'
        },
        
        'MLP_Profond': {
            'model': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'description': 'mlp profond: 128 -> 64 -> 32 -> 16 -> 1'
        },
        
        'MLP_Regularise': {
            'model': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='tanh',
                solver='lbfgs',
                alpha=0.01,  # Plus de régularisation
                max_iter=1000,
                random_state=42
            ),
            'description': 'mlp regularise: 100 -> 50 -> 1 (l2=0.01)'
        }
    }
    
    return models

def train_and_evaluate_dl(X_train, X_test, y_train, y_test, models):
    """entraine et evalue les modeles deep learning"""
    
    print(f"\nentrainement deep learning")
    print(f"donnees entrainement: {len(X_train):,} echantillons")
    print(f"donnees test: {len(X_test):,} echantillons")
    print(f"features: {X_train.shape[1]} colonnes")
    
    results = {}
    
    for i, (name, config) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] entrainement: {name}")
        print(f"   architecture: {config['description']}")
        
        # Timing précis de l'entraînement
        start_time = datetime.now()
        print(f"   debut: {start_time.strftime('%H:%M:%S')}")
        print(f"   entrainement en cours...")
        
        # Entraînement
        model = config['model']
        model.fit(X_train, y_train)
        
        # Fin du timing
        end_time = datetime.now()
        duration = end_time - start_time
        duration_minutes = duration.total_seconds() / 60
        print(f"   fin: {end_time.strftime('%H:%M:%S')}")
        print(f"   duree: {duration_minutes:.1f} minutes ({duration.total_seconds():.1f}s)")
        
        # Prédictions
        print(f"   calcul predictions...")
        y_pred = model.predict(X_test)
        
        # Métriques
        print(f"   calcul metriques...")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Validation croisée
        print(f"   validation croisee (3-fold)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        print(f"   validation terminee")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'description': config['description']
        }
        
        print(f"   f1-score: {f1:.4f}")
        print(f"   accuracy: {accuracy:.4f}")
        print(f"   cv f1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return results

def compare_ml_vs_dl(dl_results):
    """compare ml (random forest) vs deep learning"""
    
    print(f"\ncomparaison ml vs deep learning")
        
    # Chargement du modèle ML de référence
    ml_model_path = "../3-TravailModelisé/Model_bloc3.pkl"
    
    if os.path.exists(ml_model_path):
        # Charger le modèle et ses métadonnées
        try:
            with open(ml_model_path, 'rb') as f:
                ml_model_data = pickle.load(f)
            
            ml_name = ml_model_data['model_name']
            ml_f1 = ml_model_data['performance']['f1_score']
            
            print(f"machine learning ({ml_name} - bloc 3):")
            print(f"   f1-score: {ml_f1:.3f} (apres correction data leakage)")
            
        except Exception as e:
            print(f"erreur chargement modele ML: {str(e)}")
            ml_f1 = 0.805  # valeur de fallback
            print(f"machine learning (XGBoost - bloc 3):")
            print(f"   f1-score: {ml_f1:.3f} (fallback)")
        
        print(f"\ndeep learning (reseaux de neurones):")
        
        best_dl_name = None
        best_dl_f1 = 0
        
        for name, result in dl_results.items():
            f1 = result['f1_score']
            print(f"   {name}: F1 = {f1:.4f} - {result['description']}")
            
            if f1 > best_dl_f1:
                best_dl_f1 = f1
                best_dl_name = name
        
        print(f"\nmeilleur modele dl: {best_dl_name} (f1 = {best_dl_f1:.4f})")
        
        # Comparaison (ml_f1 déjà défini ci-dessus)
        diff = best_dl_f1 - ml_f1
        
        if diff > 0:
            print(f"deep learning superieur de {diff*100:.2f}%")
        else:
            print(f"machine learning superieur de {abs(diff)*100:.2f}%")
            print(f"   normal sur donnees tabulaires")
            print(f"   le dl excelle sur images/texte/sequences")
        
        return best_dl_name, dl_results[best_dl_name]
    
    else:
        print(f"modele ml de reference non trouve")
        return None, None

def main():
    """Fonction principale"""

    print("Deep learning fonctionnel")
    print("solution scikit-learn")
        
    # 1. Chargement des données
    df = load_data()
    if df is None:
        return
    
    # 2. Préparation des données
    print(f"\npreparation des donnees")
        
    # Features numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'entree_pmr'
    
    if target_col not in df.columns:
        print(f"colonne cible '{target_col}' non trouvee")
        return
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Testé ->  verif_data_leakage.py, pourquoi du datleakage??? 
    data_leakage_cols = ['accessibilite_categorie', 'score_accessibilite', 'stationnement_pmr']
    excluded_cols = [col for col in data_leakage_cols if col in numeric_cols]
    
    if excluded_cols:
        print(f"exclusion data leakage: {excluded_cols}")
        numeric_cols = [col for col in numeric_cols if col not in data_leakage_cols]
    
    X = df[numeric_cols]
    y = df[target_col]
    
    print(f"features: {len(X.columns)} colonnes numeriques")
    print(f"distribution cible: {dict(y.value_counts())}")
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation des datas 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"donnees divisees: train={len(X_train)}, test={len(X_test)}")
    print(f"normalisation appliquee (standardscaler)")
    
    # 3. Création des modèles Deep Learning
    models = create_deep_learning_models()
    
    # 4. Entraînement et évaluation
    dl_results = train_and_evaluate_dl(X_train_scaled, X_test_scaled, y_train, y_test, models)
    
    # 5. Comparaison ML vs DL
    best_name, best_model_data = compare_ml_vs_dl(dl_results)
    
    # 6. Sauvegarde du meilleur modèle
    if best_model_data:
        model_path = "Model_DL.pkl"
        
        model_data = {
            'model': best_model_data['model'],
            'scaler': scaler,
            'feature_names': X.columns.tolist(),
            'performance': {
                'f1_score': best_model_data['f1_score'],
                'accuracy': best_model_data['accuracy'],
                'cv_mean': best_model_data['cv_mean'],
                'cv_std': best_model_data['cv_std']
            },
            'architecture': best_model_data['description'],
            'timestamp': datetime.now().isoformat(),
            'framework': 'scikit-learn MLPClassifier'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nmodele sauvegarde: {model_path}")
        print(f"   architecture: {best_model_data['description']}")
        print(f"   performance: f1 = {best_model_data['f1_score']:.4f}")
    
    # 7. Résumé final
    print(f"\nbloc 5 deep learning termine")
    print(f"   framework: scikit-learn")
    print(f"   modeles testes: {len(models)} architectures")
    print(f"   meilleur modele: {best_name}")
    print(f"   comparaison: ml vs dl effectuee")
    print(f"   certification: objectifs atteints")

if __name__ == "__main__":
    main()
