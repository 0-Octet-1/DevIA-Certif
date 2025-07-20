#!/usr/bin/env python3
"""
Script de test du modèle ML - Prédiction d'accessibilité PMR
Charge le modèle sauvegardé et teste sur des exemples concrets
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def load_model_and_data():
    """charge le modèle et les données de test"""
    print("chargement du modèle sauvegardé...")
    
    try:
        # charger le modèle
        file_path_model = os.path.join('..', '3-TravailModelisé', 'Model_bloc3.pkl')
        model_data = joblib.load(file_path_model)
        
        # extraire le modèle et les features du dictionnaire
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_names = model_data.get('features', [])  # récupérer les features sauvegardées
            print(f"modèle chargé: {type(model).__name__}")
            print(f"algorithme: {model_data.get('model_name', 'inconnu')}")
            
            # récupérer le f1-score depuis performance
            performance = model_data.get('performance', {})
            f1_score = performance.get('f1_score', 'inconnu')
            print(f"f1-score: {f1_score}")
            print(f"features sauvegardées: {len(feature_names)} colonnes")
        else:
            model = model_data
            feature_names = []
            print(f"modèle chargé: {type(model).__name__}")
            print("attention: pas de métadonnées de features trouvées")
        
        # charger les données préparées pour récupérer la structure
        file_path = os.path.join('..', '2-TravailPreparationdesDonnées', 'data_prepared_EXPLORATION.csv')
        df = pd.read_csv(file_path)
        print(f"données chargées: {len(df)} lignes")
        
        return model, df, feature_names
        
    except Exception as e:
        print(f"erreur chargement: {e}")
        return None, None, []

def prepare_test_sample_simple(df, feature_names):
    """prépare un échantillon de test en utilisant les données déjà préparées"""
    
    print("préparation échantillon de test...")
    
    # chercher des établissements avec entree_pmr connu pour validation
    df_with_target = df[df['entree_pmr'].notna()].copy()
    
    if len(df_with_target) == 0:
        print("aucun établissement avec vérité terrain trouvé")
        sample_indices = np.random.choice(df.index, size=min(4, len(df)), replace=False)
        sample_original = df.loc[sample_indices].copy()
    else:
        # prendre un échantillon équilibré
        accessible = df_with_target[df_with_target['entree_pmr'] == True]
        non_accessible = df_with_target[df_with_target['entree_pmr'] == False]
        
        n_accessible = min(2, len(accessible))
        n_non_accessible = min(2, len(non_accessible))
        
        sample_parts = []
        if n_accessible > 0:
            sample_parts.append(accessible.sample(n=n_accessible, random_state=42))
            print(f"trouvé {n_accessible} établissements accessibles")
        
        if n_non_accessible > 0:
            sample_parts.append(non_accessible.sample(n=n_non_accessible, random_state=42))
            print(f"trouvé {n_non_accessible} établissements non accessibles")
        
        if sample_parts:
            sample_original = pd.concat(sample_parts, ignore_index=True)
        else:
            sample_indices = np.random.choice(df.index, size=min(4, len(df)), replace=False)
            sample_original = df.loc[sample_indices].copy()
    
    print(f"échantillon final: {len(sample_original)} établissements")
    
    # utiliser exactement les features sauvegardées dans le bon ordre
    if feature_names and len(feature_names) > 0:
        print(f"utilisation des {len(feature_names)} features sauvegardées")
        
        # vérifier que toutes les features sont disponibles
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            print(f"erreur: features manquantes: {missing_features}")
            return None, None, None
        
        # utiliser exactement les features dans l'ordre sauvegardé
        test_samples = sample_original[feature_names].copy()
        
        # ENCODER LES VARIABLES CATEGORIELLES (même logique que 03_model_training.py)
        from sklearn.preprocessing import LabelEncoder
        
        print("encodage des variables catégorielles...")
        for col in test_samples.columns:
            if test_samples[col].dtype == 'object':  # variable catégorielle
                print(f"  encodage colonne: {col}")
                # remplacer les valeurs manquantes
                test_samples[col] = test_samples[col].fillna('Unknown')
                # encoder avec LabelEncoder
                le = LabelEncoder()
                test_samples[col] = le.fit_transform(test_samples[col].astype(str))
        
        # gérer les valeurs manquantes numériques
        test_samples = test_samples.fillna(0)
        
        # vérifier que tout est numérique maintenant
        for col in test_samples.columns:
            if test_samples[col].dtype == 'object':
                print(f"attention: colonne {col} encore textuelle")
        
        print(f"données préparées: {test_samples.shape}")
        
    else:
        print("erreur: pas de features sauvegardées")
        return None, None, None
    
    return test_samples, feature_names, sample_original

def test_model_predictions(model, test_samples, original_df):
    """teste le modèle sur les échantillons avec détails complets"""
    print(f"\ntest du modèle sur {len(test_samples)} échantillons:")
    print("=" * 80)
    
    # prédictions avec seuil optimisé (correction du biais 94% non accessible)
    probabilities = model.predict_proba(test_samples)
    
    # appliquer le seuil optimal de 0.2 (au lieu de 0.5 par défaut)
    # amélioration f1-score: 0.208 → 0.373 (+79%)
    optimal_threshold = 0.35
    predictions = (probabilities[:, 1] >= optimal_threshold).astype(int)
    
    print(f"🎯 seuil de décision optimisé: {optimal_threshold} (correction biais)")
    print(f"   amélioration f1-score: +79% vs modèle standard")
    
    # afficher les résultats détaillés
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        result = "ACCESSIBLE" if pred == 1 else "NON ACCESSIBLE"
        confidence = prob[pred] * 100
        
        # récupérer les infos de l'établissement original
        etablissement = original_df.iloc[i]
        
        print(f"\n🏢 ÉTABLISSEMENT {i+1}:")
        print(f"   Nom: {etablissement.get('name', 'Non renseigné')}")
        print(f"   Activité: {etablissement.get('activite', 'Non renseigné')}")
        print(f"   Commune: {etablissement.get('commune', 'Non renseigné')}")
        
        print(f"\n📊 CARACTÉRISTIQUES D'ACCESSIBILITÉ:")
        print(f"   Largeur entrée: {etablissement.get('entree_largeur_mini', 'Non renseigné')} cm")
        print(f"   Nombre de marches: {etablissement.get('entree_marches', 'Non renseigné')}")
        
        # sanitaires
        sanitaires = etablissement.get('sanitaires_presence')
        sanitaires_txt = 'Oui' if sanitaires == 1 else 'Non' if sanitaires == 0 else 'Non renseigné'
        print(f"   Sanitaires présents: {sanitaires_txt}")
        
        # stationnement PMR
        parking = etablissement.get('stationnement_pmr')
        parking_txt = 'Oui' if parking == 1 else 'Non' if parking == 0 else 'Non renseigné'
        print(f"   Stationnement PMR: {parking_txt}")
        
        # transport
        transport = etablissement.get('transport_station_presence')
        transport_txt = 'Oui' if transport == 1 else 'Non' if transport == 0 else 'Non renseigné'
        print(f"   Transport accessible: {transport_txt}")
        
        print(f"\n🎯 PRÉDICTION DU MODÈLE (seuil {optimal_threshold}):")
        print(f"   Résultat: {result}")
        print(f"   Probabilité accessible: {prob[1]:.3f}")
        print(f"   Seuil standard (0.5): {'ACCESSIBLE' if prob[1] >= 0.5 else 'NON ACCESSIBLE'}")
        print(f"   Seuil optimisé ({optimal_threshold}): {result}")
        
        # vérité terrain si disponible
        if 'entree_pmr' in etablissement and pd.notna(etablissement['entree_pmr']):
            vraie_valeur = "ACCESSIBLE" if etablissement['entree_pmr'] else "NON ACCESSIBLE"
            correct = "✅ CORRECT" if (etablissement['entree_pmr'] and pred == 1) or (not etablissement['entree_pmr'] and pred == 0) else "❌ INCORRECT"
            print(f"   Vérité terrain: {vraie_valeur} {correct}")
        
        print("-" * 80)

def diagnostic_modele(model, feature_cols):
    """diagnostic approfondi du comportement du modèle"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC DU MODÈLE")
    print("=" * 80)
    
    # charger quelques vrais échantillons accessibles du dataset
    print("chargement d'échantillons réels accessibles...")
    df_train = pd.read_csv('../2-TravailPreparationdesDonnées/data_prepared_EXPLORATION.csv')
    
    # prendre des échantillons réels accessibles
    vrais_accessibles = df_train[df_train['entree_pmr'] == True].head(3)
    
    print(f"trouvé {len(vrais_accessibles)} échantillons réels accessibles")
    
    # afficher leurs caractéristiques
    total_tests = 0
    tests_reussis = 0
    
    for i, (idx, row) in enumerate(vrais_accessibles.iterrows()):
        total_tests += 1
        
        print(f"\n📊 ÉCHANTILLON RÉEL ACCESSIBLE {i+1}:")
        print(f"   Nom: {row.get('name', 'N/A')}")
        print(f"   Activité: {row.get('activite', 'N/A')}")
        print(f"   Largeur entrée: {row.get('entree_largeur_mini', 'N/A')} cm")
        print(f"   Marches: {row.get('entree_marches', 'N/A')}")
        print(f"   Sanitaires: {row.get('sanitaires_presence', 'N/A')}")
        print(f"   Stationnement PMR: {row.get('stationnement_pmr', 'N/A')}")
        print(f"   Transport: {row.get('transport_station_presence', 'N/A')}")
        
        # préparer pour prédiction
        test_row = row[feature_cols].copy()
        
        # encoder les variables catégorielles
        from sklearn.preprocessing import LabelEncoder
        for col in test_row.index:
            if df_train[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df_train[col].astype(str))
                test_row[col] = le.transform([str(test_row[col])])[0]
        
        # gérer les valeurs manquantes
        test_row = test_row.fillna(0)
        
        # prédiction avec seuil optimisé
        prob = model.predict_proba([test_row])[0]
        
        # appliquer le seuil optimal de 0.35
        optimal_threshold = 0.35
        pred = 1 if prob[1] >= optimal_threshold else 0
        
        result = "ACCESSIBLE" if pred == 1 else "NON ACCESSIBLE"
        prob_accessible = prob[1]
        
        # vérifier si c'est correct
        is_correct = (pred == 1)  # car on teste des échantillons accessibles
        if is_correct:
            tests_reussis += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"\n🎯 PRÉDICTION SUR CET ÉCHANTILLON RÉEL:")
        print(f"   Résultat: {result}")
        print(f"   Confiance: {prob_accessible*100:.1f}%")
        print(f"   Probabilités: Non={prob[0]:.3f}, Oui={prob[1]:.3f}")
        print(f"   Vérité: ACCESSIBLE {status}")
        print("-" * 60)
    
    # afficher le taux de réussite global
    taux_reussite = (tests_reussis / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 TAUX DE RÉUSSITE SUR ÉCHANTILLONS RÉELS ACCESSIBLES:")
    print(f"   Tests réussis: {tests_reussis}/{total_tests}")
    print(f"   Taux de réussite: {taux_reussite:.1f}%")
    
    if taux_reussite >= 70:
        print(f"   Évaluation: ✅ EXCELLENT (≥70%)")
    elif taux_reussite >= 50:
        print(f"   Évaluation: ⚠️ CORRECT (≥50%)")
    else:
        print(f"   Évaluation: ❌ INSUFFISANT (<50%)")

    # test avec un échantillon "parfait"
    print(f"\n🧪 TEST AVEC ÉCHANTILLON 'PARFAIT':")
    
    # créer un échantillon avec toutes les meilleures valeurs
    perfect_sample = {}
    for col in feature_cols:
        if col == 'activite':
            # utiliser la valeur la plus fréquente chez les accessibles
            activites_accessibles = df_train[df_train['entree_pmr'] == True]['activite'].value_counts()
            if len(activites_accessibles) > 0:
                perfect_sample[col] = activites_accessibles.index[0]
            else:
                perfect_sample[col] = 'Commerce'
        elif col == 'entree_largeur_mini':
            perfect_sample[col] = 150  # très large
        elif col == 'entree_marches':
            perfect_sample[col] = 0    # pas de marches
        elif col in ['sanitaires_presence', 'stationnement_pmr', 'transport_station_presence', 
                     'cheminement_ext_presence', 'entree_plain_pied', 'stationnement_ext_pmr']:
            perfect_sample[col] = 1    # tout présent
        else:
            # valeur par défaut
            perfect_sample[col] = 1
    
    print(f"   Activité: {perfect_sample['activite']}")
    print(f"   Largeur: {perfect_sample['entree_largeur_mini']} cm")
    print(f"   Marches: {perfect_sample['entree_marches']}")
    print(f"   Sanitaires: {perfect_sample['sanitaires_presence']}")
    print(f"   Stationnement: {perfect_sample['stationnement_pmr']}")
    
    # encoder et prédire avec seuil optimisé
    test_perfect = pd.Series(perfect_sample)
    for col in test_perfect.index:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(df_train[col].astype(str))
            try:
                test_perfect[col] = le.transform([str(test_perfect[col])])[0]
            except:
                test_perfect[col] = 0  # valeur par défaut si inconnue
    
    # prédiction avec seuil optimisé
    prob_perfect = model.predict_proba([test_perfect])[0]
    
    # appliquer le seuil optimal de 0.2
    optimal_threshold = 0.35
    pred_perfect = 1 if prob_perfect[1] >= optimal_threshold else 0
    
    result_perfect = "ACCESSIBLE" if pred_perfect == 1 else "NON ACCESSIBLE"
    prob_accessible_perfect = prob_perfect[1]
    
    print(f"\n PRÉDICTION SUR L'ÉCHANTILLON PARFAIT (seuil {optimal_threshold}):")
    print(f"   Résultat: {result_perfect}")
    print(f"   Probabilité accessible: {prob_accessible_perfect:.3f}")
    print(f"   Seuil standard (0.5): {'ACCESSIBLE' if prob_accessible_perfect >= 0.5 else 'NON ACCESSIBLE'}")
    print(f"   Seuil optimisé ({optimal_threshold}): {result_perfect}")
    
    # analyser les importances des features
    if hasattr(model, 'feature_importances_'):
        print(f"\n TOP 10 VARIABLES LES PLUS IMPORTANTES:")
        importances = model.feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"   {i+1:2d}. {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    

def create_custom_test(model, feature_cols):
    """crée un test avec des valeurs personnalisées"""
    print("\n" + "=" * 80)
    print("TEST AVEC ÉTABLISSEMENTS PERSONNALISÉS")
    print("=" * 80)
    
    # charger le dataset pour l'encodage
    df_train = pd.read_csv('../2-TravailPreparationdesDonnées/data_prepared_EXPLORATION.csv')
    
    # exemple d'établissement accessible
    etablissement_accessible = {
        'entree_largeur_mini': 120,     
        'entree_marches': 0,             
        'sanitaires_presence': 1,        
        'stationnement_pmr': 1,          
        'activite': 'Commerce',          
        'transport_station_presence': 1   
    }
    
    # exemple d'établissement non accessible
    etablissement_non_accessible = {
        'entree_largeur_mini': 60,       
        'entree_marches': 5,             
        'sanitaires_presence': 0,        
        'stationnement_pmr': 0,          
        'activite': 'Restaurant',        
        'transport_station_presence': 0   
    }
    
    etablissements = [
        ('ACCESSIBLE', etablissement_accessible),
        ('NON ACCESSIBLE', etablissement_non_accessible)
    ]
    
    for attendu, etab in etablissements:
        print(f"\n ÉTABLISSEMENT THÉORIQUEMENT {attendu}:")
        print(f"   Largeur entrée: {etab['entree_largeur_mini']} cm")
        print(f"   Nombre de marches: {etab['entree_marches']}")
        print(f"   Sanitaires présents: {'Oui' if etab['sanitaires_presence'] else 'Non'}")
        print(f"   Stationnement PMR: {'Oui' if etab['stationnement_pmr'] else 'Non'}")
        print(f"   Activité: {etab['activite']}")
        print(f"   Transport accessible: {'Oui' if etab['transport_station_presence'] else 'Non'}")
        
        # préparer les données pour le modèle
        test_data = {}
        for col in feature_cols:
            if col in etab:
                test_data[col] = etab[col]
            else:
                # valeurs par défaut pour les colonnes manquantes
                if col in ['entree_marches_main_courante', 'entree_marches_rampe', 'entree_marches_reperage', 'entree_marches_sens']:
                    test_data[col] = 0
                elif col in ['accueil_retrecissement', 'accueil_visibilite', 'accueil_cheminement_plain_pied']:
                    test_data[col] = 1
                elif col in ['entree_porte_manoeuvre', 'entree_porte_type', 'entree_porte_presence']:
                    test_data[col] = 1
                elif col in ['entree_ascenseur']:
                    test_data[col] = 0
                elif col in ['stationnement_presence', 'stationnement_ext_presence']:
                    test_data[col] = 1
                else:
                    test_data[col] = 0
        
        # convertir en Series et encoder
        test_series = pd.Series(test_data)
        
        # encoder les variables catégorielles
        from sklearn.preprocessing import LabelEncoder
        for col in test_series.index:
            if df_train[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df_train[col].astype(str))
                try:
                    test_series[col] = le.transform([str(test_series[col])])[0]
                except:
                    test_series[col] = 0  # valeur par défaut si inconnue
        
        # faire la prédiction avec seuil optimisé
        prob = model.predict_proba([test_series])[0]
        
        # appliquer le seuil optimal de 0.2
        optimal_threshold = 0.35
        pred = 1 if prob[1] >= optimal_threshold else 0
        
        result = "ACCESSIBLE" if pred == 1 else "NON ACCESSIBLE"
        prob_accessible = prob[1]
        correct = "✅ CORRECT" if result == attendu else "❌ INCORRECT"
        
        print(f"\n PRÉDICTION DU MODÈLE (seuil {optimal_threshold}):")
        print(f"   Résultat: {result}")
        print(f"   Probabilité accessible: {prob_accessible:.3f}")
        print(f"   Seuil standard (0.5): {'ACCESSIBLE' if prob_accessible >= 0.5 else 'NON ACCESSIBLE'}")
        print(f"   Seuil optimisé ({optimal_threshold}): {result}")
        print(f"   Attendu: {attendu} {correct}")
        

def main():
    """fonction principale de test du modèle"""
    
    print("TEST DU MODÈLE ML - ACCESSIBILITÉ PMR")
    
    
    # charger le modèle
    model, df, feature_names = load_model_and_data()
    if model is None:
        return
    
    # préparer les échantillons de test
    test_samples, _, sample_original = prepare_test_sample_simple(df, feature_names)
    
    if test_samples is None:
        print("erreur lors de la préparation des échantillons de test")
        return
    
    # tester le modèle
    test_model_predictions(model, test_samples, sample_original)
    
    # diagnostic du modèle
    diagnostic_modele(model, feature_names)
    
    # exemple personnalisé
    create_custom_test(model, feature_names)
    
    print("TEST TERMINÉ")

if __name__ == "__main__":
    main()
