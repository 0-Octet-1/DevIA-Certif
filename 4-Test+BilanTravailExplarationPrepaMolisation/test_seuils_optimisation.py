#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test et optimisation des seuils de décision
Projet: Prédiction accessibilité PMR
Auteur: Certification RNCP 38616
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def charger_modele_et_donnees():
    """Charge le modèle et les données de test"""
    print("🔄 Chargement du modèle et des données...")
    
    # Charger le modèle
    with open('../3-TravailModelisé/Model_bloc3.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features_list = model_data['features']
    
    # Charger les données préparées
    file_path = os.path.join('..', '2-TravailPreparationdesDonnées', 'data_prepared_EXPLORATION.csv')
    df = pd.read_csv(file_path)
    
    print(f"✅ Modèle chargé: {type(model).__name__}")
    print(f"✅ Dataset: {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    
    return model, features_list, df

def preparer_donnees_test(df, features_list):
    """Prépare les données de test avec le même encodage que l'entraînement"""
    print("🔄 Préparation des données de test...")
    
    # Filtrer les données avec target non-nulle
    df_test = df[df['entree_pmr'].notna()].copy()
    
    # Encoder les variables catégorielles
    label_encoders = {}
    for col in df_test.columns:
        if df_test[col].dtype == 'object' and col in features_list:
            le = LabelEncoder()
            df_test[col] = le.fit_transform(df_test[col].astype(str))
            label_encoders[col] = le
    
    # Sélectionner les features
    X_test = df_test[features_list]
    y_test = df_test['entree_pmr'].astype(int)
    
    print(f"✅ Données de test: {X_test.shape[0]:,} échantillons")
    print(f"✅ Distribution: {y_test.value_counts().to_dict()}")
    
    return X_test, y_test

def tester_seuils(model, X_test, y_test, seuils=None):
    """Teste différents seuils et calcule les métriques"""
    if seuils is None:
        seuils = np.arange(0.1, 0.55, 0.05)
    
    print(f"🔄 Test de {len(seuils)} seuils...")
    
    # Prédictions probabilistes
    y_proba = model.predict_proba(X_test)[:, 1]
    
    resultats = []
    
    for seuil in seuils:
        # Prédictions avec seuil personnalisé
        y_pred = (y_proba >= seuil).astype(int)
        
        # Calcul des métriques
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        resultats.append({
            'seuil': seuil,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
        
        print(f"Seuil {seuil:.2f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    return pd.DataFrame(resultats)

def creer_graphique_seuils(resultats_df):
    """Crée un graphique comparatif des seuils"""
    print("📊 Création du graphique comparatif...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1-Score
    ax1.plot(resultats_df['seuil'], resultats_df['f1_score'], 'o-', linewidth=2, markersize=6)
    ax1.set_title('F1-Score par Seuil', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Seuil de décision')
    ax1.set_ylabel('F1-Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Meilleur seuil
    best_idx = resultats_df['f1_score'].idxmax()
    best_seuil = resultats_df.loc[best_idx, 'seuil']
    best_f1 = resultats_df.loc[best_idx, 'f1_score']
    ax1.axvline(x=best_seuil, color='red', linestyle='--', alpha=0.7)
    ax1.annotate(f'Optimal: {best_seuil:.2f}\nF1: {best_f1:.3f}', 
                xy=(best_seuil, best_f1), xytext=(best_seuil+0.05, best_f1-0.1),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
    
    # Precision vs Recall
    ax2.plot(resultats_df['seuil'], resultats_df['precision'], 'o-', label='Precision', linewidth=2)
    ax2.plot(resultats_df['seuil'], resultats_df['recall'], 's-', label='Recall', linewidth=2)
    ax2.set_title('Precision vs Recall par Seuil', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Seuil de décision')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Vrais/Faux Positifs
    ax3.plot(resultats_df['seuil'], resultats_df['tp'], 'o-', label='Vrais Positifs', linewidth=2)
    ax3.plot(resultats_df['seuil'], resultats_df['fp'], 's-', label='Faux Positifs', linewidth=2)
    ax3.set_title('Vrais vs Faux Positifs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Seuil de décision')
    ax3.set_ylabel('Nombre')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Vrais/Faux Négatifs
    ax4.plot(resultats_df['seuil'], resultats_df['tn'], 'o-', label='Vrais Négatifs', linewidth=2)
    ax4.plot(resultats_df['seuil'], resultats_df['fn'], 's-', label='Faux Négatifs', linewidth=2)
    ax4.set_title('Vrais vs Faux Négatifs', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Seuil de décision')
    ax4.set_ylabel('Nombre')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = 'optimisation_seuils_comparaison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {filename}")
    
    plt.show()
    
    return best_seuil, best_f1

def tester_etablissements_fictifs(model, features_list, seuils_a_tester):
    """Teste les établissements fictifs avec différents seuils"""
    print("🏢 Test des établissements fictifs...")
    
    # Établissements de test
    etablissements = [
        {
            'nom': 'Restaurant Parfait',
            'activite': 'Restaurant',
            'entree_largeur_mini': 120,
            'entree_marches': 0,
            'sanitaires_presence': 1,
            'stationnement_pmr': 1,
            'transport_station_presence': 1,
            'attendu': 'ACCESSIBLE'
        },
        {
            'nom': 'Café Problématique',
            'activite': 'Restaurant', 
            'entree_largeur_mini': 75,
            'entree_marches': 3,
            'sanitaires_presence': 0,
            'stationnement_pmr': 0,
            'transport_station_presence': 0,
            'attendu': 'NON ACCESSIBLE'
        },
        {
            'nom': 'Pharmacie Accessible',
            'activite': 'Commerce',
            'entree_largeur_mini': 100,
            'entree_marches': 0,
            'sanitaires_presence': 1,
            'stationnement_pmr': 1,
            'transport_station_presence': 1,
            'attendu': 'ACCESSIBLE'
        }
    ]
    
    # Créer DataFrame avec toutes les features nécessaires
    test_data = []
    for etab in etablissements:
        # Initialiser avec des valeurs par défaut
        row = {feature: 0 for feature in features_list}
        
        # Remplir avec les valeurs spécifiées
        for key, value in etab.items():
            if key in features_list:
                if key == 'activite':
                    # Encoder l'activité (approximation)
                    row[key] = 1 if value == 'Restaurant' else 2
                else:
                    row[key] = value
        
        test_data.append(row)
    
    X_fictif = pd.DataFrame(test_data)
    
    # Prédictions
    y_proba_fictif = model.predict_proba(X_fictif)[:, 1]
    
    print("\n" + "="*80)
    print("RÉSULTATS SUR ÉTABLISSEMENTS FICTIFS")
    print("="*80)
    
    for i, etab in enumerate(etablissements):
        print(f"\n🏢 {etab['nom'].upper()}:")
        print(f"   Largeur: {etab['entree_largeur_mini']}cm, Marches: {etab['entree_marches']}")
        print(f"   Sanitaires: {'Oui' if etab['sanitaires_presence'] else 'Non'}")
        print(f"   Probabilité accessible: {y_proba_fictif[i]:.3f}")
        print(f"   Attendu: {etab['attendu']}")
        
        print("   Prédictions par seuil:")
        for seuil in seuils_a_tester:
            pred = "ACCESSIBLE" if y_proba_fictif[i] >= seuil else "NON ACCESSIBLE"
            correct = "✅" if pred == etab['attendu'] else "❌"
            print(f"     Seuil {seuil:.2f}: {pred} {correct}")

def main():
    """Fonction principale"""
    print("="*80)
    print("🎯 OPTIMISATION DES SEUILS DE DÉCISION")
    print("="*80)
    
    try:
        # 1. Charger modèle et données
        model, features_list, df = charger_modele_et_donnees()
        
        # 2. Préparer données de test
        X_test, y_test = preparer_donnees_test(df, features_list)
        
        # 3. Tester différents seuils
        seuils = np.arange(0.1, 0.55, 0.05)
        resultats = tester_seuils(model, X_test, y_test, seuils)
        
        # 4. Créer graphique et identifier le meilleur seuil
        best_seuil, best_f1 = creer_graphique_seuils(resultats)
        
        # 5. Afficher le résumé
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DE L'OPTIMISATION")
        print("="*80)
        print(f"🎯 Meilleur seuil: {best_seuil:.2f}")
        print(f"🎯 Meilleur F1-Score: {best_f1:.3f}")
        
        # Top 3 des seuils
        top3 = resultats.nlargest(3, 'f1_score')
        print(f"\n🏆 TOP 3 DES SEUILS:")
        for i, row in top3.iterrows():
            print(f"   {row['seuil']:.2f}: F1={row['f1_score']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")
        
        # 6. Tester sur établissements fictifs
        seuils_test = [0.2, best_seuil, 0.3, 0.5]
        tester_etablissements_fictifs(model, features_list, seuils_test)
        
        # 7. Recommandations
        print("\n" + "="*80)
        print("💡 RECOMMANDATIONS")
        print("="*80)
        
        if best_seuil < 0.25:
            print("⚠️  Seuil optimal très bas - Risque de faux positifs élevé")
            print("💡 Considérer des règles métier complémentaires")
        elif best_seuil > 0.4:
            print("⚠️  Seuil optimal élevé - Risque de manquer des établissements accessibles")
            print("💡 Vérifier la qualité des données d'entraînement")
        else:
            print("✅ Seuil optimal dans une plage raisonnable")
        
        print(f"\n🎯 Seuil recommandé pour la production: {best_seuil:.2f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
