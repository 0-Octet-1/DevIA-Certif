#!/usr/bin/env python3
"""
DIAGNOSTIC SURAPPRENTISSAGE
Analyse si les modèles Deep Learning sont en surapprentissage
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def load_and_analyze_data():
    """Charge et analyse les données"""
    print("Analyse des données")
    
    df = pd.read_csv("../data_prepared_COMPLET.csv")
    
    # Analyse de base
    print(f"Dataset: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    # Target
    target_col = 'entree_pmr'
    y = df[target_col]
    
    print(f"Distribution cible:")
    for val, count in y.value_counts().items():
        pct = count / len(y) * 100
        print(f"  {val}: {count:,} ({pct:.1f}%)")
    
    # Déséquilibre ?
    class_ratio = y.value_counts().min() / y.value_counts().max()
    print(f"Ratio déséquilibre: {class_ratio:.3f}")
    
    if class_ratio < 0.1:
        print("Déséquilibre extrême détecté !")
    elif class_ratio < 0.3:
        print("Déséquilibre modéré")
    else:
        print("Classes relativement équilibrées")
    
    return df

def test_overfitting():
    """Test spécifique du surapprentissage"""
    print("\nTest de surapprentissage")
        
    df = pd.read_csv("../data_prepared_COMPLET.csv")
    
    # Préparation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'entree_pmr'
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Exclusion data leakage (comme dans les scripts principaux)
    leakage_vars = ['accessibilite_categorie', 'score_accessibilite', 'stationnement_pmr']
    for var in leakage_vars:
        if var in numeric_cols:
            numeric_cols.remove(var)

    X = df[numeric_cols]
    y = df[target_col]
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèle simple pour test (paramètres alignés avec courbes d'apprentissage)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,  # Assez d'itérations pour voir le surapprentissage
        random_state=42,
        early_stopping=False,  # Désactivé pour voir le surapprentissage
        validation_fraction=0.2
    )
    
    print("Entraînement du modèle...")
    model.fit(X_train_scaled, y_train)
    
    # Prédictions sur train et test
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Métriques
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nRésultats:")
    print(f"Train F1: {train_f1:.4f}")
    print(f"Test F1:  {test_f1:.4f}")
    print(f"Écart F1: {abs(train_f1 - test_f1):.4f}")
    
    print(f"\nTrain Acc: {train_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Écart Acc: {abs(train_acc - test_acc):.4f}")
    
    # Diagnostic unifié
    f1_gap = abs(train_f1 - test_f1)
    acc_gap = abs(train_acc - test_acc)
    
    print(f"\nDiagnostic:")
    
    # Seuils plus réalistes pour détecter le surapprentissage
    if f1_gap > 0.03 or acc_gap > 0.03:
        print("Surapprentissage détecté !")
        print("   → Écart train/test > 3%")
        print("   → Solutions: régularisation, dropout, early stopping")
    elif f1_gap > 0.015 or acc_gap > 0.015:
        print("Léger surapprentissage")
        print("   → Écart train/test > 1.5%")
        print("   → Surveillance recommandée")
    else:
        print("Pas de surapprentissage détecté")
        print("   → Modèle bien généralisé")
    
    # Performance suspecte ?
    if test_f1 > 0.999:
        print("\n Bizarre la performance, a checker les datas:")
        print(f"   → F1 = {test_f1:.4f} (99.9%+)")
        print("   → Vérifier la qualité des données")
        print("   → Possible fuite de données (data leakage)")
    
    return train_f1, test_f1, train_acc, test_acc

def analyze_learning_curves():
    """Analyse les courbes d'apprentissage avec visualisation"""
    print("\nCourbes d'apprentissage")
        
    df = pd.read_csv("../data_prepared_COMPLET.csv")
    
    # Préparation (échantillon pour rapidité)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'entree_pmr'
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Exclusion data leakage (cohérence avec autres scripts)
    leakage_vars = ['accessibilite_categorie', 'score_accessibilite', 'stationnement_pmr']
    for var in leakage_vars:
        if var in numeric_cols:
            numeric_cols.remove(var)
    
    # Échantillon pour rapidité
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    X = df_sample[numeric_cols]
    y = df_sample[target_col]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modèle (paramètres alignés avec test principal)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        random_state=42,
        early_stopping=False  # Pour voir l'évolution complète
    )
    
    print("🔄 Calcul des courbes d'apprentissage...")
    
    # Courbes d'apprentissage
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_scaled, y, 
        cv=3, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1',
        n_jobs=-1
    )
    
    # Moyennes et écarts-types
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Création du graphique
    plt.figure(figsize=(12, 8))
    
    # Courbes avec zones d'incertitude
    plt.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Score Entraînement')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2E86AB')
    
    plt.plot(train_sizes, val_mean, 'o-', color='#F24236', linewidth=2, markersize=6, label='Score Validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#F24236')
    
    # Mise en forme
    plt.xlabel('Taille du jeu d\'entraînement', fontsize=12, fontweight='bold')
    plt.ylabel('Score F1', fontsize=12, fontweight='bold')
    plt.title('Courbes d\'Apprentissage - Diagnostic Surapprentissage\nMLP (64→32) - Dataset Accessibilité PMR', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Annotations diagnostiques
    final_gap = abs(train_mean[-1] - val_mean[-1])
    
    # Zone de surapprentissage si détecté
    if final_gap > 0.05:
        plt.axhspan(val_mean[-1], train_mean[-1], alpha=0.2, color='red', 
                   label=f'Zone surapprentissage ({final_gap:.1%})')
        diagnostic_text = "Surapprentissage détecté"
        diagnostic_color = '#E74C3C'
    elif final_gap > 0.02:
        diagnostic_text = "Léger surapprentissage"
        diagnostic_color = '#F39C12'
    else:
        diagnostic_text = "Apprentissage optimal"
        diagnostic_color = '#27AE60'
    
    # Annotation du diagnostic
    plt.text(0.02, 0.98, diagnostic_text, transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', color=diagnostic_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Statistiques dans le coin
    stats_text = f"Écart final: {final_gap:.1%}\nTrain: {train_mean[-1]:.3f}\nValid: {val_mean[-1]:.3f}"
    plt.text(0.02, 0.15, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarde du graphique
    output_path = "diagnostic_surapprentissage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_path}")
    
    # Affichage si possible
    try:
        plt.show()
    except:
        print("📱 Affichage graphique non disponible (mode serveur)")
    
    plt.close()
    
    # Analyse textuelle
    print(f"\nANALYSE DES COURBES:")
    print(f"Score final train: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
    print(f"Score final valid: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
    print(f"Écart final: {final_gap:.4f} ({final_gap:.1%})")
    
    if final_gap > 0.05:
        print("Courbes divergentes → Surapprentissage confirmé")
        print("   → Recommandation: Early stopping, régularisation L2")
    elif final_gap > 0.02:
        print("Léger écart → Surveillance recommandée")
        print("   → Recommandation: Monitoring continu")
    else:
        print("Courbes convergentes → Apprentissage optimal")
        print("   → Modèle bien généralisé")
    
    return output_path

def main():
    """Fonction principale"""
    print("Diagnostic surapprentissage - Deep Learning")
        
    try:
        # 1. Analyse des données
        load_and_analyze_data()
        
        # 2. Test surapprentissage
        test_overfitting()
        
        # 3. Courbes d'apprentissage avec visualisation
        graph_path = analyze_learning_curves()
        
        print(f"\nDiagnostic terminé")
        print(f"   → Vérifiez les écarts train/test")
        print(f"   → Performance > 99% = suspect")
        print(f"   → Solutions: régularisation, early stopping")
        print(f"   → Graphique généré: {graph_path}")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
