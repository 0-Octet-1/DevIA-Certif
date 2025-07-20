#!/usr/bin/env python3
"""
VÉRIFICATION DATA LEAKAGE
Détecte les features suspectes qui pourraient causer la performance parfaite
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

def analyze_features():
    """Analyse les features pour détecter le data leakage"""
    print("🔍 VÉRIFICATION DATA LEAKAGE")
    print("=" * 50)
    
    df = pd.read_csv("../data_prepared_COMPLET.csv")
    
    # Target
    target_col = 'entree_pmr'
    y = df[target_col]
    
    # Features numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols]
    
    print(f"📊 Features analysées: {len(numeric_cols)}")
    print(f"📊 Target: {target_col}")
    
    # 1. Corrélations avec la target
    print(f"\n🔗 CORRÉLATIONS AVEC LA TARGET:")
    print("-" * 40)
    
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(y)
        correlations.append((col, abs(corr)))
    
    # Tri par corrélation décroissante
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 corrélations les plus fortes:")
    for i, (col, corr) in enumerate(correlations[:10], 1):
        status = "🚨" if corr > 0.9 else "⚠️" if corr > 0.7 else "✅"
        print(f"{i:2d}. {col:<30} {corr:.4f} {status}")
    
    # 2. Information mutuelle
    print(f"\n🧠 INFORMATION MUTUELLE:")
    print("-" * 40)
    
    # Normalisation pour l'information mutuelle
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcul information mutuelle
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    
    # Tri par information mutuelle
    mi_results = list(zip(numeric_cols, mi_scores))
    mi_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 features les plus informatives:")
    for i, (col, score) in enumerate(mi_results[:10], 1):
        status = "🚨" if score > 1.5 else "⚠️" if score > 1.0 else "✅"
        print(f"{i:2d}. {col:<30} {score:.4f} {status}")
    
    # 3. Features parfaitement corrélées (suspects)
    print(f"\n🚨 FEATURES SUSPECTES (corrélation > 90%):")
    print("-" * 50)
    
    suspects = [col for col, corr in correlations if corr > 0.9]
    
    if suspects:
        print("⚠️  Features avec corrélation > 90% (possibles fuites):")
        for col in suspects:
            corr = df[col].corr(y)
            print(f"   • {col}: {corr:.4f}")
            
            # Analyse des valeurs
            unique_vals = df[col].nunique()
            print(f"     → Valeurs uniques: {unique_vals}")
            
            if unique_vals <= 10:
                print(f"     → Distribution: {dict(df[col].value_counts().head())}")
    else:
        print("✅ Aucune feature avec corrélation > 90%")
    
    # 4. Analyse des noms de colonnes
    print(f"\n🏷️  ANALYSE DES NOMS DE FEATURES:")
    print("-" * 40)
    
    suspicious_keywords = ['id', 'key', 'index', 'target', 'label', 'result', 'outcome']
    
    suspicious_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        for keyword in suspicious_keywords:
            if keyword in col_lower:
                suspicious_cols.append((col, keyword))
    
    if suspicious_cols:
        print("⚠️  Noms de colonnes suspects:")
        for col, keyword in suspicious_cols:
            print(f"   • {col} (contient '{keyword}')")
    else:
        print("✅ Noms de colonnes normaux")
    
    return correlations, mi_results, suspects

def main():
    """Fonction principale"""
    print("🕵️‍♂️ DÉTECTION DATA LEAKAGE - DEEP LEARNING")
    print("=" * 60)
    
    try:
        correlations, mi_results, suspects = analyze_features()
        
        print(f"\n📋 RÉSUMÉ:")
        print("-" * 20)
        
        if suspects:
            print(f"🚨 {len(suspects)} feature(s) suspecte(s) détectée(s)")
            print("   → Corrélation > 90% avec la target")
            print("   → Possible data leakage !")
            print("\n💡 RECOMMANDATIONS:")
            print("   1. Vérifier l'origine de ces features")
            print("   2. S'assurer qu'elles sont disponibles AVANT la prédiction")
            print("   3. Les exclure si elles contiennent l'info future")
        else:
            print("✅ Pas de data leakage évident détecté")
            print("   → Performance élevée peut être légitime")
            print("   → Dataset de qualité avec features pertinentes")
        
        print(f"\n🎯 CONCLUSION:")
        if len(suspects) > 0:
            print("   Performance parfaite = Data leakage probable")
        else:
            print("   Performance parfaite = Dataset de très haute qualité")
            print("   → Félicitations pour le preprocessing ! 🎉")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
