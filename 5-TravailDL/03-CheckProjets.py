#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pas sur de mettre ce script pour le moment pour la soutenance, voir l'interet sans presenter deep learning ????

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime

print("Vérification complète du pipeline...")
print()

# Configuration
BASE_DIR = Path(__file__).parent
COMPUTER_VISION_DIR = BASE_DIR.parent / "6-DL-VisionComputer"

# Résultats de validation
validation_results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "scores": {},
    "status": "EN_COURS"
}

def test_file_exists(filepath, description):
    """test si un fichier existe"""
    exists = Path(filepath).exists()
    status = "pass" if exists else " Faux"
    print(f"{status} - {description}")
    return exists

def test_model_loading(model_path, description):
    """test le chargement d'un modele"""
    try:
        model = joblib.load(model_path)
        print(f"pass - {description}")
        return True, model
    except Exception as e:
        print(f" Faux - {description} - erreur: {str(e)[:50]}...")
        return False, None

def test_data_quality(data_path, description):
    """Test la qualité des données"""
    try:
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        else:
            return False, None
            
        # Tests qualité
        n_rows, n_cols = data.shape
        missing_pct = (data.isnull().sum().sum() / (n_rows * n_cols)) * 100
        
        quality_ok = n_rows > 1000 and missing_pct < 50
        status = "PASS" if quality_ok else "WARN"
        print(f"{status} {description} - {n_rows:,} lignes, {n_cols} cols, {missing_pct:.1f}% manquant")
        
        return quality_ok, {
            "rows": n_rows,
            "cols": n_cols,
            "missing_pct": missing_pct
        }
    except Exception as e:
        print(f"Faux {description} - Erreur: {str(e)[:50]}...")
        return False, None

print("Test 1: Structure du projet")
print("-" * 30)

# Test structure principale (pas sur de mettre pour le moment)
structure_tests = [
    (BASE_DIR / "01-DL_ok.py", "Script validation DL principal"),
    (BASE_DIR / "02-Diagnostic_surapprentissage.py", "Script diagnostic surapprentissage"),
    (BASE_DIR / "03-CheckProjets.py", "Script validation finale"),
    (BASE_DIR / "diagnostic_surapprentissage.png", "Graphique diagnostic"),
    (COMPUTER_VISION_DIR, "Dossier Computer Vision"),
]

structure_score = 0
for filepath, desc in structure_tests:
    if test_file_exists(filepath, desc):
        structure_score += 1

validation_results["tests"]["structure"] = {
    "score": structure_score,
    "total": len(structure_tests),
    "percentage": (structure_score / len(structure_tests)) * 100
}

print(f"\nScore Structure: {structure_score}/{len(structure_tests)} ({validation_results['tests']['structure']['percentage']:.1f}%)")

print("\nTest 2: Computer Vision")
# Test Computer Vision (YOLO)
cv_tests = [
    (COMPUTER_VISION_DIR / "3-yolov8_train.py", "Script YOLO entraînement"),
    (COMPUTER_VISION_DIR / "4-test_validation.py", "Script test validation YOLO"),
    (COMPUTER_VISION_DIR / "5-rapport_complet.txt", "Rapport complet YOLO"),
    (COMPUTER_VISION_DIR / "yolov8n.pt", "Modèle YOLO pré-entraîné"),
    (COMPUTER_VISION_DIR / "0-readme.md", "Documentation Computer Vision"),
]

cv_score = 0
for filepath, desc in cv_tests:
    if test_file_exists(filepath, desc):
        cv_score += 1

validation_results["tests"]["computer_vision"] = {
    "score": cv_score,
    "total": len(cv_tests),
    "percentage": (cv_score / len(cv_tests)) * 100
}

print(f"\nScore Computer Vision: {cv_score}/{len(cv_tests)} ({validation_results['tests']['computer_vision']['percentage']:.1f}%)")

print("\nTest 3: Modèles entraînés")


# Test modèles (mise à jour avec nouveaux modèles)
models_to_test = [
    (BASE_DIR.parent / "3-TravailModelisé" / "modele_logistic_regression.pkl", "Modèle ML Principal (Random Forest)"),
    (BASE_DIR / "Model_DL.pkl", "Modèle DL Principal (MLP)"),
    (COMPUTER_VISION_DIR / "yolov8n.pt", "Modèle YOLO pré-entraîné"),
    (COMPUTER_VISION_DIR / "models" / "best.pt", "Modèle YOLO entraîné"),
]

models_score = 0
loaded_models = {}

for model_path, desc in models_to_test:
    success, model = test_model_loading(model_path, desc)
    if success:
        models_score += 1
        loaded_models[desc] = model

validation_results["tests"]["models"] = {
    "score": models_score,
    "total": len(models_to_test),
    "percentage": (models_score / len(models_to_test)) * 100
}

print(f"\nScore Modèles: {models_score}/{len(models_to_test)} ({validation_results['tests']['models']['percentage']:.1f}%)")

print("\nTest 4: Résultats réels DL")

# Test des résultats réels obtenus
results_tests = {
    "ml_vs_dl_comparison": False,
    "overfitting_detection": False,
    "performance_validation": False,
    "documentation_complete": False
}

# Vérification comparaison ML vs DL
try:
    # Recherche des résultats dans la documentation
    doc_path = BASE_DIR / "docs" / "ANALYSE_01_DL_OK_RESULTATS_REELS.md"
    if doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "F1 = 0.9438" in content and "Random Forest" in content:
                results_tests["ml_vs_dl_comparison"] = True
                print("Comparaison ML vs DL documentée avec résultats réels")
            else:
                print("Comparaison ML vs DL incomplète")
    else:
        print("Documentation résultats réels manquante")
except Exception as e:
    print(f"Erreur vérification ML vs DL: {str(e)[:50]}...")

# Vérification diagnostic surapprentissage
try:
    graph_path = BASE_DIR / "diagnostic_surapprentissage.png"
    doc_path = BASE_DIR / "docs" / "DIAGNOSTIC_SURAPPRENTISSAGE_COMPLET.md"
    
    if graph_path.exists() and doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "5.8%" in content and "surapprentissage" in content.lower():
                results_tests["overfitting_detection"] = True
                print("Diagnostic surapprentissage avec visualisation")
            else:
                print("Diagnostic surapprentissage incomplet")
    else:
        print("Graphique ou documentation diagnostic manquant")
except Exception as e:
    print(f"Erreur vérification diagnostic: {str(e)[:50]}...")

# Vérification performance modèles
try:
    model_dl_path = BASE_DIR / "Model_DL.pkl"
    if model_dl_path.exists():
        # Vérification que le modèle DL a été entraîné récemment
        import os
        from datetime import datetime, timedelta
        
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_dl_path))
        if datetime.now() - mod_time < timedelta(days=7):
            results_tests["performance_validation"] = True
            print("Modèle DL récemment entraîné et sauvegardé")
        else:
            print("Modèle DL ancien (> 7 jours)")
    else:
        print("Modèle DL principal manquant")
except Exception as e:
    print(f"Erreur vérification modèle: {str(e)[:50]}...")

# Vérification documentation complète
try:
    docs_dir = BASE_DIR / "docs"
    required_docs = [
        "ANALYSE_01_DL_OK_RESULTATS_REELS.md",
        "DIAGNOSTIC_SURAPPRENTISSAGE_COMPLET.md",
        "RAPPORT_FINAL_DEEP_LEARNING.md"
    ]
    
    docs_found = 0
    for doc in required_docs:
        if (docs_dir / doc).exists():
            docs_found += 1
    
    if docs_found == len(required_docs):
        results_tests["documentation_complete"] = True
        print(f"Documentation complète ({docs_found}/{len(required_docs)} fichiers)")
    else:
        print(f"Documentation incomplète ({docs_found}/{len(required_docs)} fichiers)")
except Exception as e:
    print(f"Erreur vérification documentation: {str(e)[:50]}...")

# Score résultats réels
results_score = sum(results_tests.values())
results_total = len(results_tests)

validation_results["tests"]["real_results"] = {
    "score": results_score,
    "total": results_total,
    "percentage": (results_score / results_total) * 100,
    "details": results_tests
}

print(f"\nScore Résultats Réels: {results_score}/{results_total} ({validation_results['tests']['real_results']['percentage']:.1f}%)")

print("\nTest 5: DONNÉES")


# Test données (mise à jour avec nouveaux datasets)
data_files = [
    (BASE_DIR.parent / "data_prepared_ECHANTILLON.csv", "Dataset Échantillon"),
    (BASE_DIR.parent / "data_prepared_COMPLET.csv", "Dataset Complet"),
    (BASE_DIR.parent / "3-TravailModelisé" / "data_prepared_COMPLET.csv", "Dataset ML/DL Pipeline"),
]

data_score = 0
data_info = {}

for data_path, desc in data_files:
    success, info = test_data_quality(data_path, desc)
    if success:
        data_score += 1
        data_info[desc] = info

validation_results["tests"]["data_quality"] = {
    "score": data_score,
    "total": len(data_files),
    "percentage": (data_score / len(data_files)) * 100,
    "details": data_info
}

print(f"\nScore Données: {data_score}/{len(data_files)} ({validation_results['tests']['data_quality']['percentage']:.1f}%)")

print("\nTest 6: PERFORMANCE MODÈLES")


# Test performance (simulation avec données disponibles)
try:
    # Chargement données échantillon
    data_path = BASE_DIR.parent / "data_prepared_ECHANTILLON.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
        
        # Séparation features/target
        if 'accessibilite_pmr' in data.columns:
            X = data.drop(['accessibilite_pmr'], axis=1)
            y = data['accessibilite_pmr']
            
            # Test avec modèle principal si disponible
            model_path = BASE_DIR.parent / "3-TravailModelisé" / "modele_logistic_regression.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                
                # Prédiction sur échantillon
                sample_size = min(1000, len(X))
                X_sample = X.iloc[:sample_size]
                y_sample = y.iloc[:sample_size]
                
                predictions = model.predict(X_sample)
                accuracy = (predictions == y_sample).mean()
                
                print(f" Pass performance modèle - Accuracy: {accuracy:.3f}")
                
                validation_results["scores"]["model_accuracy"] = accuracy
                performance_score = 1 if accuracy > 0.8 else 0
            else:
                print("Probleme :Modèle principal non trouvé")
                performance_score = 0
        else:
            print("probleme : Colonne target non trouvée")
            performance_score = 0
    else:
        print("probleme : Données échantillon non trouvées")
        performance_score = 0
        
except Exception as e:
    print(f"Problème performance - Erreur: {str(e)[:50]}...")
    performance_score = 0

validation_results["tests"]["performance"] = {
    "score": performance_score,
    "total": 1,
    "percentage": performance_score * 100
}

print("\n\n")
print("Rapport Final de validation")


# Calcul score global
total_tests = sum(test["total"] for test in validation_results["tests"].values())
total_passed = sum(test["score"] for test in validation_results["tests"].values())
global_score = (total_passed / total_tests) * 100

validation_results["global_score"] = {
    "passed": total_passed,
    "total": total_tests,
    "percentage": global_score
}

print(f"Score global : {total_passed}/{total_tests} ({global_score:.1f}%)")
print()

# Détail par catégorie
#for category, results in validation_results["tests"].items():
#    status = "" if results["percentage"] >= 80 else "attention" if results["percentage"] >= 60 else "erreur"
#    print(f"{status} {category.upper()}: {results['score']}/{results['total']} ({results['percentage']:.1f}%)")

print()

# Statut final
if global_score >= 90:
    final_status = "EXCELLENT"
    validation_results["status"] = "EXCELLENT"
elif global_score >= 80:
    final_status = "VALIDÉ"
    validation_results["status"] = "VALIDE"
elif global_score >= 60:
    final_status = "ACCEPTABLE"
    validation_results["status"] = "ACCEPTABLE"
else:
    final_status = "INSUFFISANT"
    validation_results["status"] = "INSUFFISANT"

print(f" Statut du modele: {final_status}")
print()

# Recommandations
print("Recommandations:")
if validation_results["tests"]["structure"]["percentage"] < 100:
    print("- Vérifier la structure complète du projet")
if validation_results["tests"]["models"]["percentage"] < 100:
    print("- Réentraîner les modèles manquants")
if validation_results["tests"]["data_quality"]["percentage"] < 100:
    print("- Vérifier la qualité des datasets")
if validation_results["tests"]["performance"]["percentage"] < 100:
    print("- Optimiser les performances des modèles")

if global_score >= 80:
    print("Projet prêt pour présentation jury certification!")
else:
    print("Corrections nécessaires avant certification")

print()

# Sauvegarde rapport
report_path = BASE_DIR / "validation_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print(f"Rapport sauvegardé: {report_path}")
print()
print("Validation terminée")

