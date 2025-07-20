#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPLICATION DE DÉPLOIEMENT - ACCESSIBILITÉ PMR
Certification RNCP 38616 - Grégory LE TERTE

Application FastAPI pour déployer le modèle de prédiction d'accessibilité PMR
basé sur l'analyse d'images d'entrées d'établissements.
"""

import os
import sys
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
import logging

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", "6-DL-VisionComputer", "models", "yolov8_accessibility_full", "weights", "best.pt")
DETECTION_CORRECTOR_PATH = os.path.join(BASE_DIR, "detection_corrector.py")

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("acceslibre-api")

# Création de l'application FastAPI
app = FastAPI(
    title="AccesLibre API",
    description="API pour la prédiction d'accessibilité PMR à partir d'images d'entrées d'établissements",
    version="1.0.0"
)

# Configuration des fichiers statiques et templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Import des modules de détection
sys.path.append(os.path.join(BASE_DIR, "..", "6-DL-VisionComputer"))
from ultralytics import YOLO
from detection_corrector import DetectionCorrector

# Chargement du modèle YOLOv8
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    detection_corrector = DetectionCorrector()
    logger.info("✅ Modèle YOLOv8 et correcteur chargés avec succès")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle YOLOv8: {str(e)}")
    yolo_model = None
    detection_corrector = None

def detect_accessibility_elements(image):
    """
    detection des elements d'accessibilite avec yolov8
    
    args:
        image: image pil
        
    returns:
        dict: resultats de detection avec corrections
    """
    try:
        if yolo_model is None or detection_corrector is None:
            return {
                "success": False,
                "error": "modele yolov8 non disponible",
                "detections": [],
                "analysis": None
            }
        
        # conversion en array numpy pour yolo
        import cv2
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # detection yolov8
        results = yolo_model(img_rgb, conf=0.01, verbose=False)
        
        # extraction des detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # mapping des classes
                class_names = ['step', 'stair', 'ramp', 'handrail', 'grab_bar']
                class_name = class_names[cls_id] if cls_id < len(class_names) else 'unknown'
                
                detections.append({
                    'type': class_name,
                    'confiance': round(conf, 3),
                    'position': [int(x1), int(y1), int(x2), int(y2)],
                    'corrige': False
                })
        
        # application des corrections
        if detection_corrector:
            corrected_detections, corrections = detection_corrector.correct_detections(
                detections, img_array.shape[:2]
            )
            
            # analyse d'accessibilite
            analysis = detection_corrector.analyze_accessibility(corrected_detections)
            analysis['corrections_appliquees'] = corrections
        else:
            corrected_detections = detections
            analysis = {
                'score_accessibilite': 0.5,
                'niveau': 'inconnu',
                'elements_detectes': detections,
                'statistiques': {},
                'recommandations': ['modele de correction non disponible'],
                'corrections_appliquees': []
            }
        
        return {
            "success": True,
            "detections": corrected_detections,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"erreur lors de la detection: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "analysis": None
        }

def draw_detections(image, detections):
    """
    dessine les detections sur l'image
    
    args:
        image: image pil
        detections: liste des detections
        
    returns:
        image pil avec detections dessinees
    """
    try:
        import cv2
        from PIL import ImageDraw, ImageFont
        
        # conversion pour opencv
        img_array = np.array(image)
        img_draw = img_array.copy()
        
        # couleurs par classe
        colors = {
            'step': (255, 0, 0),      # rouge
            'stair': (0, 255, 0),     # vert
            'ramp': (0, 0, 255),      # bleu
            'handrail': (255, 255, 0), # jaune
            'grab_bar': (255, 0, 255)  # magenta
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['position']
            class_name = detection['type']
            confidence = detection['confiance']
            corrected = detection.get('corrige', False)
            
            # couleur selon la classe
            color = colors.get(class_name, (128, 128, 128))
            
            # rectangle
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # texte
            label = f"{class_name} {confidence:.2f}"
            if corrected:
                label += " (corr)"
            
            # fond du texte
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_draw, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1), 
                color, -1
            )
            
            # texte
            cv2.putText(
                img_draw, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # conversion retour en pil
        return Image.fromarray(img_draw)
        
    except Exception as e:
        logger.error(f"erreur lors du dessin: {str(e)}")
        return image

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "AccesLibre - Analyse d'accessibilité PMR"}
    )

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    detection d'elements d'accessibilite avec yolov8
    """
    import time
    start_time = time.time()
    
    try:
        # lecture de l'image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # detection des elements d'accessibilite
        detection_result = detect_accessibility_elements(image)
        
        if not detection_result["success"]:
            return {
                "error": detection_result["error"],
                "detections": [],
                "analysis": None,
                "inference_time_ms": 0
            }
        
        # dessin des detections sur l'image
        image_with_detections = draw_detections(image, detection_result["detections"])
        
        # conversion en base64
        buffered = BytesIO()
        image_with_detections.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # temps d'inference
        inference_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "detections": detection_result["detections"],
            "analysis": detection_result["analysis"],
            "image_with_detections": img_base64,
            "inference_time_ms": inference_time,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"erreur lors de la detection: {str(e)}")
        return {
            "error": str(e),
            "detections": [],
            "analysis": None,
            "inference_time_ms": 0
        }

@app.get("/health")
async def health_check():
    """verification de l'etat de l'api"""
    return {
        "status": "ok",
        "yolo_model_loaded": yolo_model is not None,
        "detection_corrector_loaded": detection_corrector is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # lancement du serveur
    logger.info("demarrage du serveur yolov8 accessibility api")
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
