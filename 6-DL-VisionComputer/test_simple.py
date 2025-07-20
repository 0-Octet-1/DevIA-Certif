#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test simple et direct du modele yolov8
"""

from ultralytics import YOLO
import cv2

# charger modele
model = YOLO('models/yolov8_accessibility_trained.pt')

# prediction avec seuil tres bas
results = model('C:/Users/grego/Downloads/Chapeau_230_972_img_9486.jpg', conf=0.01)
result = results[0]

# afficher resultats
print(f"detections: {len(result.boxes) if result.boxes else 0}")

if result.boxes:
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        print(f"{i+1}. {name}: {conf:.3f}")

# sauvegarder image
annotated = result.plot()
cv2.imwrite('test_outputs/final_test.jpg', annotated)
print("image sauvee: test_outputs/final_test.jpg")
