#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modele yolo pour inference seulement
version simplifiee sans dependances dataset pour l'api
"""

import torch
import torch.nn as nn

class yolo_detection_head(nn.Module):
    """tete de detection yolo simplifiee"""
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # sortie : (x, y, w, h, conf, classes) par ancre
        out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.conv(x)

class simple_yolo(nn.Module):
    """architecture yolo simplifiee pour detection accessibilite"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # backbone cnn simple
        self.backbone = nn.Sequential(
            # bloc 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # bloc 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # bloc 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # bloc 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # bloc 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # tete de detection
        self.detection_head = yolo_detection_head(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections

class accessibility_yolo_inference:
    """version inference du modele yolo"""
    
    def __init__(self, num_classes=4, device=None):
        self.num_classes = num_classes
        self.device = device or self._get_device()
        self.model = simple_yolo(num_classes)
        self.model.to(self.device)
        
        # mapping labels
        self.label_map = {0: 'step', 1: 'stair', 2: 'grab_bar', 3: 'ramp'}
        
        print(f"modele yolo inference cree sur device: {self.device}")
    
    def _get_device(self):
        """detection automatique du device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            # pour amd gpu avec directml
            try:
                import torch_directml
                if torch_directml.is_available():
                    return torch_directml.device()
            except ImportError:
                pass
            return torch.device('cpu')
    
    def load_model(self, model_path):
        """charge un modele sauvegarde"""
        try:
            # chargement sur cpu d'abord pour eviter problemes device
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"modele charge depuis: {model_path}")
                print(f"epoch: {checkpoint.get('epoch', 'inconnu')}")
                print(f"loss: {checkpoint.get('loss', 'inconnu')}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"modele charge depuis: {model_path}")
            
            # deplacer vers le bon device apres chargement
            self.model.to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"erreur chargement modele: {e}")
            return False
    
    def predict(self, image_tensor):
        """prediction sur un tensor d'image"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(image_tensor.to(self.device))
        
        # retourner format compatible avec dossier 6
        return {
            'raw_predictions': predictions.cpu().numpy(),
            'predictions_tensor': predictions
        }
    
    def __call__(self, x):
        """permet d'utiliser l'objet comme une fonction"""
        return self.predict(x)
