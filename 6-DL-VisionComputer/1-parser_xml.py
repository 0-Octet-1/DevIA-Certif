import xml.etree.ElementTree as ET
import os
from collections import defaultdict, Counter
import json

class annotation_parser:
    """parser pour les annotations cvat xml"""
    
    def __init__(self, xml_path, images_dir):
        self.xml_path = xml_path
        self.images_dir = images_dir
        self.labels_map = {
            'step': 0,
            'stair': 1, 
            'grab_bar': 2,
            'ramp': 3
        }
        self.annotations = []
        self.stats = defaultdict(int)
        
    def parse_xml(self):
        """extrait toutes les annotations du fichier xml"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        for image_elem in root.findall('image'):
            image_info = self._parse_image(image_elem)
            if image_info:
                self.annotations.append(image_info)
                
        self._compute_stats()
        return self.annotations
    
    def _parse_image(self, image_elem):
        """parse une image et ses boites"""
        image_name = image_elem.get('name')
        if not image_name.startswith('images/'):
            return None
            
        # enlever le prefixe 'images/'
        image_name = image_name.replace('images/', '')
        image_path = os.path.join(self.images_dir, image_name)
        
        if not os.path.exists(image_path):
            return None
            
        width = int(image_elem.get('width'))
        height = int(image_elem.get('height'))
        
        boxes = []
        for box_elem in image_elem.findall('box'):
            box_info = self._parse_box(box_elem, width, height)
            if box_info:
                boxes.append(box_info)
        
        return {
            'image_name': image_name,
            'image_path': image_path,
            'width': width,
            'height': height,
            'boxes': boxes
        }
    
    def _parse_box(self, box_elem, img_width, img_height):
        """parse une boite et ses attributs"""
        label = box_elem.get('label')
        if label not in self.labels_map:
            return None
            
        # coordonnees absolues
        xtl = float(box_elem.get('xtl'))
        ytl = float(box_elem.get('ytl'))
        xbr = float(box_elem.get('xbr'))
        ybr = float(box_elem.get('ybr'))
        
        # normalisation [0,1]
        x_center = (xtl + xbr) / (2 * img_width)
        y_center = (ytl + ybr) / (2 * img_height)
        box_width = (xbr - xtl) / img_width
        box_height = (ybr - ytl) / img_height
        
        # attributs
        attributes = {}
        for attr_elem in box_elem.findall('attribute'):
            attr_name = attr_elem.get('name')
            attr_value = attr_elem.text
            attributes[attr_name] = attr_value
            
        return {
            'label': label,
            'label_id': self.labels_map[label],
            'bbox_abs': [xtl, ytl, xbr, ybr],
            'bbox_norm': [x_center, y_center, box_width, box_height],
            'attributes': attributes
        }
    
    def _compute_stats(self):
        """calcule les statistiques du dataset"""
        self.stats['total_images'] = len(self.annotations)
        self.stats['images_with_boxes'] = sum(1 for ann in self.annotations if ann['boxes'])
        self.stats['total_boxes'] = sum(len(ann['boxes']) for ann in self.annotations)
        
        # stats par label
        label_counts = Counter()
        attribute_stats = defaultdict(Counter)
        
        for ann in self.annotations:
            for box in ann['boxes']:
                label = box['label']
                label_counts[label] += 1
                
                # stats attributs
                for attr_name, attr_value in box['attributes'].items():
                    attribute_stats[f"{label}_{attr_name}"][attr_value] += 1
        
        self.stats['labels'] = dict(label_counts)
        self.stats['attributes'] = dict(attribute_stats)
    
    def print_stats(self):
        """affiche les statistiques"""
        print(f"total images: {self.stats['total_images']}")
        print(f"images avec annotations: {self.stats['images_with_boxes']}")
        print(f"total boites: {self.stats['total_boxes']}")
        print()
        
        print("repartition par label:")
        for label, count in self.stats['labels'].items():
            print(f"  {label}: {count}")
        print()
        
        print("attributs par label:")
        for attr_key, values in self.stats['attributes'].items():
            print(f"  {attr_key}:")
            for value, count in values.items():
                print(f"    {value}: {count}")
        print()
    
    def save_annotations(self, output_path):
        """sauvegarde les annotations parsees"""
        data = {
            'annotations': self.annotations,
            'stats': self.stats,
            'labels_map': self.labels_map
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"annotations sauvegardees: {output_path}")

def main():
    """test du parser"""
    xml_path = "data/wm_annotations.xml"
    images_dir = "data/images"
    
    parser = annotation_parser(xml_path, images_dir)
    annotations = parser.parse_xml()
    
    parser.print_stats()
    parser.save_annotations("data/parsed_annotations.json")

if __name__ == "__main__":
    main()
