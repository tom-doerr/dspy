#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/tom/git/x_twitter_production')
from pathlib import Path
import json

training_data_dir = "/home/tom/git/x_twitter_production/data/training_data"
annotations_dir = Path(training_data_dir) / "annotations"
screenshots_dir = Path(training_data_dir) / "screenshots"

print(f"Annotations dir exists: {annotations_dir.exists()}")
print(f"Screenshots dir exists: {screenshots_dir.exists()}")

ann_files = list(annotations_dir.glob("*_annotation.json"))
print(f"Found {len(ann_files)} annotation files")

if ann_files:
    # Check first annotation
    ann_file = ann_files[0]
    print(f"\nChecking first annotation: {ann_file.name}")
    
    with open(ann_file) as f:
        ann = json.load(f)
    
    print(f"Keys in annotation: {list(ann.keys())}")
    print(f"Image ID: {ann.get('image_id')}")
    
    if 'image_id' in ann:
        img_path = screenshots_dir / f"{ann['image_id']}_original.png"
        print(f"Screenshot path: {img_path}")
        print(f"Screenshot exists: {img_path.exists()}")
    
    print(f"Bbox: {ann.get('bbox')}")
    print(f"Image size: {ann.get('image_size')}")