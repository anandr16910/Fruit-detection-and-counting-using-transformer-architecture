#----------
#IMPORTANT: please load the dataset given here [https://drive.google.com/drive/folders/1NpmRvD39MCmF7CS71nCTM1wVsbyGF919?usp=drive_link] alongwith this code to work.
#-----
import os
import sys
import glob
import json
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import defaultdict, Counter

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QMessageBox, QSlider, QVBoxLayout, QListWidget, QDialog
)
from PyQt5.QtCore import Qt

from ultralytics import YOLO
import logging

# ---------------------------
# 1. Silence Logs
# ---------------------------
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# ---------------------------
# 2. YOLO Counter Class
# ---------------------------
class YOLOv5ObjectCounter:
    def __init__(self, model_path="yolov5s.pt", device="mps"):
        # Auto-download YOLOv5s if not present
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device

    def _run(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(
            img_rgb,
            conf=0.1,        
            iou=0.5,
            imgsz=640,
            max_det=300,
            agnostic_nms=True,
            device=self.device,
            verbose=False    
        )
        return results[0]

    def count_objects(self, image_bgr):
        r = self._run(image_bgr)
        return float(len(r.boxes))

    def density_map(self, image_bgr):
        h, w = image_bgr.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        r = self._run(image_bgr)
        if r.boxes is None or len(r.boxes) == 0:
            return heat

        boxes = r.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            heat[y1:y2, x1:x2] += 1.0

        if heat.max() > 0:
            heat /= heat.max()
        return heat.astype(np.float32)

# ---------------------------
# 3. Custom Dialog for Montage
# ---------------------------
class MontageDialog(QDialog):
    def __init__(self, images, query, total_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Montage Results for '{query}'")
        self.resize(1000, 850) 
        
        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # --- NEW: Info Label for Count ---
        info_text = f"Total Found: {total_count} images for '{query}'\n(Displaying preview of top {len(images)})"
        self.count_label = QLabel(info_text)
        self.count_label.setAlignment(Qt.AlignCenter)
        # Make it look nice and bold
        self.count_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin: 10px;")
        layout.addWidget(self.count_label)
        
        # Matplotlib Figure and Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Plotting
        self.plot_images(images)

    def plot_images(self, images):
        self.figure.clear()
        
        n = len(images)
        if n == 0:
            return

        # Calculate grid (5 columns fixed)
        cols = 5
        rows = int(np.ceil(n / cols))
        
        for i, img in enumerate(images):
            # Add subplot
            ax = self.figure.add_subplot(rows, cols, i+1)
            ax.imshow(img)
            ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()

# ---------------------------
# 4. Main GUI Application
# ---------------------------
class FruitSearchApp(QWidget):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        
        self.dataset_index = defaultdict(list)
        self.all_found_classes = Counter()
        
        self.filtered_files = []
        self.montage_window = None
        
        self.init_ui()
        self.index_dataset()

    def init_ui(self):
        self.setWindowTitle("Fruit Search & Count (JSON Linked)")
        self.setGeometry(100, 100, 600, 600)

        self.layout = QVBoxLayout()
        
        self.info_label = QLabel("Initializing dataset... please wait.", self)
        self.layout.addWidget(self.info_label)

        self.search_label = QLabel('Enter fruit name (e.g., "avocado", "orange"):', self)
        self.layout.addWidget(self.search_label)

        self.edit = QLineEdit(self)
        self.layout.addWidget(self.edit)

        self.button = QPushButton("Search", self)
        self.button.clicked.connect(self.search_images)
        self.layout.addWidget(self.button)

        self.list_label = QLabel("Available Classes (Double-click to search):", self)
        self.layout.addWidget(self.list_label)
        
        self.class_list_widget = QListWidget(self)
        self.class_list_widget.itemDoubleClicked.connect(self.on_list_item_clicked)
        self.layout.addWidget(self.class_list_widget)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(self.update_image_display)
        self.slider.setMinimum(1)
        self.slider.setEnabled(False)
        self.layout.addWidget(self.slider)

        self.image_idx_label = QLabel("0 images found", self)
        self.layout.addWidget(self.image_idx_label)

        self.setLayout(self.layout)

    def index_dataset(self):
        print("--- Indexing Dataset... ---")
        json_files = glob.glob(os.path.join(self.data_dir, "**", "*.json"), recursive=True)
        
        if not json_files:
            self.info_label.setText("Error: No JSON files found.")
            return

        valid_images_count = 0
        
        for j_path in json_files:
            try:
                with open(j_path, 'r') as f:
                    data = json.load(f)
                
                objects = []
                if isinstance(data, dict):
                    objects = data.get('objects', []) or data.get('shapes', [])
                elif isinstance(data, list):
                    objects = data

                found_classes_in_file = set()
                for obj in objects:
                    # Look for 'classTitle' first as per your dataset
                    name = obj.get('classTitle') or obj.get('label') or obj.get('name')
                    if name:
                        found_classes_in_file.add(name.lower().strip())

                if not found_classes_in_file:
                    continue

                # Path Linking Strategy
                base_name = j_path.replace('.json', '')
                if os.path.exists(base_name):
                    img_path = base_name
                else:
                    parts = base_name.split(os.sep)
                    if 'ann' in parts:
                        possible_img_path = base_name.replace(f'{os.sep}ann{os.sep}', f'{os.sep}img{os.sep}')
                        if os.path.exists(possible_img_path):
                            img_path = possible_img_path
                        else:
                            continue
                    else:
                        continue

                for cls in found_classes_in_file:
                    self.dataset_index[cls].append(img_path)
                    self.all_found_classes[cls] += 1
                
                valid_images_count += 1

            except Exception:
                continue

        self.info_label.setText(f"Indexed {valid_images_count} images successfully.")
        
        self.class_list_widget.clear()
        sorted_classes = sorted(self.all_found_classes.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_classes:
            self.class_list_widget.addItem(f"{cls} ({count} images)")
            
        print(f"Index complete. Found classes: {list(self.all_found_classes.keys())}")

    def on_list_item_clicked(self, item):
        text = item.text()
        fruit_name = text.split(' (')[0]
        self.edit.setText(fruit_name)
        self.search_images()

    def search_images(self):
        query = self.edit.text().lower().strip()
        if not query:
            return

        self.filtered_files = []
        for cls_name, files in self.dataset_index.items():
            if query in cls_name:
                self.filtered_files.extend(files)
        
        self.filtered_files = sorted(list(set(self.filtered_files)))

        if not self.filtered_files:
            QMessageBox.warning(self, "No Results", f"No images found for '{query}'")
            self.image_idx_label.setText("0 images found")
            return

        QMessageBox.information(self, "Found", f"Found {len(self.filtered_files)} images containing '{query}'.")
        
        # Reset slider
        self.slider.setRange(1, len(self.filtered_files))
        self.slider.setValue(1)
        self.slider.setEnabled(True)
        
        # Show montage in Dialog (Passing total count now)
        self.show_montage(self.filtered_files[:15], query, len(self.filtered_files))
        
        # Process first image
        self.update_image_display(1)

    def show_montage(self, files, pattern, total_count):
        """Creates a dialog box with the montage grid."""
        images = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not images:
            return

        # Initialize and show the custom dialog, passing total_count
        self.montage_window = MontageDialog(images, pattern, total_count, self)
        self.montage_window.show()

    def update_image_display(self, idx):
        i = idx - 1
        if i < 0 or i >= len(self.filtered_files):
            return

        filepath = self.filtered_files[i]
        self.image_idx_label.setText(f"Image {idx} / {len(self.filtered_files)}")

        img = cv2.imread(filepath)
        if img is None:
            return
            
        h, w = img.shape[:2]
        if h > 800 or w > 800:
            img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)

        try:
            counter = YOLOv5ObjectCounter() 
            count = counter.count_objects(img)
            density = counter.density_map(img)
            
            overlay = self.create_overlay(img, density, count)
            
            plt.figure("MainViewer")
            plt.clf()
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"File: {os.path.basename(filepath)}\nCount: {count:.0f}")
            plt.axis("off")
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Error processing image: {e}")

    def create_overlay(self, image, density_map, count):
        norm_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        text = f"Count: {count:.0f}"
        cv2.putText(overlay, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (255, 255, 255), 3)
        return overlay

def main():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "FruitDetectionDataset")
    
    if not os.path.isdir(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = FruitSearchApp(data_dir)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
