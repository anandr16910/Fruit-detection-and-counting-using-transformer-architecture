import os
import re
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox, QSlider, QVBoxLayout
)
from PyQt5.QtCore import Qt

# ---------------------------
# Stub for CounTR-like model
# ---------------------------

class CounTRObjectCounter:
    """
    Placeholder for a CounTR-style object counting model.
    Replace with a real implementation using PyTorch and a CounTR checkpoint.
    """
    def __init__(self, image, exemplar_bboxes):
        self.exemplar_bboxes = exemplar_bboxes
        # load model, extract patches, etc.

    def count_objects(self, image):
        # TODO: implement real neural counting
        return float(np.random.randint(1, 100))

    def density_map(self, image):
        # TODO: return real density map (H x W float32)
        h, w = image.shape[:2]
        return np.random.rand(h, w).astype("float32")


def anomaly_map_overlay(image, density_map, blend="equal"):
    """
    Rough equivalent of MATLAB anomalyMapOverlay + insertShape.
    """
    norm_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    if blend == "equal":
        alpha = 0.5
    else:
        alpha = 0.7
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def overlay_density_map(image, density_map, count):
    over = anomaly_map_overlay(image, density_map, blend="equal")
    h, w = over.shape[:2]
    font_scale = max(min(w // 30, 200) / 30.0, 0.5)
    text = f"count={count:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x = max(w - (tw + 20), 5)
    y = max(th + 10, 20)
    cv2.putText(
        over, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return over


# ---------------------------
# Core search & GUI logic
# ---------------------------

class FruitSearchApp(QWidget):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.filtered_files = []
        self.counts = []
        self.density_maps = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Fruit Search")
        self.setGeometry(100, 100, 500, 200)

        self.label = QLabel(
            'Enter a pattern to search for (e.g., "blueberry", "kiwis", "apple", "cherry","strawberry"):',
            self
        )
        self.edit = QLineEdit(self)
        self.button = QPushButton("Search", self)
        self.button.clicked.connect(self.search_images)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(self.update_image)
        self.slider.setMinimum(1)
        self.slider.setEnabled(False)

        self.index_label = QLabel("", self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        layout.addWidget(self.slider)
        layout.addWidget(self.index_label)

        self.setLayout(layout)

    def search_images(self):
        user_pattern = self.edit.text().strip()
        if not user_pattern:
            QMessageBox.warning(self, "Input Error", "Please enter a pattern.")
            return

        regex = re.compile(rf"\w*{re.escape(user_pattern)}\w*", re.IGNORECASE)
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)

        self.filtered_files = [
            f for f in all_files
            if regex.search(os.path.basename(f)) and not re.search(r"json", f, re.IGNORECASE)
        ]

        QMessageBox.information(
            self,
            "Search Result",
            f"Found {len(self.filtered_files)} images containing {user_pattern}."
        )

        if not self.filtered_files:
            return

        # Show montage of up to 15 images
        self.show_montage(self.filtered_files[:15], user_pattern)

        # Take first exemplar image
        exemplar_image_path = self.filtered_files[0]
        I = cv2.imread(exemplar_image_path)
        I = cv2.resize(I, (512, 512))

        # Let user draw exemplar bboxes (rough equivalent of selectBoundingBoxes)
        exemplar_bboxes = self.select_bounding_boxes(I)

        # Initialize CounTR-like counter
        counter = CounTRObjectCounter(I, exemplar_bboxes)

        # Single-image count + density
        count = counter.count_objects(I)
        density_map = counter.density_map(I)
        QMessageBox.information(
            self,
            "Number of fruits detected",
            f"Number of objects detected: {count:.3f}"
        )

        # Show density map
        plt.figure("DensityMap")
        plt.imshow(density_map, cmap="jet")
        plt.axis("off")
        plt.show(block=False)

        # Overlay density + count text
        overlay = overlay_density_map(I, density_map, count)
        plt.figure("DensityOverlay")
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show(block=False)

        # Show montage of original, overlay, normalized density
        norm_dm = cv2.normalize(density_map, None, 0, 1, cv2.NORM_MINMAX)
        fig, axes = plt.subplots(1, 3, num="DensityOverlayannotated")
        axes[0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Overlay")
        axes[2].imshow(norm_dm, cmap="jet")
        axes[2].set_title("Density")
        for ax in axes:
            ax.axis("off")
        plt.show(block=False)

        # Batch counts over all filtered images
        self.counts = []
        self.density_maps = []
        for p in self.filtered_files:
            img = cv2.imread(p)
            if img is None:
                self.counts.append(0.0)
                self.density_maps.append(None)
                continue
            c = counter.count_objects(img)
            dm = counter.density_map(img)
            self.counts.append(c)
            self.density_maps.append(dm)

        # Scatter plot of counts
        plt.figure("FilteredCounts")
        plt.scatter(range(1, len(self.filtered_files) + 1), self.counts)
        plt.xlabel("Image index")
        plt.ylabel("Count")
        plt.show(block=False)
        # print(f"Updating image index {index}: count={c}, dm is None={dm is None}")

        # Slider to browse overlays
        self.slider.setMaximum(len(self.filtered_files))
        self.slider.setValue(1)
        self.slider.setEnabled(True)
        self.update_image(1)

    def show_montage(self, file_list, pattern):
        imgs = []
        for fp in file_list:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.resize(img, (500, 500))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        n = len(imgs)
        if n == 0:
            return
        cols = min(5, n)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), num="Image Montage")
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")

        for i, img in enumerate(imgs):
            axes[i].imshow(img)
            axes[i].axis("off")

        fig.suptitle(f"Montage of up to {n} images containing '{pattern}'")
        plt.tight_layout()
        plt.show(block=False)

    def select_bounding_boxes(self, image):
        """
        Simple multi-ROI selector (approximate selectBoundingBoxes).
        Use matplotlib's ginput or cv2.selectROIs.
        """
        clone = image.copy()
        bboxes = cv2.selectROIs("Select Exemplars", clone, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Exemplars")
        # bboxes: N x 4, each [x, y, w, h]
        return list(bboxes)

    def update_image(self, index):
        if not self.filtered_files or not self.density_maps:
            return
        idx = int(index) - 1
        idx = max(0, min(idx, len(self.filtered_files) - 1))

        img = cv2.imread(self.filtered_files[idx])
        if img is None:
            return
        dm = self.density_maps[idx]
        c = self.counts[idx]

        if dm is None:
            overlay = img
        else:
            overlay = overlay_density_map(img, dm, c)

        plt.figure("imageIdx")
        plt.clf()
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

        self.index_label.setText(f"Image {idx + 1} of {len(self.filtered_files)}")


def main():
    data_dir = os.path.join(os.getcwd(), "FruitDetectionDataset")
    app = QApplication(sys.argv)
    w = FruitSearchApp(data_dir)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
