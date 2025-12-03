# Fruit-detection-and-counting-using-transformer-architecture
A few-shot fruit counting app using the CounTR model, built for ARM devices. Combines RGB and synthetic NIR images (deepNIR dataset) for robust detection. Class-agnostic, no retraining needed—counts fruits by exemplar matching with attention mechanism. Ideal for edge-based agricultural monitoring and yield assessment.

# Few-Shot Fruit Counting with CounTR on ARM
A class-agnostic fruit counting application built for ARM devices using the Counting Transformer (CounTR) model. Leverages few-shot learning—counts objects by matching image regions with exemplars, without retraining. Integrates RGB and synthetic NIR images from the deepNIR dataset for robust fruit detection across difficult conditions. Ideal for edge-based agricultural yield monitoring.

Features
Few-shot, class-agnostic counting: Counts any fruit provided sample exemplars.

No retraining needed: The CounTR model compares patches with examples using attention.

Multi-modal input: Utilises RGB and NIR images (deepNIR dataset) for improved accuracy.

Edge deployment: Runs efficiently on ARM hardware for on-site yield assessment.

Open-source implementation: popular deep learning libraries.

# YOLO based detection

- Core AI Components
The YOLOv5ObjectCounter class loads a pretrained YOLOv5s model (yolov5s.pt) for detecting objects like fruits in images, running inference with parameters tuned for high recall (e.g., conf=0.05, imgsz=1024). It generates object counts and density heatmaps by processing bounding boxes from the model's predictions, leveraging neural network outputs for tasks like blueberry or avocado counting. This aligns with AI-powered fruit detection systems using YOLO architectures.

- Application Features
The FruitSearchApp GUI searches a dataset directory for images matching user patterns (e.g., "blueberry"), applies YOLO detection across filtered images, and visualizes results via matplotlib plots including montages, overlays, density maps, and interactive sliders. Density maps accumulate heatmap values from box predictions, normalized for visualization with OpenCV overlays. Batch processing computes counts for all matches, producing scatter plots of detection results.

- Technical Context
Built with PyQt5 for the interface, OpenCV for image handling, and NumPy for array operations, it targets MPS (Apple Silicon GPU) acceleration via model.to("mps:0"), common in ML workflows like those with Jupyter/Anaconda/YOLOv5 for object detection projects. No fine-tuning is shown; it uses COCO-pretrained weights for general fruit detection.


# References

Liu, Chang, Yujie Zhong, Andrew Zisserman, and Weidi Xie. "Countr: Transformer-based generalised visual counting." arXiv preprint arXiv:2208.13721 (2022). [Read Paper]

Ranjan, Viresh, Udbhav Sharma, Thu Nguyen, and Minh Hoai. "Learning to count everything." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3394-3403. 2021. [Read Paper]

Sa, Inkyu, et al. "deepNIR: Datasets for Generating Synthetic NIR Images and Improved Fruit Detection System Using Deep Learning Techniques." Sensors, vol. 22, no. 13, June 2022, p. 4721. DOI: [https://doi.org/10.3390/s22134721]
