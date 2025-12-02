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


# References

Liu, Chang, Yujie Zhong, Andrew Zisserman, and Weidi Xie. "Countr: Transformer-based generalised visual counting." arXiv preprint arXiv:2208.13721 (2022). [Read Paper]

Ranjan, Viresh, Udbhav Sharma, Thu Nguyen, and Minh Hoai. "Learning to count everything." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3394-3403. 2021. [Read Paper]

Sa, Inkyu, et al. "deepNIR: Datasets for Generating Synthetic NIR Images and Improved Fruit Detection System Using Deep Learning Techniques." Sensors, vol. 22, no. 13, June 2022, p. 4721. DOI: [https://doi.org/10.3390/s22134721]
