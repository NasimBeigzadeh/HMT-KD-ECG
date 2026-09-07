# HMT-KD-ECG

## Hierarchical Multi-Teacher Knowledge Distillation Framework for Lightweight ECG Classification


This repository contains the official implementation of the proposed **Hierarchical Multi-Teacher Knowledge Distillation (HMT-KD)** framework for lightweight and efficient ECG classification.

The increasing demand for real-time healthcare applications requires deep learning models that can achieve high diagnostic performance while maintaining low computational complexity. Although deep neural networks provide powerful feature extraction capabilities for ECG analysis, their large number of parameters and high computational requirements limit their deployment on resource-constrained and wearable devices.

To address this challenge, we propose a hierarchical multi-teacher knowledge distillation framework that transfers knowledge from multiple powerful teacher networks into a compact student model. The proposed framework combines complementary knowledge sources from different architectures through both response-based and feature-based knowledge distillation strategies.

The HMT-KD framework employs two primary teacher networks, including **ResNet-18 with SENet attention mechanism** and **MobileNetV2**, to extract rich ECG representations. A **Teacher Assistant (TA)** network is introduced as an intermediate learning bridge to improve the knowledge transfer process between high-capacity teachers and the lightweight student model.

The student network is designed as an efficient CNN architecture incorporating depthwise separable convolutions, residual learning blocks, and attention mechanisms to achieve an effective balance between classification accuracy and computational efficiency.

The proposed framework is evaluated on ECG image representations generated from benchmark ECG datasets, including **PTB-XL** and **Chapman ECG datasets**. Experimental results demonstrate that the proposed approach can significantly reduce model complexity while preserving competitive diagnostic performance, making it suitable for real-time and low-resource healthcare applications.


## Framework Overview

The overall architecture of the proposed HMT-KD framework is illustrated below:

<p align="center">
<img src="PictureHMT-KD.png" width="850">
</p>

**Fig. 6.** A schematic overview of the proposed Hierarchical Multi-Teacher Knowledge Distillation (HMT-KD) framework, integrating response-based and feature-based pathways. The response-based KD exploits the outputs of ResNet-18 with SENet, MobileNetV2, and the Teacher Assistant, while the feature-based KD transfers intermediate representations from the TA to the student.


## Main Components

- **Teacher 1:** ResNet-18 with SENet attention module  
- **Teacher 2:** MobileNetV2  
- **Teacher Assistant:** Intermediate knowledge transfer network  
- **Student Model:** Lightweight CNN optimized for ECG classification  


## Knowledge Distillation Strategy

The proposed framework performs knowledge transfer through two parallel pathways:

- **Response-based Knowledge Distillation:**  
  The student learns from the soft prediction outputs of multiple teachers.

- **Feature-based Knowledge Distillation:**  
  Intermediate feature representations extracted from the Teacher Assistant are transferred to guide the student model.


## Repository Structure


## Models

The repository includes implementations of:

- MobileNetV2 teacher network
- ResNet18-SENet teacher network
- Teacher Assistant network
- Lightweight student CNN network


## Datasets

The framework supports ECG image classification using:

- PTB-XL ECG dataset
- Chapman ECG dataset


## Training

The complete training pipeline is implemented using PyTorch.

Example:

```bash
python src/training/train.py
## Results

The proposed HMT-KD framework was evaluated on the **PTB-XL ECG dataset** by comparing the performance of the teacher networks, Teacher Assistant (TA), and the lightweight student model with and without knowledge distillation.

The results demonstrate that the proposed knowledge distillation strategy enables the lightweight student model to achieve competitive diagnostic performance while significantly reducing model complexity.

### Performance Comparison on PTB-XL Dataset

| Model | Precision | Recall | F1-score | Accuracy | Specificity | AUROC | Parameters |
|------|-----------|--------|----------|----------|-------------|-------|------------|
| MobileNetV2 (Teacher) | 0.7790 | 0.7090 | 0.7390 | 0.8539 | 0.9538 | 0.9619 | 2,230,277 |
| ResNet18 + SENet (Teacher) | 0.8052 | 0.7267 | 0.7565 | 0.8527 | 0.9518 | 0.9675 | 11,220,853 |
| Teacher Assistant (TA) | 0.7933 | 0.7150 | 0.7420 | 0.8558 | 0.9565 | 0.9687 | 194,880 |
| Student (Without KD) | 0.6829 | 0.5819 | 0.6178 | 0.7424 | 0.9092 | 0.8898 | 101,600 |
| Student (With HMT-KD) | 0.8074 | 0.7433 | 0.7622 | 0.8545 | 0.9569 | 0.9653 | 101,600 |


The proposed HMT-KD framework significantly improves the performance of the lightweight student model. Without knowledge distillation, the student network achieves an accuracy of **74.24%** and an AUROC of **88.98%**. After applying the proposed hierarchical multi-teacher knowledge transfer strategy, the student model reaches **85.45% accuracy** and **96.53% AUROC**, demonstrating the effectiveness of both response-based and feature-based knowledge transfer pathways.

Moreover, the proposed student model contains only **101,600 parameters**, which represents a reduction of approximately **110× compared with the ResNet18 + SENet teacher network (11,220,853 parameters)** while maintaining comparable classification performance.

Compared with the ResNet18 + SENet teacher, the HMT-KD student reduces the model size by approximately **99.1%**, making it suitable for deployment on resource-constrained devices such as wearable healthcare systems and real-time ECG monitoring platforms.

The experimental results confirm that HMT-KD successfully achieves an optimal balance between diagnostic accuracy and computational efficiency, enabling lightweight ECG classification without significant loss of performance.
