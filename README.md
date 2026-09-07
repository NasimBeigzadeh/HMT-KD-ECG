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
