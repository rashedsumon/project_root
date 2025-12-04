# Causal Multimodal Diagnostic Agent (Medical AI)

## Overview
This project integrates chest X-ray images and clinical text reports to diagnose medical conditions using causal graph learning. The system utilizes powerful pre-trained models for image (ResNet50) and text (BioBERT) encoding, with a Graph Neural Network (GNN) for causal fusion of the modalities.

## Features
- **Image Encoder**: ResNet50 for feature extraction from X-ray images.
- **Text Encoder**: BioBERT for processing radiology reports.
- **Causal Fusion**: GNN to model relationships between image and text.
- **Explainability**: Grad-CAM heatmaps for image features and SHAP values for text features.

## Setup

1. Install dependencies:
