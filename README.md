# TPOT Breast Cancer AutoML

This repository demonstrates the use of **Automated Machine Learning (AutoML)** with [TPOT](http://epistasislab.github.io/tpot/) on the **Breast Cancer dataset** from scikit-learn.  
The project applies **genetic programming** to automatically design and optimize machine learning pipelines.

---

## Overview
- **Objective**: Classify breast cancer tumors as malignant or benign using AutoML.  
- **Approach**: Leverage TPOT to evolve pipelines over several generations.  
- **Result**: Achieves high test accuracy and exports the optimized pipeline for reproducibility.  

---

## Key Highlights
- Automated pipeline search with **TPOTClassifier**  
- Dataset: Breast Cancer (scikit-learn built-in)  
- Evaluation metric: Test set accuracy  
- Export of the **best-performing pipeline** to `tpot_pipeline.py`  
- Reproducible, lightweight, and easy to extend to other datasets  

---

## Benefits
- **Demonstrates AutoML expertise**: Showcases ability to automate model selection and optimization.  
- **Portfolio-ready project**: Clean, structured, and well-documented for professional visibility.  
- **Healthcare relevance**: Breast cancer classification is a widely studied and impactful application.  
- **Extensible framework**: Can be adapted to other datasets and ML tasks.  

---

## Usage
Run the main script to start the AutoML search:

```bash
python tpot_breast_cancer.py
