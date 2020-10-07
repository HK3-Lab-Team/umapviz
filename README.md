# umapviz

[![Coverage Status](https://coveralls.io/repos/github/HK3-Lab-Team/umapviz/badge.svg?branch=master)](https://coveralls.io/github/HK3-Lab-Team/umapviz?branch=master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HK3-Lab-Team/umapviz.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/umapviz/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HK3-Lab-Team/umapviz.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/umapviz/context:python)

## WIP ⚠️

UmapViz is a toolset for multimodal unsupervised analysis in combination with the UMAP dimensionality reduction approach. 

Following the original proposal by L. McInnes, the HDBScan (Hierarchical Density-Based Spatial Clustering of Applications with Noise), is applied to low dimensional projections computed by UMAP. 

## Main Features
- Hybrid metric (based on Tanimoto similarity coefficient and Gower distance) to compute distances between data points with multiple feature types (e.g. boolean, categorical, numerical)
- Automatic provisioning of sample features to appropriate metrics according to their type.
- Fine tuning of metrics and calibration of feature type weights
