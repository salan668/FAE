# FAE

FeAture Explorer (FAE), a radiomics (or medical analysis) tool that helps radiologists extract features, preprocess feature matrix, develop machine learning models (Binary Classification & Survival Analysis) with one-click, and evaluate models qualitatively  and quantitatively. This project was inspired on the [Radiomics](http://www.radiomics.io/), and provides a GUI with convenient process. FAE was initially developed by East China Normal University and Siemens Healthineers Ltd. 

If FAE could help in your project, We appreciate that you could cite this work:

> Y. Song, J. Zhang, Y. Zhang, Y. Hou, X. Yan, Y. Wang, M. Zhou, Y. Yao, G. Yang. FeAture Explorer (FAE): A tool for developing and comparing radiomics models. PLoS One. 2020. DOI: https://doi.org/10.1371/journal.pone.0237587
        
        
        
        
        
        

Welcome any issues and PR. 

![Python](https://img.shields.io/badge/python-v3.7-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-GPL3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Release

The Windows64 version and the Ubuntut 20.04 release could be found [Google Drive](https://drive.google.com/open?id=1htts7YsfaxKtN1NeDcNU4iksXfjr_XyK) or [SourceForce](https://sourceforge.net/projects/feature-explorer/). A short [tutorial video](https://www.bilibili.com/video/BV1yt4y1S79S/) with Chinese version may help.

## Pre-install
The below modules must be installed first to make the FAE work. 

```
- imbalanced-learn=0.6.2
- lifelines=0.27.7
- matplotlib=3.2.0
- numpy=1.21.0
- pandas=2.0.1
- pdfdocument=3.3
- pillow=7.0.0
- pycox=0.2.3
- PyQt5=5.14.1
- PyQtGraph=0.10.0
- pyradiomics=3.0
- reportlab=3.5.34
- scikit-learn=1.2.2
- scikit-image=0.18.3
- scipy=1.4.1
- seaborn=0.12.2
- statsmodels=0.11.1
- pytorch=2.0.1
- trimesh=3.9.29
- yaml=6.0
```

### Installing
Just clone it by typing in:

```
git clone https://github.com/salan668/FAE.git
```
The .ui file has to be transferred to the .py file by pyuic manually. For example, GUI/HomePage.ui should be transferred to GUI/HomePage.py file. 

### Main Architecture of Project 
- **HomePage**: The ui file and the Starting page for all modules.
- **Feature**
  - **SeriesMatcher**. The File matcher to help determine the image files and the ROI files for each folder.
  - **GUI**. The ui file and the corresponding logical files including feature extraction and feature merge.
- **BC**: Binary Classification Pipeline
  - **DataContainer**. The data structure including feature array, label, cases ID, and feature names. 
  - **Description**. The PDF generator to describe the developed BC model.
  - **FeatureAnalysis**. The module of the feature pipeline, including Data Balance, Normalization, Dimension Reduction, Feature Selector, Classifier, Cross Validation and the Pipeline Structure.
  - **Visualization**. The common visualized plots, like ROC curve and the plot of AUC against different parameters.
  - **GUI**. The ui files and the corresponding logical files including pre-process, model development, and visualization.
- **SA**: Survival Analysis Pipeline
  - **Pipeline**. Similar structure to BC and different modules.
  - **GUI**. The ui files and the corresponding logical files including model development, and visualization.
- **Plugin**: Plugin manager

## License 
This project is licensed under the GPL 3.0 License

## Contributor and Acknowledge List
- Developer:
  - [**Yang Song**](https://github.com/salan668)
  - [**Jing Zhang**](https://github.com/zhangjingcode)
  - [**Guang Yang**](https://github.com/yg88)
  - **Chengxiu Zhang**
  - [**Xue-xiang Cao**](mailto:xuer_cao@hotmail.com)
- Tester: 
  - Xu Yan
  - [**Tian-jing Zhang**](mailto:tianjingz@nvidia.com)
  - Yi-lai Pei
  - [**jmtaysom**](https://github.com/jmtaysom)
  - Zhiyong Zhang
  - Yu-dong Zhang
  - [**Wei Guo**](mailto:guowei_fy@fjmu.edu.cn)
 
