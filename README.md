# FAE

Feature Analysis Explorer (FAE) can help researchers develop a classification model with comparison among different methods. This project was inspired on the [Radiomics](http://www.radiomics.io/), and provides a GUI to help analyze the feature matrix, including feature matrix pre-process, model development, and results visualization.

If you publish any work which uses this package, I will appreciate that you could give the following link (https://github.com/salan668/FAE)

Welcome any issues and PR. 

![Python](https://img.shields.io/badge/python-v3.7-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-GPL3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Release

The Windows 64 version was release here https://drive.google.com/open?id=1htts7YsfaxKtN1NeDcNU4iksXfjr_XyK
(Alternative link is: https://pan.baidu.com/s/1ha66TajeoT6dA-a4Qdt8fA)

If FAE could help in your research, please refer to  

> Y. Song, J. Zhang, Y. Zhang, Y. Hou, X. Yan, Y. Wang, M. Zhou, Y. Yao, G. Yang. FeAture Explorer (FAE): A tool for developing and comparing radiomics models. PLoS One. 2020. DOI: https://doi.org/10.1371/journal.pone.0237587

## Getting Started

### Pre-install
The below modules must be installed first to make the FAE work. 

```
- imbalanced-learn=0.6.2
- matplotlib=3.2.0
- numpy=1.18.1
- pandas=1.0.1
- pdfdocument=3.3
- pillow=7.0.0
- PyQt5=5.14.1
- PyQtGraph=0.10.0
- pyradiomics=3.0
- reportlab=3.5.34
- scikit-learn=0.22.2
- scipy=1.4.1
- seaborn=0.10.0
- statsmodels=0.11.1
```

### Installing
Just clone it by typing in:

```
git clone https://github.com/salan668/FAE.git
```
### Architecture of Project 
- **DataContainer**
    - **DataContainer**. The structure to contain the data, which also includes methods like saving/loading.
    - **DataSeparate**. Including functions to separate data into training part and testing part.
- **Feature Analysis**
    - **DataBalance**, Sample the cases to make the binary-labels balance.
    - **Normalization**. Normalize the data to avoid the scale effect of different features.
    - **DimensionReduction**. Reduce the feature dimension, including PCA. 
    - **Classifier**. Map the features onto the labels. 
    - **CrossValidation**. Estimate model by using cross-validation on the training data set.
    - **FeatureSelector**. Select the sub-features from the feature matrix.
    - **Pipelines**. The class to estimate the model with different feature selected method and classifier. 
- **Image2Feature**
    - **RadiomicsFeatureExtractor**. This class help extract features from image and ROI with batch process. This class should be more "smart" in the future. 
- **Visulization**. 
    - **DrawDoubleLine**. This function helps draw double-y plot. e.g. plot accuracy and error against the number of iterations.
    - **DrawROCList**. This function helps draw different ROC curves. AUC will be calculated automaticly and labeled on the legend. 
    - **FeatureRelationship**. This function helps draw the distribution of the values of different features. I can only show at most 3 features in one figure. 
    - **FeatureSort**. This function helps draw the features and the weights of them on the classifiction model. 
    - **PlotMetricVsFeatureNumber**. This function helps draw the AUC / Accuracy / other metrics against the number of chosen features. This can help find the adaptive number of the features. 
- **Report**
    - To Generate the report with PDF format. 

## Document
TODO

## Author
- [**Yang Song**](https://github.com/salan668)
- [**Jing Zhang**](https://github.com/zhangjingcode)
- [**Guang Yang**](https://github.com/yg88)
- **Chengxiu Zhang**

## License 
This project is licensed under the GPL 3.0 License - see the [LICENSE.md](https://github.com/salan668/FAE/blob/master/LICENSE) file for details

## Acknowledge
- Contributor:
    - Yi-lai Pei
    - [**jmtaysom**](https://github.com/jmtaysom)
    - Zhiyong Zhao
- Demo data support. 
    - Yu-dong Zhang, Xu Yan. 
