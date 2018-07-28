> Due to some reasons, FAE will not be updated now. Thanks for everyone's STAR and FORK. Especially, thank **jmtaysom** for giving the suggestion of the name of the project; thank **Jing** and **Yi-lai** for coding; thank **Prof. Yang** for the desgning of the project and thank **Dr. Yu-dong Zhang** and **Dr. Xu Yan** for providing the demo data. 
>
> I'm still welcome any PR and Issues. If you have any interests or co-opration intention, please feel free to connect with me. 
>
> Thank you.
>
> Sincerely, 
>
> Yang Song
>
> songyangmri@gmail.com
>
> BTW: Maybe this project will be activated in the future. :)

# FAE

Feature Analysis Explorer (FAE) can help researchers develop a classification model with comparision among diffferent methods. This project was inspired on the [Radiomics](http://www.radiomics.io/), and provides some functions to help extract features with batch process.

A demo of features and the corresponding result are shown below

Demo of Features:

![DemoFeatures](https://github.com/salan668/FAE/blob/master/Example/DemoFeatures.png)

Result processed by FAE

![Result](https://github.com/salan668/FAE/blob/master/Example/Result.png)

If you publish any work which uses this package, I will appreciate that you could cite the following publication: [Song Y, Zhang YD, Yan X, Liu H, Zhou M, Hu B, Yang G, Computer-aided diagnosis of prostate cancer using a deep convolutional neural network from multiparametric MRI. J Magn Reson Imaging. 2018 Apr 16. doi: 10.1002/jmri.26047.](https://www.ncbi.nlm.nih.gov/pubmed/29659067) 

Welcome any issues and PR. 

![Python](https://img.shields.io/badge/python-v3.6-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-GPL3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisties
The below modules must be installed first to make the FAE work. 

```
- pyradiomics
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib (seaborn)
```

### Installing
Just clone it by typing in:

```
git clone https://github.com/salan668/FAE.git
```
If you would like to use FAE in any project, please add the path in your system envirement. A trick method is to create a .pth file in the site-packages folder (<Your own python folder>\Lib\site-packages) and add a path that point to the root folder of the FAE.

### Running the example. 
```
cd FAE
python Example\example_diff_method.py
```

### Architecutre of Project 
- **DataContainer**
    - **DataContainer**. The structure to contain the data, which also includes methods like saving/loading and data normaliztion processing.
    - **DataSeperate**. Including functions to seperate data into training part and testing part.
    - **DataBalance**, The class to deal with data imbalance. Now we provided Under-sampling, Over-sampling, and SMOTE method.
- **Feature Analysis**
    - **Classifier**. The classifier to develop the model, including SVM, AE, Random Forests, LDA. 
    - **CrossValidation**. The CV model to estimate the model. Return the metrics
    - **FeturePipeline**. The class to estimate the model with different feature selected method and classifier. 
    - **FeatureSelector**. The class to select features, which including 1) remove non-number features, e.g. the version of the pyradiomics; 2) remove non-useful features, e.g. the VolumnNum; 3) different method to select features, like ANOVA, RFE, Relief.
    - **DimensionReduction**, The class provided the feature decomposition, like PCA.
- **Image2Feature**
    - **RadiomicsFeatureExtractor**. This class help extract features from image and ROI with batch process. This class should be more "smart" in the future. 
- **Visulization**. 
    - **DrawDoubleLine**. This function helps draw doulbe-y plot. e.g. plot accuracy and error against the number of iterations.
    - **DrawROCList**. This function helps draw different ROC curves. AUC will be calculated automaticly and labeled on the legend. 
    - **FeatureRelationship**. This function helps draw the distribution of the values of different features. I can only show at most 3 features in one figure. 
    - **FeatureSort**. This function helps draw the features and the weights of them on the classifiction model. 
    - **PlotMetricVsFeatureNumber**. This function helps draw the AUC / Accuracy / other metrics against the number of chosen features. This can help find the adaptive number of the features. 

## Document
TODO

## Author
- [**Yang Song**](https://github.com/salan668)

## License 
This project is licensed under the GPL 3.0 License - see the [LICENSE.md](https://github.com/salan668/FAE/blob/master/LICENSE) file for details

## Acknowledge
- Contributor:
    - [**Guang Yang**](https://github.com/yg88)
    - Yi-lai Pei
    - [**jmtaysom**](https://github.com/jmtaysom)
- Bugs fix:
    - Jing Zhang. 
- Demo data support. 
    - Yu-dong Zhang, Xu Yan. 
