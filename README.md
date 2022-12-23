# Credit_Risk_Analysis
Supervised Machine Learning and Credit Risk

## Overview 
**Background and Purpose**

From the credit card dataset from the Lending Club, a peer-to-peer lending services company, we used six different supervised machine learning models to determine if an applicant is high or low risk for the company. Since the credit risk is an unbalanced classification problem, we use different models to train and evaluate the models by resampling. Specifically, we will evaluate the poerformance of these models and write a recommendation on whether they should be used to predict credit card risk. 

The following are the six specific models that will be tested: 

- Oversample the data with ```RandomOverSampler``` and ```SMOTE```
- Undersample the data with ```ClusterCentroids``` algorithm
- Combination Approach (Over and Under) with ```SMOTEENN```
- Compare two new machine learning models that reduce bias: ```BalancedRandomForestClassifier``` and ```EasyEnsembleClassifier```



## Resources 
- Original Data Source: [LoanStats_2019Q1.csv](https://github.com/meghanhkoon/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv)
- Software: Jupyter Notebook, ```imbalanced-learn``` and ```scikit-learn``` libraries


## Results
describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

## Summary 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
