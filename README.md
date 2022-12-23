# Credit_Risk_Analysis
Supervised Machine Learning and Credit Risk

## Overview 
**Background and Purpose**

From the credit card dataset from the Lending Club, a peer-to-peer lending services company, we used six different supervised machine learning models to determine if an applicant is high or low risk for the company. Since the credit risk is an unbalanced classification problem, we use different models to train and evaluate the models by resampling. Specifically, we will evaluate the performance of these models and write a recommendation on whether they should be used to predict credit card risk. 

The following are the six specific models that will be tested to determine the best performance: 

- Oversample the data with ```RandomOverSampler``` and ```SMOTE```
- Undersample the data with ```ClusterCentroids``` algorithm
- Combination Approach (Over and Under) with ```SMOTEENN```
- Compare two new machine learning models that reduce bias: ```BalancedRandomForestClassifier``` and ```EasyEnsembleClassifier```



## Resources 
- Original Data Source: [LoanStats_2019Q1.csv](https://github.com/meghanhkoon/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv)
- Software: Jupyter Notebook, Python, Python Libraries: ```imbalanced-learn``` and ```scikit-learn```, and Anaconda mlenv environment


## Results
After the inital cleaning of the original dataset's "loan_status" column, there were 68,470 low risk and 347 high risk loans. We then split the data into testing and training datasets using the ```train_test_split``` function. In the training set, 51,366 were considered "low risk" and the remaining 246 were "high risk" applications.
```
#Split into Train and Test Sets 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
Next, we used the six different models to determine which was the best for predicting credit risk. The results with balanced accuracy scores, precision and recall scores are as follows:

### Naive Random Oversampling 
### SMOTE Oversampling
### Undersampling
### Combination (Over and Under) Sampling
### Balanced Random Forest Classifier
### Easy Ensemble Classifier 

## Summary 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
