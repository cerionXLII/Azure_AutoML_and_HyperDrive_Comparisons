# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset is from a bank marketing campaign or similar. A number of features like age, martial status etc were the inputs to the models. The target is a binary yes/no column, which is likely to be the outcome of the marketing attempt. We seek to find if this column was a 'yes' or a 'no' answer.

The data was then cleaned by dropping any NaNs, and categorical variables were encoded as numerical features.

First we wanted to find the optimal hyper-parameters for a Logistic regression model. This was done using Azure HyperDrive. The best accuracy for this model was about 90.8%.

The AutoML found that a VotingEnsemble was the best model to use. It had a slightly better accuracy of 91.7%-


## Scikit-learn Pipeline
The first pipeline consisted of a number of steps. First the raw data was cleaned and features were converted into numerical values. The data was then split into test and train sets respectively.

A Logistic Regression model from Scikit Learn was then created using the input parameters. In this case we simply altered the maximum number of iterations and the regularization coefficient.

The model was trained on the train set and validated on the test set.

In order to find the optimal values for the model parameters, we used Azure HyperDrive. This does a search of the space of possible inputs and calculates the corresponding model accuracy.

In this case we did a simple uniform search of the regularization coefficient C in the interval [0.05 and 0.1], and a search of maximum iterations as a choice between 10, 100 and 150. 

A random sampling is much better than a grid search, since it is more likely to hit a sweet spot, rather than jumping over the best parameter combination. Also it is much quicker to do a random search rather than a complete grid search. The maximum iterations had a choice between small, medium and larget values. Not so much extra value would be gained by sampling randomly between them, since either the model converges or not.

A much larger sampling space could be used given more time and resources. 

The stopping policy was a simple BanditPolicy. It terminates the run if the primary metric (accuracy) is not improved for a number of tries. This is used to save computation time if it is clear that a model has already converged enough.

## AutoML
The Azure AutoML was set up and did run on the same dataset. It did a search between different machine learning models and scalers. The best model found was a VotingEnsemble. When looking at the different models within the ensemble we could find that it used a combination of XGBoostClassifiers and LightGBMClassifiers. The different classifiers scaled the input data a little bit differently, with StandardScaler and MaxAbsScaler as main tools.

### Inputs to AutoML
There are a number of inputs we need to specify in order to run the AutoML. Here is an example of the inputs used, with a short description of each.

```python
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,      #The maximum allowed time for the experiment in minutes
    task='classification',              #This is a classification task. Other options would be regression or forecasting
    primary_metric='accuracy',          #We want to maximize the accuracy of the model, could also be for example RMS
    training_data=trainingDataTabular,  #Tabular input data set
    label_column_name='y',              #The target name wtithin the training_data set
    n_cross_validations=3,              #How many folds we use in cross validation
    test_size=0.2,                      #How much of the available data we use for testing the model accuracy. For each fold say for example 20% of the data is used for validation
    enable_early_stopping=True,         #If we want to stop a model training run if it is obvious that the primary metric does not improve. Saves time
    compute_target=aml_compute)         #The cluster we use during the compute
```
[AutoML parameter documetiation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train).

## Pipeline comparison
When comparing the two methods, it is clear that we can run many more experiments using AutoML, given the same amount of time and resources. The HyperDrive is a good tool when you already have a sense of the model you want to use and which hyper parameters to optimize for.

The AutoML does not only search for the best hyper parameters, but also for the best combination of model type, scaler and hyper parameters.

The current problem did however not show significant improvement in favor of AutoML. It did improve a few percentage points, but a standard regression model did fairly well also.

## Future work
Any future experiments would expand the search space for the Scikit-learn HyperDrive parameters. In this case we were limited by time and budget, but the search space is much wider than the bondaries used. Also we could try a completely different ML algorithm, including deep learning models.

When it comes to AutoML we set constraints on maximum running time. A next step would be to allow it to run for a bit longer and see what it can come up with. How much would we gain in accuracy if we allowed the experiment to run twice as long for example?

## Cluster clean up
In this experiment a 'private' Azure ML Cluster was used. Hence it was not clearned up in this notebook. If running the experiment on your own, make sure to switch the clusters off, or clean them up all together in order to save costs.
