from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    # Set argument defaults and parse values
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--random_state', type=int, default=42, help="Random state")
    parser.add_argument('--model_save_path', type=str, default="outputs/model.joblib", help="Path to where the model is saved to")

    args = parser.parse_args()

    run = Run.get_context()

    # Log current argument values
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Random state:", np.int(args.random_state))
    run.log("Model save path:", args.model_save_path)

    # Download data
    path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    ds = Dataset.Tabular.from_delimited_files(path=path)
    
    # Clean data and extract features and targets
    x, y = clean_data(ds)

    # Split the data into train and test sets, ussing a random seed
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=args.random_state)

    # Create model and fit to training data
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=args.random_state).fit(x_train, y_train)

    # Validate accuracy using test set data
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    joblib.dump(model, args.model_save_path)

if __name__ == '__main__':
    main()


