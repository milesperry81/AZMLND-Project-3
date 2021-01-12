from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

if "outputs" not in os.listdir():
    os.mkdir("./outputs")
    
run = Run.get_context()
ws = run.experiment.workspace

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument("--data", type=str, help="Dataset from Azure dataset")

    args = parser.parse_args()

    # Prepare the data from the registered dataset.
    dataset = Dataset.get_by_id(ws, id=args.data)
    dataset = dataset.to_pandas_dataframe()
    x = dataset.drop(columns=['DEATH_EVENT'])
    y = dataset['DEATH_EVENT']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    joblib.dump(model,'outputs/model.joblib') # Add this code to save model

if __name__ == '__main__':
    main()
