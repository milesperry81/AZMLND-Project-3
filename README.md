# CAPSTONE PROJECT - AZURE MACHINE LEARNING ENGINEER

The aim of this project is to run both an Automated ML experiment and a customised experiment with hyperparameters tuned using the HyperDrive feature.
The best model from each experiment will be evaluated and the best overall model will be deployed as an endpoint and queried via the Python SDK.

The dataset used for this project was the Heart Failure Clinical Dataset from Kaggle https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. Based on the features in the dataset the aim of the experiment was to predict if the patient had a death event in the follow up period.

For the customised experiment a Logistic Regression classifier was used. The best overall model was the Logistic Regression model with 90% accuracy on the test data. The Automated ML best model was was a Voting Ensemble at only 86.6% accuarate. The Logistic Regression model was then deployed as a web service.

## Project Set Up and Installation

The following steps were taken to set up this project:

1. Create a new Azure ML workspace.
2. Create a compute instance on which to run Jupyter Notebook. STANDARD_DS2_V2 was chosen.
3. Create a compute cluster for training. STANDARD_DS12_V2, max 4 nodes, min 0 nodes, with low priority status was chosen.
4. The Kaggle dataset was uploaded to the workspace as a registered dataset.
5. The Automated ML experiment was executed using the notebook 'automl.ipynb'. Details:
	* Load the workspace, dataset and create a new experiment.
	* Load the compute cluster.
	* Define the Automated ML settings and configuration.
	* Submit the experiment.
	* Execute the RunDetails widget to see the output from each iteration.
	* The best model was examined.
6. The HyperDrive experiment was executed using the notebook /hyperparameter_tuning.ipynb. Details:
	* Load the workspace, dataset and create a new experiment.
	* Load the compute cluster.
	* Define an early termination policy.
	* Define the parameters and the sampling type.
	* Define an environment for training.
	* Define the ScriptRunConfig and HyperDriveConfig for training.
	* Submit the experiment.
	* Execute the RunDetails widget to see the output from each iteration.
	* The best model was examined and the model file saved.
	* The best model was regsitered in the workspace.
	* Inference and deployment config were defined for the endpoint.
	* The model was deployed.
	* Application Insights was enabled on the endpoint.
	* A JSON payload was sent to the end point to test it.
	* Service logs for the endpoint were retrieved.
7. This README.md file was created.
8. A screencast was recorded outlining the key points of the project.

## Dataset

### Overview

For this experiment I will be using Azure AutoML to make predictions regarding the likelihood of a death event based on a patient's features as found in the Heart Failure Prediction dataset on Kaggle. https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

### Task

Features and target (DEATH_EVENT) of the Heart Failure Prediction data set are:

* "age": Age of patient
* "anaemia": Decrease of red blood cells or hemoglobin (boolean)
* "creatinine_phosphokinase": Level of the CPK enzyme in the blood (mcg/L)
* "diabetes": If the patient has diabetes (boolean)
* "ejection_fraction": Percentage of blood leaving the heart at each contraction (percentage)
* "high_blood_pressure": If the patient has hypertension (boolean)
* "platelets": Platelets in the blood (kiloplatelets/mL)
* "serum_creatinine": Level of serum creatinine in the blood (mg/dL)
* "serum_sodium": Level of serum sodium in the blood (mEq/L)
* "sex": Woman or man (binary)
* "smoking": If the patient smokes or not (boolean)
* "time": Follow-up period (days)
* "DEATH_EVENT": If the patient deceased during the follow-up period (boolean)

Using the above features columns I will attempt to predict the DEATH_EVENT column i.e. the likelihood of a death event based on a patient's features.

### Access

The data was registered in the workspace as a dataset. It was then pulled into the Jupyter Notebook using the Dataset class of the Azure ML Python SDK.

## Automated ML

An overview of the Automated ML settings and configuration I used is below:

* "n_cross_validations": I used 3 so that 3 different cross validation trainings were performed, each using 2/3 of the available data. For my small dataset and demostration purposes this value was ok. For larger datasets I would go go a higher number so that we get more data used in training and a smaller validation set.
* "primary_metric": For this experiment accuracy is a suitable metric for classification task.
* "enable_early_stopping": This will end the experiment is primary_metric is not improving, which will save me time waiting for experiment to end.  
* "experiment_timeout_hours": If expeirment takes more than 1 hour it will timeout. This is to avoid length experiments running unnecessarily.
* "max_concurrent_iterations": This is the max number of concurrent iterations of the expeirment. I set it to 4 as 1 iteration can run on 1 node of the compute cluster, which has 4 nodes.
* "max_cores_per_iteration": The max number of cpu threads to use for an iteration. I set this to -1 so as to use all available threads.
* "iteration_timeout_minutes": This is the timeout for each iteration. I set this to 15 minutes as I was not interested in anything take longer than this to run for my current experiment.
* "verbosity": I choose default the level of logging for my experiment. 
* "iterations": I set the maximum number of iterations for AutoML to run to be 40. This seemed like a reasonable number of iterations for the course without taking up too much time and compute on the Azure platform. 
* "task": The task is to predict a death event which is a classification task (yes/no).
* "compute_target": I set this to my computer cluster that I created earlier in the notebook.
* "training_data": This was set to the dataset that I loaded earlier.
* "label_column_name": This was set to the column DEATH_EVENT in the imported dataset. It is the column we wish to predict.

### Results

The best model from the Automated ML experiment was a Voting Ensemble. The Voting Ensemble model was 86.6% accurate. The parameters for the winning classifier in the ensemble can be seen below:

	8 - sparsenormalizer
	{'copy': True, 'norm': 'l2'}

	8 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 0.5,
	 'eta': 0.1,
	 'gamma': 0,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 6,
	 'max_leaves': 15,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': -1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 2.0833333333333335,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

Ideas for improvement:
* Run the Automated ML for more iterations to see if the accuracy improves, or try more itermediate sampling values such as 25 or 75.
* I would like to change the number of cross validations and see how this effects the accuracy value for the different models.
* The data set is very small 300+ records. I would like to get more training data.

The RunDetails widget can bee seen below. This shows the status, duration and accuracy of each of the experiment iterations.

![title](images/1_RunDetails_widget.png)

![title](images/2_RunDetails_widget.png)

## Hyperparameter Tuning

For the HyperDrive experiment I chose a Logistic Regression classifier. Given this was a classification problem with a 0 or 1 outcome this is suited to Logistic Regression. Also, given the small size of my data set I didn't see then need for anything too complex (e.g. DNN). As such I settled with the Logistic Regression.

The model training was performed by the 'train.py'file. The HyperDrive feature was used to pass the dataset( --data), a range of regularization strength values (--C) and a range of epochs (--max_iter) to the 'train.py'file. The parameters were chosen randomly from a uniform range between 1 and 5 for --C, and from a discrete set of values [10, 50, 100] for --max_iter. The 'choice' and 'uniform' parameter expressions were passed to the RandomSamplingParameter class for this purpose.

The --max_iter value seemed like a good starting point for a small ML model. The values of --C were also starting point for this experiment. Smaller values of --C increase the regularization strength.

### Results

The best model was 90% accuracy. The value of --C was 2.4939 and --max_iter was 100. The best model run_id and metrics can bee seen below.

![title](images/11_HD_BestModel.png)

Ideas for improvement:
* Run the experiment for more iterations, e.g. 150, 200 and see the effect.
* Increase the regularisation to see if this has an effect on generalisation capability.
* The data set is very small 300+ records. I would like to get more training data.
* Try different sizes of mini-batches to see if this affects the model.

The RunDetails widget can bee seen below. This shows the status, duration and accuracy of each of the experiment iterations.

![title](images/7_HD_RunDetails.png)

![title](images/8_HD_RunDetails.png)

![title](images/9_HD_RunDetails.png)

## Model Deployment
The HyperDrive best model was registered in the workspace and can be seen below:

![title](images/10_HD_BestModel.png)

The model was deployed and tested as per the 'hyperparameter_tuning.ipynb' notebook. The endpoint in the workspace can be seen below:

![title](images/13_endpoint.png)

A set of test data was created as in the below screenshot for loading to the endpoint.

![title](images/12_TestData.png)

The test data was loaded to the endpoint as in the below screenshot. The resulting predictions for a death event based on the features provided are [1,1].

![title](images/14_json.png)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
