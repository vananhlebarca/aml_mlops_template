# Databricks notebook source
### Experiments submitted from Databricks

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt

# COMMAND ----------

import azureml.core
print("SDK Version:", azureml.core.VERSION)

# COMMAND ----------

subscription_id = "2f71beb8-0da7-42ec-9bb7-678bf7867567" #you should be owner or contributor
resource_group = "edadevarmrgp010" #you should be owner or contributor
workspace_name = "eaadevarmmlnuw2002" #your workspace name
workspace_region = "West US 2" #your region

# COMMAND ----------

from azureml.core import Workspace

ws = Workspace(workspace_name = workspace_name,
               subscription_id = subscription_id,
               resource_group = resource_group)

# COMMAND ----------

from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.pipeline.steps import DatabricksStep
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.train.hyperdrive import GridParameterSampling, choice
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, DatabricksCompute
import os

# COMMAND ----------

# Create a Python environment for the experiment and Ensure the required packages are installed
env = Environment("aml-mlops-template-env")
packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pandas', 'matplotlib'],
                            pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages

# COMMAND ----------

# Specify computer target
compute_name = 'githubcluster'

# COMMAND ----------

# retrieving a registered dataset
dataset_ds = ws.datasets['wine_dataset']  #or dataset_id = Dataset.get_by_name(ws, 'module3_dataset')

# COMMAND ----------


# Replace with your account info before running.
 
db_compute_name=os.getenv("DATABRICKS_COMPUTE_NAME", "DS_EOA_DEV") # Databricks compute name
db_resource_group=os.getenv("DATABRICKS_RESOURCE_GROUP", "edadevarmrgp010") # Databricks resource group
db_workspace_name=os.getenv("DATABRICKS_WORKSPACE_NAME", "eaadevarmmlnuw2002") # Databricks workspace name
db_access_token=os.getenv("DATABRICKS_ACCESS_TOKEN", "dapi8ab181f979f745cada4d7b4088208de6") # Databricks access token
 
try:
    databricks_compute = DatabricksCompute(workspace=ws, name=db_compute_name)
    print('Compute target {} already exists'.format(db_compute_name))
except ComputeTargetException:
    print('Compute not found, will use below parameters to attach new one')
    print('db_compute_name {}'.format(db_compute_name))
    print('db_resource_group {}'.format(db_resource_group))
    print('db_workspace_name {}'.format(db_workspace_name))
    print('db_access_token {}'.format(db_access_token))
 
    config = DatabricksCompute.attach_configuration(
        resource_group = db_resource_group,
        workspace_name = db_workspace_name,
        access_token= db_access_token)
    databricks_compute=ComputeTarget.attach(ws, db_compute_name, config)
    databricks_compute.wait_for_completion(True)

# COMMAND ----------

# Create a script config #/dbfs/FileStore/shared_uploads/anle@suncor.com/aml_scripts/train/train.py

script_config = ScriptRunConfig(source_directory='/dbfs/FileStore/shared_uploads/anle@suncor.com/aml_scripts/train/',
                                script='train.py',
                                arguments = ['--ds', dataset_ds.as_named_input('dataset')],
                                environment=env,
                                #compute_target=compute_name
                               ) 


#hyperparameters 
param_space = {
	'--penalty': choice(0.01, 1.0),
    '--kernel': choice('linear', 'rbf')

	}
param_sampling = GridParameterSampling(param_space)


hyperdrive = HyperDriveConfig(run_config=script_config ,
                              hyperparameter_sampling=param_sampling,
                              policy=None,
                              primary_metric_name='Accuracy',
                              primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                              max_total_runs=20,
                              max_concurrent_runs=4)

# Submit the experiment
experiment = Experiment(workspace = ws, name = 'aml_mlops_template_SVM_experiments_databricks')
run = experiment.submit(config=hyperdrive)

run.wait_for_completion()

# COMMAND ----------

