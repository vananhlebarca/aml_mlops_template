# Databricks notebook source
### Experiments submitted from Databricks

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt

# COMMAND ----------

import azureml.core
print("SDK Version:", azureml.core.VERSION)

# COMMAND ----------

subscription_id = "213bd947-0f6f-4418-90ba-65ddc22a594d" #you should be owner or contributor
resource_group = "ai" #you should be owner or contributor
workspace_name = "dp100" #your workspace name
workspace_region = "West US 2" #your region

# COMMAND ----------

from azureml.core import Workspace

ws = Workspace(workspace_name = workspace_name,
               subscription_id = subscription_id,
               resource_group = resource_group)

# COMMAND ----------

from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.train.hyperdrive import GridParameterSampling, choice
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal
from azureml.core.conda_dependencies import CondaDependencies

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

# Create a script config
script_config = ScriptRunConfig(source_directory='./steps',
                                script='train.py',
                                arguments = ['--ds', dataset_ds.as_named_input('dataset')],
                                environment=env,
                                compute_target=compute_name) 

#hyperparameters 
param_space = {
	'--penalty': choice(0.01, 0.1, 1.0),
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

