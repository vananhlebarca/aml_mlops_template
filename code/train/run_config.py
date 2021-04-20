from azureml.core import ComputeTarget
from azureml.train.estimator import Estimator
from azureml.core import Dataset

def main(workspace):
    # Load compute target
    print("Loading compute target")
    compute_target = ComputeTarget(
        workspace=workspace,
        name="githubcluster"
    )

    dataset_ds = Dataset.get_by_name(workspace=workspace, name='wine_dataset', version='latest')
    # Load script parameters which have been optimized during DS-experiment stage
    print("Loading script parameters")
    script_params = {
        "--kernel": "linear",
        "--penalty": 1.0,
        "--ds": dataset_ds 
    }



    # Create experiment config
    print("Creating experiment config")


    estimator = Estimator(
        source_directory="code/train",
        entry_script="train.py",
        script_params=script_params,
        compute_target=compute_target,
        pip_packages=["azureml-dataprep[pandas,fuse]", "scikit-learn", "pandas", "matplotlib"]
    )
    return estimator


#------------------------------------------
'''
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Model
from azureml.core import Workspace
from azureml.core import Datastore, Dataset

ws = Workspace.from_config()

# Create a Python environment for the experiment and Ensure the required packages are installed
env = Environment("sklearn-env")
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages


# retrieving a registered dataset
dataset_ds = Dataset.get_by_name(ws, 'module3_dataset')
# Create a script config
script_config = ScriptRunConfig(source_directory='src/train',
                                script='03-training.py',
                                arguments = ['--reg-rate', 0.002, '--ds', dataset_ds],
                                environment=env) 

# Submit the experiment
experiment = Experiment(workspace=ws, name='training-experiment-module3')
run = experiment.submit(config=script_config)
run.wait_for_completion()

run.register_model( model_name='classification_model',
                    model_path='outputs/model3.pkl', # run outputs path
                    description='A classification model for insurance',
                    tags={'data-format': 'CSV'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')

'''