from azureml.core import ComputeTarget, ScriptRunConfig, Experiment, Environment
from azureml.core import Dataset
from azureml.core.conda_dependencies import CondaDependencies	

def main(workspace):
    # Load compute target
    print("Loading compute target")
    compute_target = ComputeTarget(workspace=workspace,name="githubcluster")

    env = Environment("aml-mlops-template-env")
    packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pandas', 'matplotlib'],
                                        pip_packages=['azureml-defaults'])
    env.python.conda_dependencies = packages

    compute_name='githubcluster'

    dataset_ds = Dataset.get_by_name(workspace=workspace, name='wine_dataset', version='latest')

    # Load script parameters which have been optimized during DS-experiment stage
    print("Loading script parameters")
    script_params = {
        "--kernel": "linear",
        "--penalty": 1.0,
        "--ds": dataset_ds 
    }

    # Create a script config
    script_config = ScriptRunConfig(source_directory='code/train',
                                script='train.py',
                                arguments = ['--kernel', 'linear', '--penalty', 0.1, '--ds', dataset_ds],
                                environment=env,
                                compute_target=compute_name
                                ) 

    # Submit the experiment
    experiment = Experiment(workspace=workspace, name='aml_mlops_template')
    run = experiment.submit(config=script_config)

    return run

