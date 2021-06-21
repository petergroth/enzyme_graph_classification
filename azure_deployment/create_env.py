from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from wandb_api_key import WANDB_API_KEY
import os


if __name__=="__main__":
	# Get workspace
	ws = Workspace.from_config()

	# Create environment
	env = Environment.from_pip_requirements(name="EGC_env",
											file_path="requirements.txt")


	# Parse pip requirements
	with open('requirements.txt', 'r') as f:
		pip_packages = [pkg.strip() for pkg in f if not (pkg.startswith('#') or pkg.startswith('git'))]
	# Create Conda dependencies
	conda_dep = CondaDependencies.create(
		python_version='3.8.10',
		conda_packages=['pip==21.1.2'],
		pip_packages=pip_packages
	)
	conda_dep.set_pip_option('-e git+https://github.com/petergroth/enzyme_graph_classification@main#egg=src')
	#conda_dep.add_pip_package('azureml-defaults')
	conda_dep.add_pip_package('torch==1.8.1')
	conda_dep.add_pip_package('torch-scatter')
	conda_dep.add_pip_package('torch-sparse')
	conda_dep.add_pip_package('torch-cluster')
	conda_dep.add_pip_package('torch-spline-conv')
	conda_dep.add_pip_package('torch-geometric')
	conda_dep.set_pip_option('-f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html')

	env.python.conda_dependencies = conda_dep

	# Save environment
	env.save_to_directory(path="azure_deployment/EGC_env.env", overwrite=True)

	env.register(workspace=ws)
	#build = env.build_local(workspace=ws, useDocker=True, pushImageToWorkspaceAcr=True)
	build = env.build(workspace=ws)
	print("Saved and registered environment. Building.")