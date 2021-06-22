import os
import argparse
from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from wandb_api_key import WANDB_API_KEY
import sys

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--deploy', dest='train', action='store_false')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Separate envs for training/deployment due to conflicting dependencies
    args = parser()
    env_name = "EGC_train" if args.train else "EGC_deployment"
    print(f'Building environment: {env_name}')
    
    # Get workspace
    ws = Workspace.from_config()
    # Create environment
    env = Environment.from_pip_requirements(
        name=env_name, file_path="requirements.txt"
    )

    # Parse pip requirements
    with open("requirements.txt", "r") as f:
        pip_packages = [
            pkg.strip()
            for pkg in f
            if not (pkg.startswith("#") or pkg.startswith("git"))
        ]
    # Create Conda dependencies
    conda_dep = CondaDependencies.create(
        python_version="3.8.10",
        conda_packages=["pip==21.1.2"],
        pip_packages=pip_packages,
    )
    if not args.train:
        conda_dep.add_pip_package('azureml-defaults>=1.0.45')
    conda_dep.set_pip_option(
        "-e git+https://github.com/petergroth/enzyme_graph_classification@main#egg=src"
    )
    conda_dep.set_pip_option(
        "-f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html"
    )

    env.python.conda_dependencies = conda_dep

    env.register(workspace=ws)
    # build = env.build_local(workspace=ws, useDocker=True, pushImageToWorkspaceAcr=True)
    build = env.build(workspace=ws)
    print("Saved and registered environment. Building.")
