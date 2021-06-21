# run-azure.py
import os

from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.run import Run
from wandb_api_key import IMAGE, WANDB_API_KEY

if __name__ == "__main__":
    # Setup experiment
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="EGC_debug")

    # Setup environment
    env = Environment.from_docker_image(name="EGC_env", image=IMAGE)
    env.environment_variables = {"WANDB_API_KEY": WANDB_API_KEY}

    # Define configuration file
    config = ScriptRunConfig(
        source_directory="./src/models/",
        script="train_model.py",
        arguments=[
            "--max_steps",
            1,
            "--hidden_sizes",
            16,
            16,
            "--batch_size",
            16,
            "-azure",
        ],
        environment=env,
        compute_target="compute-v1",
    )

    # Submit experiment and wait for completion
    run = experiment.submit(config)
    run.wait_for_completion()

    # Save resulting model and register
    model_name = "debug_model.ckpt"
    os.makedirs("./outputs", exist_ok=True)
    run.download_file(
        name="outputs/" + model_name, output_file_path="./outputs/" + model_name
    )

    run.register_model(model_path="outputs/" + model_name, model_name=model_name)
