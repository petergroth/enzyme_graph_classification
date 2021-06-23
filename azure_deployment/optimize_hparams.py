from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from wandb_api_key import WANDB_API_KEY

COMPUTE_TARGET = "compute-v1"

if __name__ == "__main__":
    # Setup experiment
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="EGC_hyperparam_opt")

    # Setup environment
    env = Environment.get(workspace=ws, name="EGC_train")
    env.environment_variables = {"WANDB_API_KEY": WANDB_API_KEY}

    args = [
        "--num_workers",
        2,
        "--n_startup_trials",
        5,
        "--n_warmup_steps",
        50,
        "--n_trials",
        100,
        "--timeout",
        36000,
        "--max_epochs",
        300,
    ]

    # Define configuration file
    config = ScriptRunConfig(
        source_directory=".",
        script="src/models/train_model_optuna.py",
        environment=env,
        arguments=args,
        compute_target=COMPUTE_TARGET,
    )

    # Submit experiment and wait for completion
    run = experiment.submit(config)
    # run.wait_for_completion()
