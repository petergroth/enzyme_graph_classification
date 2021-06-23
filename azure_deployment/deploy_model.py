from azureml.core import Environment, Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from wandb_api_key import WANDB_API_KEY

# Deployment names
service_name = "egc-deploy"
model_name = "deploy_model.ckpt"

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Load model
model = ws.models[model_name]

# Load environment
env = Environment.get(workspace=ws, name="EGC_deployment")
env.environment_variables = {"WANDB_API_KEY": WANDB_API_KEY}

# Configure the scoring environment
inference_config = InferenceConfig(
    entry_script="azure_deployment/azure_scoring_script.py", environment=env
)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
# deployment_config = LocalWebservice.deploy_configuration(port=6789)

service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
)

service.wait_for_deployment(show_output=True)

print(f"{service.scoring_uri}")
print(f"{service.state}")
