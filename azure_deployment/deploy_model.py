from azureml.core.webservice import AciWebservice, LocalWebservice
from azureml.core.model import InferenceConfig
import azureml.core
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.model import Model
from wandb_api_key import WANDB_API_KEY, IMAGE_DEPLOYMENT

# Load the workspace from the saved config file
ws = Workspace.from_config()

model = ws.models['debug_model.ckpt']

# Load environment
env = Environment.from_docker_image(name="EGC_deployment", image= IMAGE_DEPLOYMENT)
env.environment_variables = {'WANDB_API_KEY': WANDB_API_KEY}


# Configure the scoring environment
inference_config = InferenceConfig(entry_script='azure_deployment/azure_scoring_script.py',
                                   environment=env)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
#deployment_config = LocalWebservice.deploy_configuration(port=6789)

service = Model.deploy(
    workspace=ws,
    name="egc-test-deploy-v3",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)

print(f"Scoring URI is : {service.scoring_uri}")



#service.wait_for_deployment(True)
print(service.state)