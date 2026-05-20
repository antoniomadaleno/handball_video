from roboflow import Roboflow

# Coloca a tua API key do Roboflow aqui (ou em variável de ambiente)
# https://app.roboflow.com/settings/api
API_KEY = "YOUR_ROBOFLOW_API_KEY"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("touni66").project("handball-detection-k4csp")
version = project.version(3)
dataset = version.download("yolov8")