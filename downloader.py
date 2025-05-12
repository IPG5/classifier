import yaml
from dotenv import load_dotenv

from helpers import model
import os


# label maps
id2label = {0: "Normal", 1: "Suspicious"} # todo see #7


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_name = config["model_name"]
    inn = input(f"Enter Huggingface Model Slug (hit enter for '{model_name}'): ")
    model_name = model_name if not inn else inn

    print(f"Downloading {model_name}...")
    path = str(os.path.join(model_name))
    load_dotenv()
    model.download_model(path,config["device"], id2label, model_name, os.getenv("hugging_face_PAG"))
    print("Done!")
