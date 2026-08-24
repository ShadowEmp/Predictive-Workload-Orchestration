import os
import urllib.request
from pathlib import Path

# We'll use the lessac medium ONNX model from Piper's huggingface repo
MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

def download():
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "en_US-lessac-medium.onnx"
    config_path = models_dir / "en_US-lessac-medium.onnx.json"
    
    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded.")
    else:
        print("Model already exists.")
        
    if not config_path.exists():
        print(f"Downloading config to {config_path}...")
        urllib.request.urlretrieve(CONFIG_URL, config_path)
        print("Config downloaded.")
    else:
        print("Config already exists.")

if __name__ == "__main__":
    download()
