import os
import subprocess
from pathlib import Path

def test_voice():
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / "en_US-lessac-medium.onnx"
    output_path = Path(__file__).parent / "test_output.wav"
    
    if not model_path.exists():
        print("Model not found. Run download_voice_model.py first.")
        return
        
    text = "Hello. I am JARVIS, your AI Cloud Operations Assistant."
    
    # Run Piper TTS via CLI (requires piper-tts installed in environment)
    command = f'echo "{text}" | piper -m {model_path} -f {output_path}'
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully generated {output_path}")
    except subprocess.CalledProcessError:
        print("Failed to run Piper. Is piper-tts installed?")
        
if __name__ == "__main__":
    test_voice()
