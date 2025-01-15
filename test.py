import torch

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU is available. Using CPU instead.")