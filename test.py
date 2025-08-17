import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Using CPU.")

print("PyTorch CUDA version:", torch.version.cuda)
