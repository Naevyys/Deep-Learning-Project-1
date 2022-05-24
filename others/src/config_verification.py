"""
Run this file to verify your setup (python, cuda, pytorch).
"""

import sys
import torch

# Configs
py_version = "3."  # Change to 3.8 if needed
torch_version = "1.10.2+cu113"
torch_version_strict = True

# Check python version
assert sys.version.startswith(py_version), "This project requires Python " + py_version
print("Good, you are running Python", py_version)

# Check pytorch version
if torch_version_strict:
    assert torch.__version__ == torch_version, "You are not using pytorch version  " + torch_version
else:
    assert torch.__version__ >= torch_version, "You are using a pytorch version below " + torch_version
print("Good, you are using pytorch version", torch.__version__)

# Check cuda installation
if torch.cuda.is_available():
    if torch.cuda.device_count() >= 1:
        current_device = torch.cuda.current_device()
        print("Cuda is available, and you are currently using gpu device number", current_device, ", which is",
              torch.cuda.get_device_name(current_device))
    else:
        print("Cuda is available but you don't have a gpu to use it with.")
else:
    print("Cuda is not available")