import os
import warnings
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import glob
from datetime import date

# def get_free_gpu(min_mem=20000):
#     torch.cuda.empty_cache()
#     try:
#         with NamedTemporaryFile() as f:
#             os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
#             memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
#         if max(memory_available) < min_mem:
#             warnings.warn("Not enough memory on GPU, using CPU")
#             return torch.device("cpu")
#         return torch.device("cuda", np.argmax(memory_available))
#     except:
#         warnings.warn("Could not get free GPU, using CPU")
#         return torch.device("cpu")

def get_free_gpu(min_mem=20000, device_num=None):
    torch.cuda.empty_cache()
    
    try:
        if device_num is not None:
            # Check if the specified device is available
            if torch.cuda.is_available() and device_num < torch.cuda.device_count():
                with NamedTemporaryFile() as f:
                    os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
                    memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
                    
                    # Check if the specified device has enough free memory
                    if memory_available[device_num] >= min_mem:
                        print(f"[INFO] Using specified device: cuda:{device_num} with {memory_available[device_num]} MiB free.")
                        return torch.device(f"cuda:{device_num}")
                    else:
                        warnings.warn(f"Specified GPU cuda:{device_num} does not have enough memory. Available: {memory_available[device_num]} MiB.")
            else:
                warnings.warn(f"Specified GPU cuda:{device_num} is not available.")
        
        # Fallback: Find the GPU with the most available memory
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        
        if max(memory_available) < min_mem:
            warnings.warn("Not enough memory on any GPU, using CPU")
            return torch.device("cpu")
        
        selected_gpu = np.argmax(memory_available)
        print(f"[INFO] Using GPU with the most free memory: cuda:{selected_gpu} with {memory_available[selected_gpu]} MiB free.")
        return torch.device(f"cuda:{selected_gpu}")
    
    except Exception as e:
        warnings.warn(f"Could not get free GPU, using CPU. Error: {e}")
        return torch.device("cpu")
    
def create_out_folder(experiment_name: str, 
                      output_path: str = "outputs"):
    date_str = date.today().strftime("%Y-%m-%d-%H:%M:%S")
    folder_name = date_str + '-' + experiment_name
    out_folder = os.path.join(output_path, folder_name)    
    os.makedirs(out_folder, exist_ok=True)
    return out_folder