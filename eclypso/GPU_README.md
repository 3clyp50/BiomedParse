# GPU Server Setup Guide for BiomedParse

This guide provides comprehensive instructions for setting up and running the BiomedParse GPU server for medical image analysis.

## 1. System Requirements

- CUDA-capable GPU (NVIDIA)
- CUDA 12.4 or compatible version
- Ubuntu/Linux OS recommended (Windows possible but may require additional steps)
- Python 3.9.19
- At least 16GB GPU memory recommended
- At least 32GB system RAM recommended

## 2. Initial System Setup

### 2.1. Install System Dependencies
```bash
# Install basic dependencies
sudo apt update
sudo apt install -y curl python3-pip libopenmpi-dev libgl1

# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash ~/Miniconda3-latest-Linux-x86_64.sh

# Clone BiomedParse repository
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse
```

### 2.2. Environment Setup
```bash
# Create and activate conda environment
conda create -n biomedparse python=3.9.19
conda activate biomedparse

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Setup MPI
conda install gxx_linux-64
conda install -c conda-forge mpi4py mpich
export MPICC=/home/ubuntu/miniconda3/envs/biomedparse/bin/mpicc
pip3 install mpi4py

# Add conda environment to PATH
echo 'export PATH="/home/ubuntu/miniconda3/envs/biomedparse/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install medical imaging libraries
conda install -c conda-forge pydicom
conda install conda-forge::nibabel
pip install SimpleITK

# Install remaining requirements
pip install -r assets/requirements/requirements.txt
```

## 3. Server Setup

### 3.1. Create Server Directory Structure
```bash
mkdir -p biomedparse_server/{models,uploads,results}
cd biomedparse_server
```

### 3.2. Create FastAPI Server Script (server.py)
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import io
import numpy as np
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
import asyncio
import aiofiles

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    global model
    # Load model configuration
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    
    # Initialize model
    model = BaseModel(opt, build_model(opt)).from_pretrained('hf_hub:microsoft/BiomedParse').eval().cuda()
    
    # Initialize text embeddings
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], 
            is_eval=True
        )

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prompts: str = None  # Comma-separated prompts
):
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process prompts
        prompt_list = prompts.split(',') if prompts else []
        
        # Run inference
        with torch.no_grad():
            pred_masks = interactive_infer_image(model, image, prompt_list)
        
        # Convert results to list format
        results = []
        for i, pred in enumerate(pred_masks):
            mask_bytes = io.BytesIO()
            mask_img = Image.fromarray((pred * 255).astype(np.uint8))
            mask_img.save(mask_bytes, format='PNG')
            
            results.append({
                'prompt': prompt_list[i],
                'mask': mask_bytes.getvalue(),
                'confidence': float(pred.mean())
            })
        
        return {"status": "success", "results": results}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
```

### 3.3. Install Server Dependencies
```bash
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart
pip install aiofiles
pip install redis  # For queue management
```

## 4. Running the Server

### 4.1. Environment Configuration
Create a `.env` file:
```bash
CUDA_VISIBLE_DEVICES=0  # Specify which GPU to use
MODEL_CACHE_DIR=/path/to/cache  # Optional: specify model cache directory
MAX_BATCH_SIZE=1  # BiomedParse currently supports batch size 1
```

### 4.2. Start the Server
```bash
# Navigate to server directory
cd biomedparse_server

# Start the server
python server.py
```

## 5. Testing

### 5.1. Test with cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/test/image.png" \
  -F "prompts=neoplastic cells,inflammatory cells"
```

### 5.2. Python Test Client
```python
import requests
import json

def test_prediction(image_path, prompts):
    url = "http://localhost:8000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'prompts': ','.join(prompts)}
        
        response = requests.post(url, files=files, data=data)
        
    return response.json()

# Test
result = test_prediction(
    'test_image.png',
    ['neoplastic cells', 'inflammatory cells']
)
print(json.dumps(result, indent=2))
```

## 6. Production Deployment

### 6.1. Process Management
Use PM2 for process management:
```bash
# Install PM2
npm install pm2 -g

# Start server with PM2
pm2 start server.py --name biomedparse-server
```

### 6.2. Load Balancing
For multiple GPUs, use NGINX as a load balancer:
```nginx
upstream biomedparse {
    server localhost:8000;
    server localhost:8001;
    # Add more servers as needed
}

server {
    listen 80;
    location / {
        proxy_pass http://biomedparse;
    }
}
```

### 6.3. Monitoring
Add Prometheus metrics:
```python
from prometheus_client import Counter, Histogram
import time

# Add to server.py
PREDICTION_TIME = Histogram('prediction_seconds', 'Time spent processing prediction')
PREDICTION_COUNT = Counter('prediction_total', 'Total number of predictions')
```

## 7. Troubleshooting

### 7.1. Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Monitor GPU memory usage with `nvidia-smi`

2. **MPI Issues**
   - Verify MPI installation: `mpirun --version`
   - Check MPICC path: `which mpicc`

3. **Image Loading Issues**
   - Ensure libgl1 is installed
   - Check image format compatibility

### 7.2. Logging
Enable detailed logging in server.py:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('biomedparse.log'),
        logging.StreamHandler()
    ]
)
```

## 8. Security Considerations

1. Replace `allow_origins=["*"]` with specific allowed origins
2. Implement authentication for API endpoints
3. Use HTTPS in production
4. Implement rate limiting
5. Validate input file types and sizes

## 9. Support

For issues and questions:
- GitHub Issues: [BiomedParse Issues](https://github.com/microsoft/BiomedParse/issues)
- Documentation: [BiomedParse Docs](https://github.com/microsoft/BiomedParse) 