from prometheus_client import Counter, Histogram, Gauge
import psutil
import torch

# Prediction metrics
PREDICTION_TIME = Histogram(
    'prediction_seconds',
    'Time spent processing prediction',
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, float('inf'))
)

PREDICTION_COUNT = Counter(
    'prediction_total',
    'Total number of predictions'
)

PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Resource metrics
GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'Current GPU memory usage in bytes',
    ['device']
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current system memory usage in bytes'
)

def update_resource_metrics():
    """Update system resource metrics"""
    # Update CPU usage
    CPU_USAGE.set(psutil.cpu_percent())
    
    # Update memory usage
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)
    
    # Update GPU metrics if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            GPU_MEMORY_USAGE.labels(device=f'cuda:{i}').set(memory_allocated) 