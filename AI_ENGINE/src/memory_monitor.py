"""
memory_monitor.py — GPU Memory Monitoring Utility

Giúp theo dõi lượng bộ nhớ GPU sử dụng trong quá trình training.
"""

import torch
import psutil
import os


class MemoryMonitor:
    """Monitor GPU and CPU memory usage"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.initial_gpu_memory = 0
        self.initial_cpu_memory = 0
    
    def start(self):
        """Record initial memory state"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        self.initial_cpu_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB
        
        if self.verbose:
            print(f"[MEMORY] Initial GPU: {self.initial_gpu_memory:.1f}MB | CPU: {self.initial_cpu_memory:.1f}MB")
    
    def get_gpu_memory(self):
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0
    
    def get_cpu_memory(self):
        """Get current CPU memory usage in MB"""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    
    def get_peak_gpu_memory(self):
        """Get peak GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0
    
    def report(self, step=""):
        """Print current memory usage"""
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()
        peak_gpu = self.get_peak_gpu_memory()
        
        gpu_increase = gpu_mem - self.initial_gpu_memory
        cpu_increase = cpu_mem - self.initial_cpu_memory
        
        if self.verbose:
            print(f"[MEMORY{step}] GPU: {gpu_mem:.1f}MB (+{gpu_increase:.1f}MB, peak: {peak_gpu:.1f}MB) | "
                  f"CPU: {cpu_mem:.1f}MB (+{cpu_increase:.1f}MB)")
        
        return {
            'gpu_current': gpu_mem,
            'gpu_increase': gpu_increase,
            'gpu_peak': peak_gpu,
            'cpu_current': cpu_mem,
            'cpu_increase': cpu_increase
        }
    
    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        if self.verbose:
            print("[MEMORY] GPU cache cleared")


def print_model_memory_size(model):
    """Print model parameter size"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Assuming float32 (4 bytes per parameter)
    param_memory_mb = (total_params * 4) / (1024 ** 2)
    
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    print(f"[MODEL] Memory footprint: {param_memory_mb:.1f}MB (float32)")
    print(f"[MODEL] Memory with gradients: {param_memory_mb * 2:.1f}MB")
    print(f"[MODEL] Memory with optimizer (Adam): {param_memory_mb * 3:.1f}MB")


def compare_memory_usage(before_mb, after_mb):
    """Compare memory usage"""
    reduction = before_mb - after_mb
    reduction_pct = (reduction / before_mb) * 100 if before_mb > 0 else 0
    
    print(f"\n[COMPARISON]")
    print(f"  Before: {before_mb:.1f}MB")
    print(f"  After:  {after_mb:.1f}MB")
    print(f"  Reduction: {reduction:.1f}MB ({reduction_pct:.1f}%)")
    
    return {'reduction_mb': reduction, 'reduction_pct': reduction_pct}


if __name__ == '__main__':
    # Example usage
    monitor = MemoryMonitor(verbose=True)
    monitor.start()
    
    # Simulate some memory usage
    if torch.cuda.is_available():
        dummy_tensor = torch.randn(1000, 1000, device='cuda')
        monitor.report(" [after dummy tensor]")
        del dummy_tensor
        monitor.cleanup()
        monitor.report(" [after cleanup]")
