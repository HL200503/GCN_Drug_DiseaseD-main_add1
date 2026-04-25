"""
check_gpu_capability.py — Kiểm tra GPU capability cho training

Giúp xác nhận GPU có hỗ trợ Mixed Precision Training và các tối ưu hóa khác.

Usage:
    python check_gpu_capability.py
"""

import torch
import sys

def check_gpu_available():
    """Check if GPU is available"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        print("   Please check:")
        print("   - NVIDIA GPU drivers installed")
        print("   - CUDA toolkit installed")
        print("   - PyTorch installed with CUDA support")
        return False
    
    print("✅ CUDA is available")
    return True


def check_gpu_info():
    """Print GPU information"""
    if not torch.cuda.is_available():
        return
    
    print(f"\n📊 GPU Information:")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Get GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"   Total GPU Memory: {total_memory:.1f} GB")


def check_amp_support():
    """Check if GPU supports Automatic Mixed Precision"""
    if not torch.cuda.is_available():
        return False
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        # Test autocast
        with autocast(dtype=torch.float16):
            x = torch.randn(10, 10, device='cuda')
            y = x @ x
        
        print("✅ Automatic Mixed Precision (AMP) is supported")
        return True
    except Exception as e:
        print(f"❌ Automatic Mixed Precision (AMP) is not supported: {e}")
        return False


def check_compute_capability():
    """Check GPU compute capability"""
    if not torch.cuda.is_available():
        return False
    
    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    cc = f"{major}.{minor}"
    
    print(f"\n🔧 Compute Capability:")
    print(f"   Compute Capability: {cc}")
    
    # Tensor Core availability
    has_tensor_cores = major >= 7  # Volta and newer
    
    if has_tensor_cores:
        print(f"   ✅ Has Tensor Cores (for fast float16 operations)")
    else:
        print(f"   ⚠️  No Tensor Cores (mixed precision will be slower)")
    
    # Check for specific features
    features = []
    if major >= 5:
        features.append("Dynamic Parallelism")
    if major >= 6:
        features.append("Unified Memory")
    if major >= 7:
        features.append("Tensor Cores")
    if major >= 8:
        features.append("Structured Sparsity")
    
    if features:
        print(f"   Features: {', '.join(features)}")
    
    return has_tensor_cores


def check_cudnn_benchmark():
    """Check if cuDNN benchmark is available"""
    print(f"\n⚙️  GPU Optimization Options:")
    print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")


def check_memory_efficient():
    """Check available GPU memory"""
    if not torch.cuda.is_available():
        return
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    print(f"\n💾 Memory Requirements:")
    print(f"   Total GPU Memory: {total_memory:.1f} GB")
    
    if total_memory >= 24:
        print(f"   ✅ Excellent - Can train with large models")
    elif total_memory >= 16:
        print(f"   ✅ Very Good - Can train with medium models")
    elif total_memory >= 8:
        print(f"   ✅ Good - Can train with standard models")
    elif total_memory >= 4:
        print(f"   ⚠️  Fair - Can train with mixed precision")
    else:
        print(f"   ❌ Limited - May require reduced model size")


def benchmark_speed():
    """Simple benchmark to compare float32 vs float16"""
    if not torch.cuda.is_available():
        return
    
    print(f"\n⚡ Performance Benchmark:")
    
    try:
        import time
        
        # Test matrix size
        size = 5000
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        
        # Warmup
        _ = x @ y
        torch.cuda.synchronize()
        
        # float32 benchmark
        start = time.time()
        for _ in range(10):
            _ = x @ y
        torch.cuda.synchronize()
        time_f32 = (time.time() - start) / 10
        
        # float16 benchmark
        x_f16 = x.half()
        y_f16 = y.half()
        
        start = time.time()
        for _ in range(10):
            _ = x_f16 @ y_f16
        torch.cuda.synchronize()
        time_f16 = (time.time() - start) / 10
        
        speedup = time_f32 / time_f16 if time_f16 > 0 else 0
        
        print(f"   float32: {time_f32*1000:.2f}ms")
        print(f"   float16: {time_f16*1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print(f"   ✅ GPU has good float16 support (Tensor Cores)")
        elif speedup > 1.1:
            print(f"   ⚠️  float16 is slightly faster than float32")
        else:
            print(f"   ℹ️  float32 and float16 have similar speed")
            
    except Exception as e:
        print(f"   ⚠️  Could not run benchmark: {e}")


def main():
    print("="*60)
    print("GPU Capability Checker for Mixed Precision Training")
    print("="*60)
    
    # Check GPU available
    if not check_gpu_available():
        print("\n" + "="*60)
        print("❌ GPU not available. Training will use CPU (very slow!)")
        return False
    
    # Get GPU info
    check_gpu_info()
    
    # Check AMP support
    if not check_amp_support():
        print("\n⚠️  Mixed Precision may not work properly")
    
    # Check compute capability
    has_tensor_cores = check_compute_capability()
    
    # Check cuDNN options
    check_cudnn_benchmark()
    
    # Check memory
    check_memory_efficient()
    
    # Benchmark
    benchmark_speed()
    
    # Summary
    print(f"\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if torch.cuda.is_available() and has_tensor_cores:
        print("✅ Your GPU is ready for optimized training!")
        print("   - Mixed Precision Training: Supported")
        print("   - cuDNN Optimization: Enabled")
        print("   - TF32 Acceleration: Available")
        print("\nYou can use all optimizations from train_DDA_base.py")
    elif torch.cuda.is_available():
        print("⚠️  Your GPU supports training but may not have Tensor Cores")
        print("   - Mixed Precision Training: Partially supported")
        print("   - Training will be slower than on newer GPUs")
        print("   - Recommend upgrading to RTX 20/30/40 series")
    else:
        print("❌ GPU training not available")
        print("   Please ensure NVIDIA drivers and CUDA toolkit are installed")
    
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
