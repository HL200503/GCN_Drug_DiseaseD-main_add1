"""
quick_test.py — Quick Test Suite

Test nhanh tất cả các thành phần của chương trình.
Chỉ cần ~5 phút để chạy xong.

Usage:
    python quick_test.py
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def run_command(cmd, description):
    """Run command and return True if successful"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    root = Path(__file__).parent
    os.chdir(root)
    
    print_header("QUICK TEST SUITE - PharmaLink GCN")
    
    results = {}
    
    # Test 1: Check GPU
    print_header("TEST 1: Check GPU Capability")
    results['gpu_check'] = run_command(
        f"{sys.executable} check_gpu_capability.py",
        "GPU capability check"
    )
    
    # Test 2: Check imports
    print_header("TEST 2: Check Python Imports")
    print("🔄 Checking imports...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
        
        import pandas
        print(f"  ✅ Pandas {pandas.__version__}")
        
        import dgl
        print(f"  ✅ DGL {dgl.__version__}")
        
        from torch.cuda.amp import autocast, GradScaler
        print(f"  ✅ Mixed Precision Training available")
        
        print("✅ All imports successful")
        results['imports'] = True
    except Exception as e:
        print(f"❌ Import error: {e}")
        results['imports'] = False
    
    # Test 3: Check data files
    print_header("TEST 3: Check Data Files")
    data_dir = root / "AMDGT_main" / "data" / "C-dataset"
    required_files = [
        "Drug_mol2vec.csv",
        "DiseaseFeature.csv",
        "Protein_ESM.csv",
        "DrugDiseaseAssociationNumber.csv"
    ]
    
    all_exist = True
    for fname in required_files:
        fpath = data_dir / fname
        if fpath.exists():
            fsize = fpath.stat().st_size / (1024**2)  # MB
            print(f"  ✅ {fname} ({fsize:.1f}MB)")
        else:
            print(f"  ❌ {fname} - NOT FOUND")
            all_exist = False
    
    results['data_files'] = all_exist
    if all_exist:
        print("✅ All data files found")
    else:
        print("❌ Some data files missing")
    
    # Test 4: Quick training (1 epoch, 1 fold)
    print_header("TEST 4: Quick Training Test (1 epoch)")
    results['training'] = run_command(
        f'{sys.executable} AI_ENGINE/src/train_DDA_base.py --dataset C-dataset --epochs 1 --k_fold 1',
        "Training test (1 epoch, 1 fold)"
    )
    
    # Test 5: Check results saved
    print_header("TEST 5: Check Results Saved")
    results_dir = root / "AI_ENGINE" / "data" / "results"
    csv_file = results_dir / "C-dataset_AMNTDDA_fold_results.csv"
    
    if csv_file.exists():
        print(f"  ✅ Results saved to {csv_file.relative_to(root)}")
        with open(csv_file) as f:
            lines = f.readlines()
            print(f"     ({len(lines)-1} folds)")
        results['results_saved'] = True
    else:
        print(f"  ❌ Results file not found")
        results['results_saved'] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    tests = [
        ("GPU Capability Check", results.get('gpu_check', False)),
        ("Python Imports", results.get('imports', False)),
        ("Data Files", results.get('data_files', False)),
        ("Training Test", results.get('training', False)),
        ("Results Saved", results.get('results_saved', False))
    ]
    
    passed = sum(1 for _, r in tests if r)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train full model.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
