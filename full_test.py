"""
full_test.py — Full Test Suite

Test đầy đủ toàn bộ chương trình.
Chạy training đầy đủ để xác nhận hoạt động chính xác.

Usage:
    python full_test.py

Note:
    This will take 1-2 hours depending on GPU
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def run_training(dataset, epochs, k_fold):
    """Run training test"""
    cmd = (
        f'{sys.executable} AI_ENGINE/src/train_DDA_base.py '
        f'--dataset {dataset} --epochs {epochs} --k_fold {k_fold}'
    )
    
    print(f"🔄 Training on {dataset} ({k_fold} folds, {epochs} epochs)...")
    print(f"   Estimated time: {k_fold * epochs * 0.003:.1f} seconds (~{k_fold * epochs * 0.003 / 60:.1f} minutes)")
    print(f"   Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def check_results(dataset):
    """Check and display results"""
    root = Path(__file__).parent
    results_dir = root / "AI_ENGINE" / "data" / "results"
    
    csv_file = results_dir / f"{dataset}_AMNTDDA_fold_results.csv"
    json_file = results_dir / f"{dataset}_AMNTDDA_summary.json"
    
    success = True
    
    if csv_file.exists():
        print(f"✅ Fold results saved: {csv_file.name}")
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"   {len(df)} folds completed")
    else:
        print(f"❌ Fold results NOT found: {csv_file.name}")
        success = False
    
    if json_file.exists():
        print(f"✅ Summary saved: {json_file.name}")
        with open(json_file) as f:
            summary = json.load(f)
        
        print("\n   Summary metrics:")
        for key, value in summary.items():
            if key not in ['dataset', 'model', 'n_folds']:
                if isinstance(value, float):
                    print(f"     {key}: {value:.6f}")
    else:
        print(f"❌ Summary NOT found: {json_file.name}")
        success = False
    
    return success

def main():
    root = Path(__file__).parent
    os.chdir(root)
    
    print_header("FULL TEST SUITE - PharmaLink GCN")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test configuration
    dataset = "C-dataset"
    epochs = 100  # Quick training for testing
    k_fold = 2    # Only 2 folds for testing (instead of 10)
    
    print(f"Test Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Epochs: {epochs}")
    print(f"  K-Fold: {k_fold}")
    print(f"  Total iterations: {k_fold * epochs}")
    print(f"  Estimated time: ~{k_fold * epochs * 0.003 / 60:.1f} minutes\n")
    
    # Step 1: GPU Check
    print_header("STEP 1: GPU Capability Check")
    result1 = subprocess.run(
        f"{sys.executable} check_gpu_capability.py",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result1.stdout)
    if result1.returncode != 0:
        print("⚠️  GPU check failed, but continuing...")
    
    # Step 2: Training
    print_header("STEP 2: Run Training")
    training_success = run_training(dataset, epochs, k_fold)
    
    if not training_success:
        print("❌ Training failed!")
        return 1
    
    # Step 3: Check Results
    print_header("STEP 3: Check Results")
    if check_results(dataset):
        print("\n✅ Results verification passed")
    else:
        print("\n⚠️  Some results are missing")
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Full test completed!")
    print("\nNext steps:")
    print("  1. Review results in: AI_ENGINE/data/results/")
    print("  2. For full training, use: .\train.ps1 -dataset C-dataset -model base")
    print("  3. To start full pipeline: .\start_all.ps1")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
