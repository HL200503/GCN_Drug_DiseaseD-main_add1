# 🧪 TEST GUIDE - How to Test the Program

Hướng dẫn chi tiết để test chương trình PharmaLink GCN.

## 📋 Test Files Available

| File | Mục đích | Thời gian | Yêu cầu |
|------|---------|----------|---------|
| **quick_test.py** | Quick validation (GPU, imports, 1 epoch) | ~5 min | GPU driver + PyTorch |
| **full_test.py** | Full training test (2 folds, 100 epochs) | ~20-30 min | GPU 8GB+ |
| **api_test.py** | Test API endpoints | ~1 min | Running API server |
| **check_gpu_capability.py** | Detailed GPU check | ~1 min | CUDA installed |

---

## 🚀 Quick Start - Choose Your Test Level

### ✅ Level 1: Quick Validation (5 minutes)
**Best for**: First-time setup, checking if everything works

```bash
python quick_test.py
```

**Checks:**
- ✅ GPU capability
- ✅ Python imports (torch, numpy, pandas, dgl)
- ✅ Data files existence
- ✅ Training works (1 epoch, 1 fold)
- ✅ Results saved correctly

**Expected output:**
```
============================================================
  TEST SUMMARY
============================================================
  ✅ PASS - GPU Capability Check
  ✅ PASS - Python Imports
  ✅ PASS - Data Files
  ✅ PASS - Training Test
  ✅ PASS - Results Saved

Total: 5/5 tests passed

🎉 All tests passed! Ready to train full model.
```

---

### 🔧 Level 2: Detailed GPU Check (1 minute)
**Best for**: Checking GPU capability and performance

```bash
python check_gpu_capability.py
```

**Shows:**
- GPU info (VRAM, compute capability)
- Mixed precision support
- Tensor cores availability
- Performance benchmark (float32 vs float16)
- Memory requirements

**Example output:**
```
============================================================
GPU Capability Checker for Mixed Precision Training
============================================================
✅ CUDA is available

📊 GPU Information:
   Device Name: NVIDIA RTX 3080
   CUDA Version: 11.8
   Total GPU Memory: 10.0 GB

✅ Automatic Mixed Precision (AMP) is supported

🔧 Compute Capability:
   Compute Capability: 8.6
   ✅ Has Tensor Cores

⚡ Performance Benchmark:
   float32: 125.45ms
   float16: 62.12ms
   Speedup: 2.02x
```

---

### 🎓 Level 3: Full Training Test (20-30 minutes)
**Best for**: Verifying the complete training pipeline

```bash
python full_test.py
```

**Runs:**
- GPU capability check
- Training for 2 folds x 100 epochs
- Results validation
- Summary statistics

**Example output:**
```
============================================================
  FULL TEST SUITE - PharmaLink GCN
============================================================

Test Configuration:
  Dataset: C-dataset
  Epochs: 100
  K-Fold: 2
  Total iterations: 200
  Estimated time: ~10 minutes

[Training output...]

============================================================
  TEST SUMMARY
============================================================
End time: 2026-04-26 14:35:22

✅ Full test completed!

Next steps:
  1. Review results in: AI_ENGINE/data/results/
  2. For full training: .\train.ps1 -dataset C-dataset -model base
  3. Start full pipeline: .\start_all.ps1
```

---

### 🌐 Level 4: API Endpoint Test (1 minute)
**Best for**: Testing FastAPI endpoints (requires API running)

```bash
# Terminal 1: Start API server
python AI_ENGINE/api.py

# Terminal 2: Run API tests
python api_test.py
```

**Tests:**
- ✅ Health check
- ✅ Get datasets
- ✅ Get drugs
- ✅ Get diseases
- ✅ Get proteins
- ✅ Predict endpoint
- ✅ Get statistics
- ✅ Get model comparison

---

## 🎯 Recommended Test Workflow

### First Time Setup

```bash
# Step 1: Check GPU (1 min)
python check_gpu_capability.py

# Step 2: Quick validation (5 min)
python quick_test.py

# Step 3: If all pass, ready for production!
```

### Before Production Training

```bash
# Step 1: Quick test (5 min)
python quick_test.py

# Step 2: Full test (20-30 min)
python full_test.py

# Step 3: Start training
.\train.ps1 -dataset C-dataset -model base
```

### Full System Testing

```bash
# Step 1: Quick test
python quick_test.py

# Step 2: Start API server (Terminal 1)
python AI_ENGINE/api.py

# Step 3: Test API (Terminal 2)
python api_test.py

# Step 4: Start full pipeline (Terminal 3)
.\start_all.ps1
```

---

## 📊 Expected Results

### After quick_test.py
```
✅ All 5 tests passed
- GPU check successful
- All libraries imported
- Data files found
- Training works
- Results saved
```

### After full_test.py (with 2 folds, 100 epochs)
```
✅ Training completed
- 2 fold results saved
- Summary JSON created
- Metrics calculated

Results location:
  AI_ENGINE/data/results/C-dataset_AMNTDDA_fold_results.csv
  AI_ENGINE/data/results/C-dataset_AMNTDDA_summary.json
```

### After api_test.py
```
✅ All 8 API endpoints working
- Health check: OK
- Datasets: 2+ available
- Drugs: Retrieved
- Diseases: Retrieved
- Proteins: Retrieved
- Predictions: Working
- Statistics: Available
- Comparisons: Available
```

---

## 🐛 Troubleshooting Test Issues

### Error: "CUDA out of memory"
```bash
# Run quick_test with reduced model
python quick_test.py

# If still fails, check available VRAM
nvidia-smi
```

### Error: "autocast not supported"
```bash
# Update PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python check_gpu_capability.py
```

### Error: "Data files not found"
```bash
# Check if data folder exists
ls AMDGT_main/data/C-dataset/

# If missing, download or extract data
# (Make sure AMDGT_main/data is populated)
```

### Error: "API connection refused"
```bash
# Make sure API server is running in another terminal
python AI_ENGINE/api.py

# Then run api_test
python api_test.py
```

### Error: "ImportError: No module named 'xyz'"
```bash
# Install missing package
pip install xyz

# Then run test again
python quick_test.py
```

---

## 💡 Tips for Testing

1. **Start with quick_test.py** - It's fast and catches most issues
2. **Check GPU first** - If GPU fails, you need better drivers/CUDA
3. **Test API separately** - Don't mix training and API tests
4. **Run once before production** - Validate pipeline before long training

---

## 🎯 Test Coverage

### ✅ quick_test.py covers:
- [x] GPU capability
- [x] Python environment
- [x] Data availability
- [x] Basic training
- [x] Results persistence

### ✅ full_test.py covers:
- [x] Extended training (2 folds)
- [x] Results accuracy
- [x] Summary calculation
- [x] Performance metrics

### ✅ api_test.py covers:
- [x] API server connectivity
- [x] All main endpoints
- [x] Request/response handling
- [x] Data retrieval

### ✅ check_gpu_capability.py covers:
- [x] CUDA availability
- [x] GPU specifications
- [x] Mixed precision support
- [x] Performance benchmark

---

## 🚀 Next Steps After Tests Pass

1. **Run full training**
   ```bash
   .\train.ps1 -dataset C-dataset -model base
   ```

2. **Start full pipeline**
   ```bash
   .\start_all.ps1
   ```

3. **Access web interface**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:3000/api
   - FastAPI docs: http://localhost:8000/docs

---

## 📝 Test Results Location

After running tests, check these files:

```
├── AI_ENGINE/data/results/
│   ├── C-dataset_AMNTDDA_fold_results.csv  (fold metrics)
│   ├── C-dataset_AMNTDDA_summary.json      (summary stats)
│   └── C-dataset_comparison.json           (model comparison)
│
└── Logs (console output during tests)
```

---

## ⏱️ Expected Execution Times

| Test | Time | GPU | CPU |
|------|------|-----|-----|
| check_gpu_capability.py | ~1 min | Minimal | ~50MB |
| quick_test.py | ~5 min | 100-200MB | ~50MB |
| full_test.py (2 folds) | ~20-30 min | ~300-400MB | ~100MB |
| api_test.py | ~1 min | Minimal | ~50MB |

---

## ✅ Checklist Before Production

- [ ] Ran `python quick_test.py` successfully
- [ ] Ran `python full_test.py` successfully
- [ ] All tests passed (5/5 or 8/8)
- [ ] GPU memory available
- [ ] Data files present
- [ ] No errors in console output

If all checked ✅, you're ready to:
```bash
.\train.ps1 -dataset C-dataset -model base
```

---

**Version**: 1.0  
**Last Updated**: 2026-04-26  
**Status**: ✅ Production Ready
