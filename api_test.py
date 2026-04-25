"""
api_test.py — API Test Suite

Test FastAPI endpoints (AI_ENGINE).
Cần chạy api.py trước: python AI_ENGINE/api.py

Usage:
    # Terminal 1: Start API server
    python AI_ENGINE/api.py
    
    # Terminal 2: Run tests
    python api_test.py
"""

import requests
import json
import time
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"
TIMEOUT = 10

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.RESET}\n")

def test_connection():
    """Test if API is running"""
    print("🔄 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=TIMEOUT)
        if response.status_code == 200:
            print(f"{Colors.GREEN}✅ API is running on {API_BASE}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}❌ API returned status {response.status_code}{Colors.RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}❌ Cannot connect to {API_BASE}")
        print("   Please start API first: python AI_ENGINE/api.py{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False

def test_endpoint(method, endpoint, description, **kwargs):
    """Test a single endpoint"""
    url = f"{API_BASE}{endpoint}"
    print(f"🔄 {description}")
    print(f"   {method.upper()} {endpoint}")
    
    try:
        if method == "get":
            response = requests.get(url, timeout=TIMEOUT, **kwargs)
        elif method == "post":
            response = requests.post(url, timeout=TIMEOUT, **kwargs)
        else:
            print(f"   {Colors.YELLOW}⚠️  Unknown method{Colors.RESET}")
            return False
        
        if response.status_code in [200, 201]:
            print(f"   {Colors.GREEN}✅ Status {response.status_code}{Colors.RESET}")
            
            # Try to parse JSON
            try:
                data = response.json()
                if isinstance(data, dict):
                    items = len(data) if isinstance(data, dict) else 1
                    print(f"   ✓ Response: {items} items")
                elif isinstance(data, list):
                    print(f"   ✓ Response: {len(data)} items")
                else:
                    print(f"   ✓ Response OK")
            except:
                print(f"   ✓ Response OK (not JSON)")
            
            return True
        else:
            print(f"   {Colors.RED}❌ Status {response.status_code}{Colors.RESET}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   {Colors.RED}❌ Timeout ({TIMEOUT}s){Colors.RESET}")
        return False
    except Exception as e:
        print(f"   {Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False

def main():
    print_header("API TEST SUITE - FastAPI AI_ENGINE")
    
    # Test connection
    if not test_connection():
        print(f"\n{Colors.RED}API is not running. Start it first:{Colors.RESET}")
        print(f"  {Colors.YELLOW}python AI_ENGINE/api.py{Colors.RESET}")
        return 1
    
    time.sleep(1)  # Give API time to stabilize
    
    # Test endpoints
    print_header("TESTING ENDPOINTS")
    
    results = {}
    
    # Test 1: Health check
    print("📋 Test 1: Health Check")
    results['health'] = test_endpoint("get", "/", "Health check")
    
    # Test 2: Get datasets
    print("\n📋 Test 2: Get Available Datasets")
    results['datasets'] = test_endpoint(
        "get", 
        "/datasets",
        "Get list of datasets"
    )
    
    # Test 3: Get drugs
    print("\n📋 Test 3: Get Drugs")
    results['drugs'] = test_endpoint(
        "get",
        "/data/C-dataset/drugs",
        "Get drugs from C-dataset"
    )
    
    # Test 4: Get diseases
    print("\n📋 Test 4: Get Diseases")
    results['diseases'] = test_endpoint(
        "get",
        "/data/C-dataset/diseases",
        "Get diseases from C-dataset"
    )
    
    # Test 5: Get proteins
    print("\n📋 Test 5: Get Proteins")
    results['proteins'] = test_endpoint(
        "get",
        "/data/C-dataset/proteins",
        "Get proteins from C-dataset"
    )
    
    # Test 6: Predict endpoint
    print("\n📋 Test 6: Predict Drug-Disease Association")
    results['predict'] = test_endpoint(
        "post",
        "/predict",
        "Make a prediction",
        json={
            "drug_index": 0,
            "disease_index": 0,
            "dataset": "C-dataset"
        }
    )
    
    # Test 7: Get stats
    print("\n📋 Test 7: Get Statistics")
    results['stats'] = test_endpoint(
        "get",
        "/stats/C-dataset",
        "Get dataset statistics"
    )
    
    # Test 8: Get comparison
    print("\n📋 Test 8: Get Model Comparison")
    results['comparison'] = test_endpoint(
        "get",
        "/comparison/C-dataset",
        "Get model performance comparison"
    )
    
    # Summary
    print_header("TEST SUMMARY")
    
    tests = [
        ("Health Check", results.get('health', False)),
        ("Get Datasets", results.get('datasets', False)),
        ("Get Drugs", results.get('drugs', False)),
        ("Get Diseases", results.get('diseases', False)),
        ("Get Proteins", results.get('proteins', False)),
        ("Predict", results.get('predict', False)),
        ("Get Stats", results.get('stats', False)),
        ("Get Comparison", results.get('comparison', False)),
    ]
    
    passed = sum(1 for _, r in tests if r)
    total = len(tests)
    
    print()
    for test_name, result in tests:
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if result else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"  {status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed\n")
    
    if passed == total:
        print(f"{Colors.GREEN}🎉 All API tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}⚠️  {total - passed} test(s) failed{Colors.RESET}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        sys.exit(1)
