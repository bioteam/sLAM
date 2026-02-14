#!/bin/bash
# Local test script to validate the CI workflow before pushing to GitHub
# This script mimics what the GitHub Action does

set -e  # Exit on error

echo "=========================================="
echo "Testing sLAM Model Generation (CI Mode)"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "sLAM/make-slam.py" ]; then
    echo "ERROR: Please run this script from the sLAM project root directory"
    exit 1
fi

# Clean up any previous test files
echo "1. Cleaning up previous test files..."
rm -f test-model.keras test-model.pkl *.png
echo "   ✓ Cleaned up"
echo ""

# Check Python version
echo "2. Checking Python version..."
python --version
echo ""

# Check if dependencies are installed
echo "3. Checking if dependencies are installed..."
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "   WARNING: TensorFlow not found. Installing dependencies..."
    pip install -e .
fi
echo "   ✓ Dependencies OK"
echo ""

# Download NLTK data
echo "4. Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab')"
echo "   ✓ NLTK data downloaded"
echo ""

# Run make-slam.py with minimal settings
echo "5. Running make-slam.py with minimal test settings..."
echo "   (Using: 10 datasets, 1 epoch, context_size=16, d_model=64)"
python sLAM/make-slam.py \
    --num_datasets 10 \
    --epochs 1 \
    --context_size 16 \
    --d_model 64 \
    -n test-model \
    -p "This is a test" \
    -v

echo ""

# Verify model file exists
echo "6. Verifying model file exists..."
if [ ! -f "test-model.keras" ]; then
    echo "   ✗ ERROR: Model file test-model.keras was not created!"
    exit 1
fi
echo "   ✓ Model file test-model.keras exists"
ls -lh test-model.keras
echo ""

# Verify tokenizer file exists
echo "7. Verifying tokenizer file exists..."
if [ ! -f "test-model.pkl" ]; then
    echo "   ✗ ERROR: Tokenizer file test-model.pkl was not created!"
    exit 1
fi
echo "   ✓ Tokenizer file test-model.pkl exists"
ls -lh test-model.pkl
echo ""

# Test model generation
echo "8. Testing model generation..."
python sLAM/generate.py \
    -n test-model \
    -p "This is a test"
echo ""

echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "The CI workflow should work correctly."
echo "You can now push the .github/workflows/test-slam-model.yml file."
echo ""

# Cleanup test files
echo "9. Cleaning up test artifacts..."
rm -f test-model.keras test-model.pkl *.png
echo "   ✓ Test files cleaned up"
echo ""
