#!/bin/bash
set -e  # Exit on any error

echo "------------------------"
echo "Running python scripts:"
echo "------------------------"
echo ""

"$1" fibonacci/main.py --gc-frequency 1
"$1" mandelbrot/mandelbrot.py --gc-frequency 1
./run_python_tests.sh "$1"

echo ""
echo "------------------------"
echo "Testing LLVM backend:"
echo "------------------------"
echo ""

./run_llvm_python_tests.sh "$1"