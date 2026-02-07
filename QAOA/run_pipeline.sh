#!/bin/bash

# Quantum Exam Scheduling - Quick Start Script
# This script installs dependencies and runs the full QAOA pipeline

echo "============================================================"
echo "Quantum Exam Scheduling - QAOA Pipeline Setup"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please activate your virtual environment first:"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Install dependencies
echo "============================================================"
echo "Step 1: Installing Dependencies"
echo "============================================================"
echo ""

echo "Installing packages from requirements.txt..."

# Use uv for fast package installation
if command -v uv &> /dev/null; then
    echo "Using uv (fast package manager)..."
    uv pip install -r requirements.txt
else
    echo "uv not found, using standard pip..."
    python -m pip install -q -r requirements.txt
fi

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully!"
else
    echo "❌ Installation failed!"
    exit 1
fi

echo ""

# Run the pipeline
echo "============================================================"
echo "Step 2: Running QAOA Pipeline (SIMULATOR MODE)"
echo "============================================================"
echo ""

# Step 1: Generate Dataset
echo "───────────────────────────────────────────────────────────"
echo "1/5: Generating Dataset..."
echo "───────────────────────────────────────────────────────────"
python 01_generate_dataset.py
echo ""

# Step 2: Visualize Graph
echo "───────────────────────────────────────────────────────────"
echo "2/5: Visualizing Conflict Graph..."
echo "───────────────────────────────────────────────────────────"
python 02_visualize_graph.py
echo ""

# Step 3: Build QUBO
echo "───────────────────────────────────────────────────────────"
echo "3/5: Building QUBO Matrix..."
echo "───────────────────────────────────────────────────────────"
python 03_build_qubo.py
echo ""

# Step 4: Solve with QAOA
echo "───────────────────────────────────────────────────────────"
echo "4/5: Solving with QAOA (Simulator)..."
echo "───────────────────────────────────────────────────────────"
python 04_solve_qaoa.py
echo ""

# Step 5: Assign Rooms
echo "───────────────────────────────────────────────────────────"
echo "5/5: Assigning Rooms..."
echo "───────────────────────────────────────────────────────────"
python 05_assign_rooms.py
echo ""

# Summary
echo "============================================================"
echo "PIPELINE COMPLETE! ✓"
echo "============================================================"
echo ""

# Read the latest run directory
if [ -f "output/latest_run.txt" ]; then
    RUN_DIR=$(cat output/latest_run.txt)
    echo "Generated files in: $RUN_DIR"
    echo ""
    echo "  📊 Datasets:          $RUN_DIR/datasets/"
    echo "  📈 Visualizations:    $RUN_DIR/visualizations/"
    echo "  ⚛️  Solutions:         $RUN_DIR/solutions/"
    echo "  📅 Timetables:        $RUN_DIR/timetables/"
    echo ""
    echo "Check the timetable:"
    echo "  cat $RUN_DIR/timetables/final_timetable_tiny_K3.csv"
else
    echo "  📊 Dataset:           output/run_*/datasets/"
    echo "  📈 Visualizations:    output/run_*/visualizations/"
    echo "  ⚛️  Solutions:         output/run_*/solutions/"
    echo "  📅 Timetables:        output/run_*/timetables/"
fi

echo ""
echo "Next steps:"
echo "  1. Check the generated timetable CSV file"
echo "  2. Review visualizations (PNG files)"
echo "  3. For real IBM Quantum hardware, see README.md"
echo ""
echo "============================================================"

