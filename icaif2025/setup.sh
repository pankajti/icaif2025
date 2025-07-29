#!/bin/bash

# Define the flat package directory
PACKAGE_DIR="timesfm_forecasting"

# Create the directory
mkdir -p $PACKAGE_DIR

# Create __init__.py to make it a package
touch $PACKAGE_DIR/__init__.py

# Create module files with brief headers
echo "# Data provider module (e.g., portfolio holdings, price loader)" > $PACKAGE_DIR/data_provider.py
echo "# TimesFM model factory for v1 and v2" > $PACKAGE_DIR/model_factory.py
echo "# Core forecaster using TimesFM" > $PACKAGE_DIR/forecaster.py
echo "# Evaluation metrics (RMSE, MAE, DA)" > $PACKAGE_DIR/evaluator.py
echo "# Visualization logic (matplotlib)" > $PACKAGE_DIR/plotter.py
echo "# Hyperparameter tuning logic" > $PACKAGE_DIR/tuner.py
echo "# Main executable script" > $PACKAGE_DIR/app.py

# Create run launcher
cat <<EOF > run.sh
#!/bin/bash
python3 timesfm_forecasting/app.py
EOF
chmod +x run.sh

echo "✅ Flat package structure created in './$PACKAGE_DIR'. Use './run.sh' to execute."
