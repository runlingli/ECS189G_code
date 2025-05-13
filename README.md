# ECS189G Winter 2025 Source Code Template

This repository contains the source code template for ECS189G course at UC Davis.

## Copyright Notice
This code is based on the original work by:
```
Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
```

This code is for TA review purposes only.

## Project Structure

```
.
├── data/                    # Dataset files
│   ├── stage_1_data/       # Stage 1 datasets
│   ├── stage_2_data/       # Stage 2 datasets
│   ├── stage_3_data/       # Stage 3 datasets (MNIST, CIFAR-10, ORL)
│   ├── stage_4_data/       # Stage 4 datasets (to be implemented)
│   └── stage_5_data/       # Stage 5 datasets (to be implemented)
├── local_code/             # Source code
│   ├── base_class/         # Base classes
│   ├── stage_1_code/       # Stage 1 implementations
│   ├── stage_2_code/       # Stage 2 implementations (MLP model)
│   ├── stage_3_code/       # Stage 3 implementations (CNN models)
│   ├── stage_4_code/       # Stage 4 implementations (to be implemented)
│   └── stage_5_code/       # Stage 5 implementations (to be implemented)
├── result/                 # Results and visualizations
│   ├── stage_1_result/     # Stage 1 results
│   ├── stage_2_result/     # Stage 2 results
│   ├── stage_3_result/     # Stage 3 results
│   ├── stage_4_result/     # Stage 4 results (to be implemented)
│   └── stage_5_result/     # Stage 5 results (to be implemented)
├── script/                 # Scripts to run experiments
│   ├── stage_1_script/     # Stage 1 scripts
│   ├── stage_2_script/     # Stage 2 scripts
│   ├── stage_3_script/     # Stage 3 scripts
│   ├── stage_4_script/     # Stage 4 scripts (to be implemented)
│   └── stage_5_script/     # Stage 5 scripts (to be implemented)
└── README.md              # Project documentation
```

## Current Progress

### Stage 2 Implementation
- MLP model with configurable layers
- Training visualization
- Performance evaluation metrics
- Dataset handling and preprocessing

### Stage 3 Implementation
- CNN models for different datasets:
  - MNIST CNN model
  - CIFAR-10 CNN model
  - ORL CNN model
- Features:
  - Convolutional layers with configurable parameters
  - Pooling layers
  - Dropout for regularization
  - Batch normalization
  - Training and evaluation pipeline
  - Performance metrics calculation
  - Model saving and loading
  - Multi-process data loading
  - GPU support (CUDA)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm (for progress bars)
- CUDA (optional, for GPU acceleration)

## Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib tqdm
   ```
3. Run the scripts in the `stage_#_script` directory:
   ```bash
      python script_***.py
   ```

## Results

The current implementation includes:
- Training loss visualization
- Model performance metrics
- Ablation studies results
- CNN model performance on different datasets
- Training progress visualization
- Model architecture visualization

For detailed results, please refer to the report in the repository.
