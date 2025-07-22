```
88888888ba                                            88                     88           
88      "8b                                     ,d    88                     88           
88      ,8P                                     88    88                     88           
88aaaaaa8P'  ,adPPYba,   ,adPPYba,  ,adPPYba, MM88MMM 88          ,adPPYYba, 88,dPPYba,   
88""""""8b, a8"     "8a a8"     "8a I8[    ""   88    88          ""     `Y8 88P'    "8a  
88      `8b 8b       d8 8b       d8  `"Y8ba,    88    88          ,adPPPPP88 88       d8  
88      a8P "8a,   ,a8" "8a,   ,a8" aa    ]8I   88,   88          88,    ,88 88b,   ,a8"  
88888888P"   `"YbbdP"'   `"YbbdP"'  `"YbbdP"'   "Y888 88888888888 `"8bbdP"Y8 8Y"Ybbd8"'   
```

# Supervised Learning with Gradient Boosting

## Features

### Dataset
- **Management**: load, save, import and information
- **Visualization**: raw values, heatmap, statistics, PCA
- **Filtering**: features and class selection, resampling
- **Balancing**: parametric oversample and undersample
### Model
- **Training**: early stopping support and result plots
- **Tuning**: automatic hyperparameter optimization
- **Inspection**: tree graph and feature importance
- **Export**: generate executable C++/Python code
### Evaluation
- **Performance**: classification report and confusion matrix
- **Graphs**: metric curves and probability histogram
- **Optimize**: optimal thresholds with various metrics
- **Export**: model predictions and output probabilities

## Installation
1. Install latest **Miniconda** from [official site](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Create a new virtual environment:
   1. Install required packages (choose one of the following options):
      - **CPU only**: `conda env create -f environment_cpu.yml`
      - **GPU acceleration**: `conda env create -f environment_gpu.yml`
   2. Activate the environment: `conda activate bl`

## Usage
Run main script from inside `bl` environment:
```bash
python boostlab.py
```

**Note**: If images inside `icons` folder are modified, runtime resources need to be updated before running the application:
```bash
pyside6-rcc resources.qrc -o resources.py
```
