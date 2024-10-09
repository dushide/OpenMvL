
## OpenViewer
OpenMvL is a framework for open-set multi-view learning that integrates view-specific attributes, dynamic uncertainty estimation, and evidential fusion. This repository contains the code and datasets used for the experiments in our paper.

### Installation
1. **Clone the repository:**
- git clone https://anonymous.4open.science/r/OpenMvL-1C66/
- cd OpenMvL
2. **Set up the environment:**
Use Conda to create an environment with the required dependencies:
- conda create -n openmvl python=3.7.2
- conda activate openmvl
- pip install -r requirements.txt

### Datasets Preparation
- For all datasets, please obtain them from the following links: <https://drive.google.com/drive/folders/1ew_1h023jA-bqnZrbWnsRFEfjRFnqdpD?usp=sharing>;
- Download datasets from the provided links.
- Place the datasets in the `/data` directory.

### Usage Instructions
1. **Training and Testing the Model:** To train the model on a specific dataset, run the following command:
- python test_OpenMvL.py --dataset DatasetName --config ./config/config.yaml
2. **Modifying Configurations:**
- Edit `config/config.yaml` to modify settings such as the dataset, fusion method, or loss functions, or set default values directly in test_OpenMvL.py.

### Results Reproduction
For example, **Scene15** dataset:
- python train_OpenMvL.py --dataset scene15--config ./config/scene15.yaml to see the performance under the unseen classes are one.

### Notes
 - Ensure all dependencies are installed, as listed in the requirements.txt.   
 - The code is designed to run on both **CPU** and **GPU**.   
 - For custom datasets, modify the dataset loader in loadMatData.py.

### Contact
For further questions or clarifications, please raise an issue in this repository.


