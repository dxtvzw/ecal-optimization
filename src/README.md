# Sensor Size Optimization for Particle Physics

This repository contains the implementation and experiments conducted for the project titled "Sensor size optimization for particle physics." The project aims to optimize the sensor size of an electromagnetic calorimeter (ECAL) in the context of high-energy physics experiments, specifically focusing on the reconstruction of photon energy and position within the sensor cell.

## Project Overview

In high-energy physics experiments, accurately reconstructing the energy and position of particles such as photons is crucial. This project addresses this challenge by exploring the optimization of sensor size through deep learning methods. The primary goal is to find the best matrix shape that allows for precise reconstruction while considering the trade-off between resolution and cost.

## Dataset

The data used in this project is generated using the GEANT4 simulator and consists of samples with varying matrix shapes ($10 \times 10, 15 \times 15, 20 \times 20, 25 \times 25, 30 \times 30, 40 \times 40$). Each sample includes a matrix of positive real values representing ECAL measurements, the initial energy of a photon uniformly distributed between $1$ and $100$ GeV, and the position where the photon entered the sensor cell.

## File Structure

- `checkpoints/`: Contains PyTorch model checkpoints saved during the training process.
- `data/`: Directory for GEANT4 simulation files.
- `data_numpy/`: Directory for preprocessed data files in '.npz' format for faster loading.
- `config.py`: Python script to load configurations from 'config.yaml' file.
- `config.yaml`: Default configuration file containing parameters for data paths, logging, model settings, training, and data processing.
- `data.py`: Module with functions to load data from '.root' or '.npz' files, create datasets, dataloaders, and implement data augmentation.
- `ensemble.py`: Script to create basic ensembles of multiple model runs and perform inference for metric evaluation.
- `main.py`: Main script responsible for data loading, model initialization, training, and metric computation. Supports simultaneous independent runs using `torchrun`. Path to custom configuration file can be specified as a command-line argument
- `model.py`: Module containing implementations of model architectures, including ResNet, Vision Transformer, and others, as well as learning rate schedulers.
- `prepare_data.py`: Script to convert '.root' files into '.npz' format for faster data loading.
- `requirements.txt`: File listing dependencies required to run the project.
- `run_experiments.py`: Script to execute specific experiments defined in the file. Experiment IDs are specified as command-line arguments.
- `tmp.ipynb`: Jupyter notebook used for testing and exploring data and code components.
- `utils.py`: Module with utility functions for saving model checkpoints, computing model parameters, custom loss functions, and metric computation.

## Running the Code

To run the project, follow these steps:

Install Dependencies: First, ensure that you have all the dependencies listed in requirements.txt installed. You can do this by running:

```shell
pip install -r requirements.txt
```

Preprocess Data (if necessary): If you haven't already preprocessed the data, execute `prepare_data.py` to convert `.root` files into `.npz` format for faster data loading:

```shell
python prepare_data.py
```

Configure Settings: Adjust configurations in `config.yaml` according to your requirements. This file contains parameters for data paths, logging, model settings, training, and data processing.

Run Experiments: Use `run_experiments.py` to execute specific experiments defined in the file. Specify the experiment ID as a command-line argument:

```shell
python run_experiments.py --exp_id <experiment_id>
```

Replace `<experiment_id>` with the ID of the experiment you want to run.

Training: Alternatively, run the main training process using `main.py`. If you have multiple GPUs available, you can use `torchrun` to run simultaneous independent training processes. Specify the number of processes with the `--nproc_per_node` flag (each process will use a single separate GPU available):

```shell
torchrun --nproc_per_node <num_processes> main.py
```

Replace `<num_processes>` with the number of processes you want to run.

Monitor Training with `Wandb`: Throughout the training process, metrics and logs are automatically logged to Wandb. You can monitor the training progress, visualize metrics, and compare experiments on the Wandb dashboard.

Most of the experiments were recorded using `wandb`, the runs can be viewed using the following [link](https://wandb.ai/dxtvzw/ECAL%20optimization).

Evaluate Models: After training or running experiments, you can evaluate model performance using `ensemble.py`. This script creates basic ensembles of multiple model runs and performs inference for metric evaluation.

Explore Results: Analyze the results, metrics, and model performance to gain insights into the optimization of sensor size for particle physics experiments.

By following these steps, you can effectively run and evaluate the codebase for the "Sensor size optimization for particle physics" project. For more detailed instructions and explanations, refer to the respective files and documentation within the repository.

## Metrics

The project evaluates model performance using the following metrics:

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- RMSE/E (Weighted RMSE for energy reconstruction)
- MAE/E (Weighted MAE for energy reconstruction)
- RMSLE (Root Mean Squared Logarithmic Error)

These metrics provide insights into the accuracy and precision of both energy and position reconstruction across different sensor sizes.

## Model Architectures

Various deep learning models are implemented for experimentation, including ResNet, Vision Transformer, and custom architectures. Each model consists of two heads for reconstructing energy and position.

## Contributing

Contributions to the project are welcome. Feel free to open issues for bug reports, feature requests, or general discussions.

## Acknowledgments

We acknowledge the use of GEANT4 simulator for generating simulation data and the contributions of the open-source community for the libraries and tools used in this project.
