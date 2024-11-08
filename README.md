# FairEM 

## Overview

This repository contains the code and supplementary resources for the paper [**"Mitigating Matching Biases Through Score Calibration"**](https://arxiv.org/abs/2411.01685). Our work addresses biases in entity matching models by proposing a score calibration technique that minimizes disparities across different groups. By applying optimal transport-based calibration, we ensure fairness across matching outcomes, with minimal loss in model accuracy.

## Introduction

Record matching, the task of identifying records that correspond to the same real-world entities across databases, is critical for data integration in domains like healthcare, finance, and e-commerce. While traditional record matching models focus on optimizing accuracy, fairness issues have attracted increasing attention. Biased outcomes in record matching can result in unequal error rates across demographic groups, raising ethical and legal concerns. Existing research primarily addresses fairness at specific decision thresholds. However, threshold-specific metrics may overlook cumulative biases across varying thresholds. In this project, we adapt fairness metrics traditionally applied in regression models to evaluate cumulative bias across all thresholds in record matching. We propose a novel post-processing calibration method, leveraging optimal transport theory and Wasserstein barycenters. This approach treats any matching model as a black box, making it applicable to a wide range of models without access to their training data. Also, to address limitations in reducing EOD and EO differences, we introduce a conditional calibration method, which empirically achieves fairness across widely used benchmarks and state-of-the-art matching methods.


## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
## Introduction
- [Citation](#citation)


This project addresses fairness issues in entity matching by calibrating score distributions to mitigate biases while maintaining model accuracy. It includes all necessary code to reproduce the experiments described in the paper.


## Requirements

To run the code, you need the following dependencies:

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

You can install all the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/mhmoslemi2338/mitigating-matching-bias.git
cd mitigating-matching-bias
```

## Usage

1. **Data Preparation**: Prepare the dataset for calibration. You can use the sample dataset provided in the `data/` directory or your own dataset. Make sure it is pre-processed as per the instructions in the paper.

2. **Training the Model**: To train the model on the prepared dataset, run:

   ```bash
   python train.py --config configs/train_config.yaml
   ```

3. **Score Calibration**: To apply score calibration on trained models, use the following command:

   ```bash
   python calibrate.py --model models/model.pth --data data/test_data.csv
   ```

## Experiments

Scripts for conducting experiments and evaluating the calibration's impact on fairness and accuracy are available in the `experiments/` directory. For example:

```bash
python experiments/run_experiments.py --config configs/experiment_config.yaml
```

## Results

The main results of our experiments can be reproduced using the scripts in the `results/` directory. The expected output includes:

- Accuracy and fairness metrics before and after calibration
- Plots showing the disparity in scores across demographic groups

All plots and evaluation metrics will be saved to the `results/` directory.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{moslemi2024mitigating,
  title={Mitigating Matching Biases Through Score Calibration},
  author={Moslemi, Mohammad Hossein and Milani, Mostafa},
  journal={arXiv preprint arXiv:2411.01685},
  year={2024}
}
```

## Contact

For any questions or issues, feel free to open an issue on this repository or contact me via email: [mohammad.moslemi@uwo.ca](mailto:mohammad.moslemi@uwo.ca).


