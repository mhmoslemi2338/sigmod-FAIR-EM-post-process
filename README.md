# FairEM 

## Overview

This repository contains the code and supplementary resources for the paper [**"Mitigating Matching Biases Through Score Calibration"**](https://arxiv.org/abs/2411.01685). Our work addresses biases in entity matching models by proposing a score calibration technique that minimizes disparities across different groups. By applying optimal transport-based calibration, we ensure fairness across matching outcomes, with minimal loss in model accuracy.

## Introduction

Record matching, the task of identifying records that correspond to the same real-world entities across databases, is critical for data integration in domains like healthcare, finance, and e-commerce. While traditional record matching models focus on optimizing accuracy, fairness issues have attracted increasing attention. Biased outcomes in record matching can result in unequal error rates across demographic groups, raising ethical and legal concerns. Existing research primarily addresses fairness at specific decision thresholds. However, threshold-specific metrics may overlook cumulative biases across varying thresholds. In this project, we adapt fairness metrics traditionally applied in regression models to evaluate cumulative bias across all thresholds in record matching. We propose a novel post-processing calibration method, leveraging optimal transport theory and Wasserstein barycenters. This approach treats any matching model as a black box, making it applicable to a wide range of models without access to their training data. Also, to address limitations in reducing EOD and EO differences, we introduce a conditional calibration method, which empirically achieves fairness across widely used benchmarks and state-of-the-art matching methods.


## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)


## Requirements

The following dependencies are required to run the calibration code:

- Python 3.8+
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- SciPy
- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- [Gender Guesser](https://pypi.org/project/gender-guesser/)

------
## Usage

1. **Data Preparation**: Obtain the dataset from the DeepMatcher library: [link](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). Place the dataset in the `DataDir` directory. For each dataset, create a new subdirectory inside `DataDir` containing `train.csv`, `valid.csv`, and `test.csv` files.

2. **Preparing Matching Scores**: We use various state-of-the-art entity matching methods implemented in their respective repositories. You can use any matching method of your choice. Save the matching scores in the `scores` directory. For each matching method and dataset, create a subdirectory within `scores`, named as `[matching_method_name]_[dataset_name]`, and place three files inside: `score_train.csv`, `score_valid.csv`, and `score_test.csv`. Each CSV file should contain two columns: the matching score for each row and the corresponding actual label.

3. **Creating Sensitive Vector**: You need to create a boolean vector for each dataset, where 1 indicates that the entity belongs to a minority group and 0 indicates it belongs to a majority group. You can generate this vector by running the following command:

   ```bash
   python3 0_preProcess.py
   ```

4. **Initial Bias Measurement**: To measure the initial bias present in the dataset and models, use the `2_Biases in Record Matching Scores.ipynb` notebook.

5. **Calibration**: For the calibration process, refer to the `3_Calibration Analysis.ipynb` notebook. This notebook contains detailed documentation, and all results are saved to the `FIGURES` directory.

6. **Conditional Calibration**: For conditional calibration, use the `4_Conditional Calibration Analysis.ipynb` notebook. This notebook also contains detailed documentation, and all results are saved to the `FIGURES` directory.

**Note**: Each calibration method saves the results in a pickle file in the `saved_params` folder.

**Note**: The `Calibrate.py` script provides functions for calibration, and `fairness.py` contains functions for fairness measurement. Also, fig_params.py provides parameters for plot savings.


------## Results

All figures can be found in the `FIGURES` directory.

The next table is the complete version of Table 2 in Section 6.2.1 (Biases in Record Matching Scores) of the original paper, comparing distributional parity measures and traditional measures for various thresholds, as well as the AUC of models and datasets.

![Table 2 - Biases in Record Matching Scores](FIGURES/table2_complete.png)


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


