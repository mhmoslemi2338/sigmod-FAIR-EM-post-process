
"""
mhmoslemi2338@gmail.com
This script processes sensitive attribute vectors for various datasets and saves them for later use. 
It reads train, validation, and test CSV files for each specified task, generates sensitive attribute vectors 
using the `make_sens_vector` function, and saves the results in pickle files. The main purpose of this script 
is to prepare sensitive attribute data for further analysis in the context of fairness research.
"""

import os
import pickle
import pandas as pd
from fairness import make_sens_vector, sens_dict  # Assuming make_sens_vector is defined in fairness module

# Base directory for data files
path_base = os.getcwd()

# Define tasks for which sensitivity attributes are calculated
tasks = [
    'Fodors-Zagat', 'DBLP-GoogleScholar', 'iTunes-Amazon', 'Walmart-Amazon',
    'Amazon-Google', 'Beer', 'DBLP-ACM']

# Function to process datasets and save sensitive attribute vectors
def process_sens_attributes(task_list, dataset_type):
    """
    Process sensitive attribute vectors for a given list of tasks and dataset type.

    Parameters:
    - task_list: list of task names (datasets) to process
    - dataset_type: type of dataset to process (e.g., 'train', 'valid', 'test')
    """
    sens_attributes = {}

    for task in task_list:
        print(f"Processing {task} - {dataset_type.upper()}")
        
        # Read the dataset CSV file based on dataset type
        file_path = f"{path_base}/DATA_VLDB/{task}/{dataset_type}.csv"
        df = pd.read_csv(file_path)
        
        # Generate sensitive attribute vector for the dataset
        sens_vector = make_sens_vector(df, task, sens_dict)
        
        # Store the sensitive attribute vector
        sens_attributes[task] = sens_vector
        print("DONE\n")

    # Save the sensitive attribute vectors as a pickle file
    output_file = f"saved_params/sens_attr_dict_{dataset_type}.pkl"
    with open(output_file, 'wb') as file:
        pickle.dump(sens_attributes, file)

# Process train, validation, and test datasets
process_sens_attributes(tasks, 'train')
process_sens_attributes(tasks, 'valid')
process_sens_attributes(tasks, 'test')
