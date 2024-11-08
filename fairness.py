import gender_guesser.detector as gender
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score

# List of female names used for gender detection
female_names = ['adriana', 'agma', 'alexandra', 'alice', 'aya', 'barbara', 'betty', 'bhavani', 'carol',
                'carole', 'cass', 'cecilia', 'chia-jung', 'christine', 'clara', 'claudia', 'debra',
                'diane', 'dimitra', 'ebru', 'elaheh', 'elena', 'elisa', 'elke', 'esther', 'fatima',
                'fatma', 'felicity', 'françoise', 'gillian', 'hedieh', 'helen', 'ilaria', 'isabel', 
                'jeanette', 'jeanine', 'jennifer', 'jenny', 'joann', 'julia', 'juliana', 'julie', 
                'kelly', 'kimberly', 'laura', 'letizia', 'ljiljana', 'louiqa', 'lynn', 'maria',
                'marianne', 'melissa', 'meral', 'monica', 'myra', 'pamela', 'patricia', 'paula',
                'pierangela', 'pina', 'rachel', 'sandra', 'sheila', 'sihem', 'silvana', 'sophie',
                'sorana', 'sunita', 'susan', 'teresa', 'tova', 'ulrike', 'vana', 'véronique', 'ya-hui', 'yelena', 'zoé']

# Function to determine gender based on a given name
def gender_rev(name):
    # Stripping any whitespace and splitting the name to get the first part
    name = name.strip().split()[0].strip()
    d = gender.Detector()
    # Capitalizing the first letter of the name for consistency
    modified_name = name[0].upper() + name[1:]
    # Mapping gender categories from gender-guesser library to simplified categories
    gen_dict = {
        'male': 'male',
        'female': 'female',
        'andy': 'other',
        'mostly_male': 'male',
        'mostly_female': 'female',
        'unknown': 'other'
    }
    return gen_dict[d.get_gender(modified_name)]

# Dictionary mapping sensitive attributes for different datasets
sens_dict = {
    'Walmart-Amazon': ['category', 'printers'],  # Equal
    'Beer': ['Beer_Name', 'Red'],  # Continuous
    'Amazon-Google': ['manufacturer', 'Microsoft'],  # Continuous
    'Fodors-Zagat': ['type', 'asian'],  # Equal
    'iTunes-Amazon': ['Genre', 'Dance'],  # Continuous
    'DBLP-GoogleScholar': ['venue', 'vldb j'],  # Continuous
    'DBLP-ACM': ['authors', 'female'],  # Functional
    'COMPAS': ['Ethnic_Code_Text', 'African-American']
}

# Function to create a sensitive attribute vector for a given dataset
def make_sens_vector(df, dataset, sens_dict):
    if dataset in ['Walmart-Amazon', 'Fodors-Zagat', 'COMPAS']:
        # For datasets with equal comparison, check if the attribute values are equal to the sensitive value
        df['left_contains_s'] = df['left_' + sens_dict[dataset][0]] == sens_dict[dataset][1]
        df['right_contains_s'] = df['right_' + sens_dict[dataset][0]] == sens_dict[dataset][1]
    elif dataset in ['Beer', 'Amazon-Google', 'iTunes-Amazon', 'DBLP-GoogleScholar']:
        # For datasets with continuous comparison, convert attribute values to string and check for substring match
        df['left_' + sens_dict[dataset][0]] = df['left_' + sens_dict[dataset][0]].astype(str)
        df['right_' + sens_dict[dataset][0]] = df['right_' + sens_dict[dataset][0]].astype(str)
        df['left_contains_s'] = df['left_' + sens_dict[dataset][0]].str.lower().str.contains(sens_dict[dataset][1].lower())
        df['right_contains_s'] = df['right_' + sens_dict[dataset][0]].str.lower().str.contains(sens_dict[dataset][1].lower())
    else:


        df['left_' + sens_dict[dataset][0]] = df['left_' + sens_dict[dataset][0]].astype(str)
        df['right_' + sens_dict[dataset][0]] = df['right_' + sens_dict[dataset][0]].astype(str)
        # Handling special characters and splitting the attribute values
        df['left_contains_s'] = df['left_' + sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;', '').replace('&#214;', '').replace('&#237;', ',').split(',')[-1].strip())
        df['right_contains_s'] = df['right_' + sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;', '').replace('&#214;', '').replace('&#237;', ',').split(',')[-1].strip())



        left_sens = []
        right_sens = [] 

        L = len(df['left_contains_s'])
        pre = -1
        for i,row in enumerate(list(df['left_contains_s'])):
            try:
                gen = gender_rev(row.split(' ')[0])
            except:
                gen = 'other'
        
            if (100*i)//L %10 ==0:
                if pre != (100*i)//L: 
                    print((100*i)//L,end = ', ')
                    pre = (100*i)//L
            left_sens.append(gen)
            
        L = len(df['right_contains_s'])
        pre = -1
        for i,row in enumerate(list(df['right_contains_s'])):
            try:
                gen = gender_rev(row.split(' ')[0])
            except:
                gen = 'other'
            if (100*i)//L %10 ==0:
                if pre != (100*i)//L: 
                    print((100*i)//L,end = ', ')
                    pre = (100*i)//L
            right_sens.append(gen)


        df['left_contains_s'] = left_sens
        df['right_contains_s'] = right_sens

        # Checking if 'female' is present in the gender mapping for each side
        df['left_contains_s'] = df['left_contains_s'].apply(lambda x: 'True' if 'female' in str(x) else 'False')
        df['right_contains_s'] = df['right_contains_s'].apply(lambda x: 'True' if 'female' in str(x) else 'False')
        # Converting the presence of 'female' to boolean
        df['left_contains_s'] = df['left_contains_s'].apply(lambda x: any(item in x for item in ['True']))
        df['right_contains_s'] = df['right_contains_s'].apply(lambda x: any(item in x for item in ['True']))

    # Creating a resulting vector based on the presence of the sensitive attribute
    result_vector = np.logical_or(df['left_contains_s'], df['right_contains_s']).astype(int)
    sens_attr = np.array(result_vector).reshape(-1)

    return sens_attr








def calculate_additional_fairness_metrics(y_true, y_pred, sensitive_att):
    # Initialize dictionaries to hold metrics for sensitive and non-sensitive groups
    metrics_sensitive = {}
    metrics_nonsensitive = {}
    
    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) # True Positive Rate
        fpr = fp / (fp + tn) # False Positive Rate
        fnr = fn / (fn + tp) # False Negative Rate
        tnr = tn / (tn + fp) # True Negative Rate
        ppv = tp / (tp + fp) # Positive Predictive Value
        npv = tn / (tn + fn) # Negative Predictive Value
        fdr = fp / (fp + tp) # False Discovery Rate
        for_ = fn / (fn + tn) # False Omission Rate
        return {'TPR': tpr, 'FPR': fpr, 'FNR': fnr, 'TNR': tnr, 'PPV': ppv, 'NPV': npv, 'FDR': fdr, 'FOR': for_}
    

    # Separate data into sensitive and non-sensitive groups
    indices_sensitive = np.where(sensitive_att == 1)
    indices_nonsensitive = np.where(sensitive_att == 0)
    
    # Calculate metrics for sensitive group
    metrics_sensitive = calculate_metrics(y_true[indices_sensitive], y_pred[indices_sensitive])
    e_odds_sens = metrics_sensitive['TPR'] +metrics_sensitive['FPR']
    e_opp__sens = metrics_sensitive['TPR']
    # Calculate metrics for non-sensitive group
    metrics_nonsensitive = calculate_metrics(y_true[indices_nonsensitive], y_pred[indices_nonsensitive])
    e_odds__non_sens = metrics_nonsensitive['TPR'] +metrics_nonsensitive['FPR']
    e_opp__non_sens = metrics_nonsensitive['TPR']
    # Calculate parity differences
    parity_differences = {metric: metrics_sensitive[metric] - metrics_nonsensitive[metric] for metric in metrics_sensitive}
    modified_dict = {key + ' partiy': value for key, value in parity_differences.items()}



    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_att = np.array(sensitive_att)
    y_true_sensitive = y_true[sensitive_att == 1]
    y_pred_sensitive = y_pred[sensitive_att == 1]
    y_true_nonsensitive = y_true[sensitive_att == 0]
    y_pred_nonsensitive = y_pred[sensitive_att == 0]
    sp = y_pred_sensitive.mean() - y_pred_nonsensitive.mean()
    accuracy_sensitive = (y_pred_sensitive == y_true_sensitive).mean()
    accuracy_nonsensitive = (y_pred_nonsensitive == y_true_nonsensitive).mean()
    accuracy_parity = accuracy_sensitive - accuracy_nonsensitive
    
    modified_dict['Statistical Parity'] = sp
    modified_dict['Accuracy Parity'] = accuracy_parity

    modified_dict['e_odds_sens'] = e_odds_sens
    modified_dict['e_odds__non_sens'] = e_odds__non_sens


    modified_dict['e_opp__sens'] = e_opp__sens
    modified_dict['e_opp__non_sens'] = e_opp__non_sens



    return modified_dict, 


def calculate_additional_fairness_metrics2(y_true, y_pred, sensitive_att):
    # Initialize dictionaries to hold metrics for sensitive and non-sensitive groups
    metrics_sensitive = {}
    metrics_nonsensitive = {}
    
    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) # True Positive Rate
        fpr = fp / (fp + tn) # False Positive Rate
        fnr = fn / (fn + tp) # False Negative Rate
        tnr = tn / (tn + fp) # True Negative Rate

        return {'TPR': tpr, 'FPR': fpr, 'FNR': fnr, 'TNR': tnr}



    
    # Separate data into sensitive and non-sensitive groups
    indices_sensitive = np.where(sensitive_att == 1)
    indices_nonsensitive = np.where(sensitive_att == 0)
    
    # Calculate metrics for sensitive group
    metrics_sensitive = calculate_metrics(y_true[indices_sensitive], y_pred[indices_sensitive])
    e_odds_sens = metrics_sensitive['TPR'] +metrics_sensitive['FPR']
    e_opp__sens = metrics_sensitive['TPR']

    # Calculate metrics for non-sensitive group
    metrics_nonsensitive = calculate_metrics(y_true[indices_nonsensitive], y_pred[indices_nonsensitive])
    e_odds__non_sens = metrics_nonsensitive['TPR'] +metrics_nonsensitive['FPR']
    e_opp__non_sens = metrics_nonsensitive['TPR']
    # Calculate parity differences
    parity_differences = {metric: metrics_sensitive[metric] - metrics_nonsensitive[metric] for metric in metrics_sensitive}
    modified_dict = {key + ' partiy': value for key, value in parity_differences.items()}



    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_att = np.array(sensitive_att)
    y_true_sensitive = y_true[sensitive_att == 1]
    y_pred_sensitive = y_pred[sensitive_att == 1]
    y_true_nonsensitive = y_true[sensitive_att == 0]
    y_pred_nonsensitive = y_pred[sensitive_att == 0]
    sp = y_pred_sensitive.mean() - y_pred_nonsensitive.mean()
    accuracy_sensitive = (y_pred_sensitive == y_true_sensitive).mean()
    accuracy_nonsensitive = (y_pred_nonsensitive == y_true_nonsensitive).mean()
    accuracy_parity = accuracy_sensitive - accuracy_nonsensitive
    
    modified_dict['Statistical Parity'] = sp
    modified_dict['Accuracy Parity'] = accuracy_parity

    modified_dict['e_odds_sens'] = e_odds_sens
    modified_dict['e_odds__non_sens'] = e_odds__non_sens


    modified_dict['e_opp__sens'] = e_opp__sens
    modified_dict['e_opp__non_sens'] = e_opp__non_sens



    return modified_dict, 










def F1_make(F1_total, F1_g1, F1_g2,y_true, y_pred ,sens_attr):

    accuracy = f1_score(y_true, y_pred)
    F1_total.append(accuracy)

    accuracy = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    F1_g1.append(accuracy)

    accuracy = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    F1_g2.append(accuracy)
    return F1_total, F1_g1, F1_g2

from sklearn.metrics import roc_curve, roc_auc_score


def E_make(E_op_g1, E_op_g2,E_od_g1, E_od_g2, y_true,y_pred,  sens_attr):
    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1.append(additional_fairness_metrics['e_opp__sens'])
    E_op_g2.append(additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1.append(additional_fairness_metrics['e_odds_sens'])
    E_od_g2.append(additional_fairness_metrics['e_odds__non_sens'])
    return E_op_g1, E_op_g2,E_od_g1, E_od_g2



def AUC_make(y_true, y_score, sens_attr):
    auc = roc_curve(y_true, y_score)
    auc_sens = roc_curve(y_true[sens_attr ==1], y_score[sens_attr ==1])
    auc_non_snes = roc_curve(y_true[sens_attr ==0], y_score[sens_attr ==0])
    return auc, auc_sens, auc_non_snes






def calc_p(sens_train_in,sens_test_in, split_base='train'):
    if split_base == 'train':
        p_minor= np.sum(sens_train_in == 1) / sens_train_in.shape[0]
        p_major = np.sum(sens_train_in == 0) / sens_train_in.shape[0]
    else:
        p_minor= np.sum(sens_test_in == 1) / sens_test_in.shape[0]
        p_major = np.sum(sens_test_in == 0) / sens_test_in.shape[0]
        
    return p_major, p_minor