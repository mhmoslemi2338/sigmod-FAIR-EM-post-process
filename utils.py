import gender_guesser.detector as gender
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score


# parameters for plots:
tick_l = 10
tick_w = 1.5
F_ax = 25
F_lbl = 35
fig_size = (8,6)
fig_alpha = 0.65
F_legend = 28


bar_fig_size =  (10, 7.8)
bar_border_l = 2
bar_yax_F = 40
bar_xax_F = 36
bar_legend_F = 35
tick_l_bar = 14
tick_w_bar = 2



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







# Function to calculate Distributional Parity for True Positive Rate (TPR)
def calc_DP_TPR(sens_attr, y_true, y_score):
    nums = 1000
    Distributional_Parity_TPR = 0
    groups = list(np.unique(sens_attr))
    # Iterating over groups of sensitive attribute values
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            # Extracting true positive rate for each group
            y_g1 = y_true[sens_attr == g1]
            S_g1 = y_score[sens_attr == g1]
            y_g2 = y_true[sens_attr == g2]
            S_g2 = y_score[sens_attr == g2]
            S_g1_TPR = S_g1[y_g1 == 1]
            S_g2_TPR = S_g2[y_g2 == 1]
            expected_val = 0
            # Calculating expected difference in true positive rate across thresholds
            for thresh in np.linspace(0, 1, nums):
                P1 = np.sum((S_g1_TPR > thresh)) / len(S_g1_TPR)
                P2 = np.sum((S_g2_TPR > thresh)) / len(S_g2_TPR)
                expected_val += np.abs(P1 - P2)
            expected_val = expected_val / nums
            Distributional_Parity_TPR += expected_val
    return Distributional_Parity_TPR

# Function to calculate Equalized Odds Disparity (EO_disp)
def calc_EO_disp(sens_attr, y_true, y_score):
    nums = 1000
    Distributional_Parity_FPR = 0
    groups = list(np.unique(sens_attr))
    # Extracting predicted scores for each sensitive attribute group
    y_g1 = y_true[sens_attr == 1]
    S_g1 = y_score[sens_attr == 1]
    y_g2 = y_true[sens_attr == 0]
    S_g2 = y_score[sens_attr == 0]
    expected_val = 0
    # Calculating expected difference in equalized odds across thresholds
    for thresh in np.linspace(0, 1, nums):
        FP_g1 = np.sum((S_g1 > thresh)[y_g1 == 0])
        FP_g2 = np.sum((S_g2 > thresh)[y_g2 == 0])
        TN_g1 = np.sum((S_g1 <= thresh)[y_g1 == 0])
        TN_g2 = np.sum((S_g2 <= thresh)[y_g2 == 0])
        fpr_g1 = FP_g1 / (FP_g1 + TN_g1)
        fpr_g2 = FP_g2 / (FP_g2 + TN_g2)
        FN_g1 = np.sum((S_g1 <= thresh)[y_g1 == 1])
        TP_g1 = np.sum((S_g1 > thresh)[y_g1 == 1])
        FN_g2 = np.sum((S_g2 <= thresh)[y_g2 == 1])
        TP_g2 = np.sum((S_g2 > thresh)[y_g2 == 1])
        tpr_g1 = TP_g1 / (FN_g1 + TP_g1)
        tpr_g2 = TP_g2 / (FN_g2 + TP_g2)
        expected_val += np.abs((tpr_g1 + fpr_g1) - (tpr_g2 + fpr_g2))
    expected_val = expected_val / nums
    Distributional_Parity_FPR += expected_val
    return Distributional_Parity_FPR
  
from sklearn.metrics import roc_curve, roc_auc_score

def calc_DP_PR(sens_attr, y_true, y_score):
    nums = 1000
    PR = 0
    # Extracting predicted scores for each group
    S_g1 = y_score[sens_attr == 1]
    S_g2 = y_score[sens_attr == 0]
    expected_val = 0
    # Calculating expected difference in positive rate across thresholds
    for thresh in np.linspace(0, 1, nums):
        P1 = np.sum((S_g1 > thresh)) / len(S_g1)
        P2 = np.sum((S_g2 > thresh)) / len(S_g2)
        expected_val += np.abs(P1 - P2)
    expected_val = expected_val / nums
    return expected_val


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



def PR_make(PR_total, PR_g1, PR_g2,y_true, y_pred ,sens_attr):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    PR_total.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2.append((tp + fp) / (tp+tn+fp+fn))
    return PR_total, PR_g1, PR_g2

