import gender_guesser.detector as gender
import copy
import ot
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

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

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


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

# Function to calculate Distributional Parity for False Positive Rate (FPR)
def calc_DP_FPR(sens_attr, y_true, y_score):
    nums = 1000
    Distributional_Parity_FPR = 0
    groups = list(np.unique(sens_attr))
    # Iterating over groups of sensitive attribute values
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            # Extracting false positive rate for each group
            y_g1 = y_true[sens_attr == g1]
            S_g1 = y_score[sens_attr == g1]
            y_g2 = y_true[sens_attr == g2]
            S_g2 = y_score[sens_attr == g2]
            S_g1_FPR = S_g1[y_g1 == 0]
            S_g2_FPR = S_g2[y_g2 == 0]
            expected_val = 0
            # Calculating expected difference in false positive rate across thresholds
            for thresh in np.linspace(0, 1, nums):
                P1 = np.sum((S_g1_FPR > thresh)) / len(S_g1_FPR)
                P2 = np.sum((S_g2_FPR > thresh)) / len(S_g2_FPR)
                expected_val += np.abs(P1 - P2)
            expected_val = expected_val / nums
            Distributional_Parity_FPR += expected_val
    return Distributional_Parity_FPR

# Function to calculate Statistical Parity Difference (SDD)
def calc_SDD(sens_attr, y_true, y_score):
    nums = 1000
    SDD = 0
    groups = list(np.unique(sens_attr))
    # Iterating over groups of sensitive attribute values
    for i, g1 in enumerate(groups):
        # Extracting true positive rate for each group
        y_g1 = y_true[sens_attr == g1]
        S_g1 = y_score[sens_attr == g1]
        expected_val_BG = 0
        # Calculating expected difference in true positive rate between the group and the whole dataset
        for thresh in np.linspace(0, 1, nums):
            P1 = np.sum((S_g1 > thresh)) / len(S_g1)
            P_BG = np.sum((y_score > thresh)) / len(y_score)
            expected_val_BG += np.abs(P1 - P_BG)
        expected_val_BG = expected_val_BG / nums
        SDD += expected_val_BG
    return SDD


# Function to calculate Statistical Parity Difference for Positive Predictive Difference (SPDD)
def calc_SPDD(sens_attr, y_true, y_score):
    nums = 1000
    SPDD = 0
    groups = list(np.unique(sens_attr))
    # Iterating over groups of sensitive attribute values
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            # Extracting predicted scores for each group
            y_g1 = y_true[sens_attr == g1]
            S_g1 = y_score[sens_attr == g1]
            y_g2 = y_true[sens_attr == g2]
            S_g2 = y_score[sens_attr == g2]
            expected_val = 0
            # Calculating expected difference in positive predictive value across thresholds
            for thresh in np.linspace(0, 1, nums):
                P1 = np.sum((S_g1 > thresh)) / len(S_g1)
                P2 = np.sum((S_g2 > thresh)) / len(S_g2)
                expected_val += np.abs(P1 - P2)
            expected_val = expected_val / nums
            SPDD += expected_val
    return SPDD

# Function to calculate Difference in Positive Rate (DP_PR)
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




########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


# Function to calculate the number of bins for histogram calculation
def calc_bin(data):
    try:
        # Calculate first and third quartiles
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        # Calculate bin width based on Freedman-Diaconis rule
        bin_width = 2 * IQR / (len(data) ** (1/3))
        # Calculate the number of bins
        data_range = np.max(data) - np.min(data)
        num_bins = int(np.round(data_range / bin_width))
    except:
        # If calculation fails, default to 20 bins
        num_bins = 20
    return num_bins

# Function to calculate the barycenter of two distributions
def calc_bary(score, sens_attr, R=False):
    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]
    # Determine the number of bins for histogram calculation
    num = min(int(max(calc_bin(score_g1) * 1.5, calc_bin(score_g2) * 1.5)), 500)
    # Calculate histograms for each group
    hist1, bin_edges1 = np.histogram(score_g1, bins=np.linspace(0, 1, num + 1))
    hist2, bin_edges2 = np.histogram(score_g2, bins=np.linspace(0, 1, num + 1))
    # Calculate bin centers
    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    hist1[hist1 == 0] = 1e-20
    hist2[hist2 == 0] = 1e-20
    a1 = hist1
    a2 = hist2
    # Calculate the distance matrix
    M = ot.utils.dist(hist1.reshape(-1, 1), metric='cityblock')
    M /= M.max()
    A = np.vstack((hist1, hist2)).T
    weight = 0.5 # 0<=weight<=1
    weights = np.array([1 - weight, weight])
    # Compute Wasserstein barycenter
    reg = 1e-10
    alpha = 1
    bary_wass = ot.unbalanced.barycenter_unbalanced(A, M, reg, alpha, weights=weights)
    if R:
        return bary_wass, A, bin_centers1, bin_centers2
    return bary_wass

# Function to map scores to the distribution of the barycenter
def map_scores(bary_wass, score, sens_attr):
    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]
    # Fit transport maps
    mapper1 = ot.da.MappingTransport(mu=.001, eta=1e-8, bias=False, max_iter=300, verbose=True, kernel='gaussian', sigma=2)
    mapper1.fit(Xs=score_g1.reshape(-1, 1), Xt=bary_wass.reshape(-1, 1))
    mapper2 = ot.da.MappingTransport(mu=.001, eta=1e-8, bias=False, max_iter=300, verbose=True, kernel='gaussian', sigma=2)
    mapper2.fit(Xs=score_g2.reshape(-1, 1), Xt=bary_wass.reshape(-1, 1))
    # Transform scores to the distribution of the barycenter
    scores_list_1_mapped = mapper1.transform(Xs=score_g1.reshape(-1, 1)).ravel()
    scores_list_2_mapped = mapper2.transform(Xs=score_g2.reshape(-1, 1)).ravel()
    map_score = np.zeros(score.shape)
    map_score[sens_attr == 1] = scores_list_1_mapped
    map_score[sens_attr == 0] = scores_list_2_mapped
    return map_score


def PR_make(PR_total, PR_g1, PR_g2,y_true, y_pred ,sens_attr):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    PR_total.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2.append((tp + fp) / (tp+tn+fp+fn))
    return PR_total, PR_g1, PR_g2


from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score


def ACC_make(ACC_total, ACC_g1, ACC_g2,y_true, y_pred ,sens_attr):

    accuracy = accuracy_score(y_true, y_pred)
    ACC_total.append(accuracy)

    accuracy = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    ACC_g1.append(accuracy)

    accuracy = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    ACC_g2.append(accuracy)
    return ACC_total, ACC_g1, ACC_g2



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





def calc_metric_plt(score_in, y_true,sens_attr):

    y_score = score_in
    AUC_tot , AUC_g1, AUC_g2 = AUC_make(y_true, y_score, sens_attr)


    EO_opps_distri = calc_DP_TPR(sens_attr, y_true, y_score)
    EO_odds_distri = calc_EO_disp(sens_attr, y_true, y_score)

    PR_total, PR_g1, PR_g2 = [], [], [] 
    ACC_total, ACC_g1, ACC_g2 = [], [], [] 
    F1_total, F1_g1, F1_g2 = [], [], [] 

    E_op_g1,E_op_g2 =[], []
    E_od_g1, E_od_g2 =[], []
    range = np.linspace(0, 1, 100)
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        E_op_g1, E_op_g2,E_od_g1, E_od_g2 = E_make(E_op_g1, E_op_g2,E_od_g1, E_od_g2, y_true,y_pred,  sens_attr)
        PR_total, PR_g1, PR_g2 = PR_make(PR_total, PR_g1, PR_g2,y_true, y_pred , sens_attr)
        ACC_total, ACC_g1, ACC_g2 = ACC_make(ACC_total, ACC_g1, ACC_g2,y_true, y_pred ,sens_attr)
        F1_total, F1_g1, F1_g2 = F1_make(F1_total, F1_g1, F1_g2,y_true, y_pred ,sens_attr)

    METRICS = [AUC_tot,AUC_g1, AUC_g2, E_op_g1, E_op_g2,E_od_g1,E_od_g2,PR_total, PR_g1, PR_g2,
                  ACC_total, ACC_g1, ACC_g2, F1_total, F1_g1, F1_g2, EO_opps_distri, EO_odds_distri]
    
    METRICS = {
        'AUC_tot': AUC_tot,
        'AUC_g1': AUC_g1,
        'AUC_g2': AUC_g2,
        'E_op_g1': E_op_g1,
        'E_op_g2': E_op_g2,
        'E_od_g1': E_od_g1,
        'E_od_g2': E_od_g2,
        'PR_total': PR_total,
        'PR_g1': PR_g1,
        'PR_g2': PR_g2,
        'ACC_total': ACC_total,
        'ACC_g1': ACC_g1,
        'ACC_g2': ACC_g2,
        'F1_total': F1_total,
        'F1_g1': F1_g1,
        'F1_g2': F1_g2,
        'EO_opps_distri': EO_opps_distri,
        'EO_odds_distri': EO_odds_distri
    }

    return METRICS




import matplotlib.pyplot as plt
import os
def my_plt(G_dict,color_dict, KEYS,METRICS, F, F_title, L , size , F_legend, DATASET, MODEL, stage):
    if not os.path.exists('Paper_figures'):
        os.makedirs('Paper_figures')
    if not os.path.exists('Paper_figures/'+stage):
        os.makedirs('Paper_figures/'+stage)
    metric_pre = []
    for k in KEYS:
        VAL = METRICS[k]
        if k in ['EO_opps_distri','EO_odds_distri']: continue
        

        if 'ACC' in k:metric = 'ACC'
        elif 'AUC' in k:metric = 'AUC'
        elif 'E_op' in k:metric = 'Equalized opportunity'
        elif 'E_od' in k:metric = 'Equalized odds'
        elif 'F1' in k:metric = 'F1-score'
        elif 'PR' in k: metric = 'Positive Rate'
        else: metric = ''

        if 'g1' in k:group = 'minority'
        elif 'g2' in k:group = 'majority'
        elif 'tot' in k:group = 'total'
        else:group =''

        if group not in G_dict[metric]: continue
        if metric not in metric_pre:
            metric_pre.append(metric)
            G_flag = [group]
            plt.figure(figsize=size)
            plt.xticks(fontsize = F)
            plt.yticks(fontsize = F)
            if metric != 'AUC':
                if metric == 'Equalized opportunity':  plt.ylabel('EO', fontsize = F_title)
                elif metric == 'Equalized odds':  plt.ylabel('EOD', fontsize = F_title)
                elif metric == 'F1-score': plt.ylabel('F1', fontsize = F_title)
                else: plt.ylabel(metric, fontsize = F_title)
                plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
            else:
                plt.xlabel('FPR', fontsize =F_title)  
                plt.ylabel('TPR', fontsize =F_title)        
        else: G_flag.append(group)

        
        if metric == 'AUC':
            x = VAL[0]
            y = VAL[1]
        else:
            x= np.linspace(0, 1, 100)
            y = VAL
            
        plt.plot(x,y,label =group, color = color_dict[group], linewidth = L)
        if sorted(G_flag) == sorted(G_dict[metric]):
            if metric!='Equalized odds':
                plt.ylim([0,1.03])
            plt.xlim([-0.02,1])
            legend =plt.legend(fontsize = F_legend)
            legend.get_frame().set_edgecolor('black')
            plt.gca().get_xticklabels()[0].set_horizontalalignment('right')
            plt.gca().get_yticklabels()[0].set_verticalalignment('center')
            plt.tight_layout()
            plt.savefig('Paper_figures/'+stage+'/'+DATASET+'_'+MODEL+'_'+metric+'_'+stage+'.pdf')
            plt.close()





def calc_bary2(score,sens_attr, R = False):
    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]



    num  = min(int(max(calc_bin(score_g1), calc_bin(score_g2))), 400)
    
    hist1, bin_edges1 = np.histogram(score_g1, bins=np.linspace(0, 1, num+1 ))
    hist2, bin_edges2 = np.histogram(score_g2, bins=np.linspace(0, 1, num+1 ))


    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])


    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # hist1[hist1 == 0 ] = 1e-20
    # hist2[hist2 == 0 ] = 1e-20




    M  =ot.dist(hist2.reshape(-1,1,),hist2.reshape(-1,1),metric='cityblock')
    # M  =ot.utils.dist0(hist1.shape[0])
    M /= M.max()
    A = np.vstack((hist1, hist2)).T

    weight = 0.5 # 0<=weight<=1
    weights = np.array([1 - weight, weight])

    # wasserstein
    reg = 8e-4
    alpha = 1e-4
    # bary_wass = ot.unbalanced.barycenter_unbalanced(A, M, reg, alpha, weights=weights)
    # bary_wass = ot.barycenter(A, M, reg, weights=weights, method='sinkhorn', numItermax = 10000000, stopThr= 1e-12)
    # bary_wass = ot.bregman.barycenter(A, M, reg, weights=weights, method='sinkhorn', numItermax = 10000000, stopThr= 1e-12)
    bary_wass=  ot.lp.barycenter(A,M, weights=weights, solver='interior-point')

    if R:
        return bary_wass, A, bin_centers1, bin_centers2

    return bary_wass





################################################################
################################################################
################################################################
################################################################
################################################################



def do_job(df , score,sens_attr, y_true, dataset, model, G_dict, color_dict, F, F_title, L , size , F_legend, stage,gamma):
    if not os.path.exists('Paper_figures'):
        os.makedirs('Paper_figures')

    auc_g1 = roc_auc_score(y_true[sens_attr==1], score[sens_attr ==1])
    auc_g2 = roc_auc_score(y_true[sens_attr==0], score[sens_attr ==0])
    auc = roc_auc_score(y_true, score)
    
    

    Eodd_disp = calc_EO_disp(sens_attr, y_true, score)
    Eop_disp = calc_DP_TPR(sens_attr, y_true, score)
    PR_disp = calc_DP_PR(sens_attr, y_true, score)

    y_pred = np.array([1 if score > 0.5 else 0 for score in score])

    f1_g1_5 = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    f1_g2_5 = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    f1_5 = f1_score(y_true, y_pred)
    
    accuracy_g1_5 = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    accuracy_g2_5 = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    accuracy_5 = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1_5 = (tp + fp) / (tp+tn+fp+fn)
    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2_5 = (tp + fp) / (tp+tn+fp+fn)

    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1 = (additional_fairness_metrics['e_opp__sens'])
    E_op_g2 = (additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1 = (additional_fairness_metrics['e_odds_sens'])
    E_od_g2 = (additional_fairness_metrics['e_odds__non_sens'])


    E_op_5 = np.abs(E_op_g1 - E_op_g2)
    E_od_5 = np.abs(E_od_g2 - E_od_g1)
    PR_5 = np.abs(PR_g1_5 - PR_g2_5)
    delta_auc = np.abs(auc_g1 - auc_g2)


################################

    y_pred = np.array([1 if score > 0.9 else 0 for score in score])

    f1_g1_9 = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    f1_g2_9 = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    f1_9 = f1_score(y_true, y_pred)

    accuracy_g1_9 = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    accuracy_g2_9 = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    accuracy_9 = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1_9 = (tp + fp) / (tp+tn+fp+fn)
    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2_9 = (tp + fp) / (tp+tn+fp+fn)

    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1 = (additional_fairness_metrics['e_opp__sens'])
    E_op_g2 = (additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1 = (additional_fairness_metrics['e_odds_sens'])
    E_od_g2 = (additional_fairness_metrics['e_odds__non_sens'])



    E_op_9 = np.abs(E_op_g1 - E_op_g2)
    E_od_9 = np.abs(E_od_g2 - E_od_g1)
    PR_9 = np.abs(PR_g1_9 - PR_g2_9)











    METRICS_dict = {
    'Dataset': dataset,
    'Model': model,
    'optimal lambda': gamma,
    'Distributioanl disparity: Equal opportunity (TPR)': Eop_disp ,
    'Distributioanl disparity: Equalized odds': Eodd_disp,
    'Distributioanl disparity: PR': PR_disp,

    'Positive Rate Parity (Threshold = 0.5)': PR_5,
    'Equalized odds Parity (Threshold = 0.5)': E_od_5,
    'Equal opportunity Parity (Threshold = 0.5)': E_op_5,
    
    'Total Accuracy (Threshold = 0.5)': accuracy_5,
    'Minority Accuracy (Threshold = 0.5)': accuracy_g1_5,
    'Majority Accuracy (Threshold = 0.5)': accuracy_g2_5,

    'Total F1 (Threshold = 0.5)': f1_5,
    'Minority F1 (Threshold = 0.5)': f1_g1_5,
    'Majority F1 (Threshold = 0.5)': f1_g2_5,


    'Positive Rate Parity (Threshold = 0.9)': PR_9,
    'Equalized odds Parity (Threshold = 0.9)': E_od_9,
    'Equal opportunity Parity (Threshold = 0.9)': E_op_9,
    
    'Total Accuracy (Threshold = 0.9)': accuracy_9,
    'Minority Accuracy (Threshold = 0.9)': accuracy_g1_9,
    'Majority Accuracy (Threshold = 0.9)': accuracy_g2_9,

    'Total F1 (Threshold = 0.9)': f1_9,
    'Minority F1 (Threshold = 0.9)': f1_g1_9,
    'Majority F1 (Threshold = 0.9)': f1_g2_9,


    'Total AUC': auc,
    'Minority AUC': auc_g1,
    'Majority AUC': auc_g2,
    'Delta AUC': np.abs(auc_g1 - auc_g2),
    


    }


    df_new = pd.DataFrame(METRICS_dict, index=[0])


    try:
        df = pd.concat([df, df_new], ignore_index=True)
    except:
        df = copy.deepcopy(df_new)

    # if dataset =='Amazon-Google' and model =='HierMatcher':
    # METRICS = calc_metric_plt(score, y_true,sens_attr)
    # my_plt(G_dict,color_dict, list(METRICS.keys()),METRICS, F, F_title, L , size , F_legend, dataset, model, stage)





    return df










def plot_bef_after(score_optimal_Eop,score_optimal_Eodd,score_optimal_PR, model,dataset,sens_attr, y_true, score):


    y_score = score



    Eodd_disp_init = calc_EO_disp(sens_attr, y_true, y_score)
    auc_init = roc_auc_score(y_true, y_score)
    PR_disp_init = calc_DP_PR(sens_attr, y_true, y_score)
    Eop_disp_init = calc_DP_TPR(sens_attr, y_true, y_score)


    range = np.linspace(0, 1, 200)
    Eop_disp = calc_DP_TPR(sens_attr, y_true, score_optimal_Eop)
    auc_Eop = roc_auc_score(y_true, score_optimal_Eop)

    E_op_g1,E_op_g2 =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        E_op_g1, E_op_g2,_, _ = E_make(E_op_g1, E_op_g2,[], [], y_true,y_pred,  sens_attr)


    E_op_g1_calib,E_op_g2_calib =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_Eop])
        E_op_g1_calib, E_op_g2_calib,_, _ = E_make(E_op_g1_calib, E_op_g2_calib,[], [], y_true,y_pred,  sens_attr)
        

    AUC_init ,_, _ = AUC_make(y_true, y_score, sens_attr)
    AUC_Eop ,_, _ = AUC_make(y_true, score_optimal_Eop, sens_attr)


    ########################################  

    Eodd_disp = calc_EO_disp(sens_attr, y_true, score_optimal_Eodd)
    auc_Eodd = roc_auc_score(y_true, score_optimal_Eodd)
    AUC_Eod ,_, _ = AUC_make(y_true, score_optimal_Eodd, sens_attr)


    E_od_g1, E_od_g2 =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        _, _,E_od_g1, E_od_g2 = E_make([], [],E_od_g1, E_od_g2, y_true,y_pred,  sens_attr)
        
    E_od_g1_calib, E_od_g2_calib =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_Eodd])
        _, _,E_od_g1_calib, E_od_g2_calib = E_make([], [],E_od_g1_calib, E_od_g2_calib, y_true,y_pred,  sens_attr)
        


    ########################################  

    PR_disp = calc_DP_PR(sens_attr, y_true, score_optimal_PR)
    auc_PR = roc_auc_score(y_true, score_optimal_PR)


    AUC_PR ,_, _ = AUC_make(y_true, score_optimal_PR, sens_attr)


    PR_g1, PR_g2 = [], []

    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        _, PR_g1, PR_g2 = PR_make([], PR_g1, PR_g2,y_true, y_pred , sens_attr)



    PR_g1_calib, PR_g2_calib = [], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_PR])
        _, PR_g1_calib, PR_g2_calib = PR_make([], PR_g1_calib, PR_g2_calib,y_true, y_pred , sens_attr)










    L = 1.5
    F = 28
    F_legend = 22
    F_title = 32
    size = (8,6)




    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, E_op_g1,E_op_g2, color='red', alpha=0.2, label = 'before: '+ str(round(100*Eop_disp_init,2)))
    plt.fill_between(range, E_op_g1_calib,E_op_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*Eop_disp,2)))
    plt.ylabel('EO', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.tight_layout()
    plt.legend(loc = 'lower left',fontsize = F_legend)
    plt.savefig('FIGURES/EO'+'_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()



    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, E_od_g1,E_od_g2, color='red', alpha=0.2, label = 'before: '+ str(round(100*Eodd_disp_init,2)))
    plt.fill_between(range, E_od_g1_calib,E_od_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*Eodd_disp,2)))
    plt.ylabel('EOD', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.tight_layout()
    plt.savefig('FIGURES/EOD_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()



    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, PR_g2,PR_g1, color='red', alpha=0.2, label = 'before: '+ str(round(100*PR_disp_init,2)))
    plt.fill_between(range, PR_g1_calib,PR_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*PR_disp,2)))
    plt.ylabel('PR', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.tight_layout()
    plt.savefig('FIGURES/PR_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()




    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.plot(AUC_init[0],AUC_init[1],label = 'initial: '+str(round(100*auc_init,2)), color = 'black', linewidth = L) 
    plt.plot(AUC_PR[0],AUC_PR[1],label = 'PR: '+str(round(100*auc_PR,2)), color = 'red', linewidth = L) 
    plt.plot(AUC_Eop[0],AUC_Eop[1],label = 'EO: '+str(round(100*auc_Eodd,2)), color = 'green', linewidth = L) 
    plt.plot(AUC_Eod[0],AUC_Eod[1],label = 'EOD: '+str(round(100*auc_Eop,2)), color = 'blue', linewidth = L) 
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.xlabel('FPR', fontsize =F_title)  
    plt.ylabel('TPR', fontsize =F_title)    
    plt.tight_layout()
    plt.savefig('FIGURES/AUC_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()



import bisect
import pickle
from fairness import * 


def plot_bef_after2(score_optimal_PR, model,dataset,sens_attr, y_true, score, if_plot = True):


    y_score = score



    auc_init = roc_auc_score(y_true, y_score)
    PR_disp_init = calc_DP_PR(sens_attr, y_true, y_score)


    range = np.linspace(0, 1, 100)



    ########################################  

    PR_disp = calc_DP_PR(sens_attr, y_true, score_optimal_PR)
    auc_PR = roc_auc_score(y_true, score_optimal_PR)


    if if_plot:

        PR_g1, PR_g2 = [], []

        for TH in range:
            y_pred = np.array([1 if score > TH else 0 for score in y_score])
            _, PR_g1, PR_g2 = PR_make([], PR_g1, PR_g2,y_true, y_pred , sens_attr)



        PR_g1_calib, PR_g2_calib = [], []
        for TH in range:
            y_pred = np.array([1 if score > TH else 0 for score in score_optimal_PR])
            _, PR_g1_calib, PR_g2_calib = PR_make([], PR_g1_calib, PR_g2_calib,y_true, y_pred , sens_attr)

        L = 1.5
        F = 28
        F_legend = 22
        F_title = 32
        size = (8,6)

    
        
        plt.figure(figsize=size)
        plt.xticks(fontsize = F)
        plt.yticks(fontsize = F)
        plt.fill_between(range, PR_g2,PR_g1, color='red', alpha=0.2, label = 'before: '+ str(round(100*PR_disp_init,2)))
        plt.fill_between(range, PR_g1_calib,PR_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*PR_disp,2)))
        plt.ylabel('PR', fontsize = F_title)
        plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
        plt.legend(fontsize = F_legend, loc = 'best')
        plt.tight_layout()
        plt.savefig('FIGURES/PR_'+str(model)+'_'+str(dataset)+'.pdf')
        plt.close()




        AUC_init ,_, _ = AUC_make(y_true, y_score, sens_attr)
        AUC_PR ,_, _ = AUC_make(y_true, score_optimal_PR, sens_attr)

        plt.figure(figsize=size)
        plt.xticks(fontsize = F)
        plt.yticks(fontsize = F)
        plt.plot(AUC_init[0],AUC_init[1],label = 'auc before: '+str(round(100*auc_init,2)), color = 'black', linewidth = L) 
        plt.plot(AUC_PR[0],AUC_PR[1],label = 'auc after: '+str(round(100*auc_PR,2)), color = 'red', linewidth = L) 
        plt.legend(fontsize = F_legend, loc = 'best')
        plt.xlabel('FPR', fontsize =F_title)  
        plt.ylabel('TPR', fontsize =F_title)    
        plt.tight_layout()
        plt.savefig('FIGURES/AUC_'+str(model)+'_'+str(dataset)+'.pdf')
        plt.close()


   

    return str(round(100*auc_PR,4)), str(round(100*auc_init,4)), str(round(100*PR_disp,4)), str(round(100*PR_disp_init,4))





import pandas as pd
import numpy as np

def split_data(DATA_train_in,DATA_test_in,sens_train_in,sens_test_in, split_base='train', label_dependant=True, threshold = 0.5):
    df_tmp = DATA_train_in.copy() if split_base == 'train' else DATA_test_in.copy()
    if split_base == 'train':
        sens_tmp = sens_train_in.copy()
    else:
        sens_tmp = sens_test_in.copy()


    df_minor = df_tmp[sens_tmp == 1]
    df_major = df_tmp[sens_tmp == 0]




    def split_by_label_dirty(df_in,threshold):
        df = df_in.copy()
        df['label_dirty'] = -1
        # df[df['score'] >= threshold]['label_dirty'] = 1
        # df[df['score'] < threshold]['label_dirty'] = 0

        df.loc[df['score'] < threshold, 'label_dirty'] = 0
        df.loc[df['score'] >= threshold, 'label_dirty'] = 1


        df_true = df[df['label_dirty'] == 1]
        df_false = df[df['label_dirty'] == 0]
        return (
            df_true.sample(frac=0.5), df_true.drop(df_true.sample(frac=0.5).index),
            df_false.sample(frac=0.5), df_false.drop(df_false.sample(frac=0.5).index)
        )


    def split_by_label(df):
        df_true = df[df['label'] == 1]
        df_false = df[df['label'] == 0]
        return (
            df_true.sample(frac=0.5), df_true.drop(df_true.sample(frac=0.5).index),
            df_false.sample(frac=0.5), df_false.drop(df_false.sample(frac=0.5).index)
        )

    if label_dependant:
        if split_base == 'test':
            df_minor_true_0, df_minor_true_1, df_minor_false_0, df_minor_false_1 = split_by_label_dirty(df_minor, threshold)
            df_major_true_0, df_major_true_1, df_major_false_0, df_major_false_1 = split_by_label_dirty(df_major, threshold)
        elif split_base == 'train':
            df_minor_true_0, df_minor_true_1, df_minor_false_0, df_minor_false_1 = split_by_label(df_minor)
            df_major_true_0, df_major_true_1, df_major_false_0, df_major_false_1 = split_by_label(df_major)

        df_minor_0 = pd.concat([df_minor_true_0, df_minor_false_0])
        df_minor_1 = pd.concat([df_minor_true_1, df_minor_false_1])

        df_major_0 = pd.concat([df_major_true_0, df_major_false_0])
        df_major_1 = pd.concat([df_major_true_1, df_major_false_1])
    else:
        df_minor_0 = df_minor.sample(frac=0.5)
        df_minor_1 = df_minor.drop(df_minor_0.index)
        df_major_0 = df_major.sample(frac=0.5)
        df_major_1 = df_major.drop(df_major_0.index)

    ar_minor_0 = list(df_minor_0['score_noisy'].to_numpy())
    ar_minor_1 = list(df_minor_1['score_noisy'].to_numpy())

    ar_major_0 = list(df_major_0['score_noisy'].to_numpy())
    ar_major_1 = list(df_major_1['score_noisy'].to_numpy())


    ar_minor_0.sort(reverse = False)
    ar_minor_1.sort(reverse = False)

    ar_major_0.sort(reverse = False)
    ar_major_1.sort(reverse = False)
    
    size_minor = len(ar_minor_0) + len(ar_minor_1)
    size_major = len(ar_major_0) + len(ar_major_1)

    return [ar_minor_0, ar_minor_1, ar_major_0, ar_major_1], [size_minor, size_major]


def calc_p(sens_train_in,sens_test_in, split_base='train'):
    if split_base == 'train':
        p_minor= np.sum(sens_train_in == 1) / sens_train_in.shape[0]
        p_major = np.sum(sens_train_in == 0) / sens_train_in.shape[0]
    else:
        p_minor= np.sum(sens_test_in == 1) / sens_test_in.shape[0]
        p_major = np.sum(sens_test_in == 0) / sens_test_in.shape[0]
        
    return p_major, p_minor