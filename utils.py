import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef
import sys
sys.path.append("..")

# Calculate performance metric
def calculate_metric(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    # Calculate Sensitivity (True Positive Rate, TPR) and Specificity (True Negative Rate, TNR)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    # return acc, f1, auc, mcc
    return acc, auc, sen, spe

# Check mean and standard deviation
def check_mean_std_performance(result):
    return_list = []


    for m in ['ACC', 'F1', 'AUC', 'MCC']:
        return_list.append('{:.2f}+-{:.2f}'.format(np.array(result[m]).mean() * 100, np.array(result[m]).std() * 100))

    return return_list

# Setting random seed
def set_seed(random_seed):
    # Seed Setting
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For early stopping
class EarlyStopping:
    def __init__(self, patience=100, delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.early_stop = False

        else:
            self.best_score = score
            self.counter = 0

class Toy_Dataset:
    def __init__(self, random_seed):
        data1 = pd.read_csv('ADNI/AV45.csv')
        data2 = pd.read_csv('ADNI/MRI.csv')
        data3 = pd.read_csv('ADNI/SNP.csv')
        data4 = pd.read_csv('ADNI/FDG.csv')
        Sta = pd.read_csv('ADNI/Statistic3M.csv')

        # comman ID
        common_iids = data1.merge(data2, on='IID').merge(data3, on='IID').merge(data4, on='IID').merge(Sta, on='IID')[['IID']]
        filtered_Sta = Sta.merge(common_iids, on='IID')

        # 1. Subtable with diagnosis of AD, CN, and missing (None)
        subset = filtered_Sta[filtered_Sta['diagnosis'].isin(['AD', 'CN']) | filtered_Sta['diagnosis'].isna()]
        # subset = filtered_Sta[filtered_Sta['diagnosis'].isin(['AD', 'MCI']) | filtered_Sta['diagnosis'].isna()]
        # subset = filtered_Sta[filtered_Sta['diagnosis'].isin(['MCI', 'CN']) | filtered_Sta['diagnosis'].isna()]

        subset['label'] = subset['diagnosis'].map({'AD': 1, 'CN': 0, np.nan: -1}).fillna(-1).astype(int)
        # 1 for labeled, 0 for unlabeled.
        subset['mask'] = np.where(subset['diagnosis'].isin(['AD', 'CN']), 1, 0)

        # focus on AD CN and missing label
        data1N = data1[data1['IID'].isin(subset['IID'])]
        data2N = data2[data2['IID'].isin(subset['IID'])]
        data3N = data3[data3['IID'].isin(subset['IID'])]
        data4N = data4[data4['IID'].isin(subset['IID'])]

        full_data = data1N.merge(data2N, on='IID', suffixes=('_data1', '_data2'))
        full_data = full_data.merge(data3N, on='IID', suffixes=('', '_data3'))
        full_data = full_data.merge(data4N, on='IID', suffixes=('', '_data4'))
        full_data = full_data.merge(subset[['IID', 'label', 'mask']], on='IID')

        data1N = full_data.iloc[:, 1:91]
        data2N = full_data.iloc[:, 91:181]
        data3N = full_data.iloc[:, 181:239]
        data4N = full_data.iloc[:, 239:329]

        self.labels = full_data.iloc[:, 329].to_numpy().reshape(-1, 1) # AD as 1, CN as 0, NaN as -1 for missing
        self.masks = full_data.iloc[:, 330].to_numpy().reshape(-1, 1)  # 1 for labeled, 0 for unlabeled

        self.data1 = data1N.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        self.data2 = data2N.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        self.data3 = data3N.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        self.data4 = data4N.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()

        self.dataset = {'cv1': None, 'cv2': None, 'cv3': None, 'cv4': None, 'cv5': None}
        # Stratified K-Fold on the labels, including NaN handling in labels
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        for i, (train_val_index, test_index) in enumerate(kf.split(np.zeros(len(self.labels)), self.labels)):
            # Extract data for each modality
            x_train_val_1, x_test_1 = self.data1[train_val_index], self.data1[test_index]
            x_train_val_2, x_test_2 = self.data2[train_val_index], self.data2[test_index]
            x_train_val_3, x_test_3 = self.data3[train_val_index], self.data3[test_index]
            x_train_val_4, x_test_4 = self.data4[train_val_index], self.data4[test_index]

            # Extract labels
            y_train_val, y_test = self.labels[train_val_index], self.labels[test_index]
            # Extract masks
            mask_train_val, mask_test = self.masks[train_val_index], self.masks[test_index]

            # Split training into actual training and validation sets
            x_train_1, x_val_1, y_train, y_val = train_test_split(x_train_val_1, y_train_val, test_size=0.2,
                                                                  random_state=random_seed)
            x_train_2, x_val_2, _, _ = train_test_split(x_train_val_2, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_3, x_val_3, _, _ = train_test_split(x_train_val_3, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_4, x_val_4, _, _ = train_test_split(x_train_val_4, y_train_val, test_size=0.2,
                                                        random_state=random_seed)

            # Split masks similarly
            mask_train, mask_val = train_test_split(mask_train_val, test_size=0.2, random_state=random_seed)

            cv_dataset = [[x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2],
                          [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4], [y_train, y_val, y_test],
                          [mask_train, mask_val, mask_test]]

            self.dataset['cv' + str(i + 1)] = cv_dataset

    def __call__(self, cv, tensor=True, device=None):
        # Extract datasets for the specific cross-validation fold
        [x_train_1, x_val_1, x_test_1], \
            [x_train_2, x_val_2, x_test_2], \
            [x_train_3, x_val_3, x_test_3], \
            [x_train_4, x_val_4, x_test_4], \
            [y_train, y_val, y_test], \
            [mask_train, mask_val, mask_test] = self.dataset['cv' + str(cv)]

        if tensor:
            # Convert Numpy arrays to Tensors and move to specified device
            # Modality 1
            x_train_1 = torch.tensor(x_train_1).float().to(device)
            x_val_1 = torch.tensor(x_val_1).float().to(device)
            x_test_1 = torch.tensor(x_test_1).float().to(device)

            # Modality 2
            x_train_2 = torch.tensor(x_train_2).float().to(device)
            x_val_2 = torch.tensor(x_val_2).float().to(device)
            x_test_2 = torch.tensor(x_test_2).float().to(device)

            # Modality 3
            x_train_3 = torch.tensor(x_train_3).float().to(device)
            x_val_3 = torch.tensor(x_val_3).float().to(device)
            x_test_3 = torch.tensor(x_test_3).float().to(device)

            # Modality 4
            x_train_4 = torch.tensor(x_train_4).float().to(device)
            x_val_4 = torch.tensor(x_val_4).float().to(device)
            x_test_4 = torch.tensor(x_test_4).float().to(device)

            # Labels
            y_train = torch.tensor(y_train).long().to(device)
            y_val = torch.tensor(y_val).long().to(device)
            y_test = torch.tensor(y_test).long().to(device)

            # Masks
            mask_train = torch.tensor(mask_train).long().to(device)
            mask_val = torch.tensor(mask_val).long().to(device)
            mask_test = torch.tensor(mask_test).long().to(device)
        else:
            return [x_train_1, x_val_1, x_test_1], \
                [x_train_2, x_val_2, x_test_2], \
                [x_train_3, x_val_3, x_test_3], \
                [x_train_4, x_val_4, x_test_4], \
                [y_train, y_val, y_test], \
                [mask_train, mask_val, mask_test]

        # Return the processed data
        return [x_train_1, x_val_1, x_test_1], \
            [x_train_2, x_val_2, x_test_2], \
            [x_train_3, x_val_3, x_test_3], \
            [x_train_4, x_val_4, x_test_4], \
            [y_train, y_val, y_test], \
            [mask_train, mask_val, mask_test]
