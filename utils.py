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
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    # Calculate Sensitivity (True Positive Rate, TPR) and Specificity (True Negative Rate, TNR)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return acc, auc, sen, spe

# Check mean and standard deviation
def check_mean_std_performance(result):
    return_list = []
    for m in ['ACC', 'AUC', 'SEN', 'SPE']:
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

# Toy Dataset Class
class Toy_Dataset:
    def __init__(self, random_seed):
        # # clean the data
        # all = pd.read_csv('data/all.csv')
        # con = pd.read_csv('data/connect.csv')
        # all = all.iloc[:, [0, 2, 8]]
        # # 删除'diagnosis'列中值为空的数据
        # #all_df1 = all[all['diagnosis'] != '']
        # all_df1 = all.dropna(subset=['diagnosis'])
        # # 保留第一次出现的重复行，删除后续出现的相同'id'和'diagnosis'的重复行，保留其他列的数据
        # all_df = all_df1.drop_duplicates(subset=['IID', 'diagnosis'], keep='first')
        # con = con.iloc[:, [0, 5, 6, 7, 8]]
        # merged_allcon = all_df.merge(con, on='IID')
        # condition = (
        #         (merged_allcon['diagnosis'] == 'CN') & (merged_allcon['DXGrp'] == 1) |
        #         (merged_allcon['diagnosis'] == 'AD') & (merged_allcon['DXGrp'] .isin([4, 5])) |
        #         (merged_allcon['diagnosis'] == 'MCI') & (merged_allcon['DXGrp'].isin([2, 3]))
        # )
        # # Create a new column 'ComparisonResult' based on the condition
        # merged_allcon['ComparisonResult'] = np.where(condition, 1, 0)
        # #merged_allcon.to_csv('data/final1.csv', index=False)
        # merged_allcon = merged_allcon.drop_duplicates(subset='IID', keep=False).copy()
        # merged_allcon.to_csv('data/final.csv', index=False)
        # merged_allcon = merged_allcon[merged_allcon['ComparisonResult'] == 1]
        # merged_allcon.to_csv('data/label.csv', index=False)
        # # clean data finished

        f_label = pd.read_csv('data/label.csv')
        f_label = f_label.drop_duplicates(subset=['IID'], keep=False)
        f_label = f_label.iloc[:, [0, 2]]
        data1N = pd.read_csv('data/AV45.csv')
        data2N = pd.read_csv('data/FDG.csv')
        data3N = pd.read_csv('data/VBM.csv')
        data4N = pd.read_csv('data/SNP.csv')

        merged_all = data1N.merge(data2N, on='IID').merge(data3N, on='IID').merge(data4N, on='IID')
        merged_df1 = data1N.merge(data2N, on='IID').merge(data3N, on='IID').merge(data4N, on='IID').merge(f_label, on='IID')
        # diagnosis_counts = merged_df1['diagnosis'].value_counts()

        #get unlabeled samples
        merged_unlabeled = merged_all[~merged_all['IID'].isin(merged_df1['IID'])]
        # 1-90 91-180 181-270 271-328 329
        data1N_un = merged_unlabeled.iloc[:, 1:90]
        data2N_un = merged_unlabeled.iloc[:, 91:180]
        data3N_un = merged_unlabeled.iloc[:, 181:270]
        data4N_un = merged_unlabeled.iloc[:, 271:328]
        data1_un = data1N_un.apply(lambda x: (x - x.mean()) / (x.std()))
        data1_un = data1_un.fillna(0)
        self.data1_un = np.array(data1_un)
        data2_un = data2N_un.apply(lambda x: (x - x.mean()) / (x.std()))
        data2_un = data2_un.fillna(0)
        self.data2_un = np.array(data2_un)
        data3_un = data3N_un.apply(lambda x: (x - x.mean()) / (x.std()))
        data3_un = data3_un.fillna(0)
        self.data3_un = np.array(data3_un)
        data4_un = data4N_un.apply(lambda x: (x - x.mean()) / (x.std()))
        data4_un = data4_un.fillna(0)
        self.data4_un = np.array(data4_un)

        # # AD vs CN
        # filtered_df = merged_df1[merged_df1['diagnosis'].isin(['CN', 'AD'])]
        # filtered_df['label'] = filtered_df['diagnosis'].map({'CN': 0, 'AD': 1})

        # # MCI vs CN
        # filtered_df = merged_df1[merged_df1['diagnosis'].isin(['CN', 'MCI'])]
        # filtered_df['label'] = filtered_df['diagnosis'].map({'CN': 0, 'MCI': 1})

        # AD vs MCI
        filtered_df = merged_df1[merged_df1['diagnosis'].isin(['MCI', 'AD'])]
        filtered_df['label'] = filtered_df['diagnosis'].map({'MCI': 0, 'AD': 1})

        #1-90 91-180 181-270 271-328 329
        data1N = filtered_df.iloc[:, 1:90]
        data2N = filtered_df.iloc[:, 91:180]
        data3N = filtered_df.iloc[:, 181:270]
        data4N = filtered_df.iloc[:, 271:328]
        label = filtered_df.iloc[:, -1]
        label = np.array(label)

        data1 = data1N.apply(lambda x:(x-x.mean())/(x.std()))
        data1 = data1.fillna(0)
        data1 = np.array(data1)

        data2 = data2N.apply(lambda x: (x - x.mean()) / (x.std()))
        data2 = data2.fillna(0)
        data2 = np.array(data2)

        data3 = data3N.apply(lambda x: (x - x.mean()) / (x.std()))
        data3 = data3.fillna(0)
        data3 = np.array(data3)

        data4 = data4N.apply(lambda x: (x - x.mean()) / (x.std()))
        data4 = data4.fillna(0)
        data4 = np.array(data4)

        # 5CV Dataset
        self.dataset = {'cv1': None, 'cv2': None, 'cv3': None, 'cv4': None, 'cv5': None}

        # Split Train,Validation and Test with 5 CV Fold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        for i, (train_val_index, test_index) in enumerate(kf.split(data1, label)):
            x_train_val_1, x_test_1, y_train_val, y_test = \
                data1[train_val_index], data1[test_index], label[train_val_index], label[test_index]
            x_train_val_2, x_test_2 = data2[train_val_index], data2[test_index]
            x_train_val_3, x_test_3 = data3[train_val_index], data3[test_index]
            x_train_val_4, x_test_4 = data4[train_val_index], data4[test_index]

            # Split Train and Validation
            x_train_1, x_val_1, y_train, y_val = train_test_split(x_train_val_1, y_train_val, test_size=0.2,
                                                                  random_state=random_seed)
            x_train_2, x_val_2, _, _ = train_test_split(x_train_val_2, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_3, x_val_3, _, _ = train_test_split(x_train_val_3, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_4, x_val_4, _, _ = train_test_split(x_train_val_4, y_train_val, test_size=0.2,
                                                        random_state=random_seed)

            # CV Dataset
            cv_dataset = [[x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2],
                          [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4], [y_train, y_val, y_test]]
            self.dataset['cv' + str(i + 1)] = cv_dataset

    def __call__(self, cv, tensor=True, device=None):
        [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4],\
        [y_train, y_val, y_test] = self.dataset['cv' + str(cv + 1)]

        # Numpy to tensor
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

        # Label
        y_train = torch.tensor(y_train).long().to(device)
        y_val = torch.tensor(y_val).long().to(device)
        y_test = torch.tensor(y_test).long().to(device)

        return [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4],\
        [y_train, y_val, y_test]


    def get_unlabeled_data(self):
        return self.data1_un, self.data2_un, self.data3_un, self.data4_un
