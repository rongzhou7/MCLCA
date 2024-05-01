from tqdm.auto import tqdm
from utils import *
from MCLCA import MCLCA
import torch.nn as nn


import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
# import matplotlib.pyplot as plt

import os
import pandas as pd

# Seed Setting
random_seed = 100
set_seed(random_seed)
# PATH = "./AttDGCCA.pt"
def train_MCLCA(hyper_dict):
    # Return List
    ensemble_list = {'ACC': [], 'AUC': [], 'SEN': [], 'SPE': []}
    metric_list = ['ACC', 'AUC', 'SEN', 'SPE']
    hyper_param_list = []
    best_hyper_param_list = []

    # Prepare Toy Dataset
    dataset = Toy_Dataset(hyper_dict['random_seed'])

    # 5 CV
    for cv in tqdm(range(5), desc='CV...'):
        # Prepare Dataset
        [x_train_1, x_val_1, x_test_1], \
        [x_train_2, x_val_2, x_test_2], \
        [x_train_3, x_val_3, x_test_3], \
        [x_train_4, x_val_4, x_test_4], \
        [y_train, y_val, y_test], \
        [mask_train, mask_val, mask_test]= dataset( cv + 1, tensor=True, device=hyper_dict['device'])

        # Define Deep neural network dimension of the each modality
        m1_embedding_list = [x_train_1.shape[1]] + hyper_dict['embedding_size']
        m2_embedding_list = [x_train_2.shape[1]] + hyper_dict['embedding_size']
        m3_embedding_list = [x_train_3.shape[1]] + hyper_dict['embedding_size'][1:]
        m4_embedding_list = [x_train_4.shape[1]] + hyper_dict['embedding_size']

        # Train Label -> One_Hot_Encoding
        y_train_onehot = torch.zeros(y_train.shape[0], 2).float().to(hyper_dict['device'])
        y_train_onehot[range(y_train.shape[0]), y_train.squeeze()] = 1

        # Find Best K by Validation MCC
        val_auc_result_list = []
        test_ensemble_dict = {'ACC': [], 'AUC': [], 'SPE': [], 'SEN': []}

        output_list = []

        if not os.path.exists('resultAH'):
            os.makedirs('resultAH')

        # Grid search for find best hyperparameter by AUC
        for conLamda in hyper_dict['conLamda']:
            for lr in hyper_dict['lr']:
                for reg in hyper_dict['reg']:
                    for lcont in hyper_dict['lcont']:
                        hyper_param_list.append([conLamda, lr, reg, lcont])
                        early_stopping = EarlyStopping(patience=hyper_dict['patience'], delta=hyper_dict['delta'])
                        best_loss = np.Inf

                        # Define MCLCA with 4 modality
                        model = MCLCA(m1_embedding_list, m2_embedding_list, m3_embedding_list, m4_embedding_list, conLamda).to(
                            hyper_dict['device'])

                        # Optimizer
                        clf_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

                        # Cross Entropy Loss
                        criterion = nn.CrossEntropyLoss()

                        epoch_cont_losses = []
                        epoch_clf_losses = []

                        # Model Train
                        for i in range(hyper_dict['epoch']):
                            model.train()

                            # Calculate contrastive loss
                            total_contrastive_loss = model(x_train_1, x_train_2, x_train_3, x_train_4)

                            # record the contrastive loss
                            epoch_cont_losses.append(total_contrastive_loss)
                            epoch_cor_losses_int = [item.item() for item in epoch_cont_losses]

                            # Calculate classification loss
                            clf_optimizer.zero_grad()

                            y_hat = model.predict(x_train_1, x_train_2, x_train_3, x_train_4)

                            # Squeeze y_train to ensure its dimension matches y_hat
                            y_train_squeezed = y_train.squeeze()
                            # use mask to select subjects with label
                            valid_indices = (mask_train == 1).squeeze()
                            valid_y_train = y_train_squeezed[valid_indices]
                            valid_y_hat = y_hat[valid_indices]

                            # calculate entropy loss
                            if valid_y_train.numel() > 0:
                                clf_loss = criterion(valid_y_hat, valid_y_train.squeeze())
                            else:
                                clf_loss = torch.tensor(0.0).to(y_hat.device)

                            # record loss
                            epoch_clf_losses.append(clf_loss.item())
                            epoch_clf_losses_int = [loss for loss in epoch_clf_losses]

                            clf_loss.backward()
                            clf_optimizer.step()

                            # validation
                            with torch.no_grad():
                                model.eval()
                                y_hat = model.predict(x_val_1, x_val_2, x_val_3, x_val_4)

                                # Squeeze y_val to ensure its dimension matches y_hat
                                y_val_squeezed = y_val.squeeze()
                                # Use mask to select subjects with label
                                valid_indices = (mask_val == 1).squeeze()
                                valid_y_val = y_val_squeezed[valid_indices]
                                valid_y_hat = y_hat[valid_indices]

                                # Calculate Cross Entropy Loss only for valid samples
                                if valid_y_val.numel() > 0:
                                    val_loss = criterion(valid_y_hat, valid_y_val)
                                else:
                                    val_loss = torch.tensor(0.0).to(y_hat.device)  # If no valid samples, return 0 loss

                                early_stopping(val_loss)
                                if val_loss < best_loss:
                                    best_loss = val_loss

                                if early_stopping.early_stop:
                                    break

                        # # draw the loss figure
                        # plt.figure()
                        # # plt.plot(epoch_cor_losses_int)
                        # # plt.plot(epoch_clf_losses_int)
                        #
                        # plt.title("cv {}".format(cv))
                        # plt.plot(epoch_cor_losses_int, label='epoch_cont_losses')
                        # plt.plot(epoch_clf_losses_int, label='epoch_clf_losses')
                        # plt.ylabel('loss')
                        # plt.xlabel('epoch')
                        # plt.legend()
                        # #plt.show()
                        # plt.savefig(f'images/iteration_{conLamda}_{lr}_{reg}.png')

                        # Load Best Model
                        model.eval()

                        # Model Validation
                        y_hat = model.predict(x_val_1, x_val_2, x_val_3, x_val_4)
                        y_val_squeezed = y_val.squeeze()

                        # Use mask to select subjects with label
                        valid_indices = (mask_val == 1).squeeze()
                        valid_y_val = y_val_squeezed[valid_indices]
                        valid_y_hat = y_hat[valid_indices]

                        # Calculate Cross Entropy Loss only for valid samples
                        if valid_y_val.numel() > 0:
                            individual_val_losses = criterion(valid_y_hat, valid_y_val)
                            val_loss = individual_val_losses.mean()  # Directly compute the mean since all samples are valid
                        else:
                            val_loss = torch.tensor(0.0).to(y_hat.device)  # If no valid samples, return 0 loss

                        # Evaluate model performance only on valid samples
                        if valid_y_val.numel() > 0:
                            y_pred_ensemble = torch.argmax(valid_y_hat, 1).cpu().numpy()
                            y_pred_proba_ensemble = valid_y_hat[:, 1].detach().cpu().numpy()
                            val_acc, val_auc, val_sen, val_spe = calculate_metric(valid_y_val.cpu().numpy(),
                                                                                  y_pred_ensemble,
                                                                                  y_pred_proba_ensemble)
                            validation_result = [val_acc, val_auc, val_sen, val_spe]
                            val_auc_result_list.append(val_auc)
                        else:
                            validation_result = [0, 0, 0, 0]  # Default to zero or suitable values if no valid samples

                        # Model Test
                        y_hat = model.predict(x_test_1, x_test_2, x_test_3, x_test_4)
                        y_hat = y_hat.detach()  # Detach all operations to avoid tracking history in autograd


                        # Preprocess Predicted Outputs
                        y_pred_ensemble = torch.argmax(y_hat, 1).cpu().numpy()
                        y_pred_proba_ensemble = y_hat[:, 1].cpu().numpy()  # Assuming binary classification

                        # Use mask to select subjects with label
                        valid_indices = (mask_test == 1).squeeze()
                        y_test_squeezed = y_test.squeeze()
                        valid_y_test = y_test_squeezed[valid_indices]
                        valid_y_hat = y_hat[valid_indices]
                        # Calculate Metrics Only for Valid Samples
                        if valid_y_test.numel() > 0:
                            valid_y_pred_ensemble = y_pred_ensemble[valid_indices]
                            valid_y_pred_proba_ensemble = y_pred_proba_ensemble[valid_indices]
                            test_acc, test_auc, test_sen, test_spe = calculate_metric(valid_y_test.cpu().numpy(),
                                                                                      valid_y_pred_ensemble,
                                                                                      valid_y_pred_proba_ensemble)
                            ensemble_result = [test_acc, test_auc, test_sen, test_spe]
                        else:
                            # Default values if no valid samples are present
                            ensemble_result = [0, 0, 0, 0]

                        # Optionally log or print the results for verification
                        print("Test Results - Accuracy: {}, AUC: {}, Sensitivity: {}, Specificity: {}".format(
                            *ensemble_result))

                        for k, metric in enumerate(metric_list):
                            test_ensemble_dict[metric].append(ensemble_result[k])
                        print(f'conLamda: {conLamda}, lr: {lr}, reg: {reg}, lcont: {lcont}, epoch: {i}, {ensemble_result}')
                        output_dict = {
                            'conLamda': conLamda,
                            'lr': lr,
                            'reg': reg,
                            'lcont': lcont,
                            'epoch': i,
                            'val_acc': val_acc,
                            'val_auc': val_auc,
                            'val_sen': val_sen,
                            'val_spe': val_spe,
                            'test_acc': test_acc,
                            'test_auc': test_auc,
                            'test_sen': test_sen,
                            'test_spe': test_spe
                        }

                        output_list.append(output_dict)

        output_df = pd.DataFrame(output_list)

        # output_df.to_csv(f"resultAMmini/AMminicv{cv}.csv", index=False)、

        # # Find best K
        # best_k = np.argmax(val_auc_result_list)
        #
        # # Find best hyperparameter
        # best_hyper_param_list.append(hyper_param_list[best_k])
        #
        # # torch.save(model.state_dict(), f"resultAMmini/ADGCCAmini_AM{cv}.pt")
        #
        # # Append Best K Test Result
        # for metric in metric_list:
        #     ensemble_list[metric].append(test_ensemble_dict[metric][best_k])
        # #torch.save(model.state_dict(), f"AttDGCCA{cv}.pt")

    return ensemble_list, best_hyper_param_list

if __name__ == '__main__':
    hyper_dict = {'epoch': 1000, 'delta': 0, 'random_seed': random_seed,
                  'device': torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                  'lr': [0.0001, 0.00001], 'reg': [0, 0.01, 0.001, 0.0001],
                  'patience': 30, 'embedding_size': [256, 64, 16], 'max_top_k': 10,
                  'conLamda':[0.25, 0.5, 0.75],
                  'lcont': [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0]}
                 # 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0


    ensemble_list, hyper = train_MCLCA(hyper_dict)

    # Check Performance
    performance_result = check_mean_std_performance(ensemble_list)

    print('Test Performance')
    print('ACC: {} AUC: {} SEN: {} SPE: {}'.format(performance_result[0], performance_result[1], performance_result[2],
                                                  performance_result[3]))

    print('\nBest Hyperparameter')
    for i, h in enumerate(hyper):
        print('CV: {} contLamda: {} Learning Rate: {} Regularization Term: {} lcls: {}'.format(i + 1, h[0], h[1], h[2], h[3]))
