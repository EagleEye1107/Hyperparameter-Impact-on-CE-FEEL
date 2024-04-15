import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


all_df = pd.read_csv('plots/GPU-FL-CNN-MNIST-Stat-Het-Syst-Het-C-Fraction-Tuning-Test-ACC-50Clients-GLOBAL.csv').drop('R', axis=1)


iid_systhet0 = pd.DataFrame(columns = ["C", "R", "ACC"])
iid_systhet1 = pd.DataFrame(columns = ["C", "R", "ACC"])
noniid_systhet0 = pd.DataFrame(columns = ["C", "R", "ACC"])
noniid_systhet1 = pd.DataFrame(columns = ["C", "R", "ACC"])

for col in all_df.columns:
    # IID --------------------------------------------------------------------------------------------
    if 'noniid' not in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=0' in col:
            iid_comms_round_syst0 = 1
            for acc_value in all_df[col]:
                iid_systhet0.loc[len(iid_systhet0.index)] = [col[col.find('=')+1 : col.find(',')], iid_comms_round_syst0, acc_value]
                iid_comms_round_syst0 += 1
        if 'Syst_Het=1' in col:
            iid_comms_round_syst1 = 1
            for acc_value in all_df[col]:
                iid_systhet1.loc[len(iid_systhet1.index)] = [col[col.find('=')+1 : col.find(',')], iid_comms_round_syst1, acc_value]
                iid_comms_round_syst1 += 1
    
    # NONIID --------------------------------------------------------------------------------------------
    if 'noniid' in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=0' in col:
            noniid_comms_round_syst0 = 1
            for acc_value in all_df[col]:
                noniid_systhet0.loc[len(noniid_systhet0.index)] = [col[col.find('=')+1 : col.find(',')], noniid_comms_round_syst0, acc_value]
                noniid_comms_round_syst0 += 1
        if 'Syst_Het=1' in col:
            noniid_comms_round_syst1 = 1
            for acc_value in all_df[col]:
                noniid_systhet1.loc[len(noniid_systhet1.index)] = [col[col.find('=')+1 : col.find(',')], noniid_comms_round_syst1, acc_value]
                noniid_comms_round_syst1 += 1


all_df = [iid_systhet0, iid_systhet1, noniid_systhet0, noniid_systhet1]
print()

# Pearson Correlation: Measures the strength and direction of the linear relationship between two continuous variables.
# It assumes that the variables are normally distributed and have linear relationships. It is sensitive to outliers

# Spearman Correlation: Measures the strength and direction of the monotonic relationship between two variables, whether linear or not.
# It does not assume a specific distribution and is robust to outliers and non-linear relationships. It relies on the ranks of the data rather than the raw values.')

for indx, df_val in enumerate(all_df):
    print(indx)
    x_data = np.array([float(i) for i in df_val['C']])
    y_data = np.array([int(i) for i in df_val['R']])
    z_data = np.array([float(i) for i in df_val['ACC']])
    
    # pcorr_zx, _zx = pearsonr(z_data, x_data)
    # pcorr_zy, _zy = pearsonr(z_data, y_data)
    # print(f'Pearsons correlation coefficient between ACC <-> C : {pcorr_zx}')
    # print(f'Pearsons correlation coefficient between ACC <-> R : {pcorr_zy}')

    scorr_zx, pval_zx = spearmanr(z_data, x_data)
    scorr_zy, pval_zy = spearmanr(z_data, y_data)
    print("Spearman's correlation coefficient between ACC <-> C :", scorr_zx)
    # print("p-value between ACC <-> C :", pval_zx)
    print("Spearman's correlation coefficient between ACC <-> R :", scorr_zy)
    # print("p-value between ACC <-> R :", pval_zy)

    print()