import pandas as pd

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


for indx, df_val in enumerate(all_df):
    print(indx)
    cost_list = []
    print('-----------------------------------------------------------------------------')
    for i in range(len(df_val)):
        cost_list.append([int(float(df_val.loc[i, "C"]) * 50 * int(df_val.loc[i, "R"])), [df_val.loc[i, "ACC"], df_val.loc[i, "C"], df_val.loc[i, "R"]]])

    try:
        # print(next(x for x in sorted(cost_list, key=lambda x: x[0]) if x[1][0] >= 0.99))
        print(next(x for x in sorted(cost_list, key=lambda x: x[0]) if x[1][0] >= 0.97))
    except StopIteration:
        print('not found')
    
    print('-----------------------------------------------------------------------------')
    print()


print()