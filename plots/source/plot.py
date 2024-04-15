import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



all_df = pd.read_csv('plots/GPU-FL-CNN-MNIST-Stat-Het-Syst-Het-C-Fraction-Tuning-Test-ACC-50Clients-GLOBAL.csv').drop('R', axis=1)

print(all_df.columns)

plot_colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'orange', 'pink', 'gold'] # xxxxxxxxxxxx
plot_markers = ['s', '8', '^', '*', 'D', '>', 'P', 'v', 'd', 'X', 'p']
rounds_number = range(1,21)


# IID Plot ------------------------------------------------------------------------------------------------------
color_indx = 0
marker_indx = 0
filled_bool = False

for col in all_df.columns:
    if 'noniid' not in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=0' in col:
            if (color_indx > 7):
                color_indx = 0
            plt.plot(rounds_number, list(all_df[col]), color=plot_colors[color_indx], label=col, marker = plot_markers[marker_indx], mfc='none')
            color_indx += 1
            marker_indx += 1
    
        plt.axhline(y = 0.99, color = 'black', linestyle = ':')
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        # plt.yscale("log")
        # plt.ylim(0, 1)
        # plt.title(f'i.i.d.')
        plt.legend(loc="lower right", prop={'size': 5}, bbox_to_anchor=(0.95,0.25))

plt.savefig('plots/final_plots/log_IID_WO_SystHet.eps', format='eps')
# plt.show()
# End IID Plot ------------------------------------------------------------------------------------------------------
plt.clf()


# IID Plot ------------------------------------------------------------------------------------------------------
color_indx = 0
marker_indx = 0
filled_bool = False

for col in all_df.columns:
    if 'noniid' not in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=1' in col:
            if (color_indx > 7):
                color_indx = 0
            plt.plot(rounds_number, list(all_df[col]), color=plot_colors[color_indx], label=col, marker = plot_markers[marker_indx], mfc='none')
            color_indx += 1
            marker_indx += 1
    
        plt.axhline(y = 0.99, color = 'black', linestyle = ':')
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        # plt.yscale("log")
        # plt.ylim(0, 1)
        # plt.title(f'i.i.d.')
        plt.legend(loc="lower right", prop={'size': 5}, bbox_to_anchor=(0.95,0.25))

plt.savefig('plots/final_plots/log_IID_W_SystHet.eps', format='eps')
# plt.show()
# End IID Plot ------------------------------------------------------------------------------------------------------
plt.clf()



# IID Plot ------------------------------------------------------------------------------------------------------
color_indx = 0
marker_indx = 0
filled_bool = False

for col in all_df.columns:
    if 'noniid' in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=0' in col:
            if (color_indx > 7):
                color_indx = 0
            plt.plot(rounds_number, list(all_df[col]), color=plot_colors[color_indx], label=col, marker = plot_markers[marker_indx], mfc='none')
            color_indx += 1
            marker_indx += 1
    
        plt.axhline(y = 0.99, color = 'black', linestyle = ':')
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        # plt.yscale("log")
        # plt.ylim(0, 1)
        # plt.title(f'i.i.d.')
        plt.legend(loc="lower right", prop={'size': 5}, bbox_to_anchor=(0.95,0.25))

plt.savefig('plots/final_plots/log_NONIID_WO_SystHet.eps', format='eps')
# plt.show()
# End IID Plot ------------------------------------------------------------------------------------------------------
plt.clf()


# IID Plot ------------------------------------------------------------------------------------------------------
color_indx = 0
marker_indx = 0
filled_bool = False

for col in all_df.columns:
    if 'noniid' in col and 'comm_rounds' not in col and 'Unnamed: 0' not in col:
        if 'Syst_Het=1' in col:
            if (color_indx > 7):
                color_indx = 0
            plt.plot(rounds_number, list(all_df[col]), color=plot_colors[color_indx], label=col, marker = plot_markers[marker_indx], mfc='none')
            color_indx += 1
            marker_indx += 1
    
        plt.axhline(y = 0.99, color = 'black', linestyle = ':')
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        # plt.yscale("log")
        # plt.ylim(0, 1)
        # plt.title(f'i.i.d.')
        plt.legend(loc="lower right", prop={'size': 5}, bbox_to_anchor=(0.95,0.25))

plt.savefig('plots/final_plots/log_NONIID_W_SystHet.eps', format='eps')
# plt.show()
# End IID Plot ------------------------------------------------------------------------------------------------------
plt.clf()