'''
    This code implements the basic state-of-the-art FedAvg Algorithm, performing a CNN-based Handwritten Digits recognition, based on the MNIST dataset.
    It illustrates the clients federated edge learning in a sequential way.
  
    The code investigates in Hyperparameter Impact on Computational Efficiency in Federated Edge Learning,
  by evaluating the Algorithm performance in all possible environment settings including both Systems Heterogeneity and Statistical Heterogeneity
  and by vatying C : Fraction of Selected Clients each round. Thus, it invesigates the computational and communication efficiency of the Algorithm, by investigating :
        - Fraction of Selected Clients
        - Communication rounds
        - Environment configs
        - Process time
        - CPU - GPU - Memory Usage monitoring

    The System Heterogeneity is illustrated by modeling the client's incapacity to train on their entire local dataset.
  To do so we variate the amount of data processed during the local training of each client between 0-50\% of the local dataset.
  Thus, investigating performance when client data is unevenly distributed (0-50% of local data knowing that each client has 600 samples)

'''

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import layers, models
import warnings
import numpy as np
import random
import time
import tensorflow as tf
from prettytable import PrettyTable
from operator import itemgetter
import pandas as pd
warnings.filterwarnings("ignore")


# CPU - GPU - Memory Usage monitoring ---------------------------------------------------------------
import subprocess as sp
import threading
import psutil

gpu_util_list = []
gpu_memory_list = []
gpu_temp_list = []
gpu_power_list = []
cpu_util_list = []
ram_util_list = []
cpu_temp_list = []
cpu_power_list = []
carbon_emission_factor = 0.7 # kgCO2e / kWh

bool_not_found = True

def get_gpu_usage():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    # COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv"
    try:
        gpu_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    # gpu_use_values = [int(x.split()[0]) for i, x in enumerate(gpu_use_info)]
    
    gpu_use_info = gpu_use_info[0].translate( { ord(i): None for i in ",%MiBW" } )
    gpu_use_values = [float(x) for x in gpu_use_info.split()]
    return gpu_use_values


# Function to be executed in the background thread to monitor CPU usage
def monitor_cpu_gpu_memory():
    power_file_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    with open(power_file_path, "r") as power_file:
        start_power = int(power_file.read().strip()) / 1e6  # Convert to watts
    while True:
        if bool_not_found :
            gpu_use = get_gpu_usage()
            gpu_util_list.append(gpu_use[0])
            gpu_memory_list.append(gpu_use[1])
            gpu_temp_list.append(gpu_use[2])
            gpu_power_list.append(gpu_use[3])
            cpu_util_list.append(psutil.cpu_percent())
            ram_util_list.append(psutil.virtual_memory().percent)
            
            temperature_file_path = "/sys/class/thermal/thermal_zone0/temp"
            with open(temperature_file_path, "r") as temperature_file:
                cpu_temp_list.append(int(temperature_file.read().strip()) / 1000.0)  # Convert to degrees Celsius
            
            power_file_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
            with open(power_file_path, "r") as power_file:
                cpu_current_power = int(power_file.read().strip()) / 1e6
                cpu_power_list.append(cpu_current_power - start_power)  # Convert to watts
                if 0.0 in cpu_power_list: cpu_power_list.remove(0.0)
                start_power = cpu_current_power
        time.sleep(2)
# ------------------------------------------------------------------------------------



# Params ------------------------------------------------------------------------------
c_rounds = 20 # xxxxxxxxxxxx

# Hyperparams
N_CLIENTS = 50 # xxxxxxxxxxxx
E = 5
B = 10
lr = 0.01
target_acc = 0.99 # baseline benchmark ACC

client_ids = [i for i in range(N_CLIENTS)]
loss='categorical_crossentropy'
metrics = ['accuracy']

# Environment config
C_Fractions = list(np.arange(0,11) / 10) # xxxxxxxxxxxx : C_Frac = 0 means that we have only 1 selected client
Stat_Het = ['iid', 'noniid'] # xxxxxxxxxxxx
Syst_Het = [0,1] # xxxxxxxxxxxx : Syst_Het = 0 means without Syst_Het and Syst_Het = 1 means with Syst_Het 
# -------------------------------------------------------------------------------------



# Data partitioning -------------------------------------------------------------------
def iid_partition(y_train):
  """
    the data is shuffled, and then partitioned into 50 clients each receiving 1200 examples
  """
  n_per_client = int(len(y_train)/N_CLIENTS)
  indexes_per_client = {}
  indexes = np.arange(0, len(y_train))
  random.shuffle(indexes)
  for i in range(N_CLIENTS):
    start_idx = i*n_per_client
    indexes_per_client[i] = indexes[start_idx:start_idx+n_per_client]
  return indexes_per_client


def noniid_partition(y_train):
  """
    sort the data by digit label, divide it into 100 shards of size 600, and assign each of 50 clients 4 shards.
  """
  n_shards = N_CLIENTS * 2
  n_per_shard = int(len(y_train) / n_shards)
  indexes_per_client = {}

  reverse_encoding = []
  for y in y_train:
    reverse_encoding.append(np.argmax(y))
  reverse_encoding = np.array(reverse_encoding)
  indexes = reverse_encoding.argsort()
  indexes_shard = np.arange(0, n_shards)
  random.shuffle(indexes_shard)

  for i in range(N_CLIENTS):
    start_idx_shard_1 = indexes_shard[i*2]*n_per_shard
    start_idx_shard_2 = indexes_shard[i*2+1]*n_per_shard
    indexes_per_client[i] = np.concatenate((indexes[start_idx_shard_1:start_idx_shard_1+n_per_shard],
                                            indexes[start_idx_shard_2:start_idx_shard_2+n_per_shard]))
    
  return indexes_per_client
# -------------------------------------------------------------------------------------



# Algorithm config --------------------------------------------------------------------
def load_dataset():
    (trainX, trainy), (testX, testy) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX, trainy, testX, testy


def prep_data(train, test):
    train = train.astype('float32')
    test = test.astype('float32')
    train = train / 255.0
    test = test / 255.0
    return train, test


class CNN_MNIST:
    @staticmethod
    def build(channel1 = 32, channel2 = 64, kernel_size = (5, 5), max_pool_size = (2, 2), input_shape = (28, 28, 1),
                 dense_units = 512, output_units = 10):
        model = models.Sequential()
        model.add(layers.Conv2D(channel1, kernel_size, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(max_pool_size))
        model.add(layers.Conv2D(channel2, kernel_size, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(max_pool_size))

        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dense(output_units, activation='softmax'))
        return model


def ClientUpdate(client_samples_indx, model, trainX, trainy, global_weights, loss, metrics, data_percentage, epochs = E, batch_size = B):
    model = model.build()
    optimizer = SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.set_weights(global_weights)

    if data_percentage != 0.0 :
        client_trainX = trainX[client_samples_indx]
        client_trainy = trainy[client_samples_indx]
        nb_selected_rows = int(len(trainX) * data_percentage)
        client_trainX_final = client_trainX[:nb_selected_rows]
        client_trainy_final = client_trainy[:nb_selected_rows]
        model.fit(client_trainX_final, client_trainy_final, epochs=epochs, batch_size=batch_size, validation_data=(client_trainX_final, client_trainy_final), verbose=0)

    return model


def avg_aggregator(len_global_weights, client_models, selected_clients):
    avg_weights = list()
    for j in range(len_global_weights):
        weights = [client_models[k].get_weights()[j] for k in selected_clients]
        layer_mean = tf.math.reduce_mean(weights, axis=0)
        avg_weights.append(layer_mean)
    
    return avg_weights


# -------------------------------------------------------------------------------------



# MAIN --------------------------------------------------------------------------------
# data --------------------------------------------------------
trainX, trainy, testX, testy = load_dataset()
trainX, testX = prep_data(trainX, testX)

global_test_acc_per_round = []

# Create a threading Thread for monitoring CPU GPU Memory usage
cpu_gpu_memory_monitor_thread = threading.Thread(target=monitor_cpu_gpu_memory)
# Set the thread as a daemon so it doesn't prevent the program from exiting
cpu_gpu_memory_monitor_thread.daemon = True
# Start the CPU GPU Memory monitoring thread
cpu_gpu_memory_monitor_thread.start()



# Continue with the main program ------------------------------------------------------
try:
    print()
    print(f"nb_clients = {N_CLIENTS}, B = {B}, E = {E}, lr = {lr}, n_rounds = {c_rounds}, target_acc = {target_acc}")
    for Stat_Het_config in Stat_Het:
        # Statistical Heterogeneity configuration
        print()
        print(f"### Stat_Het_config = {Stat_Het_config} -------------------------------------")

        for c_frac in C_Fractions:
            print(f"### C_Fraction = {c_frac} -------------------------------------")

            for Syst_Het_config in Syst_Het:
                print(f"### Syst_Het_config = {Syst_Het_config} -------------------------------------")
                needed_rounds = None
                needed_time = None
                bool_not_found = True

                indexes_per_client = None
                if Stat_Het_config=='iid':
                    indexes_per_client = iid_partition(trainy)
                elif Stat_Het_config=='noniid':
                    indexes_per_client = noniid_partition(trainy)
                else:
                    print('Stat_Het_config {} is not defined'.format(Stat_Het_config))
                # -------------------------------------------------------------

                # Training
                # Initialize the global_model ---------------------------------
                model = CNN_MNIST()
                global_model = model.build()
                # initial_weights = global_model.get_weights()
                # -------------------------------------------------------------

                # Start FL-Training -------------------------------------------
                print("Start FL-Training")
                start = time.time()

                # To plot the evolution of the test_acc over the rounds
                rounds_number = []
                test_acc_per_round = []
                unevenly_dist_data_per_config = []


                # FL rounds ---------------------------------------------------
                for r in range(c_rounds):
                    print(f"------------------------------------- communication round {r+1}/{c_rounds} -------------------------------------")
                    global_weights = global_model.get_weights()

                    # sampling client
                    m = max(int(c_frac*N_CLIENTS), 1)
                    selected_clients = random.sample(client_ids, m)

                    client_models = {} # to prevent crashed due to not enough RAM
                    unevenly_dist_data_per_round = []

                    # client update
                    print("clients updates ------------------------- ")
                    starttime = time.time()
                    for i in selected_clients:
                        print(f"client {i} update starting...")
                        client_start_time = time.time()

                        # System Heterogeneity configuration
                        if Syst_Het_config:
                            # Generate a random percentage based on the uniform distribution between 0 and 0.5
                            dpercentage = random.uniform(0, 0.5)
                        else :
                            # Take the whole set of data present in the client
                            dpercentage = 1
                        unevenly_dist_data_per_round.append(dpercentage)

                        # ClientUpdate on the GPU
                        with tf.device('/GPU:0'):
                            client_models[i] = ClientUpdate(indexes_per_client[i], model, trainX, trainy, global_weights, loss, metrics, data_percentage = dpercentage, epochs = E, batch_size = B)
                        client_end_time = time.time()
                        print(f"client {i} update finished in {client_end_time - client_start_time} seconds.")
                    print(f"client updates done in {time.time() - starttime} seconds --------------------- ")
                    unevenly_dist_data_per_config.append(unevenly_dist_data_per_round)

                    # averaging
                    print("server averaging ------------------------ ")
                    starttime = time.time()
                    avg_weights = avg_aggregator(len(global_weights), client_models, selected_clients)
                    global_model.set_weights(avg_weights)
                    print(f"server averaging done in {time.time() - starttime} seconds ------------------- ")

                    # evaluation of the global model on both train and test sets
                    print("evaluation phase of the global model on both train and test sets ------------------------ ")
                    starttime = time.time()
                    optimizer = SGD(learning_rate=lr, momentum=0.9)
                    global_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                    # evaluate global model on full training set
                    train_loss = 0
                    train_acc = 0
                    with tf.device('/GPU:0'):
                        train_loss, train_acc = global_model.evaluate(trainX, trainy, verbose=2)

                    # evaluate global model on testing set
                    test_loss = 0
                    test_acc = 0
                    with tf.device('/GPU:0'):
                        test_loss, test_acc = global_model.evaluate(testX, testy, verbose=2)
                    print(f"evaluation phase done in {time.time() - starttime} seconds ------------------------ ")

                    elapsed = (time.time() - start)

                    print('comm_round: {}/{} | test_acc: {:.3%} | test_loss: {} | train_acc: {:.3%} | train_loss: {} | the process time : {}'.format(r+1, c_rounds, test_acc, test_loss, train_acc, train_loss, elapsed))

                    if (test_acc >= target_acc) and bool_not_found:
                        needed_rounds = r+1
                        needed_time = elapsed
                        bool_not_found = False

                    rounds_number.append(r+1)
                    test_acc_per_round.append(test_acc)
                    # ----------------------------------------------------------------------------------------------------------------------------------


                # add acc results to the global list of test_acc
                test_acc_dict = {
                    "Stat_Het_config": Stat_Het_config,
                    "c_frac": c_frac,
                    "Syst_Het_config": Syst_Het_config,
                    "test_acc_per_round": test_acc_per_round,
                    "train_acc_per_round": train_acc,
                    "needed_rounds" : needed_rounds,
                    "needed_time" : needed_time,
                    "process_time" : time.time() - start,
                    "avg_cpu_util" : sum(cpu_util_list) / len(cpu_util_list),
                    "avg_ram_util" : sum(ram_util_list) / len(ram_util_list),
                    "avg_cpu_temp" : sum(cpu_temp_list) / len(cpu_temp_list),
                    "avg_cpu_power" : sum(cpu_power_list) / len(cpu_power_list),
                    "avg_gpu_util" : sum(gpu_util_list) / len(gpu_util_list),
                    "avg_gpu_memory" : sum(gpu_memory_list) / len(gpu_memory_list),
                    "avg_gpu_temp" : sum(gpu_temp_list) / len(gpu_temp_list),
                    "avg_gpu_power" : sum(gpu_power_list) / len(gpu_power_list)
                }
                global_test_acc_per_round.append(test_acc_dict)

                # Calculated the avg of gpu usage and reinitialize the gpu lists ------------
                gpu_util_list = []
                gpu_memory_list = []
                gpu_temp_list = []
                gpu_power_list = []
                cpu_util_list = []
                ram_util_list = []
                cpu_temp_list = []
                cpu_power_list = []



    # Ploting and desplaying the result analysis ---------------------------------------------------------------------------------------------------
    rounds_number = list(np.arange(1,c_rounds+1))
    # Create a dataframe having all the results
    acc_dict = {
        'comm_rounds' : rounds_number
    }

    list_iid = []
    list_noniid = []
    for x in global_test_acc_per_round:
        if x['Stat_Het_config'] == 'iid':
            list_iid.append(x)
        else:
            list_noniid.append(x)

    print()
    for Stat_Het_config_plot in Stat_Het:
        if Stat_Het_config_plot == 'iid':
            ploting_list = list_iid
        else:
            ploting_list = list_noniid
        
        indx_global = 0
        while (indx_global < len(ploting_list)):
            # Add the acc to the acc_dict
            acc_dict[f'{Stat_Het_config_plot} - C={ploting_list[indx_global]["c_frac"]}, Syst_Het={ploting_list[indx_global]["Syst_Het_config"]}'] = ploting_list[indx_global]["test_acc_per_round"]
            acc_dict[f'{Stat_Het_config_plot} - C={ploting_list[indx_global+1]["c_frac"]}, Syst_Het={ploting_list[indx_global+1]["Syst_Het_config"]}'] = ploting_list[indx_global+1]["test_acc_per_round"]
            indx_global += 2
    
    print(f'!!!!!!!!!!!!!!!!!!!! Test ACC Values for each Situation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    acc_df = pd.DataFrame(acc_dict)
    print(acc_df)
    # saving the dataframe
    acc_df.to_csv('plots/GPU-FL-CNN-MNIST-Stat-Het-Syst-Het-C-Fraction-Tuning-Test-ACC-50Clients-GLOBAL.csv')
    print(f'!!!!!!!!!!!!!!!!!!!! End Test ACC Values for each Situation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print()

    print('#################### Plot Summary Table ################################################################################')
    # Create a PrettyTable object
    print(f'Needed Computation to reach {target_acc*100}% of Test Acc')
    plot_summary_table = PrettyTable(['Stat Het Config', 'Syst Het Config', 'C Frac', 'Number of comms rounds', 'Final Test Acc', 'Final Train Acc', 'Execution time (secs)', 'The whole process Time (secs)', 'Avg CPU Util (%)', 'Avg RAM Util (%)', 'Avg CPU Temp (C)', 'Avg CPU Power used (W)', 'CPU Carbon Footprint (kgCO2e)', 'Avg GPU Util (%)', 'Avg GPU Memory used (MiB)', 'Avg GPU Temp (C)', 'Avg GPU Power used (W)', 'GPU Carbon Footprint (kgCO2e)'])
    global_test_acc_per_round = sorted(global_test_acc_per_round, key=itemgetter('Stat_Het_config', 'Syst_Het_config'))
    # Add rows to the table
    for data in global_test_acc_per_round:
        plot_summary_table.add_row([data['Stat_Het_config'], bool(data['Syst_Het_config']), data['c_frac'], data['needed_rounds'], round(data['test_acc_per_round'][-1]*100, 2), round(data['train_acc_per_round']*100, 2), data['needed_time'], data['process_time'], data['avg_cpu_util'], data['avg_ram_util'], data['avg_cpu_temp'], data['avg_cpu_power'], (((data['avg_cpu_power'] / 1000) * 3600) / data['process_time']) * carbon_emission_factor, data['avg_gpu_util'], data['avg_gpu_memory'], data['avg_gpu_temp'], data['avg_gpu_power'], (((data['avg_gpu_power'] / 1000) * 3600) / data['process_time']) * carbon_emission_factor])
    print(plot_summary_table)
    print('#################### End Plot Summary Table ################################################################################')
    # ----------------------------------------------------------------------------------------------------------------------------------------------


except KeyboardInterrupt:
    # Terminate the CPU monitoring thread when the program is interrupted
    print("\nProgram interrupted. Exiting...")
    cpu_gpu_memory_monitor_thread.join()
