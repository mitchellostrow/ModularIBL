import os
import glob
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

os.chdir('runs')

def get_params_and_modularities_paths():
    network_params = []
    modularity_data = []
    log_data = []
    for run in os.listdir(os.getcwd()):
        csvs  = glob.glob(run + '/**/*.csv', recursive=True) 
        jsons = glob.glob(run + '/**/*.json', recursive=True) 
        logs = glob.glob(run + '/**/*/.log',recursive=True)
        for csv in csvs: 
            if csv.endswith('modularity_data.csv'): 
                network_params.append(jsons[0])
                modularity_data.append(csv)
               
                log_data.append(Path(run, 'logging.log'))
            
    return network_params,modularity_data,log_data

def extract_data_from_directories(network_params,modularity_data,log_data):
    columns =['Architecture','Connectivity Fraction', 
                'RNN Type', 'Timescale 1', 'Timescale 2',"Timescale Difference",
                'Hidden Size', 'Training Time', 
                'Learning Rate', 'Weight Decay', 
              'Hidden Modularity', 'Weight Modularity', 
              'Hidden-Weight Correlation', 'Average Reward', 'Learning Curve']
    df = pd.DataFrame(columns = columns)
    for i,(params, modularity_scores,log) in enumerate(zip(network_params,modularity_data,log_data)):
        with open(params) as f:
            params = json.load(f)
            rnntype = params['model']['architecture']
            hidden_size = int(params['model']['kwargs']['core_kwargs']['hidden_size'])
            training_time = int(params['run']['num_grad_steps'])
            weight_decay = float(params['optimizer']['kwargs']['weight_decay'])
            lr = float(params['optimizer']['kwargs']['lr'])
            recurrent_mask = params['model']['kwargs']['connectivity_kwargs']['recurrent_mask']
            if recurrent_mask.startswith('modular'):
                conn_frac = float(recurrent_mask.split("_")[-1])
            else:
                conn_frac = 1.0
            input_mask = params['model']['kwargs']['connectivity_kwargs']['input_mask']
            readout_mask = params['model']['kwargs']['connectivity_kwargs']['readout_mask']
            if input_mask == 'none' and readout_mask == 'none':
                arch_type = 1
            elif input_mask[-1] == readout_mask[-1] and input_mask != 'none':
                arch_type = 2
            elif input_mask[-1] != readout_mask[-1] and input_mask != 'none':
                arch_type = 3
            elif input_mask == 'none' and readout_mask != 'none':
                arch_type = 4
            elif input_mask != 'none' and readout_mask == 'none':
                arch_type == 5
            timescales = params['model']['kwargs']['timescale_distributions'].split("_")[-2:]
            timescale1, timescale2 = float(timescales[0]), float(timescales[1])
            timescalediff = abs(timescale2-timescale1)
            network_data = [arch_type, conn_frac, rnntype, timescale1, timescale2, timescalediff,
                    hidden_size, training_time, lr, weight_decay]    
        mod_scores = pd.read_csv(modularity_scores)['value'].to_list()
        network_data.extend(mod_scores)
        learning_curve = []
        phrase = "INFO:root:# Correct Trials / # Total Trials: "
        with open(log,'r') as f:
            for line in f:
                if phrase in line:
                    accuracy = float(line[len(phrase):])
                    learning_curve.append(accuracy)
        network_data.append(learning_curve)
        df.loc[i] = network_data
    return df
            
def plot_modularity_data(df):
    '''
    Generates 6 plots for now:
    2 rows: hidden modularity score, r^2 between weights and hidden act
    3 columns: connectiviy fraction (sep by rnn architecture), 
                timescales (only ctrnn)
                architecture type (only vanilla?)
    '''
    fig, ax = plt.subplots(2,3,figsize=(20,10))

    for row,metric in enumerate(["Hidden Modularity", "Hidden-Weight Correlation"]):
        for col,param in enumerate(["Connectivity Fraction", "Connectivity Fraction", "Timescale Difference"]):
            if col == 0:
                hue = "RNN Type"
                data = df
            elif col == 1:
                hue = "Architecture"
                data = df
            elif col == 2:
                hue = "Architecture"
                data = df[df['RNN Type'] == 'ctrnn']
            sns.scatterplot(data=data,x=param,y=metric,hue=hue, ax=ax[row,col],
                            palette="Paired_r")
    plt.savefig("params_vs_modularity_metrics.png")
    plt.close()

def plot_reward_data(df):
    #fig, ax = plt.subplots(2,2)
    sns.scatterplot(data=df, x="Training Time", y = "Average Reward", 
                    hue = "Connectivity Fraction",size="Architecture",palette="Paired_r")
    plt.savefig("reward_vs_params.png")
    plt.close()

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def plot_learning_curves(df):
    for arch in range(1,6):
        data = df[df['Architecture'] == arch]

        if len(data) == 0:
            continue
        else:
            def plot_learning_curve(data,label):
                curves = []
                for d in data['Learning Curve']:
                    d = np.array(d)
                    curves.append(d)
                curves = tolerant_mean(curves)
                plt.plot(gaussian_filter1d(curves[0],sigma=10),label=label)
            plot_learning_curve(data,f"Architecture {arch}")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Accuracy per Trial")
    plt.legend()
    plt.savefig("learning_curves_arch.jpg")
    plt.close()

    for connectivity_frac in np.unique(df['Connectivity Fraction']):
        data = df[df['Connectivity Fraction'] == connectivity_frac]
        plot_learning_curve(data,f"Connectivity Fraction {connectivity_frac}")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Accuracy per Trial")
    plt.legend()
    plt.savefig("learning_curves_connectivity.jpg")
    plt.close()



if __name__ == "__main__":
    network_params,modularity_data,log_data = get_params_and_modularities_paths()
    df = extract_data_from_directories(network_params,modularity_data,log_data)
    df.to_csv("analysis_data.csv")
    print(df)
    plot_modularity_data(df)
    plot_learning_curves(df)
