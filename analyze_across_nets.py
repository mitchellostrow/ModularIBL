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
        for csv in csvs: 
            if csv.endswith('modularity_data.csv'): 
                network_params.append(jsons[0])
                modularity_data.append(csv)
               
                log_data.append(Path(run, 'logging.log'))
            
    return network_params,modularity_data,log_data

def extract_data_from_directories(network_params,modularity_data,log_data):
    columns =['Architecture',r"$p_{{conn}}$", "Q Modularity",
                'RNN Type', 'Timescale 1', 'Timescale 2',
                "Timescale Difference",r"$\tau_2 - \tau_1$",
                'Hidden Size', 'Training Time', 
                'Learning Rate', 'Weight Decay', 
              'Hidden Modularity', 'Weight Modularity', 
              'Hidden-Weight Correlation','PC Modularity',
              'Readout Modularity', 'Correct Trials / Action Trials', 'Learning Curve']
    df = pd.DataFrame(columns = columns)
    for i,(p, modularity_scores,log) in enumerate(zip(network_params,modularity_data,log_data)):
        with open(p) as f:
            params = json.load(f)
            rnntype = params['model']['architecture']
            hidden_size = int(params['model']['kwargs']['core_kwargs']['hidden_size'])
            training_time = int(params['run']['num_grad_steps'])
            weight_decay = float(params['optimizer']['kwargs'].get('weight_decay',0))
            lr = float(params['optimizer']['kwargs']['lr'])
            recurrent_mask = params['model']['kwargs']['connectivity_kwargs']['recurrent_mask']
            if recurrent_mask.startswith('modular'):
                conn_frac = float(recurrent_mask.split("_")[-1])
            else:
                conn_frac = 1.0
            q_modularity = 0.5 * (1 - conn_frac) / (1 + conn_frac)
            input_mask = params['model']['kwargs']['connectivity_kwargs']['input_mask']
            readout_mask = params['model']['kwargs']['connectivity_kwargs']['readout_mask']
          
            if input_mask == 'none' and readout_mask == 'none':
                arch_type = 1
            elif input_mask[-1] == readout_mask[-1] and input_mask != 'none':
                arch_type = 3
            elif input_mask[-1] != readout_mask[-1] and input_mask != 'none':
                arch_type = 2
            elif input_mask == 'none' and readout_mask != 'none':
                arch_type = 4
            elif input_mask != 'none' and readout_mask == 'none':
                arch_type == 5
            timescales = params['model']['kwargs']['timescale_distributions'].split("_")[-2:]
      
            if timescales == ['none']:
                timescale1,timescale2 = 5,5
            else:
                timescale1, timescale2 = float(timescales[0]), float(timescales[1])
            timescalediff = abs(timescale2-timescale1)
            signedtimediff = timescale2-timescale1
            network_data = [arch_type, conn_frac, q_modularity,rnntype, timescale1, 
                    timescale2, timescalediff, signedtimediff,
                    hidden_size, training_time, lr, weight_decay]    
        mod_scores = pd.read_csv(modularity_scores)['value'].to_list()
        network_data.extend(mod_scores)
        # if arch_type == 3 and mod_scores[0] > 0.6 and signedtimediff > 0:
        #     print(p[-18:-12],mod_scores[0])
        learning_curve1 = []
        learning_curve2 = []
        try:
            phrase1 = "INFO:root:# Correct Trials / # Total Trials: "
            phrase2 = "INFO:root:# Correct Trials / # Action Trials: "
            with open(log,'r') as f:
                for line in f:
                    if phrase1 in line:
                        accuracy = float(line[len(phrase1):])
                        learning_curve1.append(accuracy)
                    if phrase2 in line:
                        accuracy = float(line[len(phrase2):])
                        learning_curve2.append(accuracy)
            if len(learning_curve1) > 3000:
                learning_curve1 = learning_curve1[:1000]
            if len(learning_curve2) > 3000:
                learning_curve2 = learning_curve2[:1000]

            final_acc = np.max(learning_curve1[-2:]) #last n grad updates
        except FileNotFoundError:
            pass
        network_data.append(final_acc)
        network_data.append(learning_curve1)
        df.loc[i] = network_data
    return df
            
def plot_modularity_data(df):
    '''
    Generates 6 plots for now:
    2 rows: hidden modularity score, r^2 between weights and hidden act
    3 columns: connectiviy fraction (sep by rnn architecture), 
                timescale s (only ctrnn)
                architecture type (only vanilla?)
    '''

    fig, axes = plt.subplots(3,3,figsize=(15,10))#,sharex='col')#,sharey='row')
    #"Hidden-Weight Correlation",
    structural_modularity =  r"$p_{{conn}}$" #"Q Modularity" 
    for row,metric in enumerate(["Hidden Modularity", "PC Modularity", "Readout Modularity"]):
        for col,param in enumerate([structural_modularity, structural_modularity, r"$\tau_2 - \tau_1$"]):
            if col == 0:
                hue = "Architecture"
                data = df[df['RNN Type'] == 'ctrnn']
                palette = "Paired_r"
                sizes = None#r"$\tau_2 - \tau_1$"
               
            elif col == 1:
                hue = "Architecture"
                data = df[df['RNN Type'] == 'rnn']
                palette = "Paired_r"
                sizes = None
            elif col == 2:
                hue = "Architecture"
                data = df[df['RNN Type'] == 'ctrnn']
                palette = "Paired_r"
                sizes = None #r"$p_{{conn}}$"

            data = data.sort_values(param)
            for d in np.unique(data[hue]):
                if col == 0:
                    sigma = 1
                else:
                    sigma = 1
                sm = gaussian_filter1d(data.loc[data[hue]==d][metric],sigma=sigma)
                data.loc[data[hue]==d,metric] = sm
            #jitter 
            scale = 0.01
            if col == 2:
                scale = 0.5
            data[param] += np.random.normal(0,scale=scale,size=data[param].size)
            if row == 0:
                legend = 'auto'
            else:
                legend = False
            ax = axes[row,col]
            sns.scatterplot(data=data,x=param,y=metric,hue=hue, ax=ax,
                            palette=palette,legend=legend,size=sizes)

            #ax[row,col].legend()
    for j in range(3):
        if j in {0,2}:
            axes[0,j].set_title("CT-RNN")
        else:
            axes[0,j].set_title("Vanilla RNN")
        # for i in range(2):
        #     axes[row,i].set(xscale="log")

    sns.move_legend(axes[0,2], "upper left", bbox_to_anchor=(1, 1))

    plt.savefig("params_vs_modularity_metrics.pdf")
    plt.close()

def plot_hidden_vs_readout_modularity(df):
    sns.lmplot(x="Hidden Modularity", y="Readout Modularity", data=df,
                hue = "RNN Type", palette = "Paired_r")
    plt.savefig("hidden_vs_readout_modularity.pdf")
    plt.close()

def plot_reward_data(df):
    fig, ax = plt.subplots(1,2,figsize=(10,4),sharey=True)
    for i,param in enumerate([r"$p_{{conn}}$",r"$\tau_2 - \tau_1$"]):
        if i == 1:
            df = df[df['RNN Type'] == 'ctrnn']
            df[param] += np.random.normal(0,scale=0.5,size=df[param].size)

        if i == 0:
            df[param] += np.random.normal(0,scale=0.01,size=df[param].size)
            df[param] = np.abs(df[param])
        sns.scatterplot(data=df, x=param, y = "Correct Trials / Action Trials", 
                    hue = "Architecture",palette="Paired_r",ax=ax[i])
    plt.savefig(f"reward_vs_params.pdf")
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
                plt.plot(gaussian_filter1d(curves[0],sigma=4),label=label)
            plot_learning_curve(data,f"Architecture {arch}")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Accuracy per Trial")
    plt.legend()
    plt.savefig("learning_curves_arch.jpg")
    plt.close()

    for connectivity_frac in np.unique(df[r"$p_{{conn}}$"]):
        data = df[df[r"$p_{{conn}}$"] == connectivity_frac]
        plot_learning_curve(data,r"$p_{{conn}}$" + f"{connectivity_frac}")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Accuracy per Trial")
    plt.legend()
    plt.savefig("learning_curves_connectivity.jpg")
    plt.close()

def plot_connectivity_timescale(df):
    df = df[df['RNN Type'] == 'ctrnn']
    modularity =  "PC Modularity"#, "Readout Modularity","Hidden Modularity",
    fig, ax = plt.subplots(1,3,figsize=(10,5),sharex=True,sharey=True)
    for i in range(1,4):
        dr = df[df['Architecture'] == i]
        sns.scatterplot(data = dr,ax=ax[i-1],palette='viridis',x=r"$\tau_2 - \tau_1$",
                     y = modularity, hue = r"$p_{{conn}}$")
        ax[i-1].set_title(f"Architecture {i}")
    plt.xlabel(r"$\tau_2 - \tau_1$")
    plt.ylabel(modularity)
    plt.savefig("timescale_vs_modularity_w_connectivity.png")
    plt.close()

if __name__ == "__main__":
    network_params,modularity_data,log_data = get_params_and_modularities_paths()
    df = extract_data_from_directories(network_params,modularity_data,log_data)
    df.to_csv("analysis_data.csv")
  
    print(df)
    print("stats:")
    print("Number of each architecture")
    for rnn in ['ctrnn','rnn']:
        for i in range(1,4):
            print(f"{rnn}, arch {i}: ",sum(np.logical_and(df['Architecture'] == i,\
                                                            df['RNN Type'] == rnn)))
    plot_modularity_data(df)
    plot_learning_curves(df)
    plot_reward_data(df)
    plot_connectivity_timescale(df)
    plot_hidden_vs_readout_modularity(df)