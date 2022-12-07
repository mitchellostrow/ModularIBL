import numpy as np
from tqdm import tqdm

from train import train

train_params = {
    'model': {
        'architecture': 'rnn',
        'kwargs': {
            'input_size': 3,
            'output_size': 2,
            'core_kwargs': {
                'num_layers': 1,
                'hidden_size': 100},
            'param_init': 'default',
            'connectivity_kwargs': {
                'input_mask': 'inputblock_1',
                'recurrent_mask': 'modular_0.1_0.1',
                'readout_mask': 'readoutblock_2',
            },
            'timescale_distributions': 'none',
        },
    },
    'optimizer': {
        'optimizer': 'sgd', #try adam!
        'scheduler':{
            'gamma':0.995, #don't set this to be much less than 1
        },
        'kwargs': {
            'lr': 1e-1,
            'momentum': 0.1,
            'nesterov': False,
            'weight_decay': 0.0,
        },
        'description': 'Adam'
    },
    'loss_fn': {
        'loss_fn': 'nll'
    },
    'run': {
        'start_grad_step': 0,
        'num_grad_steps': 501,
        'seed': 2,
    },
    'env': {
        'num_sessions': 1,  # batch size
        'kwargs': {
            'num_stimulus_strength': 6,
            'min_stimulus_strength': 0,
            'max_stimulus_strength': 2.5,
            'block_side_probs': ((0.8, 0.2),
                                 (0.2, 0.8)),
            'trials_per_block_param': 1 / 50,
            'blocks_per_session': 4,
            'min_trials_per_block': 20,
            'max_trials_per_block': 100,
            'max_obs_per_trial': 10,
            'rnn_steps_before_obs': 2,
            'time_delay_penalty': -0.05,
        }
    },
    'description': 'You can add a really really really long description here.'
}



### Hyperparameter space to sweep over
hp_sweep_dict = {
    "architecture": ['ctrnn','rnn'],#'ctrnn'],
    "arch_type": [#('none', 'none'),
                #   ('inputblock_1', 'none'),
                #   ('none', 'outputblock_1'),
                ('inputblock_1', 'readoutblock_1'),
                ('inputblock_1', 'readoutblock_2'),
                  #('none','none')
                ],
    "connectivity": [#'modular_0.005_0.005',
                    #'modular_0.001_0.001',
                    'modular_0.01_0.01',
                    'modular_0.05_0.05',
                    'modular_0.25_0.25',
                    'modular_0.3_0.3',
                    'modular_0.4_0.4',
                    'modular_0.5_0.5'
                    #'modular_0.05_0.05',
                    #'modular_0.8_0.8'
                    ],
    "hidden_size": [50],
    "timescale_dist": [#'none',
                       'block_fixed_5_50',
                       'block_fixed_50_5',
                       'block_fixed_2_12',
                       'block_fixed_10_5',
                       'block_fixed_5_10',
                       'block_fixed_12_2',
                       'block_gaussian_5_25',
                       'block_gaussian_25_5']
                    #    'block_fixed_5_50',
                    #    'block_fixed_50_5',]
                    #    'block_gaussian_2_50',
                    #    'block_gaussian_50_2',
                    #    'block_fixed_2_50',
                    #    'block_fixed_50_2',]
}

def train_multiple_seeds(train_params, num_seeds=1):
    for i in range(num_seeds):
        train_params['run']['seed'] = (i+1) * 132
        try:
            train(train_params)
        except:
            continue

if __name__ == '__main__':

    num_seeds = 3

    num_rnn_seeds_total = len(hp_sweep_dict['arch_type']) * \
                          len(hp_sweep_dict['connectivity']) * \
                          len(hp_sweep_dict['hidden_size']) * \
                          num_seeds 
    num_ctrnn_seeds_total = num_rnn_seeds_total * len(hp_sweep_dict['timescale_dist'])
    print("number of total models to run: ", num_rnn_seeds_total * int('rnn' in hp_sweep_dict['architecture']) + num_ctrnn_seeds_total)
    _ = input("proceed? ")
    
    model_params = train_params['model']
    for architecture in hp_sweep_dict['architecture']:
        model_params['architecture'] = architecture
        
        for arch_type in hp_sweep_dict['arch_type']:
            connectivity_kwargs = model_params['kwargs']['connectivity_kwargs']
            connectivity_kwargs['input_mask'] = arch_type[0]
            connectivity_kwargs['readout_mask'] = arch_type[1]
            
            for connectivity in hp_sweep_dict['connectivity']:
                connectivity_kwargs['recurrent_mask'] = connectivity
                
                for hidden_size in hp_sweep_dict['hidden_size']:
                    model_params['kwargs']['core_kwargs']['hidden_size'] = hidden_size
                    
                    if architecture == 'ctrnn':
                        for timescales in hp_sweep_dict['timescale_dist']:
                            model_params['kwargs']['timescale_distributions'] = timescales
                            train_multiple_seeds(train_params, num_seeds)
                    else:
                        train_multiple_seeds(train_params, num_seeds)
