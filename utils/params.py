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
                'input_mask': 'none',
                'recurrent_mask': 'modular_0.02_0.02',
                'readout_mask': 'none',
            },
            'timescale_distributions': 'block_gaussian_5_50',
        },
    },
    'optimizer': {
        'optimizer': 'sgd',
        'scheduler':{
            'gamma':0.9999, #don't set this to be much less than 1
        },
        'kwargs': {
            'lr': 1e-2,
            'momentum': 0.1,
            'nesterov': False,
            'weight_decay': 0.0,
        },
        'description': 'Vanilla SGD'
    },
    'loss_fn': {
        'loss_fn': 'nll'
    },
    'run': {
        'start_grad_step': 0,
        'num_grad_steps': 2001,
        'seed': 6,
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
