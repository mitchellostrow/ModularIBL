from itertools import product
import logging
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#from matplotlib.colors import DivergingNorm, ListedColormap
# import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#from psytrack.plot.analysisFunctions import makeWeightPlot
import scipy.stats
import scipy.cluster.hierarchy as spc
import seaborn as sns
import utils.analysis
import utils.run

# increase resolution
plt.rcParams['figure.dpi'] = 300.
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 6


def create_rotation_matrix(theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return rotation_matrix


rotation_matrix_90 = create_rotation_matrix(theta=np.pi / 2)

# map for converting left and right to numeric -1, 1 and vice versa
side_string_map = {
    'left': -1,
    -1: 'Left',
    0: 'Left',
    'right': 1,
    1: 'Right'
}

# attempt to be color-deficient friendly
# https://gist.github.com/thriveth/8560036
side_color_map = {
    'left': '#ff7f00',  # orange
    side_string_map['left']: '#ff7f00',  # orange
    'right': '#377eb8',  # blue
    side_string_map['right']: '#377eb8',  # blue
    'neutral': '#999999',  # grey
    'correct': '#377eb8',  # blue  # previously was '#4daf4a',  # green
    'incorrect': '#e41a1c',  # red
    'timeout': '#984ea3',  # purple
    'ideal': 'k',  # black
    'ideal_correct': '#1b3f5c',  # dark blue
    'ideal_incorrect': '#720d0e',  # dark red
    'stim_readout': '#3f3f3f',  # equivalent to
    'block_readout': '#7f7f7f'  # equivalent to tab grey
}

# create orange-blue colormap
oranges_reversed = cm.get_cmap('Oranges_r', 128)
blues = cm.get_cmap('Blues', 128)
oranges_blues = np.vstack((oranges_reversed(np.linspace(0, 1, 128)),
                           blues(np.linspace(0, 1, 128))))
orange_blue_cmap = ListedColormap(oranges_blues, name='OrangeBlue')


def run_hook_and_save_fig(hook_fn, hook_input):
    logging.info(f'Calling hook_fn {str(hook_fn)}')

    # call hook function
    hook_fn(hook_input)

    # save figure(s) to disk
    fn_name = str(hook_fn).split(' ')[1] + '.jpg'
    fig = plt.gcf()  # load whatever figure was created by hook_fn
    fig.savefig(os.path.join(hook_input['tensorboard_writer'].log_dir,
                             fn_name),
                bbox_inches='tight')  # removes surrounding whitespace
    plt.close(fig)


def hook_plot_analysis_psytrack_fit(hook_input):
    session_data = hook_input['session_data']

    trial_end_data = session_data[session_data.trial_end == 1]

    try:
        psytrack_fit_output = utils.analysis.compute_psytrack_fit(
            session_data=trial_end_data)
    except RuntimeError:
        # Factor is exactly singular. can occur if model is outputting only one action
        return

    # if error was encountered, just skip
    if psytrack_fit_output is None:
        return
    wMAP, credibleInt = psytrack_fit_output['wMAP'], psytrack_fit_output['credibleInt']

    # makeWeightPlot(
    #     wMode=wMAP,
    #     outData=psytrack_data,
    #     weights_dict=weights_dict,
    #     END=len(trial_end_data),
    #     errorbar=credibleInt,
    #     perf_plot=True,
    #     bias_plot=True)

    # create subplots
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    num_trials_to_display = 500
    trial_num = np.arange(num_trials_to_display) + 1
    fig.suptitle(f'Bernoulli GLM Model (Psytrack by Roy & Pillow) (Num Points={len(trial_end_data)})')
    axes[3].set_xlabel('Trial Number')

    # plot stimuli values
    axes[0].plot(
        trial_num,
        trial_end_data['stimuli'].values[:num_trials_to_display],
        label='Stimulus Value')
    axes[1].set_ylabel('Stimulus Value')

    # plot block structure i.e. preferred side
    axes[1].plot(
        trial_num,
        trial_end_data['stimuli_preferred_sides'].values[:num_trials_to_display],
        label='Block Preferred Side')
    axes[1].scatter(
        trial_num,
        1.05 * trial_end_data['stimuli_sides'].values[:num_trials_to_display],
        alpha=0.8,
        s=1,
        c='tab:orange',
        label='Trial Correct Side')
    axes[1].set_ylabel('Block Preferred Side')
    axes[1].legend(loc="upper right")

    # plot weight time series
    stimuli_wMAP, reward_wMAP = wMAP[0, :num_trials_to_display], wMAP[1, :num_trials_to_display]
    stimuli_interval = credibleInt[0, :num_trials_to_display]
    reward_interval = credibleInt[1, :num_trials_to_display]
    axes[2].plot(
        trial_num,
        stimuli_wMAP,
        label='Stimulus Weight',
    )
    axes[2].fill_between(
        trial_num,
        stimuli_wMAP - 2 * stimuli_interval,
        stimuli_wMAP + 2 * stimuli_interval,
        alpha=0.8,
        linewidth=0)
    axes[2].set_ylabel('BernGLM Stimulus Weight')

    # add bias timeseries
    axes[3].plot(
        trial_num,
        reward_wMAP,
        label='Reward Weight')
    axes[3].fill_between(
        trial_num,
        reward_wMAP - 2 * reward_interval,
        reward_wMAP + 2 * reward_interval,
        alpha=0.8,
        linewidth=0)
    axes[3].set_ylabel('BernGLM Feedback Weight')
    hook_input['tensorboard_writer'].add_figure(
        tag='psytrack_fit',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_bayesian_coupled_observer_state_space_trajectories_within_block(hook_input):
    session_data = hook_input['session_data']
    non_blank_data = session_data[(session_data.left_stimulus != 0) &
                                  (session_data.right_stimulus != 0)]
    trial_end_indices = non_blank_data.trial_end == 1.

    coupled_observer_latents_posterior = hook_input['coupled_observer_latents_posterior']
    coupled_observer_latents_posterior = coupled_observer_latents_posterior[trial_end_indices]
    n = 100
    right_stim_posterior = coupled_observer_latents_posterior[:n, 2] \
                           + coupled_observer_latents_posterior[:n, 3]
    right_block_posterior = coupled_observer_latents_posterior[:n, 1] \
                            + coupled_observer_latents_posterior[:n, 3]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Right Trial Posterior')
    ax.set_ylabel(r'Right Block Posterior')
    for i in range(n - 1):
        ax.plot(
            right_stim_posterior[i:i + 2],
            right_block_posterior[i:i + 2],
            color=plt.cm.jet(i / n))

    hook_input['tensorboard_writer'].add_figure(
        tag='coupled_bayesian_observer_state_space_trajectories_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_dts_per_trial_by_strength(hook_input):
    session_data = hook_input['session_data']
    dts_and_signed_stimulus_strength_by_trial_df = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).agg({
        'signed_trial_strength': 'first',
        'rnn_step_index': 'size'})

    # plot trial number within block (x) vs dts/trial (y)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Signed Stimulus Contrast')
    ax.set_ylabel(r'# Steps / Trial')
    # fig.suptitle('dts/Trial by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    avg_dts_per_trial = dts_and_signed_stimulus_strength_by_trial_df.groupby(
        ['signed_trial_strength']).rnn_step_index.mean()
    sem_dts_per_trial = dts_and_signed_stimulus_strength_by_trial_df.groupby(
        ['signed_trial_strength']).rnn_step_index.sem()
    stimuli_strengths = avg_dts_per_trial.index.values

    ax.plot(
        stimuli_strengths,
        avg_dts_per_trial,
        '-o',
        color=side_color_map['neutral'],
        markersize=2,
        linewidth=1,
        label='RNN')
    ax.fill_between(
        x=stimuli_strengths,
        y1=avg_dts_per_trial - sem_dts_per_trial,
        y2=avg_dts_per_trial + sem_dts_per_trial,
        alpha=0.3,
        linewidth=0,
        color=side_color_map['neutral'])

    bayesian_session_data = hook_input['bayesian_actor_session_data']
    bayesian_dts_and_signed_stimulus_strength_by_trial_df = bayesian_session_data.groupby([
        'session_index', 'block_index', 'trial_index']).agg({
        'signed_trial_strength': 'first',
        'rnn_step_index': 'size'})

    bayesian_avg_dts_per_trial = bayesian_dts_and_signed_stimulus_strength_by_trial_df.groupby(
        ['signed_trial_strength']).rnn_step_index.mean()
    bayesian_sem_dts_per_trial = bayesian_dts_and_signed_stimulus_strength_by_trial_df.groupby(
        ['signed_trial_strength']).rnn_step_index.sem()
    bayesian_stimuli_strengths = bayesian_avg_dts_per_trial.index.values

    ax.plot(
        bayesian_stimuli_strengths,
        bayesian_avg_dts_per_trial,
        '-o',
        color=side_color_map['ideal'],
        markersize=2,
        linewidth=1,
        label='Bayesian Actor')
    ax.fill_between(
        x=stimuli_strengths,
        y1=bayesian_avg_dts_per_trial - bayesian_sem_dts_per_trial,
        y2=bayesian_avg_dts_per_trial + bayesian_sem_dts_per_trial,
        alpha=0.3,
        linewidth=0,
        color=side_color_map['neutral'])

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_dts_per_trial_by_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_bayesian_dts_per_trial_by_strength_correct_concordant(hook_input):
    # plot trial number within block (x) vs dts/trial (y)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Signed Stimulus Contrast')
    ax.set_ylabel('# Steps / Trial')
    # fig.suptitle('dts/Trial by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # session_data = hook_input['session_data']
    # dts_and_stimuli_strength_by_trial_df = session_data.groupby([
    #     'session_index', 'block_index', 'trial_index']).agg({
    #     'trial_strength': 'first',
    #     'signed_trial_strength': 'first',
    #     'rnn_step_index': 'size',
    #     'correct_action_taken': 'last',
    #     'concordant_trial': 'first'})
    #
    # for (correct_action_taken, concordant_trial), dts_and_stimuli_strength_subset in \
    #         dts_and_stimuli_strength_by_trial_df.groupby([
    #             'correct_action_taken', 'concordant_trial']):
    #
    #     if correct_action_taken == 1.:
    #         label = 'Correct'
    #         color = side_color_map['correct']
    #     else:
    #         label = 'Incorrect'
    #         color = side_color_map['incorrect']
    #
    #     if concordant_trial:
    #         label += ', Concordant Trials'
    #         style = '-o'
    #     else:
    #         label += ', Discordant Trials'
    #         style = '--o'
    #
    #     avg_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
    #         ['signed_trial_strength']).rnn_step_index.mean()
    #     sem_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
    #         ['signed_trial_strength']).rnn_step_index.sem()
    #     stimuli_strengths = avg_dts_per_trial.index.values
    #
    #     ax.plot(
    #         stimuli_strengths,
    #         avg_dts_per_trial,
    #         style,
    #         color=color,
    #         markersize=2)
    #     ax.fill_between(
    #         x=stimuli_strengths,
    #         y1=avg_dts_per_trial - sem_dts_per_trial,
    #         y2=avg_dts_per_trial + sem_dts_per_trial,
    #         alpha=0.3,
    #         linewidth=0,
    #         color=color)

    bayesian_session_data = hook_input['bayesian_actor_session_data']
    dts_and_stimuli_strength_by_trial_df = bayesian_session_data.groupby([
        'session_index', 'block_index', 'trial_index']).agg({
        'trial_strength': 'first',
        'signed_trial_strength': 'first',
        'rnn_step_index': 'size',
        'correct_action_taken': 'last',
        'concordant_trial': 'first'})

    for (correct_action_taken, concordant_trial), dts_and_stimuli_strength_subset in \
            dts_and_stimuli_strength_by_trial_df.groupby([
                'correct_action_taken', 'concordant_trial']):

        if correct_action_taken == 1.:
            label = 'Bayesian Actor Correct'
            color = side_color_map['ideal_correct']
        else:
            label = 'Bayesian Actor Incorrect'
            color = side_color_map['ideal_incorrect']

        if concordant_trial:
            label += ', Concordant Trials'
            style = '-o'
        else:
            label += ', Discordant Trials'
            style = '--o'

        avg_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
            ['signed_trial_strength']).rnn_step_index.mean()
        sem_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
            ['signed_trial_strength']).rnn_step_index.sem()
        stimuli_strengths = avg_dts_per_trial.index.values

        ax.plot(
            stimuli_strengths,
            avg_dts_per_trial,
            style,
            color=color,
            label=label,
            markersize=2)
        ax.fill_between(
            x=stimuli_strengths,
            y1=avg_dts_per_trial - sem_dts_per_trial,
            y2=avg_dts_per_trial + sem_dts_per_trial,
            alpha=0.3,
            linewidth=0,
            color=color)

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_bayesian_dts_per_trial_by_strength_correct_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_rnn_dts_per_trial_by_strength_correct_concordant(hook_input):
    # plot trial number within block (x) vs dts/trial (y)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Signed Stimulus Contrast')
    ax.set_ylabel('# Steps / Trial')
    # fig.suptitle('dts/Trial by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    session_data = hook_input['session_data']
    dts_and_stimuli_strength_by_trial_df = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).agg({
        'trial_strength': 'first',
        'signed_trial_strength': 'first',
        'rnn_step_index': 'size',
        'correct_action_taken': 'last',
        'concordant_trial': 'first'})

    for (correct_action_taken, concordant_trial), dts_and_stimuli_strength_subset in \
            dts_and_stimuli_strength_by_trial_df.groupby([
                'correct_action_taken', 'concordant_trial']):

        if correct_action_taken == 1.:
            label = 'Correct'
            color = side_color_map['correct']
        else:
            label = 'Incorrect'
            color = side_color_map['incorrect']

        if concordant_trial:
            label += ', Concordant Trials'
            style = '-o'
        else:
            label += ', Discordant Trials'
            style = '--o'

        avg_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
            ['signed_trial_strength']).rnn_step_index.mean()
        sem_dts_per_trial = dts_and_stimuli_strength_subset.groupby(
            ['signed_trial_strength']).rnn_step_index.sem()
        stimuli_strengths = avg_dts_per_trial.index.values

        ax.plot(
            stimuli_strengths,
            avg_dts_per_trial,
            style,
            color=color,
            label=label,
            markersize=2)
        ax.fill_between(
            x=stimuli_strengths,
            y1=avg_dts_per_trial - sem_dts_per_trial,
            y2=avg_dts_per_trial + sem_dts_per_trial,
            alpha=0.3,
            linewidth=0,
            color=color)

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_rnn_dts_per_trial_by_strength_correct_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_dts_within_trial(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.].copy()

    # subtract the blank dts to correctly count number of observations
    trial_end_data.rnn_step_index -= hook_input['envs'][0].rnn_steps_before_obs
    trial_end_data.loc[trial_end_data.rnn_step_index < 1, 'rnn_step_index'] = 0.

    correct_action_prob_by_num_dts = trial_end_data.groupby(
        ['rnn_step_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
        'bayesian_observer_correct_action_taken': ['mean', 'sem', 'size']})

    # drop NaN/0 SEM trials
    rnn_correct_action_prob_by_num_dts = correct_action_prob_by_num_dts[
        'correct_action_taken']
    rnn_correct_action_prob_by_num_dts = rnn_correct_action_prob_by_num_dts[
        rnn_correct_action_prob_by_num_dts['size'] > 1.]
    observer_correct_action_prob_by_num_dts = correct_action_prob_by_num_dts[
        'bayesian_observer_correct_action_taken']
    observer_correct_action_prob_by_num_dts = observer_correct_action_prob_by_num_dts[
        observer_correct_action_prob_by_num_dts['size'] > 1.]
    assert len(observer_correct_action_prob_by_num_dts) == len(rnn_correct_action_prob_by_num_dts)

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Number of Observations Within Trial')
    ax.set_xlabel('Number of Observations Within Trial')
    ax.set_ylabel('# Correct / # Trials')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0., 1 + hook_input['envs'][0].max_obs_per_trial])

    # 0th dt is 1st observation, so add +1
    ax.plot(
        rnn_correct_action_prob_by_num_dts.index.values + 1,
        rnn_correct_action_prob_by_num_dts['mean'],
        '-o',
        markersize=2,
        linewidth=1,
        label='RNN',
        color=side_color_map['correct'])

    ax.fill_between(
        x=rnn_correct_action_prob_by_num_dts.index.values + 1,
        y1=rnn_correct_action_prob_by_num_dts['mean']
           - rnn_correct_action_prob_by_num_dts['sem'],
        y2=rnn_correct_action_prob_by_num_dts['mean']
           + rnn_correct_action_prob_by_num_dts['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])

    ax.plot(
        observer_correct_action_prob_by_num_dts.index.values + 1,
        observer_correct_action_prob_by_num_dts['mean'],
        '--o',
        markersize=2,
        linewidth=1,
        label='Bayesian Observer',
        color=side_color_map['ideal_correct'])

    ax.fill_between(
        x=observer_correct_action_prob_by_num_dts.index.values + 1,
        y1=observer_correct_action_prob_by_num_dts['mean']
           - observer_correct_action_prob_by_num_dts['sem'],
        y2=observer_correct_action_prob_by_num_dts['mean']
           + observer_correct_action_prob_by_num_dts['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    bayesian_trial_session_data = hook_input['bayesian_actor_session_data']
    bayesian_trial_end_data = bayesian_trial_session_data[
        bayesian_trial_session_data.trial_end == 1.].copy()

    # subtract the blank dts to correctly count number of observations
    bayesian_trial_end_data.rnn_step_index -= hook_input['envs'][0].rnn_steps_before_obs
    bayesian_trial_end_data.loc[bayesian_trial_end_data.rnn_step_index < 1, 'rnn_step_index'] = 0.

    actor_correct_action_prob_by_num_dts = bayesian_trial_end_data.groupby(
        ['rnn_step_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size']})['correct_action_taken']

    # drop NaN/0 SEM
    actor_correct_action_prob_by_num_dts = actor_correct_action_prob_by_num_dts[
        actor_correct_action_prob_by_num_dts['size'] > 1.]

    ax.plot(
        actor_correct_action_prob_by_num_dts.index.values + 1,
        actor_correct_action_prob_by_num_dts['mean'],
        '-o',
        markersize=2,
        linewidth=1,
        label='Bayesian Actor',
        color=side_color_map['ideal_correct'])

    ax.fill_between(
        x=actor_correct_action_prob_by_num_dts.index.values + 1,
        y1=actor_correct_action_prob_by_num_dts['mean']
           - actor_correct_action_prob_by_num_dts['sem'],
        y2=actor_correct_action_prob_by_num_dts['mean']
           + actor_correct_action_prob_by_num_dts['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal_correct'])

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_dts_per_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_trial_within_block(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    correct_and_block_posterior_by_trial_index = trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
        'bayesian_observer_correct_action_taken': ['mean', 'sem', 'size'],
    })

    # drop NaN SEM rows
    rnn_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'correct_action_taken']
    observer_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'bayesian_observer_correct_action_taken']
    rnn_correct_by_trial_index = rnn_correct_by_trial_index[
        rnn_correct_by_trial_index['size'] > 1.]
    observer_correct_by_trial_index = observer_correct_by_trial_index[
        observer_correct_by_trial_index['size'] != 0.]

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Trial Within Block (All Contrast Trials)')
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('# Correct / # Trials')
    ax.set_ylim([0.5, 1.05])
    ax.set_xlim([0., 101.])
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        1 + rnn_correct_by_trial_index.index.values,
        rnn_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['correct'],
        label='RNN')

    ax.fill_between(
        x=1 + rnn_correct_by_trial_index.index.values,
        y1=rnn_correct_by_trial_index['mean']
           - rnn_correct_by_trial_index['sem'],
        y2=rnn_correct_by_trial_index['mean']
           + rnn_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])

    ax.plot(
        1 + observer_correct_by_trial_index.index.values,
        observer_correct_by_trial_index['mean'],
        '--o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Observer')

    ax.fill_between(
        x=1 + observer_correct_by_trial_index.index.values,
        y1=observer_correct_by_trial_index['mean']
           - observer_correct_by_trial_index['sem'],
        y2=observer_correct_by_trial_index['mean']
           + observer_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_trial_end_data = actor_session_data[actor_session_data.trial_end == 1.].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    actor_correct_by_trial_index = actor_trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
    })['correct_action_taken']

    # drop NaN/0. sem trials
    actor_correct_by_trial_index = actor_correct_by_trial_index[
        actor_correct_by_trial_index['size'] > 1.]

    ax.plot(
        1 + actor_correct_by_trial_index.index.values,
        actor_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Actor')

    ax.fill_between(
        x=1 + actor_correct_by_trial_index.index.values,
        y1=actor_correct_by_trial_index['mean']
           - actor_correct_by_trial_index['sem'],
        y2=actor_correct_by_trial_index['mean']
           + actor_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_trial_within_block_zero_contrast(hook_input):
    session_data = hook_input['session_data']

    # keep only zero contrast, last dts in trials
    zero_contrast_trial_end_data = session_data[
        (session_data.trial_end == 1.) & (session_data.trial_strength == 0.)].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    correct_and_block_posterior_by_trial_index = zero_contrast_trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
        'bayesian_observer_correct_action_taken': ['mean', 'sem', 'size'],
    })

    # drop NaN SEM rows
    rnn_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'correct_action_taken']
    observer_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'bayesian_observer_correct_action_taken']
    rnn_correct_by_trial_index = rnn_correct_by_trial_index[
        rnn_correct_by_trial_index['size'] > 1.]
    observer_correct_by_trial_index = observer_correct_by_trial_index[
        observer_correct_by_trial_index['size'] != 0.]

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Trial Within Block (All Contrast Trials)')
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('# Correct / # Trials\n(Zero Contrast Trials)')
    ax.set_ylim([0., 1.05])
    ax.set_xlim([0., 101.])
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        1 + rnn_correct_by_trial_index.index.values,
        rnn_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['correct'],
        label='RNN')

    ax.fill_between(
        x=1 + rnn_correct_by_trial_index.index.values,
        y1=rnn_correct_by_trial_index['mean']
           - rnn_correct_by_trial_index['sem'],
        y2=rnn_correct_by_trial_index['mean']
           + rnn_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])

    ax.plot(
        1 + observer_correct_by_trial_index.index.values,
        observer_correct_by_trial_index['mean'],
        '--o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Observer')

    ax.fill_between(
        x=1 + observer_correct_by_trial_index.index.values,
        y1=observer_correct_by_trial_index['mean']
           - observer_correct_by_trial_index['sem'],
        y2=observer_correct_by_trial_index['mean']
           + observer_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_zero_contrast_trial_end_data = actor_session_data[
        (actor_session_data.trial_end == 1.) & (actor_session_data.trial_end == 0.)].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    actor_correct_by_trial_index = actor_zero_contrast_trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
    })['correct_action_taken']

    # drop NaN/0. sem trials
    actor_correct_by_trial_index = actor_correct_by_trial_index[
        actor_correct_by_trial_index['size'] > 1.]

    ax.plot(
        1 + actor_correct_by_trial_index.index.values,
        actor_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Actor')

    ax.fill_between(
        x=1 + actor_correct_by_trial_index.index.values,
        y1=actor_correct_by_trial_index['mean']
           - actor_correct_by_trial_index['sem'],
        y2=actor_correct_by_trial_index['mean']
           + actor_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_trial_within_block_zero_contrast',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_zero_contrast_trial_within_block(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials and only keep zero contrast trials
    trial_end_data = session_data[(session_data.trial_end == 1.) &
                                  (session_data.trial_strength == 0.)]

    # plot trial number within block (x) vs probability of correct response (y)
    rnn_correct_action_prob_by_trial_num = trial_end_data.groupby(
        ['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size']
    })['correct_action_taken']

    # drop NaN SEM
    rnn_correct_action_prob_by_trial_num = rnn_correct_action_prob_by_trial_num[
        rnn_correct_action_prob_by_trial_num['size'] > 1.]

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Trial Within Block (Zero Contrast Trials)')
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('# Correct / # Trials')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0., 101.])
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        1 + rnn_correct_action_prob_by_trial_num.index.values,
        rnn_correct_action_prob_by_trial_num['mean'],
        '-o',
        color=side_color_map['correct'],
        label='Zero Contrast Trials')

    ax.fill_between(
        x=1 + rnn_correct_action_prob_by_trial_num.index.values,
        y1=rnn_correct_action_prob_by_trial_num['mean']
           - rnn_correct_action_prob_by_trial_num['sem'],
        y2=rnn_correct_action_prob_by_trial_num['mean']
           + rnn_correct_action_prob_by_trial_num['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])
    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_zero_contrast_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_by_strength_concordant(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylabel('# Correct / # Trials')
    ax.set_xlabel('Signed Stimulus Contrast')

    # keep only last dts in trials
    session_data = hook_input['session_data']
    trial_end_data = session_data[session_data.trial_end == 1.]

    for concordant, concordant_data in trial_end_data.groupby(['concordant_trial']):
        avg_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].mean()

        sem_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].sem()

        ax.plot(
            avg_correct_action_prob_by_stim_strength.index,
            avg_correct_action_prob_by_stim_strength,
            '-o' if concordant else '--o',
            # solid lines for consistent block side, trial side; dotted otherwise
            label='RNN Concordant' if concordant else 'RNN Discordant',
            color=side_color_map['correct'])
        ax.fill_between(
            x=avg_correct_action_prob_by_stim_strength.index,
            y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
            y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['correct'])

    # keep only last dts in trials
    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_trial_end_data = actor_session_data[actor_session_data.trial_end == 1.]

    for concordant, concordant_data in actor_trial_end_data.groupby(['concordant_trial']):
        avg_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].mean()

        sem_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].sem()

        ax.plot(
            avg_correct_action_prob_by_stim_strength.index,
            avg_correct_action_prob_by_stim_strength,
            '-o' if concordant else '--o',
            # solid lines for consistent block side, trial side; dotted otherwise
            label=f'Bayesian Actor Concordant' if concordant else 'Bayesian Actor Discordant',
            color=side_color_map['ideal_correct'])
        ax.fill_between(
            x=avg_correct_action_prob_by_stim_strength.index,
            y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
            y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal_correct'])

    # necessary to delete redundant legend groups
    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels, markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_by_strength_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_slope_intercept_by_prev_block_duration(hook_input):
    session_data = hook_input['session_data']

    # only take consider last dt within a trial
    session_data = session_data[session_data['trial_end'] == 1]

    num_trials_to_consider = 10

    new_data = dict(
        prev_block_durations=[],
        model_prob_correct_slopes=[],
        model_prob_correct_intercepts=[])

    # TODO: can this be refactored using multiple aggregate?
    for (session_index, block_index), block_session_data in session_data.groupby(
            ['session_index', 'block_index']):

        # skip first block because they have no preceding block!
        if block_index == 0:
            continue

        # skip truncated blocks with fewer than the minimum number of trials
        if len(block_session_data) < num_trials_to_consider:
            continue

        # keep only the first ten trials
        first_n_trials = block_session_data[block_session_data.trial_index < num_trials_to_consider]

        prev_block_duration = max(
            session_data[(session_data.session_index == session_index) &
                         (session_data.block_index == (block_index - 1))].trial_index)

        # calculate slope of best fit
        # TODO: this is the wrong regression
        coefficients = np.polyfit(
            x=first_n_trials.trial_index.values.astype(np.float32),
            y=first_n_trials.correct_action_taken.values.astype(np.float32),  # need to convert
            deg=1)
        slope, intercept = coefficients[0], coefficients[1]

        # plot the best fit line
        # plt.plot(first_n_trials.trial_index.values.astype(np.float32),
        #          first_n_trials.correct_action_prob.values.astype(np.float32))
        # plt.plot(first_n_trials.trial_index.values.astype(np.float32),
        #          np.poly1d(coefficients)(first_n_trials.trial_index.values.astype(np.float32)))

        new_data['prev_block_durations'].append(prev_block_duration)
        new_data['model_prob_correct_slopes'].append(slope)
        new_data['model_prob_correct_intercepts'].append(intercept)

    means_sem = pd.DataFrame(new_data).groupby('prev_block_durations').agg(['mean', 'sem'])

    fig, axes = plt.subplots(nrows=2, figsize=(4, 3), sharex=True)
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for i, column_str in enumerate(['model_prob_correct_slopes', 'model_prob_correct_intercepts']):
        ax = axes[i]
        if i == 0:
            ax.set_title(f'Slope of P(Correct Action) (First {num_trials_to_consider}'
                         f' Trials) by Previous Block Duration')
            ax.set_ylabel('Slope of Model P(Correct Action)')
        elif i == 1:
            ax.set_title(f'Intercept of P(Correct Action) (First {num_trials_to_consider}'
                         f' Trials) by Previous Block Duration')
            ax.set_ylabel('Intercept of Model P(Correct Action)')
        else:
            raise ValueError('Impermissible axis number')

        # plot mean
        ax.plot(means_sem.index.values,
                means_sem[column_str]['mean'].values,
                '-o',
                markersize=2,
                color=side_color_map['neutral'])

        # add SEM
        ax.fill_between(
            x=means_sem.index.values,
            y1=means_sem[column_str]['mean'].values - means_sem[column_str]['sem'].values,
            y2=means_sem[column_str]['mean'].values + means_sem[column_str]['sem'].values,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['neutral'])
    ax.set_xlabel('Previous Block Duration')
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_slope_intercept_by_prev_block_duration',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_reward_rate(hook_input):
    session_data = hook_input['session_data']
    reward_data = session_data.loc[
        session_data.trial_end == 1.,
        ['rnn_step_index', 'reward', 'bayesian_observer_reward']]

    # don't count blank dts, then add 1 because rnn step indexing starts with 0
    # for the first observation
    reward_data['rnn_step_index'] += 1 - hook_input['envs'][0].rnn_steps_before_obs

    # step penalty
    step_penalty = hook_input['envs'][0].time_delay_penalty
    rnn_step_penalty = reward_data['rnn_step_index'] * step_penalty
    reward_data['penalized_reward'] = reward_data['reward'] + rnn_step_penalty
    reward_data['penalized_bayesian_observer_reward'] = \
        reward_data['bayesian_observer_reward'] + rnn_step_penalty

    penalized_reward_by_num_obs = reward_data.groupby('rnn_step_index').agg({
        'penalized_reward': ['mean', 'sem', 'size'],
        'penalized_bayesian_observer_reward': ['mean', 'sem', 'size']})

    # drop NaN/0 SEM
    rnn_penalized_reward_by_num_obs = penalized_reward_by_num_obs['penalized_reward']
    rnn_penalized_reward_by_num_obs = rnn_penalized_reward_by_num_obs[
        rnn_penalized_reward_by_num_obs['size'] > 1.]
    observer_penalized_reward_by_num_obs = penalized_reward_by_num_obs[
        'penalized_bayesian_observer_reward']
    observer_penalized_reward_by_num_obs = observer_penalized_reward_by_num_obs[
        observer_penalized_reward_by_num_obs['size'] > 1.]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Reward Rate')
    ax.plot(
        rnn_penalized_reward_by_num_obs.index,
        rnn_penalized_reward_by_num_obs['mean'],
        '-o',
        markersize=2,
        label='RNN',
        color=side_color_map['correct'])
    ax.fill_between(
        x=rnn_penalized_reward_by_num_obs.index,
        y1=rnn_penalized_reward_by_num_obs['mean']
           - rnn_penalized_reward_by_num_obs['sem'],
        y2=rnn_penalized_reward_by_num_obs['mean']
           + rnn_penalized_reward_by_num_obs['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])
    ax.plot(
        observer_penalized_reward_by_num_obs.index,
        observer_penalized_reward_by_num_obs['mean'],
        '--o',
        markersize=2,
        label='Bayesian Observer',
        color=side_color_map['ideal'])
    ax.fill_between(
        x=observer_penalized_reward_by_num_obs.index.values,
        y1=observer_penalized_reward_by_num_obs['mean']
           - observer_penalized_reward_by_num_obs['sem'],
        y2=observer_penalized_reward_by_num_obs['mean']
           + observer_penalized_reward_by_num_obs['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_reward_data = actor_session_data.loc[
        actor_session_data.trial_end == 1.,
        ['rnn_step_index', 'reward']]

    # don't count blank dts, then add 1 because rnn step indexing starts with 0
    # for the first observation
    actor_reward_data['rnn_step_index'] += 1 - hook_input['envs'][0].rnn_steps_before_obs

    # step penalty
    rnn_step_penalty = actor_reward_data['rnn_step_index'] * step_penalty
    actor_reward_data['penalized_reward'] = actor_reward_data['reward'] + rnn_step_penalty

    actor_penalized_reward_by_num_obs = actor_reward_data.groupby('rnn_step_index').agg({
        'penalized_reward': ['mean', 'sem', 'size']})['penalized_reward']

    # drop NaN/0 SEM
    actor_penalized_reward_by_num_obs = actor_penalized_reward_by_num_obs[
        actor_penalized_reward_by_num_obs['size'] > 1.]

    ax.plot(
        actor_penalized_reward_by_num_obs.index,
        actor_penalized_reward_by_num_obs['mean'],
        '-o',
        markersize=2,
        label='Bayesian Actor',
        color=side_color_map['ideal'])
    ax.fill_between(
        x=actor_penalized_reward_by_num_obs.index.values,
        y1=actor_penalized_reward_by_num_obs['mean']
           - actor_penalized_reward_by_num_obs['sem'],
        y2=actor_penalized_reward_by_num_obs['mean']
           + actor_penalized_reward_by_num_obs['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    ax.legend(markerscale=0.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_rightward_action_by_signed_contrast',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_right_action_after_error_by_right_action_after_correct(hook_input):
    # TODO: Behavioral paper 4g
    print(10)
    pass


def hook_plot_behav_right_action_by_signed_contrast(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Right Action Taken / Total Action Trials by Signed Trial Strength')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Signed Trial Strength')
    ax.set_ylabel('# Right / # Trials')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # only consider trials with actions
    session_data = hook_input['session_data']
    action_data = session_data[session_data['action_taken'] == 1].copy()

    # rescale from [-1, -1] to [0, 1]
    action_data['action_side'] = (1 + action_data['action_side']) / 2
    action_data['bayesian_observer_action_side'] = (1 + action_data['bayesian_observer_action_side']) / 2
    bayesian_action_by_signed_trial_strength = action_data.groupby(
        ['block_side', 'signed_trial_strength']).agg(
        {'action_side': ['mean', 'sem'],
         'bayesian_observer_action_side': ['mean', 'sem']})

    for block_side in action_data['block_side'].unique():
        # take cross section of block side
        action_by_signed_trial_strength_by_block_side = \
            bayesian_action_by_signed_trial_strength.xs(block_side)['action_side']

        # plot non-block conditioned
        ax.plot(
            action_by_signed_trial_strength_by_block_side.index.values,
            action_by_signed_trial_strength_by_block_side['mean'],
            '-o',
            label=side_string_map[block_side] + ' Block',
            color=side_color_map[block_side],
            markersize=2)

        ax.fill_between(
            x=action_by_signed_trial_strength_by_block_side.index.values,
            y1=action_by_signed_trial_strength_by_block_side['mean'] -
               action_by_signed_trial_strength_by_block_side['sem'],
            y2=action_by_signed_trial_strength_by_block_side['mean'] +
               action_by_signed_trial_strength_by_block_side['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])

    for block_side in action_data['block_side'].unique():
        # take cross section of block side
        optimal_action_by_signed_trial_strength_by_block_side = \
            bayesian_action_by_signed_trial_strength.xs(block_side)['bayesian_observer_action_side']

        # plot non-block conditioned
        ax.plot(
            optimal_action_by_signed_trial_strength_by_block_side.index.values,
            optimal_action_by_signed_trial_strength_by_block_side['mean'],
            '--o',
            label=f'Bayesian Observer',
            color=side_color_map['ideal'],
            markersize=2)

        ax.fill_between(
            x=optimal_action_by_signed_trial_strength_by_block_side.index.values,
            y1=optimal_action_by_signed_trial_strength_by_block_side['mean'] -
               optimal_action_by_signed_trial_strength_by_block_side['sem'],
            y2=optimal_action_by_signed_trial_strength_by_block_side['mean'] +
               optimal_action_by_signed_trial_strength_by_block_side['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal'])

    bayesian_session_data = hook_input['bayesian_actor_session_data']
    bayesian_action_data = bayesian_session_data[bayesian_session_data['action_taken'] == 1].copy()

    # rescale from [-1, -1] to [0, 1]
    bayesian_action_data['action_side'] = (1 + bayesian_action_data['action_side']) / 2
    bayesian_action_by_signed_trial_strength = bayesian_action_data.groupby(
        ['block_side', 'signed_trial_strength']).agg(
        {'action_side': ['mean', 'sem']})
    for block_side in bayesian_action_data['block_side'].unique():
        # take cross section of block side
        bayesian_action_by_signed_trial_strength_by_block_side = \
            bayesian_action_by_signed_trial_strength.xs(block_side)['action_side']

        # plot non-block conditioned
        ax.plot(
            bayesian_action_by_signed_trial_strength_by_block_side.index.values,
            bayesian_action_by_signed_trial_strength_by_block_side['mean'],
            '-o',
            label='Bayesian Actor',
            color=side_color_map['ideal'],
            markersize=2)

        ax.fill_between(
            x=bayesian_action_by_signed_trial_strength_by_block_side.index.values,
            y1=bayesian_action_by_signed_trial_strength_by_block_side['mean'] -
               bayesian_action_by_signed_trial_strength_by_block_side['sem'],
            y2=bayesian_action_by_signed_trial_strength_by_block_side['mean'] +
               bayesian_action_by_signed_trial_strength_by_block_side['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal'])

    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels, markerscale=0.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_rightward_action_by_signed_contrast',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_subj_prob_block_switch_by_signed_trial_strength(hook_input):
    session_data = hook_input['session_data']

    trial_data = pd.DataFrame(
        columns=['signed_trial_strength',
                 'left_action_prob',
                 'reward',
                 '1_back_correct',
                 '2_back_correct',
                 '3_back_correct',
                 'prev_left_action_prob',
                 'subjective_block_switch'],
        dtype=np.float16)

    for session_index, per_session_data in session_data.groupby(['session_index']):
        # TODO lambda breaks if len(x) < 2
        trial_data_within_session = per_session_data.groupby(['block_index', 'trial_index']).agg({
            'signed_trial_strength': 'first',  # arbitrary
            'left_action_prob': 'first',
            'reward': 'last',
            '1_back_correct': 'last',
            '2_back_correct': 'last',
            '3_back_correct': 'last'})

        trial_data_within_session['prev_left_action_prob'] = trial_data_within_session.left_action_prob.shift(periods=1)

        # drop rows with no previous action
        trial_data_within_session = trial_data_within_session[
            ~pd.isna(trial_data_within_session.prev_left_action_prob)]

        trial_data_within_session['subjective_block_switch'] = \
            (trial_data_within_session.prev_left_action_prob < 0.5) ^ \
            (trial_data_within_session.left_action_prob < 0.5)

        trial_data = trial_data.append(trial_data_within_session)

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Subjective P(Block Switch) by Signed Trial Strength')
    ax.set_ylim([-0.05, 1.05])
    max_trial_strength = max(hook_input['envs'][0].possible_trial_strengths)
    ax.set_xlim([-max_trial_strength, max_trial_strength])
    ax.set_xlabel('Signed Trial Strength')
    ax.set_ylabel('Subjective P(Block Switch)')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    try:
        mean_prob_block_switch_reward = trial_data[trial_data.reward == 1].groupby(['signed_trial_strength'])[
            'subjective_block_switch'].mean()
        ax.plot(
            mean_prob_block_switch_reward.index.values,
            mean_prob_block_switch_reward,
            '-o',
            linewidth=1,
            markersize=5,
            fillstyle='none',
            label='Reward',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    try:
        avg_prob_block_switch_1_back = trial_data[
            (trial_data.reward == -1) & (trial_data['1_back_correct'] == 1)].groupby([
            'signed_trial_strength'])['subjective_block_switch'].mean()
        ax.plot(
            avg_prob_block_switch_1_back.index.values,
            avg_prob_block_switch_1_back,
            '-+',
            linewidth=1,
            markersize=5,
            label='1-Back Error',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    try:
        avg_prob_block_switch_2_back = trial_data[
            (trial_data.reward == -1) & (trial_data['1_back_correct'] == -1) & (
                    trial_data['2_back_correct'] == 1)].groupby([
            'signed_trial_strength'])['subjective_block_switch'].mean()
        ax.plot(
            avg_prob_block_switch_2_back.index.values,
            avg_prob_block_switch_2_back,
            '-d',
            linewidth=1,
            markersize=5,
            label='2-Back Error',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_subj_prob_block_switch_by_signed_trial_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_trial_outcome_by_trial_strength(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.]

    trial_outcome_by_trial_strength = trial_end_data.groupby(['signed_trial_strength']).agg({
        'block_index': 'size',  # can use any column to count number of datum in each group
        'action_taken': 'sum',
        'correct_action_taken': 'sum'
    })

    observer_trial_outcome_by_trial_strength = trial_end_data.groupby(['signed_trial_strength']).agg({
        'block_index': 'size',  # can use any column to count number of datum in each group
        'bayesian_observer_correct_action_taken': 'mean'
    })

    actor_trial_session_data = hook_input['bayesian_actor_session_data']
    actor_trial_end_data = actor_trial_session_data[actor_trial_session_data.trial_end == 1.]
    actor_trial_outcome_by_trial_strength = actor_trial_end_data.groupby(['signed_trial_strength']).agg({
        'block_index': 'size',  # can use any column to count number of datum in each group
        'action_taken': 'mean',
        'correct_action_taken': 'mean'
    })

    trial_outcome_by_trial_strength.rename(
        columns={'block_index': 'num_trials'},
        inplace=True)

    trial_outcome_by_trial_strength['is_timeout'] = \
        trial_outcome_by_trial_strength['num_trials'] - trial_outcome_by_trial_strength['action_taken']

    trial_outcome_by_trial_strength['incorrect_action_taken'] = \
        trial_outcome_by_trial_strength['action_taken'] - \
        trial_outcome_by_trial_strength['correct_action_taken']

    # drop unnecessary columns
    trial_outcome_by_trial_strength.drop(
        columns=['num_trials', 'action_taken'],
        inplace=True)

    # divide each row by row sum to get percents
    trial_outcome_by_trial_strength = trial_outcome_by_trial_strength.div(
        trial_outcome_by_trial_strength.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Trial Outcome (%) by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylabel('Trial Outcome (%)')
    ax.set_xlabel('Signed Stimulus Contrast')
    ax.set_ylim([-0.05, 1.05])
    width = 0.35
    rects = ax.bar(
        trial_outcome_by_trial_strength.index,
        trial_outcome_by_trial_strength.correct_action_taken,
        width=width,
        label='RNN Correct Action',
        color=side_color_map['correct'])
    # add text below specifying the y value
    # for rect in rects:
    #     height = rect.get_height()
    #     ax.annotate('{}'.format(np.round(height, 2)),
    #                 xy=(rect.get_x() + rect.get_width() / 2, height),
    #                 xytext=(0, -12),  # 9 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')
    ax.bar(trial_outcome_by_trial_strength.index,
           trial_outcome_by_trial_strength.incorrect_action_taken,
           width=width,
           bottom=trial_outcome_by_trial_strength.correct_action_taken,
           label='RNN Incorrect Action',
           color=side_color_map['incorrect'])
    ax.bar(trial_outcome_by_trial_strength.index,
           trial_outcome_by_trial_strength.is_timeout,
           width=width,
           bottom=trial_outcome_by_trial_strength.correct_action_taken
                  + trial_outcome_by_trial_strength.incorrect_action_taken,
           label='RNN Timeout',
           color=side_color_map['timeout'])
    ax.plot(
        actor_trial_outcome_by_trial_strength.index,
        actor_trial_outcome_by_trial_strength.correct_action_taken,
        '-',
        color=side_color_map['ideal_correct'],
        label='Bayesian Actor Correct Actions',
        markersize=2)
    ax.plot(
        observer_trial_outcome_by_trial_strength.index,
        observer_trial_outcome_by_trial_strength.bayesian_observer_correct_action_taken,
        '--',
        color=side_color_map['ideal_correct'],
        label='Bayesian Observer Correct Actions',
        markersize=2)
    ax.axhline(y=0.5, color='k')
    plt.text(0.0, 0.45, 'Chance', fontsize=10, va='center', ha='center')
    ax.legend(markerscale=0.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_trial_outcome_by_trial_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_compare_all_rnns_prob_correct_by_strength_concordant(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel('# Correct / # Trials')
    ax.set_xlabel('Signed Stimulus Contrast')

    model_names = [
        'Traditionally Distilled (2 Units)',
        'Task Trained ({} Units)'.format(hook_input['model'].core.hidden_size),
        'RADD RNN (2 Units)',
        'Task Trained (2 Units)']
    session_datas = [
        hook_input['traditionally_distilled_session_data'],
        hook_input['session_data'],
        hook_input['radd_session_data'],
        hook_input['two_unit_task_trained_session_data']
    ]
    colors = ['#984ea3', 'tab:green', 'tab:pink', 'tab:cyan']
    for model_name, session_data, color in zip(model_names, session_datas, colors):

        # keep only last dts in trials
        trial_end_data = session_data[session_data.trial_end == 1.]

        for concordant, concordant_data in trial_end_data.groupby(['concordant_trial']):

            avg_correct_action_prob_by_stim_strength = concordant_data.groupby(
                'signed_trial_strength')['correct_action_taken'].mean()

            sem_correct_action_prob_by_stim_strength = concordant_data.groupby(
                'signed_trial_strength')['correct_action_taken'].sem()

            if concordant:
                label = f'{model_name}'  # Condordant
                style = '-o'
            else:
                label = f'{model_name}'  # Discordant
                style = '--o'
            ax.plot(
                avg_correct_action_prob_by_stim_strength.index,
                avg_correct_action_prob_by_stim_strength,
                style,
                markersize=2,
                # solid lines for consistent block side, trial side; dotted otherwise
                label=label,
                color=color)
            ax.fill_between(
                x=avg_correct_action_prob_by_stim_strength.index,
                y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
                y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
                alpha=0.3,
                linewidth=0,
                color=color)

    # necessary to delete redundant legend groups
    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels, markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='compare_all_rnns_prob_correct_by_strength_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_compare_all_rnns_prob_correct_by_trial_within_block(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('# Correct / # Trials')
    ax.set_ylim([0.5, 1.05])
    ax.set_xlim([0., 101.])
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    model_names = [
        'Traditionally Distilled (2 Units)',
        'Task Trained (50 Units)',
        'RADD RNN (2 Units)',
        'Task Trained (2 Units)']
    session_datas = [
        hook_input['traditionally_distilled_session_data'],
        hook_input['session_data'],
        hook_input['radd_session_data'],
        hook_input['two_unit_task_trained_session_data']
    ]

    # purple, green,
    colors = ['#984ea3', '#4daf4a', 'tab:pink', 'tab:cyan']
    for model_name, session_data, color in zip(model_names, session_datas, colors):
        # keep only last dts in trials
        trial_end_data = session_data[session_data.trial_end == 1.]

        # plot trial number within block (x) vs probability of correct response (y)
        rnn_correct_by_trial_index = trial_end_data.groupby(['trial_index']).agg({
            'correct_action_taken': ['mean', 'sem', 'size'],
        })['correct_action_taken']

        rnn_correct_by_trial_index = rnn_correct_by_trial_index[
            rnn_correct_by_trial_index['size'] > 1.]

        ax.plot(
            1 + rnn_correct_by_trial_index.index.values,
            rnn_correct_by_trial_index['mean'],
            '-o',
            markersize=1,
            linewidth=1,
            color=color,
            label=model_name)

        ax.fill_between(
            x=1 + rnn_correct_by_trial_index.index.values,
            y1=rnn_correct_by_trial_index['mean']
               - rnn_correct_by_trial_index['sem'],
            y2=rnn_correct_by_trial_index['mean']
               + rnn_correct_by_trial_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map['correct'])

    # necessary to delete redundant legend groups
    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels, markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='compare_all_rnns_prob_correct_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane(hook_input):
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_model_jacobians_by_stimulus_and_feedback(
        model=hook_input['model'],
        trial_data=hook_input['session_data'],
        fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli)

    num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=2,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    jacobian_colors = dict(
        hidden_to_hidden='tab:blue',
        stimuli_to_hidden='tab:orange',
        rewards_to_hidden='tab:green')

    for c, (side, jacobians_by_stimuli) in \
            enumerate(jacobians_by_side_by_stimuli.items()):

        for r, (stimulus, jacobians) in enumerate(jacobians_by_stimuli.items()):

            ax = axes[r, c]
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel(r'$\Re(\lambda)$')

            if c == 0:
                ax.set_ylabel(r'$\Im(\lambda)$')

            for jacobian_name, jacobian in jacobians.items():

                if jacobian_name != 'hidden_to_hidden':
                    continue

                jacobian_eigvals = np.linalg.eigvals(jacobian)
                print(max(jacobian_eigvals))

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)

                sc = ax.scatter(
                    jacobian_eigvals.real,
                    jacobian_eigvals.imag,
                    c=jacobian_colors[jacobian_name],
                    s=2,
                    label=jacobian_name)

            ax.legend()

            # add circle
            circle = plt.Circle((0, 0), radius=1, color='k', fill=False)
            ax.add_patch(circle)

    # fig.suptitle(f'Hidden to Hidden Jacobians\' Eigenvalues')
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_to_hidden_jacobian_eigenvalues_complex_plane',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_hidden_to_hidden_jacobian_time_constants(hook_input):
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_model_jacobians_by_stimulus_and_feedback(
        model=hook_input['model'],
        trial_data=hook_input['session_data'],
        fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli)

    num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=2,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    jacobian_colors = dict(
        hidden_to_hidden='tab:blue',
        stimuli_to_hidden='tab:orange',
        rewards_to_hidden='tab:green')

    for c, (side, jacobians_by_stimuli) in \
            enumerate(jacobians_by_side_by_stimuli.items()):

        for r, (stimulus, jacobians) in enumerate(jacobians_by_stimuli.items()):

            ax = axes[r, c]
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel('Eigenvalue Index')

            if c == 0:
                ax.set_ylabel(r'Time Constant ($\tau$)')

            for jacobian_name, jacobian in jacobians.items():

                if jacobian_name != 'hidden_to_hidden':
                    continue

                jacobian_eigvals = np.linalg.eigvals(jacobian)
                time_constants = np.sort(np.abs(1. / np.log(np.abs(jacobian_eigvals))))[::-1]
                eigvals_indices = np.arange(1, 1 + len(jacobian_eigvals))

                sc = ax.scatter(
                    eigvals_indices,
                    time_constants,
                    c=jacobian_colors[jacobian_name],
                    # s=2,
                    label=jacobian_name)

            ax.legend()

    # fig.suptitle('Hidden to Hidden Jacobians\' Time Constants')
    # TODO understand why this produces such inconsistent plots
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_to_hidden_jacobian_time_constants',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_model_recurrent_weight_by_cluster(hook_input):

    cutoff = 0.15
    df = pd.DataFrame(dict(
        trial_readout=hook_input['trial_readout_vector'][0],
        block_readout=hook_input['block_readout_vector'][0]))
    df['trial_bins'] = pd.cut(
        df['trial_readout'],
        bins=[-float('inf'), -cutoff, cutoff, float('inf')])
    df['block_bins'] = pd.cut(
        df['block_readout'],
        bins=[-float('inf'), -cutoff, cutoff, float('inf')])

    left_interval = pd.Interval(left=-np.inf, right=-cutoff, closed='right')
    center_interval = pd.Interval(left=-cutoff, right=cutoff, closed='right')
    right_interval = pd.Interval(left=cutoff, right=np.inf)

    df['cluster'] = 'Remainder'
    df['cluster'][(df['trial_bins'] == left_interval) &
                  (df['block_bins'] == center_interval)] = 'Stim: -, Block: 0'
    df['cluster'][(df['trial_bins'] == right_interval) &
                  (df['block_bins'] == center_interval)] = 'Stim: +, Block: 0'
    df['cluster'][(df['trial_bins'] == center_interval) &
                  (df['block_bins'] == left_interval)] = 'Stim: 0, Block: -'
    df['cluster'][(df['trial_bins'] == center_interval) &
                  (df['block_bins'] == right_interval)] = 'Stim: 0, Block: +'

    # redefine cluster as Categorical to enable sorting
    df['cluster'] = pd.Categorical(df['cluster'], [
        'Stim: 0, Block: +',
        'Stim: 0, Block: -',
        'Stim: +, Block: 0',
        'Stim: -, Block: 0',
        'Remainder'])
    df.sort_values(by=['cluster'], inplace=True)

    # create custom heatmap axis labels
    cluster_nonchange_indices = df['cluster'].eq(df['cluster'].shift())
    custom_ticklabels = df['cluster'].astype(np.str)
    custom_ticklabels[cluster_nonchange_indices] = ''

    recurrent_matrix = hook_input['model'].core.weight_hh_l0.detach().numpy()

    fig, ax = plt.subplots()
    indices = df.index.values
    ax.set_title(f'Cutoff: {cutoff}')
    sns.heatmap(recurrent_matrix[indices][:, indices],
                cmap='RdBu_r',
                center=0,
                xticklabels=custom_ticklabels,  # indices
                yticklabels=custom_ticklabels,  # indices
                square=True,
                ax=ax,
                cbar_kws={'label': 'Avg Weight', 'shrink': 0.5})

    hook_input['tensorboard_writer'].add_figure(
        tag='model_recurrent_weight_by_cluster',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_recurrent_weight_avg_by_cluster(hook_input):

    cutoff = 0.15
    df = pd.DataFrame(dict(
        trial_readout=hook_input['trial_readout_vector'][0],
        block_readout=hook_input['block_readout_vector'][0]))
    df['trial_bins'] = pd.cut(
        df['trial_readout'],
        bins=[-float('inf'), -cutoff, cutoff, float('inf')])
    df['block_bins'] = pd.cut(
        df['block_readout'],
        bins=[-float('inf'), -cutoff, cutoff, float('inf')])

    left_interval = pd.Interval(left=-np.inf, right=-cutoff, closed='right')
    center_interval = pd.Interval(left=-cutoff, right=cutoff, closed='right')
    right_interval = pd.Interval(left=cutoff, right=np.inf)

    df['cluster'] = 'Remainder'
    df['cluster'][(df['trial_bins'] == left_interval) &
                  (df['block_bins'] == center_interval)] = 'Stim: -, Block: 0'
    df['cluster'][(df['trial_bins'] == right_interval) &
                  (df['block_bins'] == center_interval)] = 'Stim: +, Block: 0'
    df['cluster'][(df['trial_bins'] == center_interval) &
                  (df['block_bins'] == left_interval)] = 'Stim: 0, Block: -'
    df['cluster'][(df['trial_bins'] == center_interval) &
                  (df['block_bins'] == right_interval)] = 'Stim: 0, Block: +'
    df.sort_values(by=['cluster', 'trial_readout', 'block_readout'], inplace=True)

    cluster_weight_averages = pd.DataFrame(
        np.nan,
        columns=df['cluster'].unique(),
        index=df['cluster'].unique())
    recurrent_matrix = hook_input['model'].core.weight_hh_l0.detach().numpy()
    for to_cluster_str, to_df in df.groupby(['cluster']):
        to_neuron_indices = to_df.index.values
        for from_cluster_str, from_df in df.groupby(['cluster']):
            from_neuron_indices = from_df.index.values
            cluster_weight_averages.loc[to_cluster_str, from_cluster_str] = np.mean(
                recurrent_matrix[to_neuron_indices][:, from_neuron_indices])

    ordering = np.array([
        'Stim: 0, Block: +',
        'Stim: 0, Block: -',
        'Stim: +, Block: 0',
        'Stim: -, Block: 0',
        'Remainder',
    ])

    fig, ax = plt.subplots()
    ax.set_title(f'Cutoff: {cutoff}')
    sns.heatmap(cluster_weight_averages.loc[ordering, ordering],
                cmap='RdBu_r',
                center=0,
                # vmin=-1.,
                # vmax=1.,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Avg Weight', 'shrink': 0.5})

    hook_input['tensorboard_writer'].add_figure(
        tag='model_recurrent_weight_avg_by_cluster',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

    #
    #
    # indices = df.index.values
    # group_indices = df.groupby(['cluster']).size().cumsum().values
    #
    # ax = sns.heatmap(recurrent_matrix[indices][:, indices],
    #             cmap='RdBu_r',
    #             center=0,
    #             # vmin=-1.,
    #             # vmax=1.,
    #             square=True,
    #             xticklabels=indices,  # indices
    #             yticklabels=indices,  # indices
    #             cbar_kws={'label': 'PC Distance', 'shrink': 0.5})
    # ax.hlines(group_indices, *ax.get_xlim())
    # ax.vlines(group_indices, *ax.get_ylim())
    # plt.show()
    #
    #
    # # df['trial_readout_pos'] = df['trial_readout'] > 0
    # # df['block_readout_pos'] = df['block_readout'] > 0
    #
    # rows = []
    # for i in range(recurrent_matrix.shape[1]):
    #     for j in range(recurrent_matrix.shape[1]):
    #         row = [
    #             hidden_states_pca_results['trial_readout_vector'][0][i],
    #             model_block_readout_vectors_results['block_readout_vector'][0][i],
    #             recurrent_matrix[i, j],
    #             hidden_states_pca_results['trial_readout_vector'][0][j],
    #             model_block_readout_vectors_results['block_readout_vector'][0][j],
    #             0.]
    #         rows.append(row)
    # df = pd.DataFrame(
    #     rows,
    #     columns=['to_trial_coef', 'to_block_coef', 'to_z', 'from_trial_coef', 'from_block_coef', 'from_z'])
    #
    # from sklearn.preprocessing import PolynomialFeatures
    # poly_features = PolynomialFeatures(interaction_only=True, include_bias=False)
    # X = df[['to_trial_coef', 'to_block_coef', 'from_trial_coef', 'from_block_coef']]
    # X_with_interactions = poly_features.fit_transform(X)
    # feature_names = poly_features.get_feature_names(['to_trial_coef', 'to_block_coef', 'from_trial_coef', 'from_block_coef'])
    # y = df['to_z']
    #
    #
    # from sklearn.linear_model import LassoCV, LinearRegression
    # reg = LassoCV(alphas=np.logspace(-10, -1, 10), fit_intercept=False, normalize=True, cv=10)
    # reg.fit(X_with_interactions, y)
    # reg.score(X_with_interactions, y)
    #
    #
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.quiver(df_subset['from_trial_coef'],
    #           df_subset['from_block_coef'],
    #           df_subset['from_z'],
    #           df_subset['to_trial_coef'] - df_subset['from_trial_coef'],
    #           df_subset['to_block_coef'] - df_subset['from_block_coef'],
    #           df_subset['to_z'])
    # plt.show()
    #
    #
    # fig, ax = plt.subplots(figsize=(5, 5))
    # for (from_trial_pos, from_block_pos), from_df in df.groupby(['trial_readout_pos', 'block_readout_pos']):
    #     from_idx = from_df.index.values
    #     for (to_trial_pos, to_block_pos), to_df in df.groupby(['trial_readout_pos', 'block_readout_pos']):
    #         to_idx = to_df.index.values
    #         recurrent_weights = recurrent_matrix[to_idx][:, from_idx]

    # import scipy.cluster.hierarchy as spc
    # pdist = spc.distance.pdist(readout_vectors)
    # linkage = spc.linkage(pdist, method='complete')
    # labels = spc.fcluster(linkage, 0.5 * np.max(pdist), 'distance')
    # indices = np.argsort(labels)
    #
    #

    #
    #
    # recurrent_matrix[indices][:, indices]
    #
    # u, s, vh = np.linalg.svd(recurrent_matrix, full_matrices=True)
    # plt.hist(s)
    # plt.show()
    # pca_vh = hidden_states_pca_results['pca'].transform(u)
    # plt.scatter(pca_vh[:, 0],
    #             pca_vh[:, 1])
    # plt.show()


def hook_plot_model_effective_circuit(hook_input):
    # hidden states shape: (num rnn steps, num layers, hidden dimension)
    hidden_states = hook_input['hidden_states']
    hidden_size = hidden_states.shape[2]

    # reshape to (num trials, num layers * hidden dimension)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    trial_side = np.expand_dims(hook_input['session_data'].trial_side.values, 1)
    trial_side_orthogonal = np.expand_dims(hook_input['session_data'].trial_side_orthogonal.values, 1)
    block_side = np.expand_dims(hook_input['session_data'].block_side.values, 1)
    feedback = np.expand_dims(hook_input['session_data'].reward.values, 1)

    # construct correlation matrix
    hidden_states_and_task_variables = np.hstack((
        hidden_states,
        trial_side,
        trial_side_orthogonal,
        block_side,
        feedback))
    hidden_states_and_task_variables_correlations = np.corrcoef(hidden_states_and_task_variables.T)
    # due to machine error, correlation matrix isn't exactly symmetric (typically has e-16 errors)
    # so make it symmetric
    hidden_states_and_task_variables_correlations = (hidden_states_and_task_variables_correlations +
                                                     hidden_states_and_task_variables_correlations.T) / 2
    hidden_state_self_correlations = hidden_states_and_task_variables_correlations[:hidden_size, :hidden_size]
    hidden_state_task_correlations = hidden_states_and_task_variables_correlations[:hidden_size, hidden_size:]

    # compute pairwise distances
    pdist = spc.distance.pdist(hidden_state_self_correlations)
    linkage = spc.linkage(pdist, method='complete')
    labels = spc.fcluster(linkage, 0.5 * np.max(pdist), 'distance')
    # indices = np.argsort(labels)
    indices = np.arange(hidden_size) #if we don't want to sort by hidden correlations
    #TODO: when applying the mask, don't sort by this?

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 1]})
    # recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # plot hidden state correlations
    ax = axes[0]
    sns.heatmap(hidden_state_self_correlations[indices][:, indices],
                cmap='RdBu_r',
                ax=ax,
                center=0,
                vmin=-1.,
                vmax=1.,
                square=True,
                # xticklabels=indices,  # indices
                # yticklabels=indices,  # indices
                cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    ax.set_title('Hidden Unit - Hidden Unit Correlations')
    ax.set_xlabel('Hidden Unit Number')
    ax.set_ylabel('Hidden Unit Number')
    ax.set_aspect("equal")  # ensures little squares don't become rectangles

    # if hook_input['tag_prefix'] == 'analyze/':
    #     ax = axes[1]
    #     normalized_readout_vectors = np.concatenate(
    #         [hook_input['trial_readout_vector'],
    #          hook_input['block_readout_vector']],
    #         axis=0)
    #     sns.heatmap(normalized_readout_vectors[:, indices].T,
    #                 cmap='RdBu_r',
    #                 ax=ax,
    #                 center=0,
    #                 xticklabels=['Trial Readout', 'Block Readout'],
    #                 yticklabels=indices,
    #                 square=True,
    #                 cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    #     ax.set_title('Readout Vectors')
    #     ax.set_ylabel('Hidden Unit Number')

    recurrent_matrix = hook_input['model'].core.weight_hh_l0.data.numpy()

    dimension_ratio = recurrent_matrix.shape[0] / recurrent_matrix.shape[1]
    # # RNN weight will have shape (hidden size, hidden size)
    if dimension_ratio == 1:
        recurrent_matrix = recurrent_matrix[indices][:, indices]
    # LSTM weight will have shape (4*hidden size, hidden_size)
    # GRU weight will have shape (3*hidden size, hidden size)
    elif dimension_ratio == 4 or dimension_ratio == 3:
        pass
        # TODO add recurrent weight
    #     # TODO unknown whether this is correct
    #     for i in range(int(dimension_ratio)):
    #         recurrent_matrix[i*hidden_size:(i+1)*hidden_size] = \
    #             recurrent_matrix[i*hidden_size + indices][:, indices]
    else:
        raise ValueError('Unknown dimension ratio for recurrent weight matrix')

    # plot recurrent matrix values
    ax = axes[1]
    sns.heatmap(recurrent_matrix,
                cmap='RdBu_r',
                ax=ax,
                center=0,
                # xticklabels=indices,
                # yticklabels=indices,
                square=True,
                cbar_kws={'label': 'Weight Strength', 'shrink': 0.5})
    ax.set_title('Recurrent Weight Strength')
    ax.set_xlabel('Hidden Unit Number')
    # ax.set_ylabel('Hidden Unit Number')
    ax.set_aspect("equal")  # ensures little squares don't become rectangles

    # hidden state vs task side, block side correlation
    # ax = axes[3]
    # sns.heatmap(hidden_state_task_correlations[indices, :],
    #             cmap='RdBu_r',
    #             ax=ax,
    #             center=0,
    #             xticklabels=['Trial', 'Ortho Trial', 'Block', 'Feedback'],
    #             yticklabels=indices,
    #             square=True,
    #             cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    # ax.set_title('Hidden Unit - Task Correlations')
    # ax.set_ylabel('Hidden Unit Number')

    # we also would like to plot the distribution of recurrent weights per cluster
    # and between clusters
    # first, set

    hook_input['tensorboard_writer'].add_figure(
        tag='model_effective_circuit',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)
        
def hook_plot_proj_readout_onto_template(hook_input):
    trial_readout = hook_input['trial_readout_vector'].squeeze()
    block_readout = hook_input['block_readout_vector'].squeeze()
    #absolute value to make our metric make sense!
    trial_readout = np.abs(trial_readout)
    block_readout = np.abs(block_readout)
    hidden_shape = trial_readout.size
    template_1, template_2 = np.zeros(hidden_shape), np.zeros(hidden_shape)
    template_1[:hidden_shape//2] = 1
    template_2[hidden_shape//2:] = 1
    angle_bw_template_and_readout = []
    angle_bw_tp_names = []
    for i,m in enumerate([template_1,template_2]):
        for j, r in enumerate([trial_readout,block_readout]):
            name = f"$m_{i} \cdot r_{'{trial}' if j == 0 else '{block}'}$"
            angle_bw_tp_names.append(name)
            c = np.dot(m,r)
            c = c / np.linalg.norm(m) / np.linalg.norm(r)
            angle_bw_template_and_readout.append(c)
    fig,ax = plt.subplots(1,1)
    ax.bar(angle_bw_tp_names,angle_bw_template_and_readout)
    ax.set_xlabel("Projection")
    ax.set_ylabel("Value")
    hook_input['tensorboard_writer'].add_figure(
        tag='readout_proj_onto_template',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_pc_proj_onto_template(hook_input):
    angle_bw_template_and_pc = hook_input['angle_bw_template_and_pc']
    angle_bw_tp_names = hook_input['angle_bw_tp_names']
    fig,ax = plt.subplots(1,1)
    ax.bar(angle_bw_tp_names,angle_bw_template_and_pc)
    ax.set_xlabel("Projection")
    ax.set_ylabel("Value")
    hook_input['tensorboard_writer'].add_figure(
        tag='pc_proj_onto_template',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_model_recurrent_weight_distributions(hook_input):
    # TODO: this is duplicated code from hook_plot_model_effective_circuit()

    # hidden states shape: (num rnn steps, num layers, hidden dimension)
    hidden_states = hook_input['hidden_states']
    hidden_size = hidden_states.shape[2]

    # reshape to (num trials, num layers * hidden dimension)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    trial_side = np.expand_dims(hook_input['session_data'].trial_side.values, 1)
    trial_side_orthogonal = np.expand_dims(hook_input['session_data'].trial_side_orthogonal.values, 1)
    block_side = np.expand_dims(hook_input['session_data'].block_side.values, 1)
    feedback = np.expand_dims(hook_input['session_data'].reward.values, 1)

    # construct correlation matrix
    hidden_states_and_task_variables = np.hstack((
        hidden_states,
        trial_side,
        trial_side_orthogonal,
        block_side,
        feedback))
    hidden_states_and_task_variables_correlations = np.corrcoef(hidden_states_and_task_variables.T)
    # due to machine error, correlation matrix isn't exactly symmetric (typically has e-16 errors)
    # so make it symmetric
    hidden_states_and_task_variables_correlations = (hidden_states_and_task_variables_correlations +
                                                     hidden_states_and_task_variables_correlations.T) / 2
    hidden_state_self_correlations = hidden_states_and_task_variables_correlations[:hidden_size, :hidden_size]
    hidden_state_task_correlations = hidden_states_and_task_variables_correlations[:hidden_size, hidden_size:]

    # compute pairwise distances
    pdist = spc.distance.pdist(hidden_state_self_correlations)
    linkage = spc.linkage(pdist, method='complete')
    labels = spc.fcluster(linkage, 0.5 * np.max(pdist), 'distance')
    indices = np.argsort(labels)

    recurrent_matrix = hook_input['model'].core.weight_hh_l0.data.numpy()
    dimension_ratio = recurrent_matrix.shape[0] / recurrent_matrix.shape[1]
    # # RNN weight will have shape (hidden size, hidden size)
    if dimension_ratio == 1:
        recurrent_matrix = recurrent_matrix[indices][:, indices]
    else:
        raise NotImplementedError

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            quadrant_indices = np.logical_and(
                (labels == row + 1)[:, np.newaxis],
                (labels == col + 1)[np.newaxis, :])
            recurrent_quadrant_weights = recurrent_matrix[quadrant_indices]
            sns.kdeplot(
                recurrent_quadrant_weights,
                color='r' if col == row else 'b',
                ax=ax)
            ax.axvline(recurrent_quadrant_weights.mean(), color='k')
            ax.set_yticklabels([])

    hook_input['tensorboard_writer'].add_figure(
        tag='model_recurrent_weight_distributions',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_weights_vs_correlations(hook_input):
    hidden_states = hook_input['hidden_states']
    hidden_size = hidden_states.shape[2]

    # reshape to (num trials, num layers * hidden dimension)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    trial_side = np.expand_dims(hook_input['session_data'].trial_side.values, 1)
    trial_side_orthogonal = np.expand_dims(hook_input['session_data'].trial_side_orthogonal.values, 1)
    block_side = np.expand_dims(hook_input['session_data'].block_side.values, 1)
    feedback = np.expand_dims(hook_input['session_data'].reward.values, 1)

    # construct correlation matrix
    hidden_states_and_task_variables = np.hstack((
        hidden_states,
        trial_side,
        trial_side_orthogonal,
        block_side,
        feedback))
    hidden_states_and_task_variables_correlations = np.corrcoef(hidden_states_and_task_variables.T)
    # due to machine error, correlation matrix isn't exactly symmetric (typically has e-16 errors)
    # so make it symmetric
    hidden_states_and_task_variables_correlations = (hidden_states_and_task_variables_correlations +
                                                     hidden_states_and_task_variables_correlations.T) / 2
    hidden_state_self_correlations = hidden_states_and_task_variables_correlations[:hidden_size, :hidden_size]
    hidden_state_task_correlations = hidden_states_and_task_variables_correlations[:hidden_size, hidden_size:]
    recurrent_matrix = hook_input['model'].core.weight_hh_l0.data.numpy()
    hidden_corrs = hidden_state_self_correlations.reshape(-1)
    weights = recurrent_matrix.reshape(-1)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    corr = np.corrcoef(hidden_corrs,weights)[0][1]
    for i in range(2):
        if i == 0:
            ax[i].scatter(weights,hidden_corrs)
            ax[i].legend([corr])
        else:
            nhidden = hidden_state_self_correlations.shape[0]
            ax[i].scatter(recurrent_matrix[:nhidden//2,:nhidden//2],
                           hidden_state_self_correlations[:nhidden//2,:nhidden//2],label="M1")
            ax[i].scatter(recurrent_matrix[nhidden//2:,nhidden//2:],
                           hidden_state_self_correlations[nhidden//2:,nhidden//2:],label="M2")
            ax[i].scatter(recurrent_matrix[nhidden//2:,:nhidden//2],
                           hidden_state_self_correlations[nhidden//2:,:nhidden//2],label="1->2")
            ax[i].scatter(recurrent_matrix[:nhidden//2,nhidden//2:],
                           hidden_state_self_correlations[:nhidden//2,nhidden//2:],label="2->1")
            ax[i].legend()

        ax[i].set_xlabel("Recurrent Weights")
        ax[i].set_ylabel("Hidden Unit Correlations")
        

    hook_input['tensorboard_writer'].add_figure(
        tag='recurrent_weight_hiden_correlation',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_hidden_unit_fraction_var_explained(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(1, 1 + len(hook_input['frac_variance_explained'])),
            hook_input['frac_variance_explained'],
            'o',
            alpha=0.8,
            ms=3,
            color=side_color_map['neutral'])
    # fig.suptitle('Fraction of Cumulative Variance Explained by Dimension')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Fraction of Cumulative Variance Explained')
    ax.set_ylim([-0.05, 1.05])
    hook_input['tensorboard_writer'].add_figure(
        tag='var_explained_by_dimension',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_weights_and_gradients(hook_input):
    weights = dict(
        input=hook_input['model'].core.weight_ih_l0.data.numpy(),
        recurrent=hook_input['model'].core.weight_hh_l0.data.numpy(),
        readout=hook_input['model'].readout.weight.data.numpy().T  # transpose for better plotting
    )

    if hook_input['tag_prefix'] != 'analyze/':
        weight_gradients = dict(
            input=hook_input['model'].core.weight_ih_l0.grad.numpy(),
            recurrent=hook_input['model'].core.weight_hh_l0.grad.numpy(),
            readout=hook_input['model'].readout.weight.grad.numpy().T  # transpose for better plotting
        )

    fig, axes = plt.subplots(nrows=2,
                             ncols=4,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
                             figsize=(9, 6))
    recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    # fig.suptitle(f'Model Weights (Recurrent Mask: {recurrent_mask_str}) and Gradients')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for i, weight_str in enumerate(weights):
        axes[0, i].set_title(f'{weight_str} Matrix')
        hm = sns.heatmap(
            weights[weight_str],
            cmap='RdBu_r',
            square=True,
            ax=axes[0, i],
            center=0,
            vmin=-0.5,
            vmax=0.5,
            cbar_ax=axes[0, -1],
            cbar_kws={'label': 'Weight Strength'})
        axes[0, i].set_aspect("equal")  # ensures little squares don't become rectangles

        if hook_input['tag_prefix'] != 'analyze/':
            axes[1, i].set_title(f'{weight_str} Gradient')
            hm = sns.heatmap(
                weight_gradients[weight_str],
                cmap='RdBu_r',
                square=True,
                ax=axes[1, i],
                center=0,
                vmin=-0.5,
                vmax=0.5,
                cbar_ax=axes[1, -1],
                cbar_kws={'label': 'Weight Strength'})
            axes[1, i].set_aspect("equal")  # ensures little squares don't become rectangles

    hook_input['tensorboard_writer'].add_figure(
        tag='model_weights_and_gradients',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_weights_community_detection(hook_input):
    utils.analysis.compute_model_weights_community_detection(hook_input['model'])

    print(10)


def hook_plot_mice_reaction_time_by_strength_correct_concordant(hook_input):
    # plot chronometric curves, conditioned on correct, concordant
    rt_by_correct_concordant_signed_contrast = hook_input['mice_behavior_df'].groupby([
        'trial_correct', 'concordant_trial', 'signed_contrast']).agg({
        'reaction_time': ['mean', 'sem']})['reaction_time']

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Signed Contrast')
    ax.set_ylabel('Reaction Time')
    # fig.suptitle('Mouse Reaction Time by Signed Contrast')

    for correct_trial, concordant_trial in product([0., 1.],
                                                   [True, False]):

        if correct_trial == 1.:
            label = 'Correct'
            color = 'tab:green'
        else:
            label = 'Incorrect'
            color = 'tab:red'

        if concordant_trial:
            label += ', Concordant Trials'
            style = '-o'
        else:
            label += ', Discordant Trials'
            style = '--o'

        mean_reaction_time = rt_by_correct_concordant_signed_contrast.loc[
            correct_trial].loc[concordant_trial]['mean']
        sem_reaction_time = rt_by_correct_concordant_signed_contrast.loc[
            correct_trial].loc[concordant_trial]['sem']
        signed_contrasts = mean_reaction_time.index
        ax.plot(
            signed_contrasts,
            mean_reaction_time,
            style,
            markersize=4,
            color=color,
            label=label)
        ax.fill_between(
            x=signed_contrasts,
            y1=mean_reaction_time - sem_reaction_time,
            y2=mean_reaction_time + sem_reaction_time,
            alpha=0.3,
            linewidth=0,
            color=color)

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='mice_reaction_time_by_strength_correct_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_mice_prob_correct_by_strength_trial_block(hook_input):
    # plot chronometric curves, conditioned on correct, concordant
    prob_correct_by_strength_trial_block = hook_input['mice_behavior_df'].groupby([
        'block_side', 'stimulus_side', 'signed_contrast']).agg({
        'trial_correct': ['mean', 'sem']})['trial_correct']

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Signed Contrast')
    ax.set_ylabel('Correct Action Trial / Total Trials')

    for block_side, trial_side in product([-1., 1.],
                                          [-1., 1.]):
        block_side_str = side_string_map[block_side]
        trial_side_str = side_string_map[trial_side]

        mean_correct_by_strength = prob_correct_by_strength_trial_block.loc[
            block_side].loc[trial_side]['mean']
        sem_correct_by_strength = prob_correct_by_strength_trial_block.loc[
            block_side].loc[trial_side]['sem']
        signed_contrasts = mean_correct_by_strength.index
        ax.plot(
            signed_contrasts,
            mean_correct_by_strength,
            '-o' if block_side == trial_side else '--o',
            markersize=4,
            label=f'{block_side_str} Block, {trial_side_str} Trial',
            color=side_color_map[block_side])
        ax.fill_between(
            x=signed_contrasts,
            y1=mean_correct_by_strength - sem_correct_by_strength,
            y2=mean_correct_by_strength + sem_correct_by_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='mice_prob_correct_by_strength_trial_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


# def hook_plot_state_space_3d(hook_input):
#
#     import plotly.graph_objects as go
#
#     trial_side_binary = (1 + hook_input['session_data'].trial_side) / 2
#     block_side_binary = (hook_input['session_data'].classifier_block_side > 0).astype(np.float)
#     color = block_side_binary * 2 + trial_side_binary
#     fig = go.Figure(data=[
#         go.Scatter3d(
#             x=hook_input['pca_hidden_states'][:, 0],
#             y=hook_input['pca_hidden_states'][:, 1],
#             z=hook_input['pca_hidden_states'][:, 2],
#             mode='markers',
#             marker=dict(
#                 size=1,
#                 color=color,  # set color to an array/list of desired values
#                 colorscale='Viridis',  # choose a colorscale
#                 opacity=0.8
#             )
#         )])
#     fig.write_html(file='state_space.html')


# def hook_plot_state_space_fixed_points(hook_input):
#     displacement_norm_cutoff = 0.5
#
#     # TODO: deduplicate with vector fields plot
#     fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']
#
#     num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
#     fig, axes = plt.subplots(nrows=num_stimuli,
#                              ncols=3,
#                              gridspec_kw={"width_ratios": [1, 1, 0.05]},
#                              figsize=(12, 8),
#                              sharex=True,
#                              sharey=True)
#
#     fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
#
#     for c, (side, fixed_points_by_stimuli_dict) in \
#             enumerate(fixed_points_by_side_by_stimuli.items()):
#
#         for r, (stimulus, fixed_points_dict) in enumerate(fixed_points_by_stimuli_dict.items()):
#
#             num_grad_steps = fixed_points_dict['num_grad_steps']
#
#             ax = axes[r, c]
#             ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
#             ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
#             if r == 0:
#                 ax.set_title(f'Block Side: {side_string_map[side]}')
#             elif r == num_stimuli - 1:
#                 ax.set_xlabel('PC #1')
#
#             if c == 0:
#                 ax.set_ylabel(f'{stimulus}')
#             # else:
#             #     ax.set_yticklabels([])
#
#             displacement_norms = fixed_points_dict['normalized_displacement_vector_norm']
#             smallest_displacement_norm_indices = displacement_norms.argsort()
#             smallest_displacement_norm_indices = smallest_displacement_norm_indices[
#                 displacement_norms[smallest_displacement_norm_indices] < displacement_norm_cutoff]
#
#             try:
#
#                 x = fixed_points_dict['pca_final_sampled_hidden_states'][smallest_displacement_norm_indices, 0]
#                 y = fixed_points_dict['pca_final_sampled_hidden_states'][smallest_displacement_norm_indices, 1]
#                 colors = fixed_points_dict['normalized_displacement_vector_norm'][smallest_displacement_norm_indices]
#
#                 sc = ax.scatter(
#                     x,
#                     y,
#                     c=colors,
#                     vmin=0,
#                     vmax=displacement_norm_cutoff,
#                     s=1,
#                     cmap='gist_rainbow')
#
#                 # emphasize the fixed point with smallest gradient
#                 sc = ax.scatter(
#                     [x[0]],
#                     [y[0]],
#                     c=[colors[0]],
#                     edgecolors='k',
#                     vmin=0,
#                     vmax=displacement_norm_cutoff,
#                     cmap='gist_rainbow'
#                 )
#
#             except IndexError:
#                 print('No fixed points below displacement norm cutoff')
#
#             add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)
#
#     fig.suptitle(f'Fixed Points (Num Grad Steps = {num_grad_steps})')
#
#     # merge the rightmost column for the colorbar
#     gs = axes[0, 2].get_gridspec()
#     for ax in axes[:, -1]:
#         ax.remove()
#     ax_colorbar = fig.add_subplot(gs[:, -1])
#     color_bar = fig.colorbar(sc, cax=ax_colorbar)
#     color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')
#     hook_input['tensorboard_writer'].add_figure(
#         tag='hook_plot_psytrack_fit',
#         figure=fig,
#         global_step=hook_input['grad_step'],
#         close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_behav_prob_correct_by_strength_concordant(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Signed Stimulus Contrast')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylabel('# Correct / # Trials')
    ax.set_xlabel('Signed Stimulus Contrast')

    # keep only last dts in trials
    session_data = hook_input['radd_session_data']
    trial_end_data = session_data[session_data.trial_end == 1.]

    for concordant, concordant_data in trial_end_data.groupby(['concordant_trial']):
        avg_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].mean()

        sem_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].sem()

        ax.plot(
            avg_correct_action_prob_by_stim_strength.index,
            avg_correct_action_prob_by_stim_strength,
            '-o' if concordant else '--o',
            # solid lines for consistent block side, trial side; dotted otherwise
            label='RADD RNN Concordant' if concordant else 'RADD RNN Discordant',
            color=side_color_map['correct'])
        ax.fill_between(
            x=avg_correct_action_prob_by_stim_strength.index,
            y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
            y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['correct'])

    # keep only last dts in trials
    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_trial_end_data = actor_session_data[actor_session_data.trial_end == 1.]

    for concordant, concordant_data in actor_trial_end_data.groupby(['concordant_trial']):
        avg_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].mean()

        sem_correct_action_prob_by_stim_strength = concordant_data.groupby(
            'signed_trial_strength')['correct_action_taken'].sem()

        ax.plot(
            avg_correct_action_prob_by_stim_strength.index,
            avg_correct_action_prob_by_stim_strength,
            '-o' if concordant else '--o',
            # solid lines for consistent block side, trial side; dotted otherwise
            label=f'Bayesian Actor Concordant' if concordant else 'Bayesian Actor Discordant',
            color=side_color_map['ideal_correct'])
        ax.fill_between(
            x=avg_correct_action_prob_by_stim_strength.index,
            y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
            y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal_correct'])

    # necessary to delete redundant legend groups
    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels, markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='radd_behav_prob_correct_by_strength_concordant',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_behav_prob_correct_by_trial_within_block(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    correct_and_block_posterior_by_trial_index = trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
        'bayesian_observer_correct_action_taken': ['mean', 'sem', 'size'],
    })

    # drop NaN SEM rows
    rnn_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'correct_action_taken']
    observer_correct_by_trial_index = correct_and_block_posterior_by_trial_index[
        'bayesian_observer_correct_action_taken']
    rnn_correct_by_trial_index = rnn_correct_by_trial_index[
        rnn_correct_by_trial_index['size'] > 1.]
    observer_correct_by_trial_index = observer_correct_by_trial_index[
        observer_correct_by_trial_index['size'] != 0.]

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Correct Action Trials / Total Trials by Trial Within Block (All Contrast Trials)')
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('# Correct / # Trials')
    ax.set_ylim([0.5, 1.05])
    ax.set_xlim([0., 101.])
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        1 + rnn_correct_by_trial_index.index.values,
        rnn_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['correct'],
        label='RADD RNN')

    ax.fill_between(
        x=1 + rnn_correct_by_trial_index.index.values,
        y1=rnn_correct_by_trial_index['mean']
           - rnn_correct_by_trial_index['sem'],
        y2=rnn_correct_by_trial_index['mean']
           + rnn_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])

    ax.plot(
        1 + observer_correct_by_trial_index.index.values,
        observer_correct_by_trial_index['mean'],
        '--o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Observer')

    ax.fill_between(
        x=1 + observer_correct_by_trial_index.index.values,
        y1=observer_correct_by_trial_index['mean']
           - observer_correct_by_trial_index['sem'],
        y2=observer_correct_by_trial_index['mean']
           + observer_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    actor_session_data = hook_input['bayesian_actor_session_data']
    actor_trial_end_data = actor_session_data[actor_session_data.trial_end == 1.].copy()

    # plot trial number within block (x) vs probability of correct response (y)
    actor_correct_by_trial_index = actor_trial_end_data.groupby(['trial_index']).agg({
        'correct_action_taken': ['mean', 'sem', 'size'],
    })['correct_action_taken']

    # drop NaN/0. sem trials
    actor_correct_by_trial_index = actor_correct_by_trial_index[
        actor_correct_by_trial_index['size'] > 1.]

    ax.plot(
        1 + actor_correct_by_trial_index.index.values,
        actor_correct_by_trial_index['mean'],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['ideal_correct'],
        label='Bayesian Actor')

    ax.fill_between(
        x=1 + actor_correct_by_trial_index.index.values,
        y1=actor_correct_by_trial_index['mean']
           - actor_correct_by_trial_index['sem'],
        y2=actor_correct_by_trial_index['mean']
           + actor_correct_by_trial_index['sem'],
        alpha=0.3,
        linewidth=0,
        color=side_color_map['ideal'])

    ax.legend(markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='radd_behav_prob_correct_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


# def hook_plot_radd_state_space_effect_of_obs_along_task_aligned_vectors(hook_input):
#     radd_session_data = hook_input['radd_session_data']

#     diff_obs = radd_session_data['right_stimulus'] - radd_session_data['left_stimulus']
#     radd_hidden_states = np.stack(radd_session_data.hidden_state.values, axis=0)
#     radd_hidden_states = radd_hidden_states.reshape(radd_hidden_states.shape[0], -1)

#     radd_deltas = np.diff(
#         radd_hidden_states,
#         n=1,
#         axis=0)

#     # here, we need to do two things. first, exclude blank dts.
#     # second, make sure we align observations and state deltas correctly.
#     # the state at an index is the state AFTER the observation.
#     # consequently, we need to shift deltas back by one

#     during_trials_indices = radd_session_data.index[
#         (radd_session_data.left_stimulus != 0) & (radd_session_data.right_stimulus != 0)]
#     diff_obs = diff_obs[during_trials_indices]
#     radd_deltas = radd_deltas[during_trials_indices - 1]

#     num_cols = 2
#     fig, axes = plt.subplots(nrows=1,
#                              ncols=num_cols,
#                              sharex=True,
#                              sharey=True,
#                              gridspec_kw={"width_ratios": [1] * num_cols},
#                              figsize=(8, 3))

#     for col in range(num_cols):
#         ax = axes[col]
#         if col == 0:
#             ax.set_ylabel('Movement Along Stimulus Readout')
#         else:
#             ax.set_ylabel('Movement Along Block Readout')

#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
#             diff_obs,
#             radd_deltas[:, col])

#         p_eqn = 'p<1e-5' if p_value < 1e-5 else f'p={np.round(p_value, 5)}'
#         line_eqn = f'y={np.round(slope, 2)}x+{np.round(intercept, 2)} ' \
#                    f'({p_eqn}, r={np.round(r_value, 2)})'

#         # seaborn's lead dev refuses to enable displaying the best fit parameters
#         ensure_centered_at_zero = DivergingNorm(vmin=-4., vcenter=0., vmax=4.)
#         ax = sns.regplot(
#             x=diff_obs,
#             y=radd_deltas[:, col],
#             ax=ax,
#             # color=side_color_map['right'],
#             ci=99,
#             scatter_kws={'s': 1,  # marker size
#                          'color': orange_blue_cmap(ensure_centered_at_zero(radd_deltas[:, col]))
#                          },
#             line_kws={'color': side_color_map['ideal'],
#                       'label': line_eqn}
#         )
#         # this needs to go down here for some reason
#         ax.set_xlabel(r'$d_{n, t} = o_{n,t}^R - o_{n,t}^L$')

#         ax.legend()
#     hook_input['tensorboard_writer'].add_figure(
#         tag='radd_state_space_effect_of_obs_along_task_aligned_vectors',
#         figure=fig,
#         global_step=hook_input['grad_step'],
#         close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_state_space_projection_on_right_block_vector_by_trial_within_block(hook_input):
    radd_session_data = hook_input['radd_session_data']

    trial_end_data = radd_session_data[radd_session_data['trial_end'] == 1.].copy()
    # assigning to slice to not overwite posterior probability
    # trial_end_data['bayesian_observer_block_posterior_right'] = \
    #     2. * trial_end_data['bayesian_observer_block_posterior_right'] - 1.
    # trial_end_data['bayesian_observer_block_posterior_right'] *= hook_input['block_scaling_parameter']

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Normalized Projection Along Right Block Vector by Trial within Block')
    ax.set_xlabel('Trial within Block')
    ax.set_ylabel('Magnitude Along Block Readout')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for block_side, block_side_trials_data in trial_end_data.groupby(['block_side']):
        temp_df = block_side_trials_data[[
            'block_side', 'trial_index',
            # 'bayesian_observer_block_posterior_right',
            'magn_along_block_vector']].copy()

        temp_groupby = temp_df.groupby(['trial_index']).agg({
            'magn_along_block_vector': ['mean', 'sem', 'size'],
            # 'bayesian_observer_block_posterior_right': ['mean', 'sem', 'size'],
        })
        magn_along_block_vector_by_trial_index = temp_groupby[
            'magn_along_block_vector']
        # observer_block_posterior_right_by_trial_index = temp_groupby[
        #     'bayesian_observer_block_posterior_right']

        # drop NaN/0 SEM
        magn_along_block_vector_by_trial_index = magn_along_block_vector_by_trial_index[
            magn_along_block_vector_by_trial_index['size'] > 1.]
        # observer_block_posterior_right_by_trial_index = observer_block_posterior_right_by_trial_index[
        #     observer_block_posterior_right_by_trial_index['size'] > 1.]

        ax.plot(
            magn_along_block_vector_by_trial_index.index,
            magn_along_block_vector_by_trial_index['mean'],
            label=f'RADD RNN {side_string_map[block_side]} Block',
            color=side_color_map[block_side]
        )

        ax.fill_between(
            x=magn_along_block_vector_by_trial_index.index,
            y1=magn_along_block_vector_by_trial_index['mean']
               - magn_along_block_vector_by_trial_index['sem'],
            y2=magn_along_block_vector_by_trial_index['mean']
               + magn_along_block_vector_by_trial_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])
        #
        # ax.plot(
        #     observer_block_posterior_right_by_trial_index.index,
        #     observer_block_posterior_right_by_trial_index['mean'],
        #     markersize=1,
        #     linewidth=1,
        #     label=f'RNN {side_string_map[block_side]} Block',
        #     color=side_color_map[block_side])
        #
        # ax.fill_between(
        #     x=observer_block_posterior_right_by_trial_index.index,
        #     y1=observer_block_posterior_right_by_trial_index['mean']
        #        - observer_block_posterior_right_by_trial_index['sem'],
        #     y2=observer_block_posterior_right_by_trial_index['mean']
        #        + observer_block_posterior_right_by_trial_index['sem'],
        #     alpha=0.3,
        #     linewidth=0,
        #     color=side_color_map['ideal'],)

    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels)
    hook_input['tensorboard_writer'].add_figure(
        tag='radd_state_space_projection_on_right_block_vector_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_state_space_distance_decoherence(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    # ax.set_title(r'Distance (L2) by Elapsed Number RNN Steps ($\Delta$)')
    jitter, total_jitter = 0.1, 0.
    ax.set_xlabel(r'Elapsed Time ($\Delta$)  (Jitter={})'.format(jitter))
    ax.set_ylabel('Distance (L2)')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    for i, (trajectory_name, trajectory_error) in enumerate(hook_input['error_accumulation_df'].groupby(['name'])):
        if trajectory_name == 'hidden_states':
            label = r'$||h_{t+\Delta} - h_t||_2$'
        elif trajectory_name == 'task_aligned_hidden_states':
            label = r'$||z_{t+\Delta} - z_n||_2$'
        elif trajectory_name == 'radd_states':
            label = r'$||\hat{z}_{t+\Delta} - \hat{z}_t||_2$'
        elif trajectory_name == 'inv_pca_radd_states':
            label = r'$||P^{-1} h_{n+\Delta}^{\prime} - P^{-1} h_n^{\prime}||_2$'
            # TODO: decide whether to skip this
            continue
        elif trajectory_name == 'control':
            label = 'Random Gaussian Weights'
            # TODO: decide whether to skip this
            continue
        else:
            raise ValueError('Invalid trajectory name')
        ax.plot(
            total_jitter + trajectory_error.delta,
            trajectory_error.norm_mean,
            label=label)
        ax.fill_between(
            x=total_jitter + trajectory_error.delta,
            y1=trajectory_error.norm_mean - trajectory_error.norm_var,
            y2=trajectory_error.norm_mean + trajectory_error.norm_var,
            alpha=0.1,
            linewidth=0)
        total_jitter += jitter
    ax.set_yscale('log')
    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='radd_state_space_distance_decoherence',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_state_space_trajectories_within_block(hook_input):
    session_data = hook_input['session_data']

    # take only last dt within a trial
    # exclude blocks that are first in the session
    session_data = session_data[session_data.trial_end == 1]

    # determine correct scaling of projected RNN & reduced dim model
    X = hook_input['task_aligned_hidden_states'][:, 0, np.newaxis]
    Y = hook_input['radd_states'][:, 0, np.newaxis]
    stimulus_scaling_parameter = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # shape = (1, 1)
    stimulus_scaling_parameter = stimulus_scaling_parameter[0, 0]

    X = hook_input['task_aligned_hidden_states'][:, 1, np.newaxis]
    Y = hook_input['radd_states'][:, 1, np.newaxis]
    block_scaling_parameter = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # shape = (1, 1)
    block_scaling_parameter = block_scaling_parameter[0, 0]

    num_rows, num_cols = 1, 2

    # select only session 0 and the last (num_rows * num_cols) blocks
    subset_session_data = session_data[(session_data.session_index == 0) &
                                       (session_data.block_index > max(session_data.block_index) - num_cols * num_rows)]
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(8, 3))
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # create possible color range
    max_block_duration = max(subset_session_data.groupby(['session_index', 'block_index']).size())

    for i, (block_idx, session_data_by_block) in enumerate(subset_session_data.groupby('block_index')):

        row, col = int(i / num_cols), int(i % num_cols)
        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        block_side = side_string_map[session_data_by_block.block_side.unique()[0]]
        ax.set_title(f'{block_side} Block')
        # ax.set_title(f'Block {1 + int(block_idx)}\n{block_side} Block')
        ax.set_title(f'{block_side} Block')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        if row == (num_rows - 1):
            ax.set_xlabel('Magnitude Along Stimulus Vector')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('Magnitude Along Block Vector')
        else:
            ax.set_yticklabels([])

        block_indices = session_data_by_block.index.values
        task_aligned_hidden_states_block = hook_input['task_aligned_hidden_states'][block_indices]
        radd_states_block = hook_input['radd_states'][block_indices]
        # stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
        # segment_text = np.where(session_data_by_block['reward'] > 0.9, 'C', 'I')
        for i in range(len(block_indices) - 1):
            ax.plot(
                stimulus_scaling_parameter * task_aligned_hidden_states_block[i:i + 2, 0],
                block_scaling_parameter * task_aligned_hidden_states_block[i:i + 2, 1],
                '--o',
                label='Projected RNN',
                color=plt.cm.jet(i / max_block_duration),
                markersize=2,
                zorder=2)

        for i in range(len(block_indices) - 1):
            ax.plot(
                radd_states_block[i:i + 2, 0],
                radd_states_block[i:i + 2, 1],
                '-o',
                label='Reduced Dim RNN',
                color=plt.cm.jet(i / max_block_duration),
                markersize=2,
                zorder=2)
        handles, labels = delete_redundant_legend_groups(ax=ax)
        ax.legend(handles, labels, markerscale=.1)
    hook_input['tensorboard_writer'].add_figure(
        tag='radd_state_space_trajectories_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_state_space_trajectories_within_trial(hook_input):
    session_data = hook_input['session_data']

    num_rows, num_cols = 3, 3
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(4, 4))
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    # plt.suptitle('State Space PCA Trajectories ({} degrees between readout vectors)'.format(
    #     hook_input['degrees_btwn_pca_trial_block_vectors']))

    # select only environment 1, first 12 trials
    subset_session_data = session_data[(session_data['session_index'] == 0) &
                                       (session_data['block_index'] == 2) &
                                       (session_data['trial_index'] < num_cols * num_rows)]

    # create possible color range
    max_trial_duration = max(subset_session_data.groupby(['session_index', 'block_index', 'trial_index']).size())

    for trial_index, session_data_by_trial in subset_session_data.groupby('trial_index'):

        if trial_index >= num_cols * num_rows:
            break

        row, col = int(trial_index / num_cols), int(trial_index % num_cols)
        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        trial_side = side_string_map[session_data_by_trial.trial_side.unique()[0]]
        title = f'{trial_side} Trial, '
        title += 'Correct Action' if bool(session_data_by_trial.tail(1).iloc[0].correct_action_taken) \
            else 'Incorrect Action'
        # ax.set_title(title)
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        if col == ((num_cols - 1) // 2):
            ax.set_xlabel('Magnitude Along Right Trial Readout')
        elif col != 0:
            ax.set_xticklabels([])

        if row == ((num_rows - 1) // 2):
            ax.set_ylabel('Magnitude Along Right Block Readout')
        elif row != (num_rows - 1):
            ax.set_yticklabels([])

        trial_indices = session_data_by_trial.index.values
        proj_hidden_states_block = hook_input['task_aligned_hidden_states'][trial_indices]

        # plot the first dt in the trial
        ax.plot(
            proj_hidden_states_block[0, 0],
            proj_hidden_states_block[0, 1],
            'o-',
            markersize=1,
            color=plt.cm.jet(0 / max_trial_duration),
            zorder=2)

        # plot the rest of the trial's dts
        for i in range(1, len(trial_indices)):
            ax.plot(
                proj_hidden_states_block[i - 1:i + 1, 0],
                proj_hidden_states_block[i - 1:i + 1, 1],
                'o-',
                markersize=1,
                color=plt.cm.jet(i / max_trial_duration))

    hook_input['tensorboard_writer'].add_figure(
        tag='radd_state_space_trajectories_within_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_radd_state_space_vector_fields_ideal(hook_input):
    # TODO: deduplicate with hook_plot_state_space_vector_fields_ideal
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_cols + 1,  # +1 for colorbar
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = np.max(hook_input['radd_fixed_point_df']['displacement_norm'])

    initial_pca_sampled_states = np.stack(
        hook_input['radd_fixed_point_df']['initial_pca_sampled_state'].values.tolist())

    # drop all except left stimulus, no stimulus, right stimulus
    fixed_point_df = hook_input['radd_fixed_point_df']
    indices_to_keep = fixed_point_df['feedback'] == -0.05
    fixed_point_df = fixed_point_df[indices_to_keep]
    initial_pca_sampled_states = initial_pca_sampled_states[indices_to_keep]

    # calculate ranges of plot
    xmin = np.min(initial_pca_sampled_states[:, 0]) - 0.1
    xmax = np.max(initial_pca_sampled_states[:, 0]) + 0.1
    ymin = np.min(initial_pca_sampled_states[:, 1]) - 0.1
    ymax = np.max(initial_pca_sampled_states[:, 1]) + 0.1

    for i, ((lstim, rstim, fdbk), fixed_point_subset) in enumerate(
            fixed_point_df.groupby(['left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        col = int(i % num_cols)

        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        if i == 0:
            title = 'Left Stimulus'
        elif i == 1:
            title = 'No Stimulus'
        elif i == 2:
            title = 'Right Stimulus'
        else:
            raise ValueError
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Unit #1')

        if col == 0:
            ax.set_ylabel('Unit #2')
        else:
            ax.set_yticklabels([])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        initial_pca_sampled_states = np.stack(
            fixed_point_subset['initial_pca_sampled_state'].values.tolist())
        displacement_pca = np.stack(fixed_point_subset['displacement_pca'].values.tolist())

        qvr = ax.quiver(
            initial_pca_sampled_states[:, 0],
            initial_pca_sampled_states[:, 1],
            displacement_pca[:, 0],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_pca[:, 1],  # / np.linalg.norm(displacement_pca, axis=1),
            fixed_point_subset['displacement_norm'],  # color
            angles='xy',  # this and the next two ensures vector scales match data scales
            scale_units='xy',
            scale=1,
            alpha=0.6,
            clim=(color_min, color_max),
            headwidth=7,
            cmap='gnuplot2')

    # merge the rightmost column for the colorbar
    color_bar = fig.colorbar(qvr, cax=axes[-1])
    color_bar.set_label(r'state velocity (a.u.)', size=9)
    color_bar.set_alpha(1)
    color_bar.draw_all()

    hook_input['tensorboard_writer'].add_figure(
        tag='radd_state_space_vector_fields_ideal',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


# def hook_plot_state_space_effect_of_obs_along_task_aligned_vectors(hook_input):
#     session_data = hook_input['session_data']

#     diff_obs = session_data['right_stimulus'] - session_data['left_stimulus']
#     task_aligned_deltas = np.diff(
#         hook_input['task_aligned_hidden_states'],
#         n=1,
#         axis=0)

#     # here, we need to do two things. first, exclude blank dts.
#     # second, make sure we align observations and state deltas correctly.
#     # the state at an index is the state AFTER the observation.
#     # consequently, we need to shift deltas back by one

#     during_trials_indices = session_data.index[
#         (session_data.left_stimulus != 0) & (session_data.right_stimulus != 0)]
#     diff_obs = diff_obs[during_trials_indices]
#     task_aligned_deltas = task_aligned_deltas[during_trials_indices - 1]

#     num_cols = 2
#     fig, axes = plt.subplots(nrows=1,
#                              ncols=num_cols,
#                              gridspec_kw={"width_ratios": [1] * num_cols},
#                              figsize=(8, 3))

#     for col in range(num_cols):
#         ax = axes[col]
#         ax.axis('equal')  # set yscale to match xscale
#         if col == 0:
#             ax.set_ylabel('Movement Along Stimulus Readout')
#         else:
#             ax.set_ylabel('Movement Along Block Readout')

#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
#             diff_obs,
#             task_aligned_deltas[:, col])

#         p_eqn = 'p<1e-5' if p_value < 1e-5 else f'p={np.round(p_value, 5)}'
#         line_eqn = f'y={np.round(slope, 2)}x+{np.round(intercept, 2)} ' \
#                    f'({p_eqn}, r={np.round(r_value, 2)})'

#         # seaborn's lead dev refuses to enable displaying the best fit parameters
#         ensure_centered_at_zero = DivergingNorm(vmin=-4., vcenter=0., vmax=4.)
#         ax = sns.regplot(
#             x=diff_obs,
#             y=task_aligned_deltas[:, col],
#             ax=ax,
#             # color=side_color_map['right'],
#             ci=99,
#             scatter_kws={'s': 1,  # marker size
#                          'color': orange_blue_cmap(ensure_centered_at_zero(task_aligned_deltas[:, col]))
#                          },
#             line_kws={'color': side_color_map['ideal'],
#                       'label': line_eqn}
#         )
#         # this needs to go down here for some reason
#         ax.set_xlabel(r'$d_{n, t} = o_{n,t}^R - o_{n,t}^L$')

#         ax.legend()
#     hook_input['tensorboard_writer'].add_figure(
#         tag='state_space_effect_of_obs_along_task_aligned_vectors',
#         figure=fig,
#         global_step=hook_input['grad_step'],
#         close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_effect_of_feedback_along_task_aligned_vectors(hook_input):
    session_data = hook_input['session_data']
    reward_indices = session_data.index[
        (session_data.reward == 1.) | (session_data.reward == -1)]
    rewards = session_data.loc[reward_indices, 'reward']
    task_aligned_deltas = np.diff(
        hook_input['task_aligned_hidden_states'],
        n=1,
        axis=0)
    task_aligned_deltas = task_aligned_deltas[reward_indices - 1]

    num_cols = 2
    fig, axes = plt.subplots(nrows=1,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(8, 3))

    for col in range(num_cols):
        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        if col == 0:
            ax.set_ylabel('Movement Along Stimulus Readout')
        else:
            ax.set_ylabel('Movement Along Block Readout')

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            rewards,
            task_aligned_deltas[:, col])

        p_eqn = 'p<1e-5' if p_value < 1e-5 else f'p={np.round(p_value, 5)}'
        line_eqn = f'y={np.round(slope, 2)}x+{np.round(intercept, 2)} ' \
                   f'({p_eqn}, r={np.round(r_value, 2)})'

        # seaborn's lead dev refuses to enable displaying the best fit parameters
        ax = sns.regplot(
            x=rewards,
            y=task_aligned_deltas[:, col],
            ax=ax,
            ci=99,
            scatter_kws={'s': 1,
                         'color': orange_blue_cmap(task_aligned_deltas[:, col])},  # marker size
            line_kws={'color': side_color_map['ideal'],
                      'label': line_eqn}
        )

        # this needs to go down here for some reason
        ax.set_xlabel(r'$r_{n, t}$')

        ax.legend()

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_effect_of_feedback_along_task_aligned_vectors',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_effect_of_feedback_along_task_aligned_vectors_by_side(hook_input):
    session_data = hook_input['session_data']
    reward_data = session_data[
        (session_data.reward == 1.) | (session_data.reward == -1)]
    # drop last dt because no ensuing state. this prevents indexing error
    reward_data = reward_data.iloc[:len(reward_data) - 1]

    task_aligned_deltas = np.diff(
        hook_input['task_aligned_hidden_states'],
        n=1,
        axis=0)

    num_rows, num_cols = 1, 2
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(8, 3))

    for (action, feedback), reward_data_subset in reward_data.groupby([
        'action_side', 'reward']):

        for col in range(2):
            ax = axes[col]
            if col == 0:
                ax.set_xlabel('Movement Along Stimulus Readout')
            else:
                ax.set_xlabel('Movement Along Block Readout')

            action_str = side_string_map[action]
            color = side_color_map[action]
            linestyle = '-' if feedback == 1 else '--'
            sns.distplot(task_aligned_deltas[reward_data_subset.index, col],
                         hist=False,
                         kde=True,
                         ax=ax,
                         label=f'{action_str} Action, Feedback: {feedback}',
                         # bins=int(180 / 5),
                         kde_kws={'linewidth': 1,
                                  'color': color,
                                  'linestyle': linestyle})

            ax.legend()

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_effect_of_feedback_along_task_aligned_vectors_by_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_fixed_point_basins_of_attraction(hook_input):
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols + 1,
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    fixed_points_basins_df = hook_input['fixed_points_basins_df']
    color_min = 0.
    color_max = max([
        max(energy_array) for energy_array in fixed_points_basins_df['energy'].values
        if isinstance(energy_array, list)])

    for i, ((lstim, rstim, fdbk), fixed_point_basin_subset) in enumerate(fixed_points_basins_df.groupby([
        'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        row, col = int(i / num_cols), int(i % num_cols)
        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        title = f'l={np.round(lstim, 2)}, r={np.round(rstim, 2)}, f={fdbk}'
        ax.set_title(title)
        if row == 1:
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        if not isinstance(fixed_point_basin_subset.loc[i, 'pca_fixed_point_state'], list):
            continue

        # plot fixed point and annotate it with displacement
        pca_fixed_point_state = np.array(
            fixed_point_basin_subset.loc[i, 'pca_fixed_point_state'])
        ax.scatter(
            pca_fixed_point_state[0],
            pca_fixed_point_state[1],
            c='k',
            marker='*',
            s=30,
            zorder=2  # put in front
        )

        ax.annotate(
            np.round(fixed_point_basin_subset.loc[i, 'fixed_point_displacement'], 3),
            (pca_fixed_point_state[0],
             pca_fixed_point_state[1] + 0.5),  # plot a little above
            weight='bold',
            fontSize=8)

        # plot energy contour within basin
        initial_pca_states_in_basin = np.array(
            fixed_point_basin_subset.loc[i, 'initial_pca_states_in_basin'])
        energy = np.array(
            fixed_point_basin_subset['energy'].tolist())
        sc = ax.scatter(
            initial_pca_states_in_basin[:, 0],
            initial_pca_states_in_basin[:, 1],
            c=energy,
            zorder=1,  # put behind
            s=6,
            cmap='copper',
            vmin=color_min,
            vmax=color_max)

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # merge the rightmost column for the colorbar
    gs = axes[0, 3].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(sc, cax=ax_colorbar)
    color_bar.set_label(r'Energy ($0.5 x^T P x$)')
    hook_input['tensorboard_writer'].add_figure(
        tag=f'state_space_fixed_point_basins',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_fixed_point_search(hook_input):
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols + 1,
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = .75  # np.max(hook_input['fixed_point_df']['displacement_norm'])

    for i, ((lstim, rstim, fdbk), fixed_point_subset) in enumerate(hook_input['fixed_point_df'].groupby([
        'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        row, col = int(i / num_cols), int(i % num_cols)

        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        title = f'l={np.round(lstim, 2)}, r={np.round(rstim, 2)}, f={fdbk}'
        ax.set_title(title)
        if row == 1:
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        final_pca_sampled_states = np.stack(
            fixed_point_subset['final_pca_sampled_state'].values.tolist())

        # indices_to_keep = fixed_point_subset['displacement_norm'] < color_max
        indices_to_keep = fixed_point_subset['jacobian_hidden_stable'].values.astype(np.bool)

        sc = ax.scatter(
            final_pca_sampled_states[indices_to_keep, 0],
            final_pca_sampled_states[indices_to_keep, 1],
            c=fixed_point_subset['displacement_norm'][indices_to_keep],
            s=3,
            vmin=color_min,
            vmax=color_max,
            cmap='copper')

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # merge the rightmost column for the colorbar
    gs = axes[0, 3].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(sc, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')

    hook_input['tensorboard_writer'].add_figure(
        tag=f'state_space_fixed_point_search',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_projection_on_right_block_vector_by_trial_within_block(hook_input):
    session_data = hook_input['session_data']

    trial_end_data = session_data[session_data['trial_end'] == 1.].copy()
    # assigning to slice to not overwite posterior probability
    trial_end_data['bayesian_observer_block_posterior_right'] = \
        2. * trial_end_data['bayesian_observer_block_posterior_right'] - 1.
    trial_end_data['bayesian_observer_block_posterior_right'] *= hook_input['block_scaling_parameter']

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Normalized Projection Along Right Block Vector by Trial within Block')
    ax.set_xlabel('Trial within Block')
    ax.set_ylabel('Magnitude Along Block Readout')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for block_side, block_side_trials_data in trial_end_data.groupby(['block_side']):
        temp_df = block_side_trials_data[[
            'block_side', 'trial_index',
            'bayesian_observer_block_posterior_right',
            'magn_along_block_vector']].copy()

        temp_groupby = temp_df.groupby(['trial_index']).agg({
            'magn_along_block_vector': ['mean', 'sem', 'size'],
            'bayesian_observer_block_posterior_right': ['mean', 'sem', 'size'],
        })
        magn_along_block_vector_by_trial_index = temp_groupby[
            'magn_along_block_vector']
        observer_block_posterior_right_by_trial_index = temp_groupby[
            'bayesian_observer_block_posterior_right']

        # drop NaN/0 SEM
        magn_along_block_vector_by_trial_index = magn_along_block_vector_by_trial_index[
            magn_along_block_vector_by_trial_index['size'] > 1.]
        observer_block_posterior_right_by_trial_index = observer_block_posterior_right_by_trial_index[
            observer_block_posterior_right_by_trial_index['size'] > 1.]

        ax.plot(
            magn_along_block_vector_by_trial_index.index,
            magn_along_block_vector_by_trial_index['mean'],
            label=f'RNN {side_string_map[block_side]} Block',
            color=side_color_map[block_side],
        )

        ax.fill_between(
            x=magn_along_block_vector_by_trial_index.index,
            y1=magn_along_block_vector_by_trial_index['mean']
               - magn_along_block_vector_by_trial_index['sem'],
            y2=magn_along_block_vector_by_trial_index['mean']
               + magn_along_block_vector_by_trial_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal'])

        ax.plot(
            observer_block_posterior_right_by_trial_index.index,
            observer_block_posterior_right_by_trial_index['mean'],
            markersize=1,
            linewidth=1,
            label='Bayesian Observer (Scaling: {})'.format(
                np.round(hook_input['block_scaling_parameter'], 2)
            ),
            color=side_color_map['ideal'])

        ax.fill_between(
            x=observer_block_posterior_right_by_trial_index.index,
            y1=observer_block_posterior_right_by_trial_index['mean']
               - observer_block_posterior_right_by_trial_index['sem'],
            y2=observer_block_posterior_right_by_trial_index['mean']
               + observer_block_posterior_right_by_trial_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])

    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels)
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_projection_on_right_block_vector_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_projection_on_right_trial_vector_by_dts_within_trial(hook_input):
    session_data = hook_input['session_data']

    during_trials_data = session_data[(session_data.left_stimulus != 0) &
                                      (session_data.right_stimulus != 0)].copy()

    # assigning to slice to not overwite posterior probability
    during_trials_data['bayesian_observer_stimulus_posterior_right'] = \
        2. * during_trials_data['bayesian_observer_stimulus_posterior_right'] - 1.

    # scale posterior to range of projection values
    # use OLS estimate
    X = during_trials_data['bayesian_observer_stimulus_posterior_right'].values[:, np.newaxis]
    Y = during_trials_data['magn_along_block_vector'].values[:, np.newaxis]
    scaling_parameter = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # shape = (1, 1)
    scaling_parameter = scaling_parameter[0, 0]
    during_trials_data['bayesian_observer_stimulus_posterior_right'] *= scaling_parameter

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle('Projection Along Right Trial Vector by dt within Trial')
    ax.set_xlabel('dt within Trial')
    ax.set_ylabel('Magnitude Along Trial Vector')
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    for trial_side, trial_side_during_trials_data in during_trials_data.groupby(['trial_side']):
        temp_groupby = trial_side_during_trials_data.groupby(['rnn_step_index']).agg({
            'magn_along_trial_vector': ['mean', 'sem', 'size'],
            'bayesian_observer_stimulus_posterior_right': ['mean', 'sem', 'size']})

        magn_along_trial_vector_by_rnn_step_index = temp_groupby[
            'magn_along_trial_vector']
        bayesian_observer_stimulus_posterior_right_by_trial_index = temp_groupby[
            'bayesian_observer_stimulus_posterior_right']

        # drop NaN/0 SEM trials
        magn_along_trial_vector_by_rnn_step_index = magn_along_trial_vector_by_rnn_step_index[
            magn_along_trial_vector_by_rnn_step_index['size'] > 1.]
        bayesian_observer_stimulus_posterior_right_by_trial_index = \
            bayesian_observer_stimulus_posterior_right_by_trial_index[
                bayesian_observer_stimulus_posterior_right_by_trial_index['size'] > 1.]

        ax.plot(
            magn_along_trial_vector_by_rnn_step_index.index,
            magn_along_trial_vector_by_rnn_step_index['mean'],
            '-o',
            markersize=2,
            label=f'RNN {side_string_map[trial_side]} Trials',
            color=side_color_map[trial_side])

        ax.fill_between(
            x=magn_along_trial_vector_by_rnn_step_index.index,
            y1=magn_along_trial_vector_by_rnn_step_index['mean']
               - magn_along_trial_vector_by_rnn_step_index['sem'],
            y2=magn_along_trial_vector_by_rnn_step_index['mean']
               + magn_along_trial_vector_by_rnn_step_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map[trial_side])

        ax.plot(
            bayesian_observer_stimulus_posterior_right_by_trial_index.index,
            bayesian_observer_stimulus_posterior_right_by_trial_index['mean'],
            '-o',
            markersize=2,
            linewidth=1,
            label=f'Bayesian Observer (Scaling: {np.round(scaling_parameter, 2)})',
            color=side_color_map['ideal'])

        ax.fill_between(
            x=bayesian_observer_stimulus_posterior_right_by_trial_index.index,
            y1=bayesian_observer_stimulus_posterior_right_by_trial_index['mean']
               - bayesian_observer_stimulus_posterior_right_by_trial_index['sem'],
            y2=bayesian_observer_stimulus_posterior_right_by_trial_index['mean']
               + bayesian_observer_stimulus_posterior_right_by_trial_index['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map['ideal'])

    handles, labels = delete_redundant_legend_groups(ax=ax)
    ax.legend(handles, labels)
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_projection_on_right_trial_vector_by_dts_within_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_within_block(hook_input):
    session_data = hook_input['session_data']

    # take only last dt within a trial
    # exclude blocks that are first in the session
    session_data = session_data[session_data.trial_end == 1]

    num_rows, num_cols = 1, 2

    # select only environment 0, last num_rows * num_cols blocks
    subset_session_data = session_data[(session_data.session_index == 0) &
                                       (session_data.block_index > max(session_data.block_index) - num_cols * num_rows)]
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(8, 3))
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    # plt.suptitle('State Space Trajectories ({} degrees between readout vectors)'.format(
    #     hook_input['degrees_btwn_pca_trial_block_vectors']))

    # create possible color range
    max_block_duration = max(subset_session_data.groupby(['session_index', 'block_index']).size())

    for i, (block_idx, session_data_by_block) in enumerate(subset_session_data.groupby('block_index')):

        row, col = int(i / num_cols), int(i % num_cols)
        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        block_side = side_string_map[session_data_by_block.block_side.unique()[0]]
        ax.set_title(f'{block_side} Block')
        # ax.set_title(f'Block {1 + int(block_idx)}\n{block_side} Block')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        if row == (num_rows - 1):
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])

        block_indices = session_data_by_block.index.values
        proj_hidden_states_block = hook_input['pca_hidden_states'][block_indices]
        # stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
        # segment_text = np.where(session_data_by_block['reward'] > 0.9, 'C', 'I')
        for i in range(len(block_indices) - 1):
            ax.plot(
                proj_hidden_states_block[i:i + 2, 0],
                proj_hidden_states_block[i:i + 2, 1],
                '-o',
                color=plt.cm.jet(i / max_block_duration),
                markersize=2,
                # linestyle='None',
                zorder=2)
            # ax.text(
            #     proj_hidden_states_block[i, 0],
            #     proj_hidden_states_block[i, 1],
            #     # str(stimuli[i]),
            #     segment_text[i]
            # )

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # TODO: add colobar without disrupting
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_block_duration))
    # color_bar = fig.colorbar(sm, cax=axes[-1])
    # color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


# def hook_plot_state_space_trajectories_within_block_smooth(hook_input):
#     trajectory_controlled_output = utils.analysis.compute_projected_hidden_state_trajectory_controlled(
#         model=hook_input['model'],
#         pca=hook_input['pca'])
#
#     session_data = trajectory_controlled_output['session_data']
#     max_block_len = max(session_data.groupby(['session_index', 'stimuli_block_number']).size())
#
#     fig, axes = plt.subplots(nrows=3,
#                              ncols=4,  # 1 row, 3 columns
#                              gridspec_kw={"width_ratios": [1, 1, 1, 1]},
#                              figsize=(18, 12))
#     fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
#     plt.suptitle(f'Model State Space (Projected) Smooth Trajectories')
#
#     for block_num, trial_data_by_block in session_data.groupby('stimuli_block_number'):
#         row, col = block_num // 4, block_num % 4  # hard coded for 2 rows, 4 columns
#         ax = axes[row, col]
#         ax.set_title(f'Block Num: {1 + block_num}')
#         ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
#         ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
#         if row == 1:
#             ax.set_xlabel('PC #1')
#         if col == 0:
#             ax.set_ylabel('PC #2')
#
#         block_indices = trial_data_by_block.index.values
#         proj_hidden_states_block = trajectory_controlled_output['projected_hidden_states'][block_indices]
#         stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
#         for i in range(len(block_indices) - 1):
#             ax.plot(
#                 proj_hidden_states_block[i:i + 2, 0],
#                 proj_hidden_states_block[i:i + 2, 1],
#                 color=plt.cm.jet(i / max_block_len))
#             ax.text(
#                 proj_hidden_states_block[i + 1, 0],
#                 proj_hidden_states_block[i + 1, 1],
#                 str(stimuli[i]))
#
#     hook_input['tensorboard_writer'].add_figure(
#         tag='hidden_state_projected_phase_space_trajectories_controlled',
#         figure=fig,
#         global_step=hook_input['grad_step'],
#         close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_state_space_trajectories_different_block_same_stimuli(hook_input):
    session_data = hook_input['session_data']

    # sorted
    dist_of_magn_along_block_vector = session_data.groupby(['trial_within_session'])[
        'magn_along_block_vector'].nth(n=1).sort_values()

    # keep top 29 left-most blocks and 15 right-most blocks
    top_k = 20
    leftmost_trial_indices = dist_of_magn_along_block_vector.index[:top_k]
    leftmost_block_session_data = session_data.groupby(['trial_within_session']).filter(
        lambda group: (group['trial_within_session'].isin(leftmost_trial_indices)).any()
    )

    rightmost_trial_indices = dist_of_magn_along_block_vector.index[-top_k:]
    rightmost_block_session_data = session_data.groupby(['trial_within_session']).filter(
        lambda group: (group['trial_within_session'].isin(rightmost_trial_indices)).any()
    )

    rnn_steps_before_obs = hook_input['envs'][0].rnn_steps_before_obs
    l2_stimulus_distances = {}

    # find two trials with as similar stimuli as possible
    for right_idx, rightmost_block_trial_data in rightmost_block_session_data.groupby(['trial_within_session']):
        right_block_left_stimuli = rightmost_block_trial_data['left_stimulus']
        right_block_right_stimuli = rightmost_block_trial_data['right_stimulus']
        right_block_diff_stimuli = right_block_right_stimuli - right_block_left_stimuli
        for left_idx, leftmost_block_trial_data in leftmost_block_session_data.groupby(['trial_within_session']):
            left_block_left_stimuli = leftmost_block_trial_data['left_stimulus']
            left_block_right_stimuli = leftmost_block_trial_data['right_stimulus']
            left_block_diff_stimuli = left_block_right_stimuli - left_block_left_stimuli
            shortest_length = min(right_block_diff_stimuli.size, left_block_diff_stimuli.size)
            l2_stimulus_distances[(right_idx, left_idx)] = np.linalg.norm(np.subtract(
                right_block_diff_stimuli[rnn_steps_before_obs:shortest_length].values,
                left_block_diff_stimuli[rnn_steps_before_obs:shortest_length].values))
            # print(left_block_diff_stimuli)
            # print(right_block_diff_stimuli)
            # print(l2_stimulus_distances[(right_idx, left_idx)])

    top_right_index, top_left_index = min(l2_stimulus_distances, key=l2_stimulus_distances.get)
    right_trial_data = rightmost_block_session_data.groupby('trial_within_session').filter(
        lambda group: (group['trial_within_session'] == top_right_index).all())
    right_trial_data = right_trial_data.iloc[1:]
    left_trial_data = leftmost_block_session_data.groupby('trial_within_session').filter(
        lambda group: (group['trial_within_session'] == top_left_index).all())
    left_trial_data = left_trial_data.iloc[1:]

    # separate by side bias
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(4, 3))

    ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
    ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
    ax.set_xlabel('PC #1')
    ax.set_ylabel('PC #2')

    pca_hidden_states = hook_input['pca_hidden_states']

    for side, trial_data in [('left', left_trial_data), ('right', right_trial_data)]:

        pca_trial_hidden_states = pca_hidden_states[trial_data.index.values]

        # plot the first dt in the trial
        ax.plot(
            pca_trial_hidden_states[0, 0],
            pca_trial_hidden_states[0, 1],
            'o-',
            markersize=2,
            color=side_color_map[side],
            zorder=2)

        # plot the rest of the trial's dts
        for i in range(1, len(trial_data)):
            ax.plot(
                pca_trial_hidden_states[i - 1:i + 1, 0],
                pca_trial_hidden_states[i - 1:i + 1, 1],
                'o-',
                markersize=2,
                color=side_color_map[side])

    add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input, add_labels=True)

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_different_block_same_stimuli',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_within_trial(hook_input):
    session_data = hook_input['session_data']

    num_rows, num_cols = 3, 3
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(4, 4))
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    # plt.suptitle('State Space PCA Trajectories ({} degrees between readout vectors)'.format(
    #     hook_input['degrees_btwn_pca_trial_block_vectors']))

    # select only environment 1, first 12 trials
    subset_session_data = session_data[(session_data['session_index'] == 0) &
                                       (session_data['block_index'] == 2) &
                                       (session_data['trial_index'] < num_cols * num_rows)]

    # create possible color range
    max_trial_duration = max(subset_session_data.groupby(['session_index', 'block_index', 'trial_index']).size())

    for trial_index, session_data_by_trial in subset_session_data.groupby('trial_index'):

        if trial_index >= num_cols * num_rows:
            break

        row, col = int(trial_index / num_cols), int(trial_index % num_cols)
        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        trial_side = side_string_map[session_data_by_trial.trial_side.unique()[0]]
        title = f'{trial_side} Trial, '
        title += 'Correct Action' if bool(session_data_by_trial.tail(1).iloc[0].correct_action_taken) \
            else 'Incorrect Action'
        ax.set_title(title)
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        if row == (num_rows - 1):
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])

        trial_indices = session_data_by_trial.index.values
        proj_hidden_states_block = hook_input['pca_hidden_states'][trial_indices]

        # plot the first dt in the trial
        ax.plot(
            proj_hidden_states_block[0, 0],
            proj_hidden_states_block[0, 1],
            'o-',
            markersize=1,
            color=plt.cm.jet(0 / max_trial_duration),
            zorder=2)

        # plot the rest of the trial's dts
        for i in range(1, len(trial_indices)):
            ax.plot(
                proj_hidden_states_block[i - 1:i + 1, 0],
                proj_hidden_states_block[i - 1:i + 1, 1],
                'o-',
                markersize=1,
                color=plt.cm.jet(i / max_trial_duration))
            # ax.text(
            #     proj_hidden_states_block[i, 0],
            #     proj_hidden_states_block[i, 1],
            #     # str(stimuli[i]),
            #     segment_text[i]
            # )

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # TODO: add colobar without disrupting
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_block_duration))
    # color_bar = fig.colorbar(sm, cax=axes[-1])
    # color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_within_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)

def hook_plot_state_space_trials_by_classifier(hook_input):
    session_data = hook_input['session_data']
    # take only last dt within a trial
    # exclude blocks that are first in the session
    trial_end_data = session_data[session_data.trial_end == 1]

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             gridspec_kw={"width_ratios": [1, 0.05]},
                             figsize=(4, 3))

    max_trials_per_block_to_consider = 50
    trial_end_data = trial_end_data[trial_end_data.trial_index < max_trials_per_block_to_consider]

    ax = axes[0]
    ax.axis('equal')  # set yscale to match xscale
    ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
    ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
    ax.set_xlabel('PC #1')
    ax.set_ylabel('PC #2')

    block_side_trial_end_rows = trial_end_data.index.values
    block_side_trial_end_proj_hidden_states = hook_input['pca_hidden_states'][block_side_trial_end_rows]
    sc = ax.scatter(
        block_side_trial_end_proj_hidden_states[:, 0],
        block_side_trial_end_proj_hidden_states[:, 1],
        alpha=0.4,
        s=1,
        c=(1. + trial_end_data.classifier_block_side) / 2,
        vmin=0.,
        vmax=1.,
        cmap=orange_blue_cmap)

    add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    cbar = fig.colorbar(sc, cax=axes[1])
    cbar.set_label('Classifier Block Side (Left=0, Right=1)')
    # cbar.set_label('Classifier (PCA Accuracy: {}, Full Accuracy: {})'.format(
    #     np.round(hook_input['pca_block_classifier_accuracy'], 4),
    #     np.round(hook_input['full_block_classifier_accuracy'], 4)))
    cbar.set_alpha(1)
    cbar.draw_all()
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trials_by_block_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_vector_fields_ideal(hook_input):
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols + 1,  # +1 for colorbar
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = np.max(hook_input['fixed_point_df']['displacement_norm'])

    for i, ((lstim, rstim, fdbk), fixed_point_subset) in enumerate(hook_input['fixed_point_df'].groupby([
        'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        row, col = int(i / num_cols), int(i % num_cols)

        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        if np.isclose(lstim, 1.2, atol=1e-02) and np.isclose(rstim, 0.2, atol=1e-02):
            title = 'Left Stimulus'
        elif np.isclose(lstim, 0.2, atol=1e-02) and np.isclose(rstim, 1.2, atol=1e-02):
            title = 'Right Stimulus'
        elif np.isclose(lstim, 0.2, atol=1e-02) and np.isclose(rstim, 0.2, atol=1e-02):
            title = 'Equal Stimulus'
        elif fdbk == -1.:
            title = 'Negative Feedback'
        elif fdbk == 1.:
            title = 'Positive Feedback'
        elif lstim == 0 and rstim == 0 and fdbk == 0:
            title = 'No Stimulus'
        else:
            # print(lstim, rstim, fdbk)
            raise ValueError

        ax.set_title(title, fontsize=8)
        if row == 1:
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        initial_pca_sampled_states = np.stack(
            fixed_point_subset['initial_pca_sampled_state'].values.tolist())

        displacement_pca = np.stack(fixed_point_subset['displacement_pca'].values.tolist())

        qvr = ax.quiver(
            initial_pca_sampled_states[:, 0],
            initial_pca_sampled_states[:, 1],
            displacement_pca[:, 0],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_pca[:, 1],  # / np.linalg.norm(displacement_pca, axis=1),
            fixed_point_subset['displacement_norm'],  # color
            angles='xy',  # this and the next two ensures vector scales match data scales
            scale_units='xy',
            scale=1,
            alpha=0.6,
            clim=(color_min, color_max),
            headwidth=7,
            cmap='gnuplot2')

        add_pca_readout_vectors_to_axis(
            ax=ax,
            hook_input=hook_input,
            add_labels=True)

    # merge the rightmost column for the colorbar
    gs = axes[0, 3].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(qvr, cax=ax_colorbar)
    color_bar.set_label('state velocity (a.u.)', size=9)
    color_bar.set_alpha(1)
    color_bar.draw_all()

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_vector_fields_ideal',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_vector_fields_real(hook_input):
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols + 1,  # +1 for colorbar
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = -np.inf
    for array in hook_input['model_state_space_vector_fields']['displacement_pca_norm'].values:
        if len(array) == 0:
            continue
        color_max = max(np.max(array), color_max)

    for i, ((lstim, rstim, fdbk), vector_field_task_condition) in \
            enumerate(hook_input['model_state_space_vector_fields'].groupby([
                'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        row, col = int(i / num_cols), int(i % num_cols)

        ax = axes[row, col]
        ax.axis('equal')  # set yscale to match xscale
        title = r'$o_{{n,t}}^L={}, o_{{n,t}}^R={}, r_{{n, t}}={}$'.format(
            np.round(lstim, 2),
            np.round(rstim, 2),
            np.round(fdbk, 2))
        ax.set_title(title, fontsize=8)
        if row == 1:
            ax.set_xlabel('PC #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('PC #2')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        pca_hidden_states_pre = np.concatenate(
            vector_field_task_condition['pca_hidden_states_pre'].values.tolist())

        displacement_pca = np.concatenate(
            vector_field_task_condition['displacement_pca'].values.tolist())

        displacement_norm = np.concatenate(
            vector_field_task_condition['displacement_pca_norm'].values.tolist())

        qvr = ax.quiver(
            pca_hidden_states_pre[:, 0],
            pca_hidden_states_pre[:, 1],
            displacement_pca[:, 0],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_pca[:, 1],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_norm,  # color
            angles='xy',  # this and the next two ensures vector scales match data scales
            scale_units='xy',
            scale=1,
            alpha=0.6,
            clim=(color_min, color_max),
            headwidth=7,
            cmap='gist_rainbow')

        add_pca_readout_vectors_to_axis(
            ax=ax,
            hook_input=hook_input,
            add_labels=False)

    # merge the rightmost column for the colorbar
    gs = axes[0, 3].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(qvr, cax=ax_colorbar)
    color_bar.set_label(r'$||h_{n, t} - RNN(h_{n, t-1}, o_{n, t}) ||_2$')
    color_bar.set_alpha(1)
    color_bar.draw_all()
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_vector_fields_real',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_block_inference_multiple_blocks(hook_input):
    n = 300
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Trial Number')
    ax.set_ylabel(r'Block Side (Rescaled)')
    session_data = hook_input['session_data']
    trial_end_data = session_data[session_data.trial_end == 1.]
    x = np.arange(1, n + 1)
    ax.plot(
        x,
        trial_end_data['magn_along_block_vector'][:n],
        '-o',
        markersize=1,
        linewidth=1,
        color=side_color_map['right'],
        label='RNN Block Belief')
    # transform from [0, 1] to whatever scaling RNN uses
    rescaled_observer_block_posterior = \
        hook_input['block_scaling_parameter'] * \
        (2 * trial_end_data['bayesian_observer_block_posterior_right'][:n] - 1)
    ax.plot(
        x,
        rescaled_observer_block_posterior,
        color=side_color_map['ideal'],
        label='Bayesian Observer')
    ax.plot(
        x,
        hook_input['block_scaling_parameter'] * trial_end_data['block_side'][:n],
        color=side_color_map['neutral'],
        label='Block Side')
    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='optimal_observer_block_posterior_multiple_blocks',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_block_inference_single_block(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Trial Number')
    ax.set_ylabel(r'Block Side (Rescaled)')
    ax.set_ylim(-5, 5)
    session_data = hook_input['session_data']
    trial_end_data = session_data[session_data.trial_end == 1.]
    block_switch_indicator = trial_end_data['block_side'].diff(periods=-1) != 0
    # index of first block switch index + the next 20 of the next block
    n = block_switch_indicator.argmax() + + hook_input['envs'][0].min_trials_per_block
    x = np.arange(1, n + 1)
    # transform from [0, 1] to whatever scaling RNN uses
    rescaled_observer_block_posterior = \
        hook_input['block_scaling_parameter'] * \
        (2 * trial_end_data['bayesian_observer_block_posterior_right'][:n] - 1)
    ax.plot(
        x,
        rescaled_observer_block_posterior,
        color=side_color_map['ideal'],
        label='Bayesian Observer')
    ax.plot(
        x,
        hook_input['block_scaling_parameter'] * trial_end_data['block_side'][:n],
        color=side_color_map['neutral'],
        label='Block Side')
    ax.plot(
        x,
        hook_input['block_scaling_parameter'] * trial_end_data['trial_side'][:n],
        'x',
        markersize=3,
        color=side_color_map['neutral'],
        label='Stimulus Side')
    ax.plot(
        x,
        trial_end_data['magn_along_block_vector'][:n],
        '-+',
        markersize=3,
        color=side_color_map['right'],
        label='RNN Block Belief',
        linewidth=1)

    non_blank_data = session_data[(session_data.left_stimulus != 0) &
                                  (session_data.right_stimulus != 0)]
    average_stimuli = non_blank_data.groupby(['session_index', 'block_index', 'trial_index']).agg({
        'left_stimulus': 'mean',
        'right_stimulus': 'mean'})
    mean_stimulus_diff = average_stimuli['right_stimulus'] - average_stimuli['left_stimulus']

    ax.plot(
        x,
        mean_stimulus_diff[:n],
        '--+',
        markersize=2,
        color=side_color_map['right'],
        label=r'Average of Diff of Observation Within Trial',
        linewidth=1)
    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='optimal_observer_block_posterior_single_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_block_side_trial_side_by_trial_number(hook_input):
    session_data = hook_input['session_data']

    # keep only session 0
    session_data = session_data[session_data.session_index == 0]
    first_dt_of_each_trial = session_data.groupby(['block_index', 'trial_index']).first()

    # keep only first 100 trials
    first_dt_of_each_trial = first_dt_of_each_trial[:150]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Stimulus Side')

    # plot trial side
    ax.scatter(np.arange(1, 1 + len(first_dt_of_each_trial)),
               np.where(first_dt_of_each_trial.trial_side == 1, 1., 0.),
               label='Trial Side',
               s=1,
               c=side_color_map['ideal'])

    # plot block side
    for x in np.arange(1, 1 + len(first_dt_of_each_trial)):
        ax.axvspan(
            x,
            x + 1,
            facecolor=side_color_map[first_dt_of_each_trial.block_side.values[int(x - 1)]],
            alpha=0.5)

    ax.set_yticklabels(['', 'Left', '', '', '', '', 'Right'])

    hook_input['tensorboard_writer'].add_figure(
        tag='task_block_side_trial_side_by_trial_number',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_by_block_side(hook_input):
    session_data = hook_input['session_data']

    fig, axes = plt.subplots(nrows=2, figsize=(4, 3))

    for i, block_side in enumerate(session_data.block_side.unique()):
        ax = axes[i]
        ax.set_title(f'{side_string_map[block_side]} Block')
        block_side_session_data = session_data[session_data.block_side == block_side]
        ax.hist(
            block_side_session_data.left_stimulus,
            bins=50,
            label='Left Stimulus',
            alpha=0.5,
            color=side_color_map['left'])
        ax.hist(
            block_side_session_data.right_stimulus,
            bins=50,
            label='Right Stimulus',
            alpha=0.5,
            color=side_color_map['right'])
        ax.legend()

    # add x label to lowest row
    ax.set_xlabel('Stimulus Value')
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_by_block_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_by_correct_trial_side(hook_input):
    session_data = hook_input['session_data']
    correct_side_stimuli = pd.concat(
        (session_data.right_stimulus[session_data.trial_side == 1],
         session_data.left_stimulus[session_data.trial_side == -1]),
        axis=0)
    incorrect_side_stimuli = pd.concat(
        (session_data.right_stimulus[session_data.trial_side == -1],
         session_data.left_stimulus[session_data.trial_side == 1]),
        axis=0)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Sampled Stimuli')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.scatter(
        correct_side_stimuli,
        incorrect_side_stimuli,
        alpha=0.5,
        color=side_color_map['neutral'])
    ax.axvline(np.mean(correct_side_stimuli.values))
    ax.axhline(np.mean(incorrect_side_stimuli.values))
    ax.set_xlabel('Correct Side Stimuli')
    ax.set_ylabel('Incorrect Side Stimuli')
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_by_correct_trial_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_and_model_prob_in_first_n_trials(hook_input):
    session_data = hook_input['session_data']

    ncols = 5
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(4, 3),
        sharex=True,
        sharey=True)
    # fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    axes[0].set_ylabel('Stimulus Contrast & Model P(Right Trial)')

    for col, (_, trial_data) in enumerate(session_data.groupby(
            ['session_index', 'block_index', 'trial_index'])):

        if col == ncols:
            break

        ax = axes[col]
        ax.axhline(0.1, color='k')
        ax.axhline(0.9, color='k')
        ax.set_xlabel('RNN Step In Trial')
        ax.set_title(f'Trial Side: {side_string_map[trial_data.trial_side.unique()[0]]}\n'
                     f'Strength: {round(trial_data.trial_strength.unique()[0], 2)}')
        ax.set_xlim(0, hook_input['envs'][0].max_rnn_steps_per_trial)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.left_stimulus,
            '+',  # necessary to ensure 1-RNN step trials visualized
            label='Left Stimulus',
            color=side_color_map['left'],
            markersize=6)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.right_stimulus,
            '+',  # necessary to ensure 1-RNN step trials visualized
            label='Right Stimulus',
            color=side_color_map['right'],
            markersize=3)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.right_action_prob,
            'o--',  # necessary to ensure 1-RNN step trials visualized
            label='Model P(Right Action)',
            color=side_color_map['right'],
            markersize=1,
            linewidth=1)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.bayesian_observer_stimulus_posterior_right,
            'o--',  # necessary to ensure 1-RNN step trials visualized
            label='Bayesian Observer',
            color=side_color_map['ideal'],
            markersize=1,
            linewidth=1)
    ax.legend()

    # add x label to lowest col
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_and_model_prob_in_first_n_trials',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_traditionally_distilled_state_space_vector_fields_ideal(hook_input):
    # TODO: deduplicate with hook_plot_state_space_vector_fields_ideal
    num_cols = 3
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_cols + 1,  # +1 for colorbar
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = np.max(hook_input['traditionally_distilled_fixed_point_df']['displacement_norm'])

    initial_pca_sampled_states = np.stack(
        hook_input['traditionally_distilled_fixed_point_df']['initial_pca_sampled_state'].values.tolist())

    # drop all except left stimulus, no stimulus, right stimulus
    fixed_point_df = hook_input['traditionally_distilled_fixed_point_df']
    indices_to_keep = fixed_point_df['feedback'] == -0.05
    fixed_point_df = fixed_point_df[indices_to_keep]
    initial_pca_sampled_states = initial_pca_sampled_states[indices_to_keep]

    # calculate ranges of plot
    xmin = np.min(initial_pca_sampled_states[:, 0]) - 0.1
    xmax = np.max(initial_pca_sampled_states[:, 0]) + 0.1
    ymin = np.min(initial_pca_sampled_states[:, 1]) - 0.1
    ymax = np.max(initial_pca_sampled_states[:, 1]) + 0.1

    for i, ((lstim, rstim, fdbk), fixed_point_subset) in enumerate(
            fixed_point_df.groupby(['left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        col = int(i % num_cols)

        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        if i == 0:
            title = 'Left Stimulus'
        elif i == 1:
            title = 'No Stimulus'
        elif i == 2:
            title = 'Right Stimulus'
        else:
            raise ValueError
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Unit #1')

        if col == 0:
            ax.set_ylabel('Unit #2')
        else:
            ax.set_yticklabels([])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        initial_pca_sampled_states = np.stack(
            fixed_point_subset['initial_pca_sampled_state'].values.tolist())
        displacement_pca = np.stack(fixed_point_subset['displacement_pca'].values.tolist())

        qvr = ax.quiver(
            initial_pca_sampled_states[:, 0],
            initial_pca_sampled_states[:, 1],
            displacement_pca[:, 0],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_pca[:, 1],  # / np.linalg.norm(displacement_pca, axis=1),
            fixed_point_subset['displacement_norm'],  # color
            angles='xy',  # this and the next two ensures vector scales match data scales
            scale_units='xy',
            scale=1,
            alpha=0.6,
            clim=(color_min, color_max),
            headwidth=7,
            cmap='gnuplot2')

    # merge the rightmost column for the colorbar
    color_bar = fig.colorbar(qvr, cax=axes[-1])
    color_bar.set_label(r'state velocity (a.u.)', size=9)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    hook_input['tensorboard_writer'].add_figure(
        tag='traditionally_distilled_state_space_vector_fields_ideal',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_traditionally_distilled_training_losses(hook_input):
    training_losses = hook_input['traditionally_distilled_training_losses']

    # exclude last point (artifact of how loss array was pre-allocated)
    training_losses = training_losses[:-1]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4, 3),
        gridspec_kw={"width_ratios": [1]})
    ax.set_xlabel('Gradient Step (4 Blocks)')
    ax.set_ylabel('Avg Loss / RNN Step')
    ax.plot(
        np.arange(len(training_losses)),
        training_losses,
        label='Traditionally Distilled (2 Units)')

    hook_input['tensorboard_writer'].add_figure(
        tag='traditionally_distilled_training_losses',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_two_unit_task_trained_state_space_vector_fields_ideal(hook_input):
    # TODO: deduplicate with hook_plot_state_space_vector_fields_ideal

    num_cols = 3
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_cols + 1,  # +1 for colorbar
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    color_min = 0.
    color_max = np.max(hook_input['two_unit_task_trained_fixed_point_df']['displacement_norm'])

    initial_pca_sampled_states = np.stack(
        hook_input['two_unit_task_trained_fixed_point_df']['initial_pca_sampled_state'].values.tolist())

    # drop all except left stimulus, no stimulus, right stimulus
    fixed_point_df = hook_input['two_unit_task_trained_fixed_point_df']
    indices_to_keep = fixed_point_df['feedback'] == -0.05
    fixed_point_df = fixed_point_df[indices_to_keep]
    initial_pca_sampled_states = initial_pca_sampled_states[indices_to_keep]

    # calculate ranges of plot
    xmin = np.min(initial_pca_sampled_states[:, 0]) - 0.1
    xmax = np.max(initial_pca_sampled_states[:, 0]) + 0.1
    ymin = np.min(initial_pca_sampled_states[:, 1]) - 0.1
    ymax = np.max(initial_pca_sampled_states[:, 1]) + 0.1

    for i, ((lstim, rstim, fdbk), fixed_point_subset) in enumerate(
            fixed_point_df.groupby([
                'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        col = int(i % num_cols)
        ax = axes[col]
        ax.axis('equal')  # set yscale to match xscale
        if i == 0:
            title = 'Left Stimulus'
        elif i == 1:
            title = 'No Stimulus'
        elif i == 2:
            title = 'Right Stimulus'
        else:
            raise ValueError
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Unit #1')

        if col == 0:
            ax.set_ylabel('Unit #2')
        else:
            ax.set_yticklabels([])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        initial_pca_sampled_states = np.stack(
            fixed_point_subset['initial_pca_sampled_state'].values.tolist())
        displacement_pca = np.stack(fixed_point_subset['displacement_pca'].values.tolist())

        qvr = ax.quiver(
            initial_pca_sampled_states[:, 0],
            initial_pca_sampled_states[:, 1],
            displacement_pca[:, 0],  # / np.linalg.norm(displacement_pca, axis=1),
            displacement_pca[:, 1],  # / np.linalg.norm(displacement_pca, axis=1),
            fixed_point_subset['displacement_norm'],  # color
            angles='xy',  # this and the next two ensures vector scales match data scales
            scale_units='xy',
            scale=1,
            alpha=0.6,
            clim=(color_min, color_max),
            headwidth=7,
            cmap='gnuplot2')

    # merge the rightmost column for the colorbar
    color_bar = fig.colorbar(qvr, cax=axes[-1])
    color_bar.set_label(r'state velocity (a.u.)', size=9)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    hook_input['tensorboard_writer'].add_figure(
        tag='two_unit_task_trained_state_space_vector_fields_ideal',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def add_pca_readout_vectors_to_axis(ax,
                                    hook_input=None,
                                    add_labels=True,
                                    vectors=None,
                                    add_vectors=False):
    if hook_input is None and vectors is None:
        raise ValueError('Must provide at least one of hook_input or vectors')

    # add readout vectors for right trial, right block
    labels = [
        'Stim Boundary',
        'Block Boundary'
    ]

    if vectors is None:
        vectors = [
            hook_input['pca_trial_readout_vector'],
            hook_input['pca_block_readout_vector']
        ]

    colors = [
        side_color_map['stim_readout'],
        side_color_map['stim_readout'],
        # side_color_map['block_readout'],
    ]

    for i, (label, vector, color) in enumerate(zip(labels, vectors, colors)):

        if add_vectors is True:
            ax.arrow(x=0.,
                     y=0.,
                     dx=2 * vector[0],
                     dy=2 * vector[1],
                     color=color,
                     length_includes_head=True,
                     head_width=0.16,
                     zorder=1)  # plot on top

        # calculate perpendicular hyperplane
        hyperplane = np.matmul(rotation_matrix_90, vector)
        np.testing.assert_almost_equal(actual=np.dot(hyperplane, vector),
                                       desired=0.)
        # scale hyperplane to ensure it covers entire plot
        hyperplane = 10 * hyperplane / np.linalg.norm(hyperplane)
        hyperplane_angle_degrees = 180 * np.arctan(hyperplane[1] / hyperplane[0]) / np.pi
        ax.plot([-hyperplane[0], 0, hyperplane[0]],
                [-hyperplane[1], 0, hyperplane[1]],
                color,
                zorder=1,  # plot on top
                # dashes=[2, 2]
                )

        if add_labels:
            # ax.annotate(
            #     label,
            #     xy=(vector[0] + 0.5,
            #         vector[1] + 0.5))

            # https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/text_rotation_relative_to_line.html
            ax.text(0, 0, label,
                    rotation=hyperplane_angle_degrees,
                    rotation_mode='anchor',
                    color=color)


def delete_redundant_legend_groups(ax):
    handles, labels = ax.get_legend_handles_labels()
    i = 1
    while i < len(labels):
        if labels[i] in labels[:i]:
            del (labels[i])
            del (handles[i])
        else:
            i += 1

    return handles, labels
