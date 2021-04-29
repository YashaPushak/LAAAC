from types import SimpleNamespace
import copy

import pandas as pd
import numpy as np
from scipy import stats

try:
    from ray.tune import Analysis
    from ray.tune.error import TuneError
    ray_available = True
except:
    ray_available = False

def _get_trial_results(analysis, iter_fidelity, gpsls):
    results = []
    if gpsls:
        columns = ['all_losses', 'all_timestamps', 
                   'new_current_runtime', 'real_configuration', 'new_timestamps',
                   'new_current_fidelity_budget', 'line_search_end_step_timestamp']
    else:
        columns = ['current_loss', 'all_timestamps', 'current_runtime', 'configuration', 'current_fidelity_budget']
    if iter_fidelity:
        columns.append('fresh')
    for key in analysis.trial_dataframes:
        df = analysis.trial_dataframes[key]
        results.append(df[columns])
    if len(results) == 0:
        return None
    # Concatenate everything
    results = pd.concat(results)
    return results


def _format_timestamps(df, df_budget, start_timestamp=None):
    """Adjusts the timestamps so that they start at 0"""
    min_ = df['Timestamp Start'].min() if start_timestamp is None else start_timestamp
    df['Timestamp End'] -= min_
    df['Timestamp Start'] -= min_
    df_budget['Timestamp'] -= min_
    min_ = 0
    max_ = df['Timestamp End'].max()
    return df, df_budget, min_, max_


def _flatten(df, columns_to_flatten, new_column_names, config_column='real_configuration'):
    flattened_columns = [[] for _ in range(len(columns_to_flatten))]
    configs = []
    for i in range(0, len(df)):
        entry = df.iloc[i]
        for j in range(len(columns_to_flatten)):
            if isinstance(entry[columns_to_flatten[j]], str):
                flattened = eval(entry[columns_to_flatten[j]].replace('inf', 'float("inf")'))
            else:
                flattened = entry[columns_to_flatten[j]]
            if isinstance(flattened, list):
                flattened_columns[j].extend(flattened)
            else:
                flattened_columns[j].append(flattened)
        configs.extend([entry[config_column]]*len(flattened))
    df = pd.DataFrame({'Configuration': configs})
    for i in range(len(columns_to_flatten)):
        df[new_column_names[i]] = flattened_columns[i]
    return df


def _extract_start_end_timestamps(df):
    df['Timestamp End'] = df['Timestamps'].map(lambda x: x[1])
    df['Timestamp Start'] = df['Timestamps'].map(lambda x: x[0])
    df = df.drop('Timestamps', axis=1)
    return df


def _get_anytime_trajectory(df, df_budget, objective='mean', min_run_ratio=4, asha_original=False):
    df, df_budget, min_, max_ = _format_timestamps(df, df_budget)
    df = df.sort_values('Timestamp End')
    incumbents = []
    wall_times = []
    target_times = []
    target_runs = []
    anytime_losses = []
    incumbent = df.head(1)['Configuration']
    incumbent_loss = df.head(1)['Loss']
    first = True
    if objective == 'mean':
        if asha_original:
            accept_challenger = _accept_challenger_mean
        else:
            accept_challenger = _accept_challenger_t_test
    else:
        accept_challenger = _accept_challenger_final
    for i in range(len(df)):
        challenger_row = df.iloc[i, :]
        time = challenger_row['Timestamp End']
        challenger = challenger_row['Configuration']
        # Get all of the data collected on this challenger 
        challenger_data = df[np.logical_and(df['Timestamp End'] <= time, df['Configuration'] == challenger)]
        challenger_loss = challenger_data['Loss']
        accept, challenger_loss_stat = accept_challenger(incumbent_loss, challenger_loss, min_run_ratio=min_run_ratio)
        if accept or first:
            first = False
            # Record it
            incumbents.append(challenger)
            wall_times.append(time)
            anytime_losses.append(challenger_loss_stat)
            # Find the total number of runs and the time spent on them for this budget
            budget = competitors = df_budget[df_budget['Timestamp'] <= time]
            target_times.append(budget['Runtime'].sum())
            target_runs.append(budget['Fidelity Budget'].sum())
            # Update the current incumbent data
            incumbent = challenger
            incumbent_loss = challenger_loss
    df_traj = pd.DataFrame({'Loss': anytime_losses,
                            'Wallclock Time': wall_times,
                            'Target Algorithm Time': target_times,
                            '# Target Algorithm Runs': target_runs,
                            'Configuration': incumbents})
    return df_traj


def _accept_challenger_t_test(incumbent, challenger, alpha=0.05, min_run_ratio=4):
    accept_challenger =  np.mean(challenger) < np.mean(incumbent)
    if len(challenger) < len(incumbent)/min_run_ratio:
        # If there are too few runs of the challenger, simply don't accept it.
        accept_challenger = False
    elif len(challenger) < len(incumbent) and min(len(challenger), len(incumbent)) > 1:
        # There may be less runs of the challenger than the incumbent; however,
        # if a t_test thinks it has better performance then we can still accept
        # it.
        _, p_value = stats.ttest_ind(incumbent, challenger, equal_var=False)
        if p_value > alpha:
            # The difference we saw might be due to random chance, so we keep
            # the current incumbent.
            accept_challenger = False
    return accept_challenger, np.mean(challenger)


def _accept_challenger_mean(incumbent, challenger, **kwargs):
    accept_challenger =  np.mean(challenger) < np.mean(incumbent)
    return accept_challenger, np.mean(challenger)


def _accept_challenger_final(incumbent, challenger, **kwargs):
    challenger = np.array(challenger)
    incumbent = np.array(incumbent)
    # Just pick based on the final performance of each.
    accept_challenger = challenger[-1] < incumbent[-1]
    return accept_challenger, challenger[-1]


def _get_gpsls_budget_and_incumbents(df):
    # FRESH: unlike with ASHA, where we first need to discard all of the
    # duplicate, non-fresh results so that we don't double-count them in
    # the budget, we do not need to do that here. This is because GPS-ASHA
    # already automatically checks to see what data is fresh and only records
    # that which is.
    # Create a flattened dataframe that contains information tracking all
    # of the data about how much budget has been spent.
    # At every step returned by GPS-ASHA, it returns a bunch of information
    # about all of the *new* configuration runs that were performed. This
    # information is not returned again in the next step of the same trial runner.
    df_budget = _flatten(df, ['new_current_runtime', 'new_current_fidelity_budget', 'new_timestamps'],
                             ['Runtime', 'Fidelity Budget', 'Timestamp'])
    # Now get only the last entry for each unique real configuration.
    # At each step of GPS-ASHA, it also returns the history of all run information
    # for the *current incumbent* of the line search. This information does not
    # include data about the configurations that are *not* the current incumbent
    # of the line search. Therefore to properly extract the overall, anytime incumbent
    # configurations and the budgets spent to obtain them, we need to extract this
    # information separately. Hence the line above extracts the budget information
    # and these lines extract the information *only* for the incumbents and their run
    # history. Note that this means that in order for a configuration to be considered
    # the overall incumbent is must have been returned as an incumbent of a line search.
    df = df.sort_values('line_search_end_step_timestamp', ascending=True).groupby('real_configuration').tail(1)
    # Create a flattened dataframe that contains information about the
    # incumbents of the various line-searches. The end Timestamps here
    # should exactly match one of the entries in the flattened budget
    # dataframe.
    df = _flatten(df, ['all_losses', 'all_timestamps'], ['Loss', 'Timestamps'])
    df = _extract_start_end_timestamps(df)
    return df_budget, df

def _get_asha_budget_and_incumbents(df):
    # First, if there is a fresh column then the dataframe contains many duplicate entries
    # because the same result was returned many times to ASHA even though nother new was
    # run because this is a scenario with an iteration-based fidelity and we didn't want to
    # re-run the same thing at every single step of ASHA's trial evaluation.
    if 'fresh' in df.columns:
        df = copy.deepcopy(df[df['fresh']])
    # Extract the start and end timestamps (there is a list of timestamps for the history of
    # runs, but we only want the most recent one because it corresponds to the run reported
    # now.)
    df['Timestamps'] = df['all_timestamps'].map(lambda t: eval(t)[-1])
    df = _extract_start_end_timestamps(df)
    # Rename the other columns
    df = df.rename({'configuration': 'Configuration',
                    'current_runtime': 'Runtime',
                    'current_fidelity_budget': 'Fidelity Budget',
                    'current_loss': 'Loss'}, axis=1)
    # make a copy of the dataframe to make the budget dataframe, in the format needed
    # for the analysis.
    df_budget = copy.deepcopy(df)
    df_budget = df_budget[['Configuration', 'Runtime', 'Fidelity Budget', 'Timestamp End']]
    df_budget = df_budget.rename({'Timestamp End': 'Timestamp'}, axis=1)
    # make a copy of the dataframe to make the "incumbent" dataframe, in the format needed
    # for the analysis. Not that here the "incumbent" dataframe really just means all of
    # the configurations, because there is no line-search-style behaviour going on to
    # pre-eliminate any of these configurations. Hence, the only reason to have these
    # separate now is so that the analysis code can handle the data in the same format
    # as for GPS-ASHA.
    df = df[['Configuration', 'Loss', 'Timestamp End', 'Timestamp Start']]
    return df_budget, df


def analyze_results(trial_analysis, objective='mean', mode='min',
                    gpsls=True, min_run_ratio=4, asha_original=False):
    """analyze_results

    Gets the anytime incumbent trajectory for a run of GPSLS+BF-ASHA, CQA+BF-ASHA or
    random search + ASHA, and also results some stats about the overall
    configuration procedure.
    
    Parameters
    ----------
    trial_analysis : ray.tune.Analysis
      The raw analysis of the configurator run (see ray.tune.Analysis).
    objective : 'mean' | 'final'
      Specifies whether the goal is to optimize the mean of the metric or the
      final value observed (i.e., the value of metric observed with the highest
      fidelity budget).
    mode : 'min' | 'max'
      Specifies whether the metric is to be minimized or maximized. Currently
      only 'min' is supported.
    gpsls : bool
      If True, the configurator run must have been GPS [BF-]ASHA. If False, the
      configurator run must have been ASHA and the GPS ASHA trial runner
      wrapper must have been used as well.
    min_run_ratio : int
      If objective == 'mean', specifies the minimum ratio of runs (challenger
      to incumbent) that is allowed for a challenger to be considered as the
      new incumbent. Not used for objective == 'final'.
    asha_original : bool
      If True and objective == 'mean', then the original ASHA incumbent
      selection mechanism is used even though it was not designed for this 
      type of objective. That is, any configuratin, regardless of the fidelity,
      can be selected as the incumbent if it has the best solution quality.
      If objective == 'final', ignored.
    
    Returns
    -------
    incumbent_trajectory: pandas.DataFrame 
      A dataframe containing the anytime incumbent trajectory of the run.
      Contains the columns:
        'Loss' : The estimated value of the metric to be optimized as of the
            selection of the configuration as the incumbent.
        'Wallclock Time' : The amount of wallclock time elapsed since the first
            configuration run began.
        'Target Algorithm Time': GPS ASHA's wrapper will estimate the wallclock
            time spent evaluating each of the configurations. This will be the
            sum of all such wallclock times. This may be larger than 
            'Wallclock Time' if the configurator has more than one worker.
        'Total Fidelity Budget Spent' : The sum of the fidelity budget spent
            as of the time of the incubment's selection.
        'Maximum Fidelity Budget' : The maximum fidelity budget on which the
            incumbent configuration was evaluated. Note that this may be a 
            larger fidelity budget than used to calculate the entry in the
            'Loss' column.
        'Configuration' : The current incumbent configuration.
    final_results : SimpleNamepace
      Contains the final incumbent configuration and the details of the total
      budget spent.
    """
    if not ray_available:
        raise ImportError('Requires ray.tune. Please install it first.')
    if mode != 'min':
        raise NotImplementedError('Currently only mode="min" is supported. Provided {}'.format(mode))
    df = _get_trial_results(trial_analysis, objective=='final', gpsls)
    if df is None:
        raise ValueError('No data was contained in the provided trial analysis.')
    # depending on the configurator, the data will be in a different format and may need
    # to be flattened.
    if gpsls:
        df_budget, df = _get_gpsls_budget_and_incumbents(df)
    else:
        df_budget, df = _get_asha_budget_and_incumbents(df)
    # Now that the formats are the same we can extract the actual anytime incumbent trajectory.
    df_traj = _get_anytime_trajectory(
        df, df_budget, objective, min_run_ratio=min_run_ratio, asha_original=asha_original)
    # Also extract the final fidelity budgets for each of the incumbents
    df_traj = _get_final_fidelity_budgets(df_budget, df_traj, objective)
    # Also extract from this the final results
    final_results = SimpleNamespace(
        incumbent=df_traj.tail(1)['Configuration'].iloc[0],
        total_fidelity_spent=df_budget['Fidelity Budget'].sum(),
        total_wallclock_time=df['Timestamp End'].max(),
        total_target_time=df_budget['Runtime'].sum())
    # And return them both
    return df_traj, final_results


def _get_final_fidelity_budgets(df_budget, df_traj, objective):
    if objective == 'final':
        df_final_budget = df_budget.sort_values('Timestamp').groupby('Configuration').tail(1)
    else:
        df_final_budget = df_budget.sort_values(['Configuration', 'Timestamp'])
        df_final_budget['Fidelity Budget'] = df_final_budget.groupby('Configuration')['Fidelity Budget'].cumsum()
        df_final_budget = df_final_budget.groupby('Configuration').tail(1)
    df_final_budget = df_final_budget[df_final_budget['Configuration'].isin(df_traj['Configuration'])]
    df_result = df_traj.merge(df_final_budget[['Configuration', 'Fidelity Budget']], on='Configuration', how='inner')
    return df_result.rename({'Fidelity Budget': 'Maximum Fidelity Budget'}, axis=1)
    
