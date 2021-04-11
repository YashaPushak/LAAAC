import glob

import numpy as np
import pandas as pd
try:
    import altair as alt
    import streamlit as st
    plot_available = True
except ImportError:
    plot_available = False

from ray.tune.analysis import Analysis

from laaac import results


trajectories = []
for optimizer in ['GPSLS', 'CQA']:
    for run in glob.glob(f'./laaac-output/{optimizer}/*/'):
        run_id = run.split('/')[-2]
        # Load the ray.tune results
        run_result = Analysis(run)
        # Analyze those results according the the optimizer
        incumbent_trajectory, final_result = results.analyze_results(run_result, gpsls=optimizer=='GPSLS')
        # Print some data about them
        print('*'*10 + f'{optimizer} Run {run_id}' + '*'*10)
        print('Final Configuration:')
        print(final_result.incumbent)
        print(f'Total Configuration Budget Spent: {final_result.total_fidelity_spent}')
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 20):
            print(f'Incumbent Trajectory: \n{incumbent_trajectory}')
        # Add a couple extra columns to the result for plotting
        incumbent_trajectory['Optimizer'] = optimizer
        incumbent_trajectory['Run ID'] = run_id
        trajectories.append(incumbent_trajectory)

if plot_available:
    # Create a single dataframe with all results
    trajectories = pd.concat(trajectories)
    trajectories['Loss'] = np.clip(trajectories['Loss'], 0.00001, None)
    selector = alt.selection_multi(fields=['Optimizer'], bind='legend')
    chart = alt.Chart(trajectories).mark_line(point=True, interpolate='step-after').encode(
        x=alt.X('# Target Algorithm Runs', scale=alt.Scale(domain=(0, 100))),
        y=alt.Y('Loss', title='Esimtated Loss from Optimizer', scale=alt.Scale(domain=[0.00001,1], type='log')),
        color='Optimizer',
        detail='Run ID',
        opacity=alt.condition(selector, alt.value(1), alt.value(0.2)),
    ).add_selection(selector)
    st.write(chart)
