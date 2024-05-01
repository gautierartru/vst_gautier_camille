#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% __________ [0] Imports
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.channels import read_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test,linear_regression, fdr_correction
from mne.viz import (plot_topomap, plot_evoked_topo, plot_compare_evokeds)
from mne.report import Report
from mne.epochs import equalize_epoch_counts
from mne.time_frequency import (
    tfr_array_morlet,
)

from mne.parallel import parallel_func
import copy

import pandas as pd

import os
import sys
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\scripts")
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\config")
import vst_config as config
import eeg_preprocessing as preprocessing


#%% __________ [1] Define directories
stat_dir    = f'{config.stat_dir}/regressions/'
report_dir  = f'../reports/'

subjects_list    = config.preprocessed_subjects_list[4:]                # list of participants (sid)
n_subjects       = len(subjects_list)             # number of participants

#_____  colors
Scmap = plt.cm.Greens
Tcmap = plt.cm.Purples
Scol  = [Scmap(100),Scmap(160), Scmap(220)]
Tcol  = [Tcmap(100),Tcmap(160), Tcmap(220)]

#____ frequencies of interest (FOI)
foi_names   = ['delta', 'theta','alpha','beta','Lgamma','Hgamma']
foi_list    = {'delta':   (1, 3),
               'theta':   (4, 7),  
               'alpha':   (8, 12), 
               'beta':    (14, 30), 
               'Lgamma':  (35,60), 
               'Hgamma':  (60, 119)}

#%%============================================================================
# --------------------------- YOUR SELECTION NEEDED ---------------------------
#==============================================================================

foi = 'beta'
condition = 'SNS'
epochs_type = "start_prod"
baseline_duration = 0.1
epoch_duration = 4
pick_type = "frontocentral"

# Number of standard deviations to define the categories of duration produced
n_std = 1

#%%============================================================================

# Takes the epochs as input and returns the power time course within given frequencies as an epochs array number of epochs x number of channels x times
def get_power_epochs(epochs, power_freq_low_bound, power_freq_high_bound):
    frequencies = np.arange(power_freq_low_bound, power_freq_high_bound)  
    cycles_per_freq = 6
    
    power = tfr_array_morlet(epochs.get_data(), freqs=frequencies, n_cycles=cycles_per_freq, output="power", sfreq=config.resample_sfreq)
    power_timecourse = np.mean(power, axis=2)

    # Transform to epochs and keep metadata
    epochs_power_timecourse = mne.EpochsArray(power_timecourse, info=epochs.info)
    epochs_power_timecourse.metadata = epochs.metadata

    return epochs_power_timecourse


#%%============================================================================
# Separate the epochs by produced duration: shorter, normal, longer than the mean of the distribution of produced durations

def produced_duration_category(epochs):
    produced_duration = epochs.metadata["duration_produced"]
    
    shorter = produced_duration < (produced_duration.mean() - n_std * produced_duration.std())
    normal = (produced_duration >= (produced_duration.mean() - n_std * produced_duration.std())) & (produced_duration <= (produced_duration.mean() + n_std * produced_duration.std()))
    longer = produced_duration > (produced_duration.mean() + n_std * produced_duration.std())
    produced_duration_category = np.zeros(len(produced_duration))
    produced_duration_category[shorter] = -1
    produced_duration_category[normal] = 0
    produced_duration_category[longer] = 1
    return produced_duration_category


#%%============================================================================

for i, subject in enumerate(subjects_list):

    #  Load, TFR transform and power time course of the epochs
    print(f'Processing: {subject}')
    for condition in config.conditions:
        report_out  = report_dir + f'{subject}_{condition}_Duration_Produced_std_{n_std}_TFR_ireg.html'
        os.makedirs(report_dir, exist_ok=True)
        report      = Report(report_out, verbose=False)
    
        condition_epochs = preprocessing.load_subject_epochs_by_type_and_condition(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)  
        
        #____ Retrieve the power time course of the epochs as an epochs object
        iepos = get_power_epochs(condition_epochs, foi_list[foi][0], foi_list[foi][1])
        iepos.average().plot(show=True, window_title=condition)
      
    # REGRESSION
    #____________________________________________________
        pred_vars = ['Duration_Produced', 'Intercept']
            
        #________________ design matrix should be (n_observations, n_regressors)
        prod_time_refs = produced_duration_category(iepos)
        
        df = {'Duration_Produced': prod_time_refs,
              'Intercept': np.ones(iepos.__len__())}
        design = pd.DataFrame.from_dict(df)   
        
        #________________ run separate regressions for induced power
        ireg = linear_regression(iepos, design_matrix=design, names=pred_vars)
        
        #___ save for pop level stats
        f_out_duration = stat_dir + subject + "_"+ pick_type + "_" + epochs_type + "_" + condition + f'_TFR_std_{n_std}_Duration_Produced_beta_reg_' + foi + 'ipower_betas.fif'
        os.makedirs(stat_dir, exist_ok=True)
        ireg['Duration_Produced'].beta.save(f_out_duration, overwrite = True)

        f_out_intercept = stat_dir + subject + "_"+ pick_type + "_" + epochs_type + "_" + condition + f'_TFR_std_{n_std}_Duration_Produced_Intercept_reg_' + foi + 'ipower_betas.fif'
        os.makedirs(stat_dir, exist_ok=True)
        ireg['Intercept'].beta.save(f_out_intercept, overwrite = True)

        #Save figures of the condition beta and condition t_val

        #For the produced duration predictor
        fig_beta = ireg['Duration_Produced'].beta.plot_joint(title='Effect of the produced duration', show=False)
        report.add_figure(fig_beta, title = f'{subject}_{condition}_regression_on_produced_duration_beta')  
        topomap_args = dict(scalings=dict(eeg=1))

        fig_t_val = ireg['Duration_Produced'].t_val.plot_joint(
            ts_args=dict(unit=False), topomap_args=topomap_args, show=False
        )
        fig_t_val.axes[0].set_ylabel("T-value")
        report.add_figure(fig_t_val, title = f'{subject}_{condition}_regression_on_produced_duration_t_value') 

        #For the intercept predictor
        fig_beta = ireg['Intercept'].beta.plot_joint(title='Intercept', show=False)
        report.add_figure(fig_beta, title = f'{subject}_{condition}_regression_intercept')  
        topomap_args = dict(scalings=dict(eeg=1))

        fig_t_val = ireg['Intercept'].t_val.plot_joint(
            ts_args=dict(unit=False), topomap_args=topomap_args, show=False
        )
        fig_t_val.axes[0].set_ylabel("T-value")
        report.add_figure(fig_t_val, title = f'{subject}_{condition}_regression_intercept_t_value') 


        report.save(fname=report_out, open_browser=False, overwrite=True)
    

# %%
