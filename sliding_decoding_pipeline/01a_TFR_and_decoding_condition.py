#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:13:36 2020
@author: vv221713


"""
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
import eeg_decoding as decoding

from sklearn.utils import shuffle


#%% __________ [1] Define directories
decoding_dir    = f'{config.stat_dir}/auc_time_courses/'
decoding_report_dir  = f'../reports_auc_time_courses/'

subjects_list    = config.preprocessed_subjects_list                # list of participants (sid)
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

#Select the classifier used for decoding
classifier = "logistic"

#%%============================================================================

# Takes the epochs as input and returns the power time course within given frequencies as an array  number of epochs x number of channels x times
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

for i, subject in enumerate(subjects_list[2:]):

    #  Load, TFR transform and power time course of the epochs
    print(f'Processing: {subject}')
    for conditions in config.condition_pairs:
        report_out  = decoding_report_dir + f'{subject}_{"_".join(conditions)}_Condition_TFR_{classifier}.html'
        os.makedirs(decoding_report_dir, exist_ok=True)
        report      = Report(report_out, verbose=False)
        iepos_list  = []
    
        for condition in conditions:
            condition_epochs = preprocessing.load_subject_epochs_by_type_and_condition(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)  
            
            #____ Retrieve the power time course of the epochs as an epochs object
            iepos = get_power_epochs(condition_epochs, foi_list[foi][0], foi_list[foi][1])
            iepos.average().plot(show=True, window_title=condition)
            
            # create list so we can use the equalize count 
            iepos_list.append(iepos)
      
    # DECODING
    #____________________________________________________
        
        
        #________________
        # X matrix
        concat_data = np.concatenate([iepos.get_data() for iepos in iepos_list])
        print(concat_data.shape)
            
        # labels
        y = np.concatenate([np.repeat(i, len(iepos_list[i])) for i in range(len(iepos_list))])
        print(y.shape)
        
        concat_data,y = shuffle(concat_data,y)
        scores = decoding.compute_decoding_scores(eeg_data=concat_data, labels=y, decoder=classifier, cv=4, ncores=-1)
        
        #___ save np arrays scores
        os.makedirs(decoding_dir, exist_ok=True)
        np.save(decoding_dir + f'{subject}_{"_".join(conditions)}_Condition_TFR_{classifier}.npy', scores)

        fig, ax = decoding.plot_decoding_scores(
            decoding_scores=scores,
            epochs=iepos_list[0],
            end_stim=0,
            plot_title = f"Sensor space decoding on {pick_type} channels on {epochs_type} epochs ({conditions[0]} vs. {conditions[1]})",
            plotting_theme="ticks"
        )
        
        report.add_figure(fig, title = f"Sensor space decoding on {pick_type} channels on {epochs_type} epochs ({conditions[0]} vs. {conditions[1]})") 
        os.makedirs(decoding_dir, exist_ok=True)
        report.save(fname=report_out, open_browser=False, overwrite=True)
    

# %%
