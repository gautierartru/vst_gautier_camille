# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:18:32 2023
@author: VV221713

Virginie.van.Wassenhove@gmail.com
=================
group level 

=================
March 30th 2024: Simplified for VST project Camille, Gautier

"""
import pandas as pd
import numpy as np
from scipy import stats  

import mne
from mne.stats import linear_regression, fdr_correction, spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test
from mne.viz import plot_compare_evokeds, plot_evoked_topo
from mne.channels import read_ch_adjacency
from mne.report import Report

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

import os
import sys
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\scripts")
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\config")
import vst_config as config
import eeg_preprocessing as preprocessing  
import eeg_decoding as decoding

#%% __________ [0] SELECT + VERIFY your options
# linear regressions variables

#%% __________ [1] Define directories
decoding_dir    = f'{config.stat_dir}/auc_time_courses/'
decoding_report_dir  = f'..reports_auc_time_courses/'

subjects_list    = config.preprocessed_subjects_list                # list of participants (sid)
n_subjects       = len(subjects_list)  


foi = 'beta'
epochs_type = "start_prod"
baseline_duration = 0.1
epoch_duration = 4
pick_type = "frontocentral"

# whether we used TFR or Hilbert to compute beta power
beta_power_computation = 'TFR'

# Select classifier
classifier = 'logistic'
metric_multilabel = "f1_macro"


# for future vizualisation, load an epochs object
epochs = preprocessing.load_subject_epochs_by_type_and_condition(subjects_list[0], 'SNS', epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)

from mne.channels import find_ch_adjacency
adjacency, _ = find_ch_adjacency(epochs.info, "eeg")

#%% __________ [2] STATS

p_accept    = .001
n_perm      = 1024   
   
#%___ load individual betas / intercept
for ic, condition in enumerate(config.conditions):
    report_out  = decoding_report_dir + f'popauc{condition}_Duration_Produced_{metric_multilabel}.html'
    report      = Report(report_out, verbose=False)    

    for isub, subject in enumerate(subjects_list[:7]):

        auc_in  = decoding_dir + subject + "_" + condition + f'_Duration_Produced_TFR_{classifier}_{metric_multilabel}.npy'           
        auc     = np.load(auc_in)
        auc = np.mean(auc, axis=0)
        
        # init or append
        if isub==0 and ic==0: 
            aucs = np.zeros((len(subjects_list[:7]), len(config.conditions), auc.data.shape[0]))
            aucs[isub,ic] = auc
        else: 
            aucs[isub, ic] = auc

    #%___ plot vizu in Report
    
    aucs_gdav      = np.mean(aucs[:,ic,:],0)
    
    figure, ax          = decoding.plot_decoding_scores(aucs_gdav, epochs, plot_title=f'Accuracy Av (N={n_subjects}) {condition} Duration Produced {metric_multilabel}', plotting_theme="ticks")  
    print('plotting')
    report.add_figure(figure, title= f'Accuracy Av (N={n_subjects}) {condition} Duration Produced {metric_multilabel}')   
    figure, ax          = decoding.plot_decoding_scores(aucs[0,0], epochs, plot_title=f'test', plotting_theme="ticks")  

    report.add_figure(figure, title= f'test one subject') 
    plt.close()
    
    #%___ spatio-temporal cluster of regression predictors
    # p_thres     = 0.05
    # thres       = - stats.distributions.t.ppf(p_thres / 2., n_subjects - 1)

    # obs x times x sensors
    aucs_condition        = aucs[:,ic,:]

    cluster_stats = permutation_cluster_1samp_test(aucs_condition - 0.3, n_permutations=n_perm, n_jobs=-1, seed=42, threshold=0.05, tail=1)        
    T_obs, clusters, p_values, H0 = cluster_stats
    if len(clusters) == 0:
        print('No significant clusters found')
        continue
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(str(len(good_cluster_inds)) + ' sign. clusters')

    p_min = np.min(p_values)
    
    #======================================================
    # if no cluster, display the lowest p of n.s. cluster (as sanity check)
    #======================================================
    if len(good_cluster_inds) == 0:
        good_cluster_inds = np.where(p_values < p_min+0.01)[0]
        print(str(len(good_cluster_inds)) + ' n.s. cluster min p')

    #======================================================
    #% loop over clusters, plot, save in report
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        
        # unpack cluster information, get unique indices
        time_inds   = clusters[clu_idx][0]
        
        title = 'Cluster #{0}, p min value {1}'.format(i_clu + 1, p_min)
        
        # plot aucs evo
        fig, ax_signals = plt.subplots(1, 1, figsize=(15, 7))
        ax_signals.plot(aucs_gdav, color='black')
    
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), time_inds[0], time_inds[-1],
                                color='grey')
            
        report.add_figure(fig, title=('sign cluster ' + '_' + str(p_values[clu_idx])))
        plt.close()
        
    report.save(fname=report_out, open_browser=False, overwrite=True)
#%% __________