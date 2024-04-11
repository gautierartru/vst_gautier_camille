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
from mne.stats import linear_regression, fdr_correction, spatio_temporal_cluster_1samp_test 
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

#%% __________ [0] SELECT + VERIFY your options
# linear regressions variables
pred_vars   = ['Duration_Produced_beta']

#%% __________ [1] Define directories
stat_dir    = f'{config.stat_dir}/regressions/'
report_dir  = f'{config.stat_dir}/reports/'

subjects_list    = config.preprocessed_subjects_list                # list of participants (sid)
n_subjects       = len(subjects_list)  


foi = 'beta'
epochs_type = "start_prod"
baseline_duration = 0.1
epoch_duration = 4
pick_type = "frontocentral"

# whether we used TFR or Hilbert to compute beta power. Choose the number of standard deviations used for duration categories
beta_power_computation = 'TFR_std_1'


# for future vizualisation, load an epochs object
epochs = preprocessing.load_subject_epochs_by_type_and_condition(subjects_list[0], 'SNS', epochs_type, baseline_duration, epoch_duration, pick_type, verbose=False)

from mne.channels import find_ch_adjacency
adjacency, _ = find_ch_adjacency(epochs.info, "eeg")

#%% __________ [2] STATS

p_accept    = .001
n_perm      = 1024 
   
#%___ load individual betas / intercept
for ic, condition in enumerate(config.conditions):

    report_out  = report_dir + f'popreg_Duration_Produced_{condition}.html'
    report      = Report(report_out, verbose=False)      

    for isub, subject in enumerate(subjects_list):
        
        for ip,p in enumerate(pred_vars):

            beta_in  = stat_dir + subject + "_"+ pick_type + "_" + epochs_type + "_" + condition + f'_{beta_power_computation}_{p}_reg_' + foi + 'ipower_betas.fif'           
            beta     = mne.read_evokeds(beta_in)[0].pick('eeg')
            
            # init or append
            if isub==0 and ip==0 and ic==0: 
                betas = np.zeros((n_subjects, len(pred_vars), len(config.conditions), beta.data.shape[0], beta.data.shape[1]))
                betas[isub,ip, ic] = beta.data
            else: 
                betas[isub,ip, ic] = beta.data

    #%___ plot vizu in Report
    for ip,p in enumerate(pred_vars):
        #selecting only the first condition pair
        betas_gdav      = np.mean(betas[:,ip,ic,:,:],0)
        betas_gdav_evo  = mne.EvokedArray(betas_gdav, epochs.info)
        
        figure          = betas_gdav_evo.plot_joint(title=p, 
                                                    ts_args=dict(time_unit='s'),
                                            topomap_args=dict(time_unit='s'))    
        
        report.add_figure(figure, title= f'Beta Av {condition} (N={n_subjects}) {p}')   
        plt.close()
    
    #%___ spatio-temporal cluster of regression predictors
    p_thres     = 0.05
    thres       = - stats.distributions.t.ppf(p_thres / 2., n_subjects - 1)

    for ip,p in enumerate(pred_vars):
    
        # obs x times x sensors
        tbetas        = np.transpose(betas[:,ip,ic,:,:], (0, 2, 1))
    
        cluster_stats = spatio_temporal_cluster_1samp_test(tbetas, 
                                                        n_permutations=n_perm, 
                                                        threshold=None, 
                                                        n_jobs=-1, 
                                                        adjacency=adjacency)        
        T_obs, clusters, p_values, H0 = cluster_stats
        if len(clusters) == 0:
            print('No significant clusters found for this predictor :' + p)
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
        #% loop over clusters, plot, save in report and CSV
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds     = np.unique(space_inds)
            time_inds   = np.unique(time_inds)
        
            # get topography for F stat
            f_map = T_obs[time_inds, ...].mean(axis=0)
        
            # get signals at the sensors contributing to the cluster
            # sig_times = blah.times[time_inds]
            sig_times = betas_gdav_evo.times[time_inds]
            
            # create spatial mask
            mask = np.zeros((f_map.shape[0], 1), dtype=bool)
            mask[ch_inds, :] = True
        
            # initialize figure
            fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
        
            # plot average test statistic and mark significant sensors
            f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
            
            f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Greys',
                                show=False,
                                colorbar=False, mask_params=dict(markersize=10))
            image = ax_topo.images[0]
            
            # create additional axes
            divider = make_axes_locatable(ax_topo)
        
            # add axes for colorbar
            ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(image, cax=ax_colorbar)
            ax_topo.set_xlabel(
                'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))
        
            # add new axis for time courses and plot time courses
            ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            
            title = 'Cluster #{0}, {1} sensor, p min value {2}'.format(i_clu + 1, len(ch_inds), p_min)
            
            plot_compare_evokeds([betas_gdav_evo], ci = True, title=title, 
                                picks=ch_inds, axes=ax_signals,
                                legend = 'upper right', split_legend=True, 
                                truncate_yaxis='auto')
        
            # plot temporal cluster extent
            ymin, ymax = ax_signals.get_ylim()
            ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                    color='grey', alpha=0.3)
        
            # clean up viz
            fig.subplots_adjust(bottom=.05)

            report.add_figure(fig, title=('sign cluster ' + p + '_' + str(p_min)))        
            fig.savefig(report_dir + f'N{n_subjects}_{p}_sigclust.pdf')
            plt.close()
            
            report.save(fname=report_out, open_browser=False, overwrite=True)    
            
            #==========================================
            # save cluster info to CSV to make publishable figures
            #==========================================
            if p_min < p_accept:
                df          = pd.DataFrame(columns = ['channels', p, 'value', 'latency'])               
                df['channels']    = ch_inds
                df['latency'].loc[0] = sig_times[0]
                df['latency'].loc[1] = sig_times[-1]
                
                tmask = np.zeros((betas_gdav_evo.data.shape[1]), dtype=bool)
                tmask[time_inds] = True
                
                for idx, ch in enumerate(ch_inds):        
                    df['value'].loc[idx] = np.mean(betas_gdav_evo.data[ch, tmask])
                    df_name = report_dir + f'N{str(n_subjects)}_reg_sigclust{str(i_clu+1)}_p{str(p_accept)}.csv'
                df.to_csv(df_name)         
            # #==========================================
#%% __________