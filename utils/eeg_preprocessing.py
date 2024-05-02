#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November, 17, 2023
@author: Camille Grasso
"""

import os
import os.path as op
import numpy as np
import statistics
import matplotlib.pyplot as plt
import glob
import pandas as pd
import mne
from mne import create_info
from mne.io import read_raw_fif, RawArray
from mne.preprocessing import ICA
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator, apply_inverse, make_inverse_operator
import csv
import vst_config as config
import json

###########
# 
###########

###########
# Import data
###########
def import_data(subject):
     # importing the raw data
    path_in = config.raw_data_paths[subject]
    raw = mne.io.read_raw_brainvision(f'{config.work_dir}/{path_in}', preload=True, verbose=False)
    
    # Remove Stimulus/255 annotation
    indices_to_remove = [idx for idx, ann in enumerate(raw.annotations) if ann['description'] == "Stimulus/s255"]
    raw.annotations.delete(indices_to_remove)

    raw.annotations.rename(config.mapped_annotations)

    if subject[-1] == "Y":
        inv_annotations = {v: v[:-1] + "S" if v[-1] == "L" else v[:-1] + "L" if v[-1] == "S" else v for k, v in config.mapped_annotations.items()}
        print(inv_annotations)
        raw.annotations.rename(inv_annotations)

    # define the montage for processing
    montage = mne.channels.read_custom_montage(f'{config.work_dir}/VST_Data/eeg data/MontageAntneuro2.bvef')
    raw.set_montage(montage, on_missing="warn", verbose=False)
    raw.set_channel_types({'EOG': 'eog'})

    return raw

###########
# Filtering and resampling raw data
###########
def filter_and_resample_raw_data(raw):

    # filtering the signal
    raw.filter(l_freq=config.f_min_filtering, h_freq=config.f_max_filtering, verbose=False)

    # resampling
    raw.resample(config.resample_sfreq, verbose=False)

#############
## Exclude Bad Channels
#############
def exclude_bad_channels(raw, subject):
    raw.info['bads'] = config.bad_channels_per_sub[subject]
    return raw

###########
## Perform ICA
###########
def perform_ICA(raw, subject):
    ica_directory = "ica_solutions/"
    file_name = subject + "-ica.fif"
    ica = mne.preprocessing.read_ica(f"{config.work_dir}/{ica_directory+file_name}")
    ica.exclude = config.ica_exclude[subject]
    ica.apply(raw)
    return raw

#########
# Returns a list of the "good" blocks of the experiment (there should be 8 for each subject): each block is a Raw object
#
# The preprocessing consists of:
# - importing the data and montage
# - filtering
# - resampling 
# - exclusion of bad channels 
# - ICA
#########
def get_preprocessed_raw(subject, bad_channels_exclusion, ICA_exclusion):
    # Importing data
    raw = import_data(subject)

    print("[DONE] Importing data")

    # Filtering and resampling in the first place to achieve shorter computations
    filter_and_resample_raw_data(raw)

    print("[DONE] Filtering and resampling")
    
    # Exclusion of bad channels
    if bad_channels_exclusion:
        exclude_bad_channels(raw, subject)
        print("[DONE] Excluding bad channels")

    # Exclusion of bad channels
    if ICA_exclusion:
        perform_ICA(raw, subject)
        print("[DONE] Performing ICA")

    return raw

#########
# Fix badly labeled annotations
#########
def solve_annotations_mismatch(blocks, subject):
    # The participant VST_02_X confused LHS and LHL in the first block
    if subject == "VST_02_X":
        blocks[0].annotations.rename({
            "start_trial_LHL":"start_trial_LHS", 
            "go_signal_LHL":"go_signal_LHS",
            "start_prod_LHL":"start_prod_LHS",
            "end_prod_LHL":"end_prod_LHS",
            "start_trial_LHS":"start_trial_LHL", 
            "go_signal_LHS":"go_signal_LHL",
            "start_prod_LHS":"start_prod_LHL",
            "end_prod_LHS":"end_prod_LHL",
            })




#########
# Returns a list of the "good" blocks of the experiment (there should be 8 for each subject): each block is a Raw object
#########
def get_good_raw_blocks(raw, subject, threshold = 100):
    ## retrieving events 
    events, revt_id = mne.events_from_annotations(raw, config.evts_id, verbose=False)

    good_blocks = []

    # 666 event id corresponds to a new segment, ie a new block
    new_segment_indices = list(np.append(np.where(events==666)[0], events.shape[0]))
    print(new_segment_indices)

    good_index = 1
    for i in range(len(new_segment_indices)-1): 
        raw_copy = raw.copy()
        start_time_block = raw_copy.times[events[new_segment_indices[i] + 1][0]]
        end_time_block = raw_copy.times[events[new_segment_indices[i+1] - 1][0]]
        # Check if the block is long enough to be kept
        if end_time_block - start_time_block > threshold:
            good_blocks.append(raw_copy.crop(tmin=start_time_block, tmax=end_time_block))
            good_index += 1
    
    print("[DONE] Extracting good blocks")

    solve_annotations_mismatch(good_blocks, subject)

    print("[DONE] Solving annotations mismatch")

    return good_blocks


###########
# Create epochs
###########

# Create a single epoch from an event with the baseline calculated from baseline_duration before another event.
# The baseline_event must be BEFORE the reference_event
# Retruns MNE Epochs with one epoch
def compute_single_epoch_with_baseline_on_second_event(raw, reference_event, baseline_event, epoch_duration, baseline_duration, sfreq):

    # Retrieve data from raw, from baseline_event - baseline_duration to reference_event + epoch_duration
    start_sample = baseline_event[0] - int(baseline_duration*sfreq)
    stop_sample = reference_event[0] + int(epoch_duration*sfreq)
    single_epoch_data = raw.get_data(start=start_sample, stop=stop_sample)

    # Perform baseline taking (baseline_event - baseline_duration, baseline_event) as baseline
    baseline = np.mean(single_epoch_data[:, :int(baseline_duration*sfreq)], axis=1, keepdims=True)
    single_epoch_data = single_epoch_data - baseline

    #Crop data from reference_event to epoch_duration
    single_epoch_data = single_epoch_data[:,-int(epoch_duration*sfreq):]

    # Reshape data to get good data
    single_epoch_data = single_epoch_data.reshape(1, single_epoch_data.shape[0], single_epoch_data.shape[1])

    #create new epoch with appropriate information and the computed data
    info = raw.info

    # Not putting any event related information in the computed epoch
    single_epoch = mne.EpochsArray(
                            data = single_epoch_data, 
                            info = info,
                            verbose=False
                            )
    
    return single_epoch


# Create a single epoch from an event
# Retruns MNE Epochs with one epoch
def compute_single_epoch_with_baseline(raw, reference_event, epoch_duration, baseline_duration, sfreq):

    # Retrieve data from raw, from reference_data - baseline_duration to reference_event + epoch_duration
    start_sample = reference_event[0] - int(baseline_duration*sfreq)
    stop_sample = reference_event[0] + int(epoch_duration*sfreq)
    single_epoch_data = raw.get_data(start=start_sample, stop=stop_sample)

    # Perform baseline taking (baseline_event - baseline_duration, baseline_event) as baseline
    baseline = np.mean(single_epoch_data[:, :int(baseline_duration*sfreq)], axis=1, keepdims=True)
    single_epoch_data = single_epoch_data - baseline

    #Crop data from reference_event to epoch_duration
    single_epoch_data = single_epoch_data[:,-int(epoch_duration*sfreq):]

    # Reshape data to get good data
    single_epoch_data = single_epoch_data.reshape(1, single_epoch_data.shape[0], single_epoch_data.shape[1])

    #create new epoch with appropriate information and the computed data
    info = raw.info

    # Not putting any event related information in the computed epoch
    single_epoch = mne.EpochsArray(
                            data = single_epoch_data, 
                            info = info,
                            verbose=False
                            )
    
    return single_epoch


# Create start_prod epochs with a baseline before the go signal
# This is currently not possible only with built-in MNE python because it does not support different baselines when concatenating epochs
# Return a dict start_prod_epochs_dict = {condition:epochs_condition_block}
def get_start_prod_epochs(block, block_number, subject, picks, baseline_duration=0.1, epoch_duration=1):
    block_copy = block.copy()

    # Pick good channels
    block_copy.pick(picks=picks)

    # Delete bad start_prod and go_signal events 
    # Check whether there are bad events for this block
    if subject in config.bad_go_signal_start_prod_events.keys():
        if block_number in config.bad_go_signal_start_prod_events[subject]:
            start_prod_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "start_prod" in ann['description'] ]
            start_prod_indices_to_delete = [start_prod_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["start_prod"]]
            if start_prod_indices_to_delete != []:
                block_copy.annotations.delete(start_prod_indices_to_delete)

            end_prod_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "end_prod" in ann['description'] ]
            end_prod_indices_to_delete = [start_prod_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["end_prod"]]
            if end_prod_indices_to_delete != []:
                block_copy.annotations.delete(end_prod_indices_to_delete)

            go_signal_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "go_signal" in ann['description'] ]
            go_signal_indices_to_delete = [go_signal_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["go_signal"]]
            if go_signal_indices_to_delete != []:
                block_copy.annotations.delete(go_signal_indices_to_delete)

    # Retrieve events and events dict for start _prod, end_prod and go_signal
    start_prod_events_dict = {k:v for k,v in config.evts_id.items() if "start_prod" in k}
    start_prod_events, start_prod_events_dict_block = mne.events_from_annotations(block_copy, event_id=start_prod_events_dict, verbose=False)

    end_prod_events_dict = {k:v for k,v in config.evts_id.items() if "end_prod" in k}
    end_prod_events, end_prod_events_dict_block = mne.events_from_annotations(block_copy, event_id=end_prod_events_dict, verbose=False)

    go_signal_events_dict = {k:v for k,v in config.evts_id.items() if "go_signal" in k}
    go_signal_events, go_signal_events_dict_block = mne.events_from_annotations(block_copy, event_id=go_signal_events_dict, verbose=False)

    #The way MNE works here is a bit odd: when cropping a raw, events onset times do not change, 
    # which results in a misalignment between the cropped raw and the events.
    # However, first_samp and last_samp change as expected: first_samp updates from 0 to the first time sample in the raw from which it was cropped
    start_prod_events[:,0] = start_prod_events[:,0] - block_copy.first_samp
    end_prod_events[:,0] = end_prod_events[:,0] - block_copy.first_samp
    go_signal_events[:,0] = go_signal_events[:,0] - block_copy.first_samp

    #instantiate the start_prod_epochs_dict
    conditions = [event.split('_').pop() for event in start_prod_events_dict_block.keys()]
    start_prod_epochs_dict = {condition:[] for condition in conditions}

    for i in range(len(start_prod_events)):

        #Create start_prod epoch
        start_prod_epoch = compute_single_epoch_with_baseline_on_second_event(
                                                                        raw=block_copy, 
                                                                        reference_event = start_prod_events[i], 
                                                                        baseline_event=go_signal_events[i], 
                                                                        epoch_duration=epoch_duration, 
                                                                        baseline_duration=baseline_duration, 
                                                                        sfreq = config.resample_sfreq)

        #Add epoch to the corresponding condition
        event_description = list(start_prod_events_dict_block.keys())[list(start_prod_events_dict_block.values()).index(start_prod_events[i][2])]
        condition = event_description.split('_').pop()

        # Remove annotations
        start_prod_epoch.set_annotations(None)

        # Add duration produced as metadata
        start_prod_epoch.metadata = pd.DataFrame({'duration_produced': (end_prod_events[i][0] - start_prod_events[i][0])/config.resample_sfreq }, index = [0])
        start_prod_epochs_dict[condition].append(start_prod_epoch)

    for condition, epochs in start_prod_epochs_dict.items():
        start_prod_epochs_dict[condition] = mne.concatenate_epochs(epochs, on_mismatch='ignore', verbose=False).drop_bad(reject=config.reject_dict)
    
    return start_prod_epochs_dict



# Create go_signal epochs with a baseline before the go signal. 
# Returns:
# - a dict go_signal_epochs_dict = {condition:epochs_condition_block} 
# - a list surpass_events = [event] corresponding to events where the start_prod - go_signal duration < epoch_duration given. 
#   In this case, the epoch is still computed but the associated event is put in this list
def get_go_signal_epochs(block, block_number, subject, picks, baseline_duration=0.1, epoch_duration=0.1):
    block_copy = block.copy()

    # Pick good channels
    block_copy.pick(picks=picks)

    # Delete bad start_prod and go_signal events 
    if subject in config.bad_go_signal_start_prod_events.keys():
        if block_number in config.bad_go_signal_start_prod_events[subject]:
            start_prod_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "start_prod" in ann['description'] ]
            start_prod_indices_to_delete = [start_prod_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["start_prod"]]
            if start_prod_indices_to_delete != []:
                block_copy.annotations.delete(start_prod_indices_to_delete)

            end_prod_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "end_prod" in ann['description'] ]
            end_prod_indices_to_delete = [end_prod_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["start_prod"]]
            if end_prod_indices_to_delete != []:
                block_copy.annotations.delete(end_prod_indices_to_delete)

            go_signal_indices = [idx for idx, ann in enumerate(block_copy.annotations) if "go_signal" in ann['description'] ]
            go_signal_indices_to_delete = [go_signal_indices[i] for i in config.bad_go_signal_start_prod_events[subject][block_number]["go_signal"]]
            if go_signal_indices_to_delete != []:
                block_copy.annotations.delete(go_signal_indices_to_delete)

    # Retrieve events and events dict for start_prod and go_signal
    start_prod_events_dict = {k:v for k,v in config.evts_id.items() if "start_prod" in k}
    start_prod_events, start_prod_events_dict_block = mne.events_from_annotations(block_copy, event_id=start_prod_events_dict, verbose=False)
    go_signal_events_dict = {k:v for k,v in config.evts_id.items() if "go_signal" in k}
    go_signal_events, go_signal_events_dict_block = mne.events_from_annotations(block_copy, event_id=go_signal_events_dict, verbose=False)

    #The way MNE works here is a bit odd: when cropping a raw, events onset times do not change, 
    # which results in a misalignment between the cropped raw and the events.
    # However, first_samp and last_samp change as expected: first_samp updates from 0 to the first time sample in the raw from which it was cropped
    start_prod_events[:,0] = start_prod_events[:,0] - block_copy.first_samp
    go_signal_events[:,0] = go_signal_events[:,0] - block_copy.first_samp

    conditions = [event.split('_').pop() for event in go_signal_events_dict_block.keys()]
    go_signal_epochs_dict = {condition:[] for condition in conditions}
    
    surpass_events = []

    for i in range(len(go_signal_events)):
        # Check whether start_prod - go_signal duration < epoch_duration
        delta_between_go_and_start = block_copy.times[start_prod_events[i][0]]-block_copy.times[go_signal_events[i][0]]
        if delta_between_go_and_start < epoch_duration:
            surpass_events.append(go_signal_events[i])

        # Create go_signal_epoch
        go_signal_epoch = compute_single_epoch_with_baseline(
                                                        raw=block_copy, 
                                                        reference_event = go_signal_events[i], 
                                                        epoch_duration=epoch_duration, 
                                                        baseline_duration=baseline_duration, 
                                                        sfreq = config.resample_sfreq)
        #Add epoch to the corresponding condition
        event_description = list(go_signal_events_dict_block.keys())[list(go_signal_events_dict_block.values()).index(go_signal_events[i][2])]
        condition = event_description.split('_').pop()
        go_signal_epoch.set_annotations(None)
        go_signal_epochs_dict[condition].append(go_signal_epoch)

    for condition, epochs in go_signal_epochs_dict.items():
        go_signal_epochs_dict[condition] = mne.concatenate_epochs(epochs, on_mismatch='ignore', verbose=False).drop_bad(reject=config.reject_dict)
    
    return go_signal_epochs_dict, surpass_events


# Save epochs
# For each epochs given, it saves it inside the corresponding subject, condition and epoch type folder
def save_epochs(epochs, subject, block_number, epochs_type, condition, baseline_duration, epoch_duration, pick_type):
    # Create an epoch_type for the epoch type if it doesn't exist
    epoch_type_folder = f"{config.work_dir}/saved_epochs/{pick_type}_{epochs_type}_{baseline_duration}_{epoch_duration}"
    os.makedirs(epoch_type_folder, exist_ok=True)

    # Create a folder for the condition if it doesn't exist
    condition_folder = f"{epoch_type_folder}/{condition}"
    os.makedirs(condition_folder, exist_ok=True)

    # Create a folder for the subject if it doesn't exist
    subject_folder = f"{condition_folder}/subject_{subject}"
    os.makedirs(subject_folder, exist_ok=True)

    # Save .fif file
    fif_file_path = os.path.join(subject_folder, f"block_{block_number}-epo.fif")
    epochs.save(fif_file_path, overwrite=True)


# Returns epochs array corresponding to given subject, condition and epochs_type
def load_subject_epochs_by_type_and_condition(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose):
    
    subject_epochs = []

    epochs_folder = f"{config.work_dir}/saved_epochs/{pick_type}_{epochs_type}_{baseline_duration}_{epoch_duration}/{condition}/subject_{subject}"

    for epoch_file_name in os.listdir(epochs_folder):
        epoch_file_path = os.path.join(epochs_folder, epoch_file_name)
        epochs = mne.read_epochs(epoch_file_path, verbose=verbose)
        subject_epochs.append(epochs)

    epochs = mne.concatenate_epochs(subject_epochs, verbose=False)
    return epochs


# Returns epochs array corresponding to given condition and epochs_type
def load_all_epochs_by_type_and_condition(condition, epochs_type, baseline_duration, epoch_duration, pick_type):
    all_epochs = []

    condition_epochs_folder = f"{config.work_dir}/saved_epochs/{pick_type}_{epochs_type}_{baseline_duration}_{epoch_duration}/{condition}"
    for subject_folder_name in os.listdir(condition_epochs_folder):
        subject_folder_path = os.path.join(condition_epochs_folder, subject_folder_name)
        if os.path.isdir(subject_folder_path):
            for epoch_file_name in os.listdir(subject_folder_path):
                epoch_file_path = os.path.join(subject_folder_path, epoch_file_name)
                epochs = mne.read_epochs(epoch_file_path, verbose=False)
                all_epochs.append(epochs)

    epochs = mne.concatenate_epochs(all_epochs, verbose=False)
    return epochs

# Returns list of epochs array corresponding to given subject, condition and epochs_type
def load_subject_epochs_by_type_and_condition_in_blocks(subject, condition, epochs_type, baseline_duration, epoch_duration, pick_type, verbose):
    
    subject_epochs = []

    epochs_folder = f"{config.work_dir}/saved_epochs/{pick_type}_{epochs_type}_{baseline_duration}_{epoch_duration}/{condition}/subject_{subject}"

    for epoch_file_name in os.listdir(epochs_folder):
        epoch_file_path = os.path.join(epochs_folder, epoch_file_name)
        epochs = mne.read_epochs(epoch_file_path, verbose=verbose)
        subject_epochs.append(epochs)

    return subject_epochs

def change_bad_channels(epochs, verbose = False):
    bad_channels = [item for key, sublist in config.bad_channels_per_sub.items() for item in sublist]
    epochs.info['bads'] = bad_channels
    return epochs