# -*- coding: utf-8 -*-

# Global variables
work_dir = r"C:\Users\gautier\Données EEG - Supervised Project"
study_name = 'VST'
figures = True
plot = True
generate_report = True
montage_path = 'VST_Data/eeg data/MontageAntneuro2.bvef'
stat_dir = r"C:\Users\gautier\Données EEG - Supervised Project\Stats"
decoding_dir = r"C:\Users\gautier\Données EEG - Supervised Project\Decoding"

subjects_list = ['VST_02_X', 'VST_03_Y', 'VST_05_X', 'VST_07_X',
                'VST_08_Y', 'VST_09_X', 'VST_10_Y', 'VST_11_X',
                'VST_12_Y', 'VST_13_X', 'VST_14_Y', 'VST_15_X',
                'VST_16_Y', 'VST_17_X', 'VST_18_Y', 'VST_19_X',
                'VST_20_Y', 'VST_21_X', 'VST_22_Y', 'VST_23_X',
                'VST_24_Y']

preprocessed_subjects_list = ['VST_02_X', 'VST_03_Y', 'VST_05_X', 'VST_07_X',
                'VST_08_Y', 'VST_09_X', 'VST_10_Y', 'VST_11_X',
                'VST_12_Y', 'VST_13_X', 'VST_14_Y', 'VST_15_X',
                'VST_16_Y', 'VST_17_X', 'VST_18_Y', 'VST_19_X',
                'VST_20_Y', 'VST_21_X', 'VST_22_Y', 'VST_23_X',
                'VST_24_Y']

raw_data_paths = { 
              'VST_02_X': './VST_Data/eeg data\VST_02_X\VST_VST_02_X_prod_task.vhdr',
              'VST_03_Y': './VST_Data/eeg data\VST_03_Y\VST_VST_03_Y_2023-06-26_10-49-33.vhdr',
              'VST_05_X': './VST_Data/eeg data\VST_05_X\VST_VST_05_X_2023-07-04_11-14-58.vhdr',
              'VST_07_X': './VST_Data/eeg data\VST_07_X\VST_VST_07_X_2023-07-10_10-12-22.vhdr',
              'VST_08_Y': './VST_Data/eeg data\VST_08_Y\VST_VST_08_Y_2023-07-10_14-56-45.vhdr',
              'VST_09_X': './VST_Data/eeg data\VST_09_X\VST_VST_09_X_2023-07-11_09-47-07.vhdr',
              'VST_10_Y': './VST_Data/eeg data\VST_10_Y\VST_VST_10_Y_2023-07-12_09-53-53.vhdr',
              'VST_11_X': './VST_Data/eeg data\VST_11_X\VST_VST_11_X_2023-07-12_13-25-51.vhdr',
              'VST_12_Y': './VST_Data/eeg data\VST_12_Y\VST_VST_12_y_2023-07-12_16-38-34.vhdr',
              'VST_13_X': './VST_Data/eeg data\VST_13_X\VST_VST_13_X_2023-07-17_09-52-46.vhdr',
              'VST_14_Y': './VST_Data/eeg data\VST_14_Y\VST_VST_14_Y_2023-07-17_13-48-18.vhdr',
              'VST_15_X': './VST_Data/eeg data\VST_15_X\VST_VST_15_X_2023-07-18_09-45-21.vhdr',
              'VST_16_Y': './VST_Data/eeg data\VST_16_Y\VST_VST_16_Y_2023-07-18_13-32-05.vhdr',
              'VST_17_X': './VST_Data/eeg data\VST_17_X\VST_VST_17_X_2023-07-18_16-34-34.vhdr',
              'VST_18_Y': './VST_Data/eeg data\VST_18_Y\VST_VST_18_Y_2023-07-19_13-12-59.vhdr',
              'VST_19_X': './VST_Data/eeg data\VST_19_X\VST_VST_19_X_2023-07-19_16-31-24.vhdr',
              'VST_20_Y': './VST_Data/eeg data\VST_20_Y\VST_VST_20_Y_2023-07-24_09-52-47.vhdr',
              'VST_21_X': './VST_Data/eeg data\VST_21_X\VST_VST_21_X_2023-07-24_13-40-06.vhdr',
              'VST_22_Y': './VST_Data/eeg data\VST_22_Y\VST_VST_22_Y_2023-07-25_13-26-33.vhdr',
              'VST_23_X': './VST_Data/eeg data\VST_23_X\VST_VST_23_X_2023-07-25_16-58-51.vhdr',
              'VST_24_Y': './VST_Data/eeg data\VST_24_Y\VST_VST_24_Y_2023-07-26_13-55-13.vhdr',             
              }

##############
# Define Events
##############
# RS = resting state
# SNS = small normal short
# SNL = small normal long
# SHS = small high short
# SHL = small high long
# LNS = large normal short
# LNL = large normal long
# LHS = large high short
# LHL = large high long
# press = rythm

conditions = ['SNS', 'SNL', 'SHS', 'SHL', 'LNS', 'LNL', 'LHS', 'LHL']
condition_pairs = [('SNS', 'SNL'), ('SHS', 'SHL'), ('LNS', 'LNL'), ('LHS', 'LHL'), ('SNS', 'SHS'), ('SNL', 'SHL'), ('LNS', 'LHS'), ('LNL', 'LHL'), ('SNS', 'LNS'), ('SNL', 'LNL'), ('SHS', 'LHS'), ('SHL', 'LHL')]

mapped_annotations = {
    'Stimulus/s11': 'start_trial_SNS',
    'Stimulus/s12': 'go_signal_SNS',
    'Stimulus/s13': 'start_prod_SNS',
    'Stimulus/s14': 'end_prod_SNS',
    'Stimulus/s21': 'start_trial_SNL',
    'Stimulus/s22': 'go_signal_SNL',
    'Stimulus/s23': 'start_prod_SNL',
    'Stimulus/s24': 'end_prod_SNL',
    'Stimulus/s51': 'start_trial_SHS',
    'Stimulus/s52': 'go_signal_SHS',
    'Stimulus/s53': 'start_prod_SHS',
    'Stimulus/s54': 'end_prod_SHS',
    'Stimulus/s61': 'start_trial_SHL',
    'Stimulus/s62': 'go_signal_SHL',
    'Stimulus/s63': 'start_prod_SHL',
    'Stimulus/s64': 'end_prod_SHL',
    'Stimulus/s111': 'start_trial_LNS',
    'Stimulus/s112': 'go_signal_LNS',
    'Stimulus/s113': 'start_prod_LNS',
    'Stimulus/s114': 'end_prod_LNS',
    'Stimulus/s121': 'start_trial_LNL',
    'Stimulus/s122': 'go_signal_LNL',
    'Stimulus/s123': 'start_prod_LNL',
    'Stimulus/s124': 'end_prod_LNL',
    'Stimulus/s151': 'start_trial_LHS',
    'Stimulus/s152': 'go_signal_LHS',
    'Stimulus/s153': 'start_prod_LHS',
    'Stimulus/s154': 'end_prod_LHS',
    'Stimulus/s161': 'start_trial_LHL',
    'Stimulus/s162': 'go_signal_LHL',
    'Stimulus/s163': 'start_prod_LHL',
    'Stimulus/s164': 'end_prod_LHL',
    'Stimulus/s210': 'press_SN',
    'Stimulus/s215': 'press_SH',
    'Stimulus/s220': 'press_HN',
    'Stimulus/s225': 'press_HH',
    'New Segment/': 'new_run'
}
evts_id = {
    "start_resting": 1,
    "stop_resting": 2,
    
    "start_trial_SNS": 11,
    "go_signal_SNS": 12,
    "start_prod_SNS": 13,
    "end_prod_SNS": 14,
    
    "start_trial_SNL": 21,
    "go_signal_SNL": 22,
    "start_prod_SNL": 23,
    "end_prod_SNL": 24,
    
    "start_trial_SHS": 51,
    "go_signal_SHS": 52,
    "start_prod_SHS": 53,
    "end_prod_SHS": 54,
    
    "start_trial_SHL":61,
    "go_signal_SHL": 62,
    "start_prod_SHL": 63,
    "end_prod_SHL": 64,
    
    "start_trial_LNS": 111,
    "go_signal_LNS": 112,
    "start_prod_LNS": 113,
    "end_prod_LNS": 114,
    
    "start_trial_LNL": 121,
    "go_signal_LNL": 122,
    "start_prod_LNL": 123,
    "end_prod_LNL": 124,
    
    "start_trial_LHS": 151,
    "go_signal_LHS": 152,
    "start_prod_LHS": 153,
    "end_prod_LHS": 154,
    
    "start_trial_LHL":161,
    "go_signal_LHL": 162,
    "start_prod_LHL": 163,
    "end_prod_LHL": 164,
    
    "press_SN": 210,
    "press_SH": 215,
    "press_HN": 220,
    "press_HH": 225,

    "new_run": 666
}

###########
# Preprocessing
###########

#Resample
resample_sfreq = 250

# Filter
f_min_filtering = 1
f_max_filtering = 40

# bad channels

bad_channels_per_sub = {
    'VST_02_X': [],
    'VST_03_Y': [],
    'VST_05_X': [],
    'VST_07_X': ['TP7', 'F5', 'AF3', 'FC3'],
    'VST_08_Y': [],
    'VST_09_X': ['AF7'],
    'VST_10_Y': ['M1'],
    'VST_11_X': ['F8', 'F6'],
    'VST_12_Y': [],
    'VST_13_X': ['M1', 'T7'],
    'VST_14_Y': ['FT7', 'AF3', 'AF7'],
    'VST_15_X': ['FC4', 'AF3', 'AF7'],
    'VST_16_Y': ['T7', 'C3'],
    'VST_17_X': ['CP2', 'AF7', 'T7'],
    'VST_18_Y': ['Pz', 'T8'],
    'VST_19_X': [],
    'VST_20_Y': [],
    'VST_21_X': [],
    'VST_22_Y': [],
    'VST_23_X': ['Cz', 'CP2', 'CP4'],
    'VST_24_Y': [],
    }

# Old flagging done by DeAurojoDaSilva
# bad_channels_per_sub = {
#     'VST_02_X': [],
#     'VST_03_Y': ['CP2', 'CP1'],
#     'VST_04_Y': ['Cz'], #['T8', 'M2', 'C2', 'C1', 'O2', 'C5', 'FC4'], 
#     'VST_05_X': [],
#     'VST_07_X': ['Pz', 'CP2'],
#     'VST_08_Y': ['Pz'],
#     'VST_09_X': ['Cz', 'CP1']
#     }


# ICA
ica_exclude = {
    'VST_02_X': [1, 17],
    'VST_03_Y': [3],
    'VST_05_X': [2],
    'VST_07_X': [0],
    'VST_08_Y': [0],
    'VST_09_X': [9, 0],
    'VST_10_Y': [0],
    'VST_11_X': [7],
    'VST_12_Y': [3,5],
    'VST_13_X': [0],
    'VST_14_Y': [16,5,10],
    'VST_15_X': [4,5],
    'VST_16_Y': [5],
    'VST_17_X': [1,3],
    'VST_18_Y': [0],
    'VST_19_X': [2,9],
    'VST_20_Y': [4,6],
    'VST_21_X': [9,10,2,1],
    'VST_22_Y': [7,10],
    'VST_23_X': [10,13],
    'VST_24_Y': [],
        }

# Bad time production blocks, ie blocks with start prod and go signal events that are not suitable for further analysis
bad_time_production_blocks = {
    'VST_02_X': [],
    'VST_03_Y': [],
    'VST_05_X': [],
    'VST_07_X': [],
    'VST_08_Y': [2,3],
    'VST_09_X': [],
    'VST_10_Y': [7],
    'VST_11_X': [],
    'VST_12_Y': [],
    'VST_13_X': [],
    'VST_14_Y': [],
    'VST_15_X': [0],
    'VST_16_Y': [],
    'VST_17_X': [],
    'VST_18_Y': [],
    'VST_19_X': [],
    'VST_20_Y': [],
    'VST_21_X': [0],
    'VST_22_Y': [],
    'VST_23_X': [],
    'VST_24_Y': [],
    }


# Bad start prod and go signal events, ie events that break the go_signal - start_prod chain (two consecutive start prod for instance)
# Add all the subjects to the dictionary

bad_go_signal_start_prod_events = {
                                    'VST_02_X':{
                                        3:{
                                            'start_prod':[0],
                                            'end_prod':[0],
                                            'go_signal':[]
                                            }
                                        },
                                    'VST_16_Y':{
                                        3:{
                                            'start_prod':[0],
                                            'end_prod':[0],
                                            'go_signal':[]
                                            }
                                        },
                                    'VST_21_X':{
                                        1:{
                                            'start_prod':[0],
                                            'end_prod':[0],
                                            'go_signal':[]
                                            }
                                        },
                                    }

reject_dict = dict(eeg=150e-6)

####
# Selected channels
####

# From Robinson & Wiener 2020 paper
frontocentral_channels = ['Fz', 'FC1', 'Cz', 'FC2', 'F1', 'C1', 'C2', 'F2', 'FCz']

# All Channels names
all_channels_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']

# Channels from cluster analysis
channels_from_cluster_analysis = ['Fpz', 'F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C4', 'CP5', 'CP1', 'CP6', 'P3', 'AF4', 'F1', 'F2', 'FCz', 'C5', 'C1', 'C2', 'C6', 'CP3', 'P1', 'P2']

# TO BE REMOVED

# # Re reference
# # reference = ['M1', 'M2']
# reference = 'average'

# # events
# first_keypress_events = ['Stimulus/s113', 'Stimulus/s123', 'Stimulus/s13', 
#                          'Stimulus/s153', 'Stimulus/s163', 'Stimulus/s23', 
#                          'Stimulus/s53', 'Stimulus/s63']
# second_keypress_events = ['Stimulus/s114', 'Stimulus/s124', 'Stimulus/s14', 
#                           'Stimulus/s154', 'Stimulus/s164', 'Stimulus/s24', 
#                           'Stimulus/s54', 'Stimulus/s64']





# channels evoked data
channels_evoked =  ['Fz', 'FC1', 'Cz', 'FC2', 'F1', 'C1', 'C2', 'F2', 'FCz']