import sys
import mne
import os
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\scripts")
sys.path.append(r"C:\Users\gautier\OneDrive - CentraleSupelec\3A - Master CNN\Supervised Project\pipeline project v0\config")
import eeg_preprocessing as preprocessing
import vst_config as config

def get_eeg_epochs_and_labels_for_binary_decoding(conditions_class_1, conditions_class_2, epochs_type, pick_type, baseline_duration, epoch_duration):
    epochs_class_1_list = []
    for condition in conditions_class_1:
        epochs = preprocessing.load_all_epochs_by_type_and_condition(
            epochs_type=epochs_type,
            condition=condition,
            baseline_duration=baseline_duration,
            epoch_duration = epoch_duration,
            pick_type=pick_type
        )
        epochs_class_1_list.append(epochs)
    epochs_class_1 = mne.concatenate_epochs(epochs_class_1_list, verbose=False)

    epochs_class_2_list = []
    for condition_high in conditions_class_2:
        epochs = preprocessing.load_all_epochs_by_type_and_condition(
            epochs_type=epochs_type,
            condition=condition_high,
            baseline_duration=baseline_duration,
            epoch_duration = epoch_duration,
            pick_type=pick_type
        )
        epochs_class_2_list.append(epochs)
    epochs_class_2 = mne.concatenate_epochs(epochs_class_2_list, verbose=False)

    decoding_epochs = mne.concatenate_epochs([epochs_class_1, epochs_class_2])
    y = ([0]*epochs_class_1.__len__()) + ([1]*epochs_class_2.__len__())

    return decoding_epochs, y


# Return decoding scores along time points for classification problems.
# Supports following decoders: logistic regression, SVC
def compute_decoding_scores(eeg_data, labels, decoder="logistic", cv=4, ncores=8, verbose=None, scoring_multilabel = "accuracy"):
    if decoder=="logistic":
        
        clf = make_pipeline(StandardScaler(),LogisticRegression(solver="liblinear"))

    elif decoder=="svc":
        
        clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))

    # Choose the scoring metric based on the number of classes
    if len(np.unique(labels)) == 2:
        scoring="roc_auc"
    else:
        scoring=scoring_multilabel

    # Slide the estimator on all time frames
    time_decod = mne.decoding.SlidingEstimator(clf, n_jobs=ncores, scoring=scoring, verbose=verbose)

    scores = mne.decoding.cross_val_multiscore(time_decod, eeg_data, labels, cv=cv, n_jobs=ncores, verbose=verbose)

    return scores


# Plot decoding results
def plot_decoding_scores(decoding_scores, epochs, plot_title, plotting_theme="ticks"):
    results_df = pd.DataFrame(decoding_scores.transpose())
    results_df["time"] = epochs.times
    results_df = pd.melt(results_df, id_vars="time", var_name="fold", value_name="score")
    results_df.head()
    
    sns.set_theme(style=plotting_theme)

    fig, ax = plt.subplots(1, figsize=[12, 6])
    
    # Plot the chance level
    ax.axhline(y=0.5, color="k", ls="--", label="Chance level")
    
    # plot the stimulus onset
    ax.axvline(x=0, color="k", ls="-")
    
    # # Plot the stimulus duration (in visual blocks)
    # ax.axvspan(0, end_stim, alpha=0.1, color="black")
    
    # plot the average decoding accuracy (over folds) with 95% confidence interval
    sns.lineplot(data=results_df, x="time", y="score", ax=ax, lw=2, estimator="mean", label="Average accuracy")
    
    # plotting timepoint significantly better than chance (to-do)
    # ax.plot(0.30, 0.2, marker="o", color="b", markersize=5)
    # ax.plot(0.31, 0.2, marker="o", color="b", markersize=5)
    # ax.plot(0.32, 0.2, marker="o", color="b", markersize=5)
    
    # filling accuracy above 0.5
    # plt.fill_between(x=results_df["time"], y1=0.5, y2=results_df["score"], alpha=0.3, where=results_df["score"]>0.5, interpolate=True)
    
    # specifying axis labels
    ax.set_title(plot_title, size=14, weight=800)
    ax.set_xlabel("Time (in seconds)", size=12)
    ax.set_ylabel("Decoding accuracy (AUC)", size=12)
    
    # adding a legend
    plt.legend(loc="upper right")

    return fig, ax