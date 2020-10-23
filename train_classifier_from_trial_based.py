import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP
from joblib import dump, load

## construct CSP filters and LDA classifier based on trial-based paradigm
# load data
main_dir = 'D:/PhD_MPI/InteractiveEEG/Data'
file_name = op.join(main_dir, 'tilman1.edf')
raw_1 = mne.io.read_raw_edf(file_name, preload=True)

clab = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7',
        'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz', 'gyroX', 'gyroY', 'gyroz']
ind_del = [clab.index(chan1) for chan1 in ['gyroX', 'gyroY', 'gyroz']]
clab = [c for i, c in enumerate(clab) if i not in ind_del]

clab_rename_dict = {'Channel 1': 'Fp1', 'Channel 2': 'Fp2', 'Channel 3': 'F3', 'Channel 4': 'F4', 'Channel 5': 'C3',
                    'Channel 6': 'C4',
                    'Channel 7': 'P3', 'Channel 8': 'P4', 'Channel 9': 'O1', 'Channel 10': 'O2', 'Channel 11': 'F7',
                    'Channel 12': 'F8', 'Channel 13': 'T7',
                    'Channel 14': 'T8', 'Channel 15': 'P7', 'Channel 16': 'P8', 'Channel 17': 'Fz', 'Channel 18': 'Cz',
                    'Channel 19': 'Pz', 'Channel 20': 'M1',
                    'Channel 21': 'M2', 'Channel 22': 'AFz', 'Channel 23': 'CPz', 'Channel 24': 'POz'}
raw_1.rename_channels(clab_rename_dict)
raw_1.pick_channels(clab)  # pick EEG channels already now!

# get events
events = mne.events_from_annotations(raw_1)
ev_id = {'OVTK_GDF_Left': 6, 'OVTK_GDF_Right': 7}  # this is specific for tilman1
eventx = events[0][(events[0][:, 2] > 5) & (events[0][:, 2] < 8)]  # this is specific for tilman1

# temporal filtering
sfreq = raw_1.info['sfreq']
iir_params = dict(order=2, ftype='butter', output='sos')
iir_params = mne.filter.construct_iir_filter(iir_params, (7, 30), None, sfreq, 'bandpass', return_copy=False)
raw_filt = raw_1.filter(l_freq=7, h_freq=30, method='iir', iir_params=iir_params)

# adjust scaling (depends on device with which online data is acquired!!!)
raw_filt._data = raw_filt._data / 10 ** 6
# raw_filt.plot(scalings='auto') # -> 10^6 times higher scale
# raw_filt.plot()

# epoching
epochs = mne.Epochs(raw_filt, events=eventx, event_id=ev_id, tmin=-1, tmax=5, baseline=(-1, 0), preload=True)
labels = epochs.events[:, -1] - 6  # set labels to 0 (left hand imagery) and 1 (right hand imagery); this is specificaly for tilman1
#labels = labels_1.copy()
epochs_data = epochs.get_data()  # n_epochs x n_channels x n_times

# extract data of interest
sfreq = raw_1.info['sfreq']
t_win_sec = np.array([0.5, 3.75])  # !!! choose time window (in sec relative to stimulus onset) !!!
t_win_pt = t_win_sec * sfreq + sfreq  # adjust to whole epoch length
epochs_data = epochs_data[:, :, int(t_win_pt[0]):int(t_win_pt[1])]  # choose time window of interest
#epochs_data_train = epochs_data.copy()


# determine number of components and cross-validate classifier performance
n_cv = 5
cv = ShuffleSplit(n_cv, test_size=0.25)
cv_split = cv.split(epochs_data)
j_iter = -1
acc_test_cv = np.zeros((n_cv,))
for train_idx, test_idx in cv_split:
    j_iter += 1
    print(j_iter)
    epochs_train = epochs_data[train_idx, :, :]
    epochs_test = epochs_data[test_idx, :, :]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    # nested CV for determining n_components
    range_n_components = range(2, 6)
    random_state_nested = np.random.randint(2**31)
    acc_n_components = np.zeros((len(list(range_n_components)),))
    for i_comp, n_components in enumerate(range_n_components):
        cv_nested = ShuffleSplit(3, test_size=0.2, random_state=random_state_nested)
        cv_split_nested = cv_nested.split(epochs_train)
        i_iter = -1
        acc_test_nested_iter = np.zeros((cv_nested.get_n_splits(),))
        for tr_idx_nested, te_idx_nested in cv_split_nested:
            i_iter += 1
            epochs_train_nested = epochs_train[tr_idx_nested, :, :]
            epochs_test_nested = epochs_train[te_idx_nested, :, :]
            labels_train_nested = labels_train[tr_idx_nested]
            labels_test_nested = labels_train[te_idx_nested]
            csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
            csp.fit(epochs_train_nested, labels_train_nested)
            epochs_train_nested_new = csp.transform(epochs_train_nested)
            epochs_test_nested_new = csp.transform(epochs_test_nested)
            lda = LinearDiscriminantAnalysis()
            lda.fit(epochs_train_nested_new, labels_train_nested)
            # lbl_train_pred_nested = lda.predict(epochs_train_nested_new)
            lbl_test_pred_nested = lda.predict(epochs_test_nested_new)
            acc_test_nested_iter[i_iter] = np.mean(lbl_test_pred_nested == labels_test_nested)
        acc_n_components[i_comp] = np.mean(acc_test_nested_iter)
    idx1 = np.argmax(acc_n_components)
    n_components = list(range_n_components)[idx1]
    print('*****************************')
    print('n_components=', n_components)
    print('acc_nested_max=', acc_n_components[idx1])
    print('*****************************')
    csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
    csp.fit(epochs_train, labels_train)
    epochs_train_new = csp.fit_transform(epochs_train, labels_train)
    epochs_test_new = csp.fit_transform(epochs_test, labels_test)
    lda = LinearDiscriminantAnalysis()
    lda.fit(epochs_train_new, labels_train)
    lbl_train_pred = lda.predict(epochs_train_new)
    lbl_test_pred = lda.predict(epochs_test_new)
    acc_test_cv[j_iter] = np.mean(lbl_test_pred == labels_test)

print('****************************************************')
print('**********  Results of cross-validation   **********')
print('Finally chosen number of components:', n_components)
print('Accuracies of cross-validation:', acc_test_cv)
print('\n\n')



# train classifier on whole training data
csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
csp.fit(epochs_data, labels)
fea_whole = csp.transform(epochs_data)

lda = LinearDiscriminantAnalysis()
lda.fit(fea_whole, labels)
lbl_pred = lda.predict(fea_whole)
print('acc on train whole data csp = ', np.mean(lbl_pred == labels))


# save csp filters + lda classifier to file
dump(csp, 'csp_stored.joblib')
dump(lda, 'lda_stored.joblib')

# plot CSP pattern for training data
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)

csp_patterns = csp.patterns_.T

fig = plt.figure()
for i in range(n_components):
    axes = fig.add_subplot(1, n_components, i + 1)
    mne.viz.plot_topomap(csp_patterns[:, i], epochs.info)
plt.suptitle('CSP patterns')

# csp.plot_filters(epochs.info)
# csp.plot_patterns(epochs.info)
