"""
V0, 20200906, by Mina
V1, 20200922, by Tilman & Mina
V2, 20200923, by Tilman (optimized preprocessing)
"""

import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP
from scipy.signal import filtfilt, butter


#main_dir = '/Users/mina/Documents/Academics/COVID19_HomeOffice/Reafference'
main_dir = 'D:/PhD_MPI/InteractiveEEG/Data'

clab = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7',
        'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz', 'gyroX', 'gyroY', 'gyroz']
ind_del = [clab.index(chan1) for chan1 in ['M1', 'M2', 'gyroX', 'gyroY', 'gyroz'] ]
clab = [c for i, c in enumerate(clab) if i not in ind_del]

# ---------------------------------------------------------------------------------
# read tilman1
# ---------------------------------------------------------------------------------
file_name = op.join(main_dir, 'tilman1.edf')
raw_1 = mne.io.read_raw_edf(file_name, preload=True)

# quickly check data appearance
raw_1.plot(scalings='auto') # -> quite high values?! unit~=µV ??


# -------------------------------------------
# create raw_info for plotting reasons
# -------------------------------------------
data1 = raw_1.get_data()
data1 = np.delete(data1, ind_del, axis=0)
raw_info = mne.create_info(ch_names=clab, sfreq=1000, ch_types=['eeg'] * len(clab))
raw1 = mne.io.RawArray(data1, raw_info)
montage = mne.channels.make_standard_montage('standard_1020')
raw1.set_montage(montage)
raw_info = raw1.info
del raw1, data1
sfreq = raw_1.info['sfreq']
b, a = butter(2, np.array([7, 30]) / sfreq * 2, 'bandpass')

# ----------------------------------
# extract events from raw
# -------------------------------------
# tilman1 : left is 6 right is 7

events = mne.events_from_annotations(raw_1)
ev_id = {'OVTK_GDF_Left': 6, 'OVTK_GDF_Right': 7}  # this is specific for tilman1
eventx = events[0][(events[0][:, 2] > 5) & (events[0][:, 2] < 8)] # this is specific for tilman1


# ---------------------------------------------------
# temporal filtering, epoching, time window selection
# ---------------------------------------------------

iir_params = dict(order=2, ftype='butter', output='sos')  
iir_params = mne.filter.construct_iir_filter(iir_params, (7,30), None, sfreq, 'bandpass', return_copy=False)
raw_filt = raw_1.filter(l_freq=7, h_freq=30, method='iir', iir_params=iir_params)

epochs = mne.Epochs(raw_filt, events=eventx, event_id=ev_id, tmin=-1, tmax=5, baseline=(-1, 0), preload=True)
labels_1 = epochs.events[:, -1] - 6  # set labels to 0 (left hand imagery) and 1 (right hand imagery); this is specificaly for tilman1
labels = labels_1.copy()
#epochs_train = epochs.copy()
epochs_data = epochs.get_data()  # n_epochs x n_channels x n_times

# ---> maybe no baseline?!
# epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

t_win_sec = np.array([0.5, 3.75]) # !!! choose time window (in sec relative to stimulus onset) !!!
t_win_pt = t_win_sec*sfreq + sfreq # adjust to whole epoch length
epochs_data = epochs_data[:, :, int(t_win_pt[0]):int(t_win_pt[1])]  # choose time window of interest
epochs_data = np.delete(epochs_data, ind_del, axis=1) # omit non-EEG channels
epochs_data_1 = epochs_data.copy()
#epochs_data_1 = filtfilt(b, a, epochs_data, axis=2) # filter before epoching



# ---------------------------------------------------------------------------------
# read tilman2
# ---------------------------------------------------------------------------------
file_name = op.join(main_dir, 'tilman2.edf')
raw_2 = mne.io.read_raw_edf(file_name, preload=True)

# quickly check data appearance
raw_2.plot(scalings='auto') # also very high amplitude values


# ------------------------------------
# extract events from raw
# ------------------------------------
# tilman2: left is 5 and right 6

events = mne.events_from_annotations(raw_2)
ev_id = {'OVTK_GDF_Left': 5, 'OVTK_GDF_Right': 6}  # this is specific for tilman2
eventx = events[0][(events[0][:, 2] > 4) & (events[0][:, 2] < 7)] # this is specificaly for tilman2


# ---------------------------------------------------
# temporal filtering, epoching, time window selection
# ---------------------------------------------------

iir_params = dict(order=2, ftype='butter', output='sos')  
iir_params = mne.filter.construct_iir_filter(iir_params, (7,30), None, sfreq, 'bandpass', return_copy=False)
raw_filt = raw_2.filter(l_freq=7, h_freq=30, method='iir', iir_params=iir_params)

epochs = mne.Epochs(raw_filt, events=eventx, event_id=ev_id, tmin=-1, tmax=5, baseline=(-1, 0), preload=True)
labels_2 = epochs.events[:, -1] - 5  # set labels to 0 (left hand imagery) and 1 (right hand imagery); this is specificaly for tilman2
#epochs_train = epochs.copy()
epochs_data = epochs.get_data()  # n_epochs x n_channels x n_times

t_win_sec = np.array([0.5, 3.75]) # !!! choose time window (in sec relative to stimulus onset) !!!
t_win_pt = t_win_sec*sfreq + sfreq # adjust to whole epoch length
epochs_data = epochs_data[:, :, int(t_win_pt[0]):int(t_win_pt[1])]  # choose time window of interest
epochs_data = np.delete(epochs_data, ind_del, axis=1) # omit non-EEG channels
epochs_data_2 = epochs_data.copy()
#epochs_data_2 = filtfilt(b, a, epochs_data, axis=2) # filter before epoching





# ---------------------------------------------------------------------------------
# predict dataset1 by dataset2
epochs_data = epochs_data_2.copy()
labels = labels_2.copy()
epochs_data_prim = epochs_data_1.copy()
labels_prim = labels_1.copy()

# predict dataset2 by dataset1
epochs_data = epochs_data_1.copy()
labels = labels_1.copy()
epochs_data_prim = epochs_data_2.copy()
labels_prim = labels_2.copy()

# ---------------------------------------------------------------------------------
# first check: apply CSP on the whole data points to check the pattern
# ---------------------------------------------------------------------------------
n_components = 4  # randomly selected
csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
csp.fit(epochs_data, labels)
csp_patterns = csp.patterns_.T

# plot patterns
fig = plt.figure()
for i in range(n_components):
    axes = fig.add_subplot(1, n_components, i + 1)
    mne.viz.plot_topomap(csp_patterns[:, i], raw_info)

# csp.plot_filters(raw_info)
# csp.plot_patterns(raw_info)
"""
note when plotting the patterns using the method plot_pattern from CSP class, the scales are made the same.
Therefore, it looks a bit different from what I manually plot.

The first 2 patterns look like what we expect to see...
"""
# check the lda accuracy if trained on whole data
fea_whole = csp.fit_transform(epochs_data, labels)
lda = LinearDiscriminantAnalysis()
lda.fit(fea_whole, labels)
lbl_pred = lda.predict(fea_whole)
print('acc on whole data csp = ', np.mean(lbl_pred == labels))


# ---------------------------------------------------------------------------------
# final pipeline for training based on this training data
# ---------------------------------------------------------------------------------
# cv for determining n_components
n_cv = 5
range_n_components = range(2, 6)
random_state_nested = np.random.randint(2**31) # didn´t work with 2**32?!
acc_n_components = np.zeros((len(list(range_n_components)),))
for i_comp, n_components in enumerate(range_n_components):
    cv = ShuffleSplit(n_cv, test_size=0.2, random_state=random_state_nested)
    cv_split = cv.split(epochs_data)
    i_iter = -1
    acc_test_iter = np.zeros((cv.get_n_splits(),))
    for tr_idx, te_idx in cv_split:
        i_iter += 1
        epochs_train = epochs_data[tr_idx, :, :]
        epochs_test = epochs_data[te_idx, :, :]
        labels_train = labels[tr_idx]
        labels_test = labels[te_idx]
        csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
        csp.fit(epochs_train, labels_train)
        epochs_train_nested_new = csp.transform(epochs_train)
        epochs_test_nested_new = csp.transform(epochs_test)
        lda = LinearDiscriminantAnalysis()
        lda.fit(epochs_train_nested_new, labels_train)
        # lbl_train_pred_nested = lda.predict(epochs_train_nested_new)
        lbl_test_pred = lda.predict(epochs_test_nested_new)
        acc_test_iter[i_iter] = np.mean(lbl_test_pred == labels_test)
    acc_n_components[i_comp] = np.mean(acc_test_iter)

idx1 = np.argmax(acc_n_components)
n_components = list(range_n_components)[idx1]

"""
csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
csp.fit(epochs_data, labels)
fea_whole = csp.fit_transform(epochs_data, labels)
lda = LinearDiscriminantAnalysis()
lda.fit(fea_whole, labels)
lbl_pred = lda.predict(fea_whole)
print('acc on whole data csp = ', np.mean(lbl_pred == labels))
"""

# apply on the other data set
csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
csp.fit(epochs_data, labels)
fea_whole = csp.transform(epochs_data)
#fea_whole_prim = csp.fit_transform(epochs_data_prim, labels_prim)
fea_whole_prim = csp.transform(epochs_data_prim) # !!!!!!!!!!?
lda = LinearDiscriminantAnalysis()
lda.fit(fea_whole, labels)
lbl_pred = lda.predict(fea_whole)
lbl_pred_prim = lda.predict(fea_whole_prim)
print('acc on test data csp = ', np.mean(lbl_pred_prim == labels_prim))
print('acc on train whole data csp = ', np.mean(lbl_pred == labels))


# ---------------------------------------------------------------------------------
# classification performance with cross-validation
# ---------------------------------------------------------------------------------
"""
Note:
the CV does not show a good performance (overfitting). It can be because of the lack of enough training data. 
"""
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




