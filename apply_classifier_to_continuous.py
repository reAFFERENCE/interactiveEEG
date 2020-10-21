##### figure out .bdf import #####
# TS 10/2020

import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP
from scipy.signal import filtfilt, butter, lfilter, lfilter_zi

## load data
main_dir = 'D:/PhD_MPI/InteractiveEEG/Data'
file_name = op.join(main_dir, 'random_trial_tilman_left1_right2_stop3.bdf')
raw = mne.io.read_raw_bdf(file_name, preload=True)

# quickly check data appearance
#raw.plot(scalings='auto')


## inspect event information in the data
stim_ch = raw['Status']
stim_ch1 = stim_ch[0]
stim_ch2 = stim_ch[1] # time in seconds?

marker_ix = np.nonzero(stim_ch1)

# extract events
events = mne.find_events(raw) # seems to work !!!
event_dict = {'left': 101, 'right': 102, 'stop':103}

## visualize event markers
#fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], event_id=event_dict)
# ~ event structure makes sense



## keep only EEG channels
#ch_sel = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7',
#	'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz']
# --> Problem: electrode names do not match with those of the laptop recording!
# --> validity of topographies/ spatial filters ??? 	
	
# For now, let´s just take all EEG channels
ch_sel = raw.info['ch_names'][0:24]
raw_sel = raw.pick_channels(ch_sel) # raw object
#raw_sel.plot(events=events, scalings='auto') # show selected raw data with events



# temporal filtering
sfreq = raw.info['sfreq']
iir_params = dict(order=2, ftype='butter', output='sos')  
iir_params = mne.filter.construct_iir_filter(iir_params, (7,30), None, sfreq, 'bandpass', return_copy=False)
test_filt = raw_sel.filter(l_freq=7, h_freq=30, method='iir', iir_params=iir_params)

test_filt.plot(events=events, scalings='auto') # show preprocessed continuous data with events


# get data
test_dat = test_filt.get_data(picks=ch_sel) # data
# --> transfer labels from laptop recording?




	
## create continuous marker of conditions
dat_length = test_dat.shape[1]
events_cont = np.zeros(dat_length, dtype='int64')
for e, event_tmp in enumerate(events): 
	if events[e,2] == 101:
		ix_start = events[e,0]
		ix_end = events[e+1,0]
		events_cont[ix_start:ix_end] = 1
		if events[e+1,2] != 103:
			print('no stop trigger after 101!')
	if events[e,2] == 102:
		ix_start = events[e,0]
		ix_end = events[e+1,0]
		events_cont[ix_start:ix_end] = 2
		if events[e+1,2] != 103:
			print('no stop trigger after 102!')
			
# augment to array with sample information
events_cont = np.vstack((np.array(range(dat_length), dtype='int64'), np.zeros(dat_length, dtype='int64'), events_cont)).T

# plot continuous triggers
time_vec = raw['Fp1'][1]
plt.plot(time_vec, events_cont[:,2])
plt.show()

	
	
	
	
## construct spatial filters and classifier based on trial-by-trial paradigm
# load data
file_name = op.join(main_dir, 'tilman1.edf')
raw_1 = mne.io.read_raw_edf(file_name, preload=True)

clab = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7',
		'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz', 'gyroX', 'gyroY', 'gyroz']
ind_del = [clab.index(chan1) for chan1 in ['gyroX', 'gyroY', 'gyroz'] ]
clab = [c for i, c in enumerate(clab) if i not in ind_del]

clab_rename_dict = {'Channel 1':'Fp1', 'Channel 2':'Fp2', 'Channel 3':'F3', 'Channel 4':'F4', 'Channel 5':'C3', 'Channel 6':'C4',
	'Channel 7':'P3', 'Channel 8':'P4', 'Channel 9':'O1', 'Channel 10':'O2', 'Channel 11':'F7', 'Channel 12':'F8', 'Channel 13':'T7',
	'Channel 14':'T8', 'Channel 15':'P7', 'Channel 16':'P8', 'Channel 17':'Fz', 'Channel 18':'Cz', 'Channel 19':'Pz', 'Channel 20':'M1',
	'Channel 21':'M2', 'Channel 22':'AFz', 'Channel 23':'CPz', 'Channel 24':'POz'}
raw_1.rename_channels(clab_rename_dict)
raw_1.pick_channels(clab) # pick EEG channels already now!


# get events
events = mne.events_from_annotations(raw_1)
ev_id = {'OVTK_GDF_Left': 6, 'OVTK_GDF_Right': 7}  # this is specific for tilman1
eventx = events[0][(events[0][:, 2] > 5) & (events[0][:, 2] < 8)] # this is specific for tilman1

# temporal filtering 
iir_params = dict(order=2, ftype='butter', output='sos')  
iir_params = mne.filter.construct_iir_filter(iir_params, (7,30), None, sfreq, 'bandpass', return_copy=False)
raw_filt = raw_1.filter(l_freq=7, h_freq=30, method='iir', iir_params=iir_params)

# adjust scaling
raw_filt._data = raw_filt._data / 10**6
	#raw_filt.plot(scalings='auto') # -> 10^6 times higher scale
	#raw_filt.plot()

# epoching
epochs = mne.Epochs(raw_filt, events=eventx, event_id=ev_id, tmin=-1, tmax=5, baseline=(-1, 0), preload=True)
labels_1 = epochs.events[:, -1] - 6  # set labels to 0 (left hand imagery) and 1 (right hand imagery); this is specificaly for tilman1
labels = labels_1.copy()
epochs_data = epochs.get_data()  # n_epochs x n_channels x n_times

# extract data of interest
sfreq = raw_1.info['sfreq']
t_win_sec = np.array([0.5, 3.75]) # !!! choose time window (in sec relative to stimulus onset) !!!
t_win_pt = t_win_sec*sfreq + sfreq # adjust to whole epoch length
epochs_data = epochs_data[:, :, int(t_win_pt[0]):int(t_win_pt[1])]  # choose time window of interest
#epochs_data = np.delete(epochs_data, ind_del, axis=1) # omit non-EEG channels
epochs_data_train = epochs_data.copy()

# determine number of components by cross-validation
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


# train classifier on training data
csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, cov_est='concat')
csp.fit(epochs_data_train, labels)
fea_whole = csp.transform(epochs_data_train)
	
lda = LinearDiscriminantAnalysis()
lda.fit(fea_whole, labels)
lbl_pred = lda.predict(fea_whole)
print('acc on train whole data csp = ', np.mean(lbl_pred == labels))
	
	
# plot CSP pattern for training data
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)

csp_patterns = csp.patterns_.T

fig = plt.figure()
for i in range(n_components):
	axes = fig.add_subplot(1, n_components, i + 1)
	mne.viz.plot_topomap(csp_patterns[:, i], epochs.info)

#csp.plot_filters(epochs.info)
#csp.plot_patterns(epochs.info)


	
## Apply spatial filters from classifier (trained on trial-by-trial paradigm) on the continuous test data
# start with 'test_filt'; to do: implement filter on the fly!
test_filt_data = test_filt.get_data()

n_samples_integrate = 20 # samples to generate classifier output from
n_output_integrate = 1 # classifier outputs to integrate (smoothing)
bias =0 # bias term
s = 1 # scaling factor
#pred_cont =[]
pred_buffer = np.empty(n_output_integrate+1)
pred_buffer[:] = np.nan

cl_out_cont = np.empty(test_filt_data.shape[1])
cl_out_cont[:] = np.nan

b,a = butter(2, np.array([0.15]) / sfreq * 2, 'lowpass')
zi_previous = lfilter_zi(b,a)

for k in range(test_filt_data.shape[1]):

	if k < n_samples_integrate-1: # beginning of data stream (not yet enough samples)
		signal_tmp = test_filt_data[:,:k+1]

	else: # enough samples collected (n_samples_integrate)
		signal_tmp = test_filt_data[:,k-(n_samples_integrate-1):k+1]

	# format data
	signal_tmp = np.expand_dims(signal_tmp,	0)  # dimensions for CSP: epochs x channels x samples (1,24,n_samples_integrate)

	# apply CSP filters + LDA
	fea_tmp = csp.transform(signal_tmp)
	#pred_tmp = lda.predict(fea_tmp)
	pred_tmp = lda.decision_function(fea_tmp)

	# put in array for prediction values
	#pred_cont.append(list(pred_tmp))
	#pred_cont = pred_cont[-n_output_integrate:] # only keep last values in buffer
	pred_buffer[:n_output_integrate] = pred_buffer[-n_output_integrate:] # shift to the left by one
	pred_buffer[n_output_integrate] = pred_tmp # add prediction of this loop

	# alternative: low-pass filter for buffer lfilter with zi // initiliaze with lfilter_zi
	cl_out_cont[k], zi_previous = lfilter(b, a, pred_tmp, axis=-1, zi=zi_previous)

	# # smooth classifier output based on previous samples
	# #if len(pred_cont) < n_output_integrate: # beginning of acquisition (not yet many samples)
	# if (k+1)/n_samples_integrate < n_output_integrate:  # beginning of acquisition (not yet many samples)
	# 	#cl_out_cont[k] = s * (sum( np.array(pred_cont) -bias ) / len(pred_cont))
	# 	cl_out_cont[k] = s * (sum(pred_buffer[-(k+1):] - bias) / (k+1))
	#
	# else:
	# 	#cl_out_cont[k] = s * (sum( np.array(pred_cont)[-n_output_integrate:] -bias ) / n_output_integrate)
	# 	cl_out_cont[k] = s * (sum(pred_buffer[-n_output_integrate:] - bias) / n_output_integrate)

print('done')


# try Savitzky-Golay filter
from scipy.signal import savgol_filter
cl_out_filt = savgol_filter(cl_out_cont, 5001, 3)
cl_out_filt = cl_out_filt / np.std(cl_out_filt) # scale

plt.figure()
plt.plot(cl_out_filt)
plt.plot(events_true)
# --> works pretty well (but no online filter)


# threshold classifier output (for evaluation of accuracy)
#thresh = 0.4
cl_out_sca = cl_out_cont / np.std(cl_out_cont) # just scale classifier output

cl_out_adj = cl_out_cont*2-1
thresh = 1 * np.std(cl_out_adj)

cl_out_thresh = cl_out_adj.copy()
cl_out_thresh[cl_out_thresh<=-thresh] = -1
cl_out_thresh[cl_out_thresh>=thresh] = 1
cl_out_thresh[(np.abs(cl_out_thresh)<thresh) & (np.abs(cl_out_thresh)!=1)] = 0

# visualize results (compare with original events)
events_true = events_cont[:,2].copy()
events_true[events_true==1] = -1
events_true[events_true==2] = 1

plt.figure()
plt.plot(cl_out_cont)
plt.plot(events_true)

plt.figure()
plt.plot(cl_out_thresh)
plt.plot(events_true)

plt.figure()
plt.plot(cl_out_adj)
plt.plot(events_true)

plt.figure()
p1, = plt.plot(time_vec, cl_out_sca)
p2, = plt.plot(time_vec, events_true)
plt.xlabel('time (sec)')
plt.legend([p1,p2], ['classifier output', 'true labels; -1:left hand imagery, +1:right hand imagery'])

#plt.hist(cl_out_adj)
#plt.hist(cl_out_adj)