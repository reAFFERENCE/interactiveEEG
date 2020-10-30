import sys
import numpy as np
import math
import random
from pylsl import StreamInlet, resolve_stream
from mainwindow import *
from functools import partial
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
# Import needed modules from pythonosc
from pythonosc.udp_client import SimpleUDPClient

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP
from scipy.signal import filtfilt, butter, lfilter, lfilter_zi
from joblib import dump, load

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

class MyWin(QtWidgets.QMainWindow):
    osc_ip = "127.0.0.1"
    osc_port = 9023
    osc_client = SimpleUDPClient(osc_ip, osc_port)  # Create client

    streams = resolve_stream('name', 'EEG')
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    timer = QtCore.QTimer()
    plot_timer = QtCore.QTimer()
    active_channels = [] # array of "active" channels - used for averaged signal
    samp_rate = 500 # sampling rate
    n_points = 0  # number of obtained points
    buffer_size = 0.5 # seconds, size of buffer
    buffer_vals = np.zeros(int(buffer_size*samp_rate))

    buffer_channels = []
    for i in range(24):
        buffer_channels.append(np.zeros(int(buffer_size*samp_rate)))

    fdb = open("delta_band.csv", "w")
    ftb = open("theta_band.csv", "w")
    fab = open("alpha_band.csv", "w")
    fbb = open("beta_band.csv", "w")
    fgb = open("gamma_band.csv", "w")
    fhgb = open("hgamma_band.csv", "w")
    timestmp = 0

    num_active_chan = 24
    average_val = 0 # averaged value - used for relative power and attention estimation
    for i in range(24):
        active_channels.append(1)

    attention_estimation_points = 5  # number of points for attention estimation
    relpower_arrays_size = 20        # size of array for storing relative power values
    theta_relpower = []
    beta_relpower = []
    alpharelpow = 0
    gammarelpow = 0
    attention_val = 0

    polygon = QPolygonF()

    #plotting variables
    size = 50
    x_vec = np.linspace(0, 50, size + 1)[0:-1]
    y_vec = np.random.randn(len(x_vec))
    line1 = []

    # for motor imagery
    motor_imagery_out = 0  # positive values reflect left hand imagery, negative values right hand
    n_output_integrate = 8  # how many motor imagery classifier outputs to integrate
    pred_buffer = np.zeros(n_output_integrate + 1)
    motor_imagery_out_save = []  # for debugging

    # set filter coefficients
    sfreq = 500  # or user already existing variable that specifies the sampling rate
    b_bp, a_bp = butter(2, np.array([8, 30]) / sfreq * 2, 'bandpass')
    zi_previous_bp = []
    for i in range(24):
        zi_previous_bp.append(lfilter_zi(b_bp, a_bp))
    zi_previous_bp = np.array(zi_previous_bp)

    b_pred, a_pred = butter(2, np.array([0.15]) / (sfreq / (sfreq * buffer_size)) * 2,
                            'lowpass')  # adapted to sampling rate of buffer
    zi_previous_pred = lfilter_zi(b_pred, a_pred)


    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setGeometry(500, 200, 900, 580)
        self.ui.pushButton.clicked.connect(self.startprocessing)
        self.ui.pushButton.setGeometry(QtCore.QRect(50, 510, 120, 28))
        self.ui.pushButton_2.clicked.connect(self.ext)
        self.ui.pushButton_2.setGeometry(QtCore.QRect(740, 510, 70, 28))
        self.ui.pushButton_3.clicked.connect(partial(self.update_allchans, True))
        self.ui.pushButton_5.clicked.connect(self.drawturtle)
        self.ui.pushButton_5.setVisible(False)
        self.ui.pushButton_6.clicked.connect(self.startdrawstream)
        self.ui.pushButton_6.setText("Plot Average signal")
        self.ui.pushButton_6.setGeometry(QtCore.QRect(180, 510, 140, 28))

        self.ui.pushButton_4.clicked.connect(partial(self.update_allchans, False))
        self.ui.doubleSpinBox.setValue(self.buffer_size)

        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0, 255)))
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setFrameStyle(0)
        self.back_brush = QBrush(Qt.gray)
        self.shape_brush = QBrush(Qt.darkGreen)
        self.pen = QPen(Qt.green)
        self.ui.graphicsView.setBackgroundBrush(self.back_brush)
        self.drawshapes()

        self.ui.checkBox.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_2.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_3.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_4.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_5.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_6.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_7.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_8.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_9.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_10.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_11.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_12.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_13.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_14.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_15.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_16.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_17.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_18.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_19.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_20.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_21.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_22.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_23.stateChanged.connect(self.update_activechannels)
        self.ui.checkBox_24.stateChanged.connect(self.update_activechannels)

        self.timer.timeout.connect(self.updatedata)
        self.plot_timer.timeout.connect(self.drawstream)

    def drawshapes(self):
        # self.scene.addEllipse(20,20,200,200,self.pen,self.shape_brush)
        self.scene.clear()
        self.polygon.clear()
        x = y = t = 0
        npoints = int(self.attention_val * 10) + 3
        radius = int(self.attention_val * 10) * 15 + 20
        for i in range(npoints):
            t = 2 * math.pi * (float(i) / npoints + 0.5)
            x = 100 + math.cos(t) * radius
            y = 100 + math.sin(t) * radius
            self.polygon.append(QPointF(x, y))
        self.scene.addPolygon(self.polygon, self.pen,
                              QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),
                                     int(100 + self.attention_val * 10)))

    def on_liveplot_close(self, event):
        self.plot_timer.stop()

    def live_plotter(self,x_vec,y1_data,line1,identifier='',pause_time=0.1):
        if line1==[]:
            # this is the call to matplotlib that allows dynamic plotting
            #plt.ion()
            fig = plt.figure(figsize=(9,2))
            fig.canvas.mpl_connect('close_event', self.on_liveplot_close)
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
            #update plot label/title
            plt.ylabel('Raw Value')
            plt.xticks([])
            plt.title('{}'.format(identifier))
            plt.show()
        
        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        # adjust limits if new data goes beyond bounds
        #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        # return line so we can update it again in the next iteration
        return line1

    def startprocessing(self):
        self.timer.start(1)

    def startdrawstream(self):
        self.line1=[]
        self.plot_timer.start(10)

    def drawstream(self):
        self.y_vec[-1] = self.average_val
        self.line1 = self.live_plotter(self.x_vec,self.y_vec,self.line1,"Average raw signal")
        self.y_vec = np.append(self.y_vec[1:],0.0)

    def drawturtle(self):
        import turtle
        MINIMUM_BRANCH_LENGTH = 5

        def build_tree(t, branch_length, shorten_by, angle):
          if branch_length > MINIMUM_BRANCH_LENGTH:
            t.speed(int(self.alpharelpow*100)+1)
            t.forward(branch_length)
            new_length = branch_length - shorten_by
            t.left(angle)
            build_tree(t, new_length, shorten_by, angle)
            t.right(angle * 2)
            build_tree(t, new_length, shorten_by, angle)
            t.left(angle)
            t.backward(branch_length)

        tree = turtle.Turtle()
        # tree.speed(100)
        tree.hideturtle()
        tree.setheading(90)
        tree.color('green')
        build_tree(tree, 50, 5, 30)
        turtle.mainloop()

    def update_allchans(self, flag):
        self.ui.checkBox.setChecked(flag)
        self.ui.checkBox_2.setChecked(flag)
        self.ui.checkBox_3.setChecked(flag)
        self.ui.checkBox_4.setChecked(flag)
        self.ui.checkBox_5.setChecked(flag)
        self.ui.checkBox_6.setChecked(flag)
        self.ui.checkBox_7.setChecked(flag)
        self.ui.checkBox_8.setChecked(flag)
        self.ui.checkBox_9.setChecked(flag)
        self.ui.checkBox_10.setChecked(flag)
        self.ui.checkBox_11.setChecked(flag)
        self.ui.checkBox_12.setChecked(flag)
        self.ui.checkBox_13.setChecked(flag)
        self.ui.checkBox_14.setChecked(flag)
        self.ui.checkBox_15.setChecked(flag)
        self.ui.checkBox_16.setChecked(flag)
        self.ui.checkBox_17.setChecked(flag)
        self.ui.checkBox_18.setChecked(flag)
        self.ui.checkBox_19.setChecked(flag)
        self.ui.checkBox_20.setChecked(flag)
        self.ui.checkBox_21.setChecked(flag)
        self.ui.checkBox_22.setChecked(flag)
        self.ui.checkBox_23.setChecked(flag)
        self.ui.checkBox_24.setChecked(flag)
        self.update_activechannels()

    def update_activechannels(self):
        self.active_channels[0] = self.ui.checkBox.isChecked()
        self.active_channels[1] = self.ui.checkBox_2.isChecked()
        self.active_channels[2] = self.ui.checkBox_3.isChecked()
        self.active_channels[3] = self.ui.checkBox_4.isChecked()
        self.active_channels[4] = self.ui.checkBox_5.isChecked()
        self.active_channels[5] = self.ui.checkBox_6.isChecked()
        self.active_channels[6] = self.ui.checkBox_7.isChecked()
        self.active_channels[7] = self.ui.checkBox_8.isChecked()
        self.active_channels[8] = self.ui.checkBox_9.isChecked()
        self.active_channels[9] = self.ui.checkBox_10.isChecked()
        self.active_channels[10] = self.ui.checkBox_11.isChecked()
        self.active_channels[11] = self.ui.checkBox_12.isChecked()
        self.active_channels[12] = self.ui.checkBox_13.isChecked()
        self.active_channels[13] = self.ui.checkBox_14.isChecked()
        self.active_channels[14] = self.ui.checkBox_15.isChecked()
        self.active_channels[15] = self.ui.checkBox_16.isChecked()
        self.active_channels[16] = self.ui.checkBox_17.isChecked()
        self.active_channels[17] = self.ui.checkBox_18.isChecked()
        self.active_channels[18] = self.ui.checkBox_19.isChecked()
        self.active_channels[19] = self.ui.checkBox_20.isChecked()
        self.active_channels[20] = self.ui.checkBox_21.isChecked()
        self.active_channels[21] = self.ui.checkBox_22.isChecked()
        self.active_channels[22] = self.ui.checkBox_23.isChecked()
        self.active_channels[23] = self.ui.checkBox_24.isChecked()
        self.num_active_chan = sum(self.active_channels)

    def bandpower(self, data, sf, band, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
            https://raphaelvallat.com/bandpower.html
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

        Return
        ------
        bp : float
            Absolute or relative band power.
        """
        from scipy.signal import welch
        from scipy.integrate import simps
        band = np.asarray(band)
        low, high = band

        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative and simps(psd, dx=freq_res) > 0:
            bp /= simps(psd, dx=freq_res)
        else:
            return 0
        return bp

    def apply_csp_lda_online(self):
        # retrieve data
        signal_tmp = np.array(self.buffer_channels)

        # filter incoming signal (and use previous filter state)
        for i in range(24):
            signal_tmp[i, ], self.zi_previous_bp[i, ] = lfilter(self.b_bp, self.a_bp, signal_tmp[i, ], axis=-1, zi=self.zi_previous_bp[i, ])

        # scale signal (depending on input; signal range should be around +-25 µV)
        signal_tmp = signal_tmp / 10 ** 6

        # bring data into right format
        signal_tmp = np.expand_dims(signal_tmp, 0)  # dimensions for CSP: epochs x channels x samples (1,24,n_samples_integrate)

        # load CSP+LDA from training (pre-allocate this to class MyWin? self.csp = ...)
        csp = load('csp_stored.joblib')
        lda = load('lda_stored.joblib')

        # apply CSP filters + LDA
        fea_tmp = csp.transform(signal_tmp)
        pred_tmp = lda.decision_function(fea_tmp)

        # put in array for prediction values
        # pred_cont.append(list(pred_tmp))
        # pred_cont = pred_cont[-n_output_integrate:] # only keep last values in buffer
        self.pred_buffer[:self.n_output_integrate] = self.pred_buffer[-self.n_output_integrate:]  # shift to the left by one
        self.pred_buffer[self.n_output_integrate] = pred_tmp  # add prediction of this loop
        # (actually, the buffer for the prediction output is not needed if we just low-pass filter but let´s keep it for now in case we switch to a moving average)

        # low-pass filter classifier outputs
        self.motor_imagery_out, self.zi_previous_pred = lfilter(self.b_pred, self.a_pred, pred_tmp, axis=-1, zi=self.zi_previous_pred)
        print(self.motor_imagery_out)
        self.osc_client.send_message("/motor_imagery", self.motor_imagery_out)
        self.motor_imagery_out_save.append(self.motor_imagery_out)  # for debugging

    def update_freqbandspower_allchans(self):
        win_sec = self.buffer_size

        self.fdb.write("{0:4.1f}".format(self.timestmp)+',')
        self.ftb.write("{0:4.1f}".format(self.timestmp)+',')
        self.fab.write("{0:4.1f}".format(self.timestmp)+',')
        self.fbb.write("{0:4.1f}".format(self.timestmp)+',')
        self.fgb.write("{0:4.1f}".format(self.timestmp)+',')
        self.fhgb.write("{0:4.1f}".format(self.timestmp)+',')

        for i in range(24):
            db_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [0.5, 4], win_sec, True)
            tb_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [4, 8], win_sec, True)
            ab_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [8, 12], win_sec, True)
            bb_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [12, 27], win_sec, True)
            gb_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [27, 60], win_sec, True)
            hgb_rel = self.bandpower(self.buffer_channels[i], self.samp_rate, [60, self.samp_rate / 2], win_sec, True)
            endch = ','
            if (i==23):
                endch = ''
            self.fdb.write("{0:4.3f}".format(db_rel)+endch)
            self.ftb.write("{0:4.3f}".format(tb_rel)+endch)
            self.fab.write("{0:4.3f}".format(ab_rel)+endch)
            self.fbb.write("{0:4.3f}".format(bb_rel)+endch)
            self.fgb.write("{0:4.3f}".format(gb_rel)+endch)
            self.fhgb.write("{0:4.3f}".format(hgb_rel)+endch)
            #print("{0:4.2f},{1:4.2f},{2:4.2f},{3:4.2f},{4:4.2f},{5:4.2f}".format(db_rel, tb_rel, ab_rel, bb_rel, gb_rel, hgb_rel))

        self.fdb.write('\n')
        self.ftb.write('\n')
        self.fab.write('\n')
        self.fbb.write('\n')
        self.fgb.write('\n')
        self.fhgb.write('\n')
        self.fdb.flush()
        self.ftb.flush()
        self.fab.flush()
        self.fbb.flush()
        self.fgb.flush()
        self.fhgb.flush()

    def update_freqbandspower(self):
        if self.num_active_chan == 0:
            return

        win_sec = self.buffer_size

        db_rel = self.bandpower(self.buffer_vals, self.samp_rate, [0.5, 4], win_sec, True)
        tb_rel = self.bandpower(self.buffer_vals, self.samp_rate, [4, 8], win_sec, True)
        ab_rel = self.bandpower(self.buffer_vals, self.samp_rate, [8, 12], win_sec, True)
        bb_rel = self.bandpower(self.buffer_vals, self.samp_rate, [12, 27], win_sec, True)
        gb_rel = self.bandpower(self.buffer_vals, self.samp_rate, [27, 60], win_sec, True)
        hgb_rel = self.bandpower(self.buffer_vals, self.samp_rate, [60, self.samp_rate / 2], win_sec, True)
        sum_rel = db_rel + tb_rel + ab_rel + bb_rel + gb_rel + hgb_rel
        self.alpharelpow = ab_rel
        self.gammarelpow = gb_rel
        # print("{0:4.2f},{1:4.2f},{2:4.2f},{3:4.2f},{4:4.2f},{5:4.2f},{6:4.2f}".format(db_rel, tb_rel, ab_rel, bb_rel,
        #                                                                              gb_rel, hgb_rel, sum_rel))

        self.update_attention(tb_rel, bb_rel)

        self.ui.progressBar.setValue(round(db_rel*100))
        self.ui.lineEdit_30.setText(str(int(db_rel*100))+"%")
        self.ui.progressBar_2.setValue(round(tb_rel * 100))
        self.ui.lineEdit_31.setText(str(int(tb_rel * 100)) + "%")
        self.ui.progressBar_3.setValue(round(ab_rel * 100))
        self.ui.lineEdit_32.setText(str(int(ab_rel * 100)) + "%")
        self.ui.progressBar_4.setValue(round(bb_rel * 100))
        self.ui.lineEdit_33.setText(str(int(bb_rel * 100)) + "%")
        self.ui.progressBar_5.setValue(round(gb_rel * 100))
        self.ui.lineEdit_34.setText(str(int(gb_rel * 100)) + "%")
        self.ui.progressBar_6.setValue(round(hgb_rel * 100))
        self.ui.lineEdit_35.setText(str(int(hgb_rel * 100)) + "%")
        # self.buffer_vals = np.zeros(self.samp_rate)

    def update_rawchannels(self, sample, timestamp):
        self.ui.lineEdit.setText(str(round(sample[0])))
        self.ui.lineEdit_2.setText(str(round(sample[1])))
        self.ui.lineEdit_3.setText(str(round(sample[2])))
        self.ui.lineEdit_4.setText(str(round(sample[3])))
        self.ui.lineEdit_5.setText(str(round(sample[4])))
        self.ui.lineEdit_6.setText(str(round(sample[5])))
        self.ui.lineEdit_7.setText(str(round(sample[6])))
        self.ui.lineEdit_8.setText(str(round(sample[7])))
        self.ui.lineEdit_9.setText(str(round(sample[8])))
        self.ui.lineEdit_10.setText(str(round(sample[9])))
        self.ui.lineEdit_11.setText(str(round(sample[10])))
        self.ui.lineEdit_12.setText(str(round(sample[11])))
        self.ui.lineEdit_13.setText(str(round(sample[12])))
        self.ui.lineEdit_14.setText(str(round(sample[13])))
        self.ui.lineEdit_15.setText(str(round(sample[14])))
        self.ui.lineEdit_16.setText(str(round(sample[15])))
        self.ui.lineEdit_17.setText(str(round(sample[16])))
        self.ui.lineEdit_18.setText(str(round(sample[17])))
        self.ui.lineEdit_19.setText(str(round(sample[18])))
        self.ui.lineEdit_20.setText(str(round(sample[19])))
        self.ui.lineEdit_21.setText(str(round(sample[20])))
        self.ui.lineEdit_22.setText(str(round(sample[21])))
        self.ui.lineEdit_23.setText(str(round(sample[22])))
        self.ui.lineEdit_24.setText(str(round(sample[23])))

        self.ui.lineEdit_25.setText(str(round(sample[24])))
        self.ui.lineEdit_26.setText(str(round(sample[25])))
        self.ui.lineEdit_27.setText(str(round(sample[26])))
        self.osc_client.send_message("/gyro/x", round(sample[24]))
        self.osc_client.send_message("/gyro/y", round(sample[25]))
        self.osc_client.send_message("/gyro/z", round(sample[26]))

        self.ui.lineEdit_28.setText(str(round(timestamp)))
        self.osc_client.send_message("/timestamp", round(timestamp))


    def update_attention(self, tb_rel, bb_rel):
        if self.num_active_chan == 0:
            return

        self.theta_relpower.append(tb_rel)
        self.beta_relpower.append(bb_rel)
        if len(self.theta_relpower) == self.relpower_arrays_size:
            self.theta_relpower.pop(0)
            self.beta_relpower.pop(0)

        if len(self.theta_relpower) < self.attention_estimation_points:
            if bb_rel < 0.01:
                bb_rel = 0.01
            self.attention_val = (1 - tb_rel / bb_rel)
        else:
            thetv = 0
            betav = 0
            for i in range(self.attention_estimation_points):
                thetv += self.theta_relpower[len(self.theta_relpower) - i - 1]
                betav += self.beta_relpower[len(self.theta_relpower) - i - 1]
            thetv /= self.attention_estimation_points
            betav /= self.attention_estimation_points
            self.attention_val = (1 - thetv / betav)
        #print(self.attention_val)

        if self.attention_val > 1:
            self.attention_val = 1
        elif self.attention_val < 0:
            self.attention_val = 0
        self.ui.progressBar_7.setValue(round(self.attention_val * 100))
        #print(int(self.attention_val * 100))
        self.osc_client.send_message("/eeg/attention", int(self.attention_val * 100))
        self.osc_client.send_message("/eeg/alpha", int(self.alpharelpow * 100))
        self.osc_client.send_message("/eeg/gamma", int(self.gammarelpow * 100))

    def updatedata(self):
        sample, timestamp = self.inlet.pull_sample()
        self.timestmp = timestamp

        self.average_val = 0
        for i in range(24):
            if self.active_channels[i] == 1:
                self.average_val += sample[i]
            self.buffer_channels[i][self.n_points] = sample[i]

        if self.num_active_chan > 0:
            self.average_val /= self.num_active_chan
        else:
            self.average_val = 0

        self.buffer_vals[self.n_points] = self.average_val
        self.n_points += 1

        self.osc_client.send_message("/eeg/raw", self.average_val)

        if self.n_points == self.buffer_size*self.samp_rate:
            self.update_rawchannels(sample, timestamp)
            self.update_freqbandspower()
            self.update_freqbandspower_allchans()
            self.apply_csp_lda_online()
            self.drawshapes()
            self.ui.lineEdit_29.setText(str(round(self.average_val)))
            self.n_points = 0
            self.buffer_size = self.ui.doubleSpinBox.value()
            self.buffer_vals = np.zeros(int(self.buffer_size*self.samp_rate))
            for i in range(24):
                self.buffer_channels[i] = np.zeros(int(self.buffer_size * self.samp_rate))

    def ext(self):
        """ exit the application """
        self.plot_timer.stop()
        self.timer.stop()
        self.fdb.close()
        self.ftb.close()
        self.fab.close()
        self.fbb.close()
        self.fgb.close()
        self.fhg.close()
        sys.exit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()

    sys.exit(app.exec_())
