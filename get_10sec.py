import numpy as np
import xmltodict
from glob import glob
import scipy.io as io
import scipy.signal as signal
import matplotlib.pyplot as plt
import multiprocessing

def get10sec(filename, lpf=100):
    print(filename)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads = np.array(leads)
############################################################################################################
# Get Raw ECG data
############################################################################################################
    
    dic = xmltodict.parse(open(filename, 'rb'))
    if 'restingecgdata' in dic:
        age = int(dic['restingecgdata']['patient']['generalpatientdata']['age']['years'])
        sex = dic['restingecgdata']['patient']['generalpatientdata']['sex']
        fs = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])
        raw_ecg = dic['restingecgdata']['waveforms']['parsedwaveforms']['#text'].split()
        if 'signalresolution' in dic['restingecgdata']['dataacquisition']['signalcharacteristics']:
            signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['signalresolution'])
        else:
            signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['resolution'])
        ecg = np.empty([len(leads), 10 * fs])
        for i in range(len(leads)):
            ecg[i, :] = raw_ecg[i * fs * 11:i * fs * 11 + fs * 10]
        ecg *= signal_res # scale ECG signal to uV
    else:
        if 'root' in dic:
            fs = int(dic['root']['ECGRecord']['Record']['RecordData'][0]['Waveforms']['XValues']['SampleRate']['#text'])
            raw_ecg = dic['root']['ECGRecord']['Record']['RecordData']
            age = int(dic['root']['ECGRecord']['PatientDemographics']['Age']['#text'])
            sex = dic['root']['ECGRecord']['PatientDemographics']['Sex']
        else:
            fs = int(dic['ECGRecord']['Record']['RecordData'][0]['Waveforms']['XValues']['SampleRate']['#text'])
            raw_ecg = dic['ECGRecord']['Record']['RecordData']
            age = int(dic['ECGRecord']['PatientDemographics']['Age']['#text'])
            sex = dic['ECGRecord']['PatientDemographics']['Sex']
        ecg = np.empty([12, 5000], dtype=float)
        leadnames = []
        for i in range(len(raw_ecg)):
            leadnames.append(raw_ecg[i]['Channel'])
        for i in range(len(leads)):
            j = leadnames.index(leads[i])
            scale = float(raw_ecg[j]['Waveforms']['YValues']['RealValue']['Scale']) * 1000
            data = raw_ecg[j]['Waveforms']['YValues']['RealValue']['Data'].split(',')
            data = ['0' if x == '' else x for x in data]
            ecg[i, :] = np.array(data)
            ecg[i, :] *= scale

    ecg_raw = ecg.copy()

############################################################################################################
# Remove Baseline Wander
############################################################################################################

    b_baseline = io.loadmat('../filters/baseline_filt.mat')['Num'][0]
    ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)
    ecg = signal.filtfilt(b_baseline, 1, ecg, axis=1)
    ecg = ecg - np.median(ecg,axis=1)[:,None]

############################################################################################################
# Remove Pacing Spikes
############################################################################################################

    b_pace, a_pace = signal.butter(2, 80, btype='highpass', fs=500)
    ecg_pace = signal.filtfilt(b_pace, a_pace, ecg, axis=-1)
    ecg = ecg[:,1000:-1000]
    ecg_pace = ecg_pace[:,1000:-1000]

    rms_pace = np.sqrt(np.sum(ecg_pace**2,axis=0)) / np.sqrt(12)
    rms_ecg = np.sqrt(np.sum(ecg**2,axis=0)) / np.sqrt(12)
    peaks_pace = signal.find_peaks(rms_pace, height=250, distance=2)[0]
    peaks_pace = peaks_pace[(peaks_pace < 4995) & (peaks_pace > 5)]

    if (len(peaks_pace) > 0) and (len(peaks_pace) < 50):

        peaks_pace1 = np.empty(len(peaks_pace), dtype=int)
        for j in range(len(peaks_pace)):
            peaks_pace1[j] = np.argmax(rms_ecg[peaks_pace[j]-2:peaks_pace[j]+2])+peaks_pace[j]-2

        # get width of each peak
        for j in range(len(peaks_pace1)):
            if ((rms_ecg[peaks_pace1[j] - 2] < 1/2 * rms_ecg[peaks_pace1[j]]) or (rms_ecg[peaks_pace1[j] - 1] < 1/2 * rms_ecg[peaks_pace1[j]])) and ((rms_ecg[peaks_pace1[j] + 2] < 1/2 * rms_ecg[peaks_pace1[j]]) or (rms_ecg[peaks_pace1[j] + 1] < 1/2 * rms_ecg[peaks_pace1[j]])):
                for lead in range(12):
                    start = np.argmin(np.abs(ecg[lead,peaks_pace1[j]-5:peaks_pace1[j]])) + peaks_pace1[j]-5
                    end = np.argmin(np.abs(ecg[lead,peaks_pace1[j]+1:peaks_pace1[j]+6])) + peaks_pace1[j]+1
                    ecg[:, start:end] = np.nan

        # interpolate over removed pacing spikes
        for lead in range(12):
            nan_idx = np.where(np.isnan(ecg[lead,:]))[0]
            if len(nan_idx) > 0:
                ecg[lead,nan_idx] = np.interp(nan_idx, np.where(~np.isnan(ecg[lead,:]))[0], ecg[lead,~np.isnan(ecg[lead,:])])
    else:
        peaks_pace1 = []
############################################################################################################
# Low Pass Filter & PLI Filters
############################################################################################################

    if lpf == 100:
        # low pass hamming window filter
        b_low = io.loadmat('../filters/lowpass100_filt.mat')['Num'][0]
        ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)
        ecg = signal.filtfilt(b_low, 1, ecg, axis=1)

        # pli filter
        b_pli50 = io.loadmat('../filters/pli50_filt.mat')['Num'][0]
        b_pli60 = io.loadmat('../filters/pli60_filt.mat')['Num'][0]

        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1)

    elif lpf == 150:
        # low pass hamming window filter
        b_low = io.loadmat('../filters/lowpass150_filt.mat')['Num'][0]
        ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)
        ecg = signal.filtfilt(b_low, 1, ecg, axis=1)

        # pli filter
        b_pli50 = io.loadmat('../filters/pli50_filt.mat')['Num'][0]
        b_pli60 = io.loadmat('../filters/pli60_filt.mat')['Num'][0]
        b_pli100 = io.loadmat('../filters/pli100_filt.mat')['Num'][0]
        b_pli120 = io.loadmat('../filters/pli120_filt.mat')['Num'][0]

        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli100, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli120, 1, ecg, axis=1)

    ecg = ecg[:,1000:-1000]
    ecg = ecg - np.median(ecg,axis=1)[:,None]
    ecg_filtered = ecg.copy()

############################################################################################################
# Remove artifacts within lead
############################################################################################################

    ecg = ecg.reshape(12,-1,625)
    num_leads = ecg.shape[0]
    num_windows = ecg.shape[1]
    
    bad_windows = np.zeros([num_leads,num_windows])
    ecg1 = np.abs(ecg)**2
    for i in range(num_leads):
        pwr = np.sum(ecg1[i,:,:], axis=-1)
        rng = np.max(ecg[i,:,:], axis=-1) - np.min(ecg[i,:,:],axis=-1)
        avg_rng = np.median(rng)
        avg_pwr = np.median(pwr)
        bad_windows[i,:] = (pwr > avg_pwr*10) | (rng > avg_rng*5) | (rng > 10000)
    ecg[bad_windows == 1] = 0

############################################################################################################
# Remove uncorrelated leads
############################################################################################################

    ecg = ecg.reshape(12,-1)
    nonzero_leads = np.where(np.sum(np.abs(ecg),axis=1) != 0)[0]
    corrs = np.abs(np.corrcoef(ecg[nonzero_leads]))
    corrs = np.sum(corrs,axis=1) - 1
    corrs /= (len(nonzero_leads) - 1)
    bad_leads = np.zeros(12)
    bad_leads[nonzero_leads[corrs < 0.05]] = 1
    bad_leads[[2,3,4,5]] = 0

    # if np.sum(bad_leads) > 0:
    #     for i in range(12):
    #         if bad_leads[i] == 1:
    #             plt.plot(ecg[i,:] - 1500*i, color='r')
    #         else:
    #             plt.plot(ecg[i,:] - 1500*i, color='k')
    #     plt.show()

    ecg[bad_leads == 1] = 0

############################################################################################################
# Remove missing leads
############################################################################################################

    ecg = ecg.reshape(12,-1,625)
    missing_leads = np.max(np.abs(ecg), axis=2)
    missing_leads = np.median(missing_leads, axis=1) < 100
    ecg = ecg.reshape(12,-1)
    ecg[missing_leads == 1] = 0
    if missing_leads[2]: #lead III = lead II - lead I
        ecg[2] = ecg[1] - ecg[0]
    if missing_leads[3]: #lead aVR = -(lead I + lead II)/2
        ecg[3] = -(ecg[0] + ecg[1])/2
    if missing_leads[4]: #lead aVL = lead I - lead II/2
        ecg[4] = ecg[0] - ecg[1]/2
    if missing_leads[5]: #lead aVF = lead II - lead I/2
        ecg[5] = ecg[1] - ecg[0]/2
    missing_leads[[2,3,4,5]] = 0

    if np.sum(missing_leads) > 1 or missing_leads[0] or missing_leads[1] or missing_leads[6] or missing_leads[11]:
        poor_quality = True
    else:
        poor_quality = False
        if missing_leads[7]:
            ecg[7] = (ecg[6] + ecg[8])/2
        elif missing_leads[8]:
            ecg[8] = (ecg[7] + ecg[9])/2
        elif missing_leads[9]:
            ecg[9] = (ecg[8] + ecg[10])/2
        elif missing_leads[10]:
            ecg[10] = (ecg[9] + ecg[11])/2


    ecg_clean = ecg.copy()

    return {'ecg_raw': ecg_raw, 'ecg_filtered': ecg_filtered, 'ecg_clean': ecg_clean, 'poor_quality': poor_quality, 'fs': fs, 'leads': leads, 'pacing_spikes': peaks_pace1, 'age': age, 'sex': sex}
    