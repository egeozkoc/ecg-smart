# 1) Combine Data
# 2) Deidentify XMLs
# 3) Get Outcomes
# 4) Find Missing Leads

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pandas as pd

from get_10sec import get10sec
from get_median import getMedianBeat
from get_fiducials import getFiducials
from get_features import getFeatures

class ECG:
    def __init__(self, filename):
        self.filename = filename
        self.id = filename.split('\\')[-1].split('.')[0]
        self.waveforms = {'ecg_10sec_raw': None, 'ecg_10sec_filtered': None, 'ecg_10sec_clean': None, 'ecg_median': None, 'beats': None}
        self.fiducials = {'local': None, 'global': None, 'pacing_spikes': None, 'r_peaks': None}
        self.features = {}
        self.demographics = {'age': None, 'sex': None}
        self.leads = None
        self.fs = None
        self.poor_quality = None

    def processRawData(self):
        data = get10sec(self.filename, 100)
        self.waveforms['ecg_10sec_raw'] = data['ecg_raw']
        self.waveforms['ecg_10sec_filtered'] = data['ecg_filtered']
        self.waveforms['ecg_10sec_clean'] = data['ecg_clean']
        self.fs = data['fs']
        self.leads = data['leads']
        self.poor_quality = data['poor_quality']
        self.fiducials['pacing_spikes'] = data['pacing_spikes']
        self.demographics['age'] = data['age']
        self.demographics['sex'] = data['sex']

    def processMedian(self):
        data = getMedianBeat(self.waveforms['ecg_10sec_clean'])
        self.waveforms['ecg_median'] = data['median_beat']
        self.waveforms['beats'] = data['beats']
        self.features['rrint'] = data['rrint']
        self.fiducials['rpeaks'] = data['rpeaks']

    def segmentMedian(self):
        data = getFiducials(self.waveforms['ecg_median'], self.features['rrint'], self.fs)
        self.fiducials['local'] = data['local']
        self.fiducials['global'] = data['global']

    def processFeatures(self):
        getFeatures(self)


def process_file(filename):
    ecg = ECG(filename)
    ecg.processRawData()
    ecg.processMedian()
    ecg.segmentMedian()
    ecg.processFeatures()
    np.save('../ecgs100/' + ecg.id, ecg, allow_pickle=True)

def ecg2csv(filenames):
    df_all = pd.DataFrame()
    # outcomes = pd.read_csv('../outcomes/outcomes.csv')
    for filename in filenames:
        print(filename)
        ecg = np.load(filename, allow_pickle=True).item()
        df_pt = pd.DataFrame(ecg.features, index=[ecg.id])
        # df_pt['outcome_omi'] = outcomes.loc[outcomes['id'] == ecg.id, 'outcome_omi'].values[0]
        # df_pt['outcome_acs'] = outcomes.loc[outcomes['id'] == ecg.id, 'outcome_acs'].values[0]
        # df_pt['poor_quality'] = ecg.poor_quality
        # df_pt['outcome_vt'] = outcomes.loc[outcomes['id'] == ecg.id, 'outcome_vt'].values[0]
        df_all = pd.concat([df_all, df_pt])

    df_all.to_csv('features100.csv')

def poor_quality2csv(filenames):
    df = pd.read_csv('../outcomes/outcomes.csv')
    for filename in filenames:
        ecg = np.load(filename, allow_pickle=True).item()
        df.loc[df['id'] == ecg.id, 'poor_quality100'] = int(ecg.poor_quality)
    df.to_csv('../outcomes/outcomes.csv', index=False)
    


# main
if __name__ == '__main__':
    # filenames = glob('../xmls/registry/*.xml')
    # # filenames1 = glob('../ecgs150/*.npy')
    # # filenames1 = [filename.replace('npy','xml').replace('ecgs150','xmls/registry') for filename in filenames1]
    # # filenames = np.setdiff1d(filenames, filenames1)

    # # idx1 = np.where(np.char.find(filenames, 'reg07300-1') != -1)[0][0]
    # # idx2 = np.where(np.char.find(filenames, 'reg07400-1') != -1)[0][0]
    # # filenames = filenames[idx1:idx2]

    # # for filename in filenames[0:]:
    # #     process_file(filename)

    # # run code on multiple cores
    # pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    # pool.map(process_file, filenames)

    filenames1 = glob('../ecgs100/*.npy')
    # poor_quality2csv(filenames1)
    ecg2csv(filenames1)
    