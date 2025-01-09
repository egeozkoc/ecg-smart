# 1) Combine Data
# 2) Deidentify XMLs
# 3) Get Outcomes
# 4) Find Missing Leads

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pandas as pd
import json

from get_10sec import get10sec
from get_median import getMedianBeat
from get_fiducials import getFiducials
from get_features import getFeatures
from tkinter.filedialog import askdirectory

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)  # Convert numpy integers to Python int
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)  # Convert numpy floats to Python float
        elif isinstance(obj, (np.bool_,)):  # Convert numpy booleans to Python bool
            return bool(obj)
        else:
            return super().default(obj)  # Use the default behavior for other types

class ECG:
    def __init__(self, filename):
        self.filename = filename
        self.id = filename.split('\\')[-1].split('.')[0]
        self.waveforms = {'ecg_10sec_clean': None, 'ecg_median': None, 'beats': None}
        self.fiducials = {'local': None, 'global': None, 'pacing_spikes': None, 'r_peaks': None}
        self.features = {}
        self.demographics = {'age': None, 'sex': None}
        self.leads = None
        self.fs = None
        self.poor_quality = None

    def processRawData(self):
        data = get10sec(self.filename, 100)
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


def process_file(args):
    filename, folder1 = args
    ecg = ECG(filename)
    ecg.processRawData()
    ecg.processMedian()
    ecg.segmentMedian()
    ecg.processFeatures()

    # convert ecg to dictionary
    ecg = {k: v for k, v in ecg.__dict__.items() if v is not None}

    np.save(folder1 + '/' + ecg['id'] + '.npy', ecg)

    # # save ecg to json
    # with open(folder1 + '/' + ecg['id'] + '.json', 'w') as f:
    #     json.dump(ecg, f, cls=NumpyEncoder, indent=4)

def ecg2csv(filenames):
    df_all = pd.DataFrame()
    for filename in filenames:
        print(filename)
        ecg = np.load(filename, allow_pickle=True).item()
        df_pt = pd.DataFrame(ecg.features, index=[ecg.id])
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

    # browse to get folders
    print('Select folder containing raw ECGs')
    folder = askdirectory()
    print('Select folder to save processed ECGs')
    folder1 = askdirectory()

    filenames = glob(folder+'/**/*.xml', recursive=True) + glob(folder+'/**/*.hea', recursive=True) + glob(folder+'/**/*.json', recursive=True)
    # ask user if they want to run in parallel
    parallel = input('Run in parallel? (y/n) ')
    parallel = parallel.lower()
    if parallel == 'y':
        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
        pool.map(process_file, [(filename, folder1) for filename in filenames])
    else:
        for filename in filenames:
            process_file((filename, folder1))

    # filenames1 = glob('../ecgs100/*.npy')
    # # poor_quality2csv(filenames1)
    # ecg2csv(filenames1)
    