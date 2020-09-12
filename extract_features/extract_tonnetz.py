import os
import numpy as np
import h5py as h5
import hdfdict
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter('ignore')

datasets_path = '../data'
#datasets = ['gtzan', 'extendedballroom', 'ismir04', 'homburg']
datasets = ['homburg']

def extract_tonnetz(x, sr=44100):
    return librosa.feature.tonnetz(y=x, sr=sr)

def process_song(song):
    try:
        x, sr = librosa.load(song, sr=44100)
    except:
        return None

    slices = [x[i * sr : (i + 1) * sr] for i in range(int(len(x)/sr))]
    if len(x) > len(slices) * sr:
        extra = len(x) - len(slices) * sr
        extra_samples = x[-extra:]
        last_slice = np.concatenate([extra_samples, np.zeros(sr - extra)])
        slices.append(last_slice)

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_tonnetz, slices)
    return result

def process_genre(i, folder):
    dataset = {
        'genre': [],
        'tonnetz': [],
        'label': []
    }

    files = sorted(os.listdir(folder))
    files = [os.path.join(folder, f) for f in files]
    for f in tqdm(files):
        chroma = process_song(f)
        if chroma:
            for c in chroma:
                dataset['genre'].append(folder)
                dataset['tonnetz'].append(c)
                dataset['label'].append(i)

    dataset['genre'] = np.array(dataset['genre']).astype('S')
    dataset['tonnetz'] = np.array(dataset['tonnetz'])
    dataset['label'] = np.array(dataset['label'])

    return dataset

def process_genres(dataset):
    dataset_path = os.path.join(datasets_path, dataset)
    folders = sorted(os.listdir(dataset_path))
    folders = [os.path.join(dataset_path, folder) for folder in folders]
    features_path = '../features/tonnetz'
    for i, folder in enumerate(tqdm(folders)):
        data = process_genre(i, folder)
        dst_path = folder.split('/')[-2]
        folder = folder.split('/')[-1]
        print('  ', folder)
        dst_path = os.path.join(features_path, dst_path, folder + '.hdf5')
        hdfdict.dump(data, dst_path)
    
if __name__ == '__main__':
    for dataset in datasets:
        print(dataset.upper(), ':')
        process_genres(dataset)
