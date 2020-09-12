import os
import hdfdict
import numpy as np

def load_data(feature=None, dataset=None):
    features_dir = '/home/anupambiswas/Yeshwant/genre_classification/features'
    feature_path = os.path.join(features_dir, feature, dataset)
    hdf5_files = os.listdir(feature_path)
    feature = 'chroma' if feature == 'chromagram' else feature
    feature_data = {'genre': [], 
                    'label': [], 
                    feature: [],}
    for f in hdf5_files:
        data = dict(hdfdict.load(os.path.join(feature_path, f)))
        feature_data['genre'] += data['genre'].tolist()
        feature_data['label'] += data['label'].tolist()
        feature_data[feature] += data[feature].tolist()

    feature_data['genre'] = np.array(feature_data['genre'])
    feature_data['label'] = np.array(feature_data['label'])
    feature_data[feature] = np.array(feature_data[feature])

    return feature_data

