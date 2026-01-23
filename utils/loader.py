'''
Dataset Builder
'''
import os
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
import torch
import torch.nn.functional as F
from utils.processor.base import Processor
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
JOINT_ID = 9
STRIDE = 64



def csvloader(file_path: str, **kwargs):
    '''
    Loads csv data
    '''
    file_data = pd.read_csv(file_path, index_col=False, header = 0).dropna().bfill()
    # num_col = file_data.shape[1]
    # num_extra_col = num_col % 
    # cols_to_select = num_col - num_extra_col
    if 'skeleton' in file_path: 
        cols = 96
    else: 
        cols  = 3
    activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs):
    '''
    Loads MatLab files 
    '''
    key = kwargs.get('key',None)
    assert key in ['d_iner' , 'd_skel'] , f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv' : csvloader, 
    'mat' : matloader
}

def avg_pool(sequence : np.array, window_size : int = 5, stride :int =1, 
             max_length : int = 512 , shape : int = None) -> np.ndarray:

    '''
    Executes average pooling to smoothen out the data

    '''
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    '''
    Pools and pads the sequence to uniform length

    Args:
        sequence : data 
        max_sequence_length(int) : the fixed length of data
        input_shape: shape of the data
    Return: 
        new_sequence: data after padding
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def sliding_window(data : np.ndarray, clearing_time_index : int, max_time : int, 
                   sub_window_size : int, stride_size : int ,label : int) -> np.ndarray:
    '''
    Sliding Window
    '''
    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data['skeleton'].shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )
    for key in data.keys():
        data[key] = data[key][sub_windows]
    data['labels'] =  np.repeat(label, len(data['skeleton']))
    return data

def sliding_window_alignment(data : np.ndarray, clearing_time_index : int, max_time : int, 
                   sub_window_size : int, stride_size : int ,label : int) -> np.ndarray:
    '''
    Sliding Window
    '''
    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )

    data = data[sub_windows]
    return data

def filter_data_by_ids(data : np.ndarray, ids : List[int]):
    '''
    Index the different modalities with only selected ids

    Arguements: 
        data : data dictionary with skeleton and inertial data
        skeleton_ids: skeleton data selected ids
        inertial_ids: inertial data selected ids
    Return : 
        changed data with selected ids
    '''
    return data[ids, :]




def filter_repeated_ids(path : List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filtering indices those match with mutliple other indices
    Arguements: 
        path : Tuple of indices defining the DTW path
    
    Return : 
        set of tuples containing the unique indices

    '''
    seen_first = set()
    seen_second = set()

    for (first , second) in path : 

        if first not in seen_first and second not in  seen_second: 
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second
def get_segment(attn_weights, accelerometer_raw, skeleton_raw) -> Tuple[np.ndarray, np.ndarray]:
# get the row, column index of the maximum value in the attention weights
# ...existing code...


    # attn_weights: [T_acc, T_skel] or [T_q, T_k]
    attn_np = attn_weights.detach().numpy()

    # Find the indices of the maximum attention value
    max_idx = np.unravel_index(np.argmax(attn_np, axis=None), attn_np.shape)
    acc_idx, skel_idx = max_idx  # (row, col) = (acc, skel)
    # get the segment of length with mi
    window_size = min(len(accelerometer_raw)-1, len(skeleton_raw)-1)
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for symmetry
    start_acc = max(0, acc_idx - window_size // 2)
    if start_acc == 0:
        end_acc = window_size
    else:   
        end_acc = min(attn_np.shape[1], acc_idx + window_size // 2)
        if end_acc  == attn_np.shape[1]:
            start_acc = end_acc - window_size
    # create a segment of skeleton and accelerometer both having the same length and the highest attention index in it
    start_skel = max(0, skel_idx - window_size // 2)
    if start_skel == 0:
        end_skel = window_size
    else:   
        end_skel = min(attn_np.shape[0], skel_idx + window_size // 2)
        if end_skel  == attn_np.shape[0]:
            start_skel = end_skel - window_size 

    segment_acc = accelerometer_raw[start_acc:end_acc]
    segment_skel = skeleton_raw[start_skel:end_skel]
    return segment_acc, segment_skel

# def align_sequence(data : Dict[str, np.ndarray] ) -> Dict[str, np.ndarray]: 
#     '''
#     Matching the skeleton and phone data using dynamic time warping 
#     Args: 
#         dataset: Dictionary containing skeleton and accelerometer data

#     '''
#     #joint_id = 9
#     joint_id = JOINT_ID
#     #seperating left wrist joint data
#     dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
#     skeleton_joint_data = data['skeleton'][:, (joint_id -1) * 3 : joint_id * 3 ]
#     inertial_data = data[dynamic_keys[0]]
#     if len(dynamic_keys) > 1: 
#         gyroscope_data = data[dynamic_keys[1]]
#         min_len = min(inertial_data.shape[0], gyroscope_data.shape[0])
#         inertial_data = inertial_data[:min_len, :]
#         data[dynamic_keys[1]] = gyroscope_data[:min_len, :]

#    # calcuating frobenis norm of skeleton and intertial data 
#     skeleton_frob_norm = norm(skeleton_joint_data, axis = 1)
#     interial_frob_norm = norm(inertial_data, axis = 1)
    
#     distance, path  = fastdtw(interial_frob_norm[:, np.newaxis], skeleton_frob_norm[:, np.newaxis],dist = euclidean)


#     interial_ids ,skeleton_idx ,= filter_repeated_ids(path)
#     data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
#     for key in dynamic_keys: 
#         data[key]= filter_data_by_ids(data[key],list(interial_ids))
#     #skeleton_after_dtw = data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
#     #plt.plot( np.arange(skeleton_before_dtw.shape[0]),skeleton_before_dtw[..., 0], '--r',
#              #np.arange(skeleton_after_dtw.shape[0]), skeleton_after_dtw[..., 0], '--g')
    
#     # plt.savefig(f'exps/comparision/comparision_before_after_dtw_{idx}.jpg')
#     # plt.close()
#     return data
def align_sequence(data : Dict[str, np.ndarray] ) -> Dict[str, np.ndarray]:
    '''
    Align skeleton and inertial data using STUMPY matrix profile
    
    Args:
        data: Dictionary containing skeleton and inertial data
    
    Returns:
        data: Dictionary with aligned sequences
    '''
    try:
        import stumpy
        from stumpy import core
    except ImportError:
        print("STUMPY not installed, skipping alignment")
        return data
    
    # Get joint data (joint 9 - wrist)
    joint_id = JOINT_ID
    skeleton_joint_data = np.array(data['skeleton'][:, (joint_id - 1) * 3 : joint_id * 3], dtype= np.float64)
    
    # Get inertial data (accelerometer)
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])
    inertial_data = np.array(data[dynamic_keys[0]], dtype = np.float64)
    
    # Calculate magnitude and z-normalize
    mag_skeleton = core.z_norm(np.sqrt(np.sum(skeleton_joint_data**2, axis=1)))
    mag_inertial = core.z_norm(np.sqrt(np.sum(inertial_data**2, axis=1)))
    
    if len(mag_skeleton) > len(mag_inertial):
        query = mag_inertial
        target = mag_skeleton
        full_target = data['skeleton']
    else:
        query = mag_skeleton
        target = mag_inertial
        full_target = inertial_data

    windowed_target = sliding_window_alignment(target, clearing_time_index=len(query)-1, max_time=len(target), 
                                    sub_window_size=len(query), stride_size=1, label=0)
    windowed_full_target = sliding_window_alignment(full_target, clearing_time_index=len(query)-1, max_time=len(target), 
                                    sub_window_size=len(query), stride_size=1, label=0)

    query = np.expand_dims(query, axis=0) #1, len_query
    windowed_target = windowed_target.reshape(windowed_target.shape[0], windowed_target.shape[1])
    pairwise_similarity = cosine_similarity(query, windowed_target)
    correlated_window = windowed_full_target[pairwise_similarity.argmax(axis = 1), :].squeeze(0)
    
    if len(mag_skeleton) > len(mag_inertial):
        data['skeleton'] = correlated_window
    else:
        data[dynamic_keys[0]] = correlated_window
    return data


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    '''Function to fitter noise '''
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0) 

class DatasetBuilder:
    '''
    Builds a numpy file for the data and labels and 

    Args: 
        Dataset: a dataset class containing all matched files
    '''
    def __init__(self , dataset: object, mode: str, max_length: int, task = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool' , 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data = defaultdict(list)
        #self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.diff = []
    def load_file(self, file_path):
        '''
        
        '''
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        #self.set_input_shape(data)
        return data
    

    def _import_loader(self, file_path:str) -> np.array :
        '''
        Reads file and loads data from
         
        '''

        file_type = file_path.split('.')[-1]

        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'

        return LOADER_MAP[file_type]
    
    def process(self, data, label):
        '''
        function implementation to process data
        '''

        if self.mode == 'avg_pool':
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length,
                                      input_shape=data.shape)
        
        else: 
            # sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis = 1))
            # if label == 1: 
            #     #phone height = 25, distance = 200
            #     #meta height = 1 distaince = 10 
            #     peaks , _ = find_peaks(sqrt_sum, height=15, distance=10)
                
            # else: 
            #     #phone height = 15, distance = 500
            #     peaks , _ = find_peaks(sqrt_sum, height=15, distance=15)

            # data = selective_sliding_window(data, window_size= self.max_length,peaks = peaks, label = label, fuse = self.fuse)
            if label == 1:
                stride = 16
            else: 
                stride = STRIDE
            data = sliding_window(data, self.max_length-1, data['skeleton'].shape[0], self.max_length, stride, label)
        return data

    def _add_trial_data(self, trial_data):

        for modality, modality_data in trial_data.items():
            self.data[modality].append(modality_data)
    
    def _len_check(self, d):
        return all(len(v) > 1 for v in d.values())

    def get_size_diff(self, trial_data):
        return trial_data['accelerometer'].shape[0]  - trial_data['skeleton'].shape[0]

    def store_trial_diff(self, difference):
        self.diff.append(difference)
    
    def viz_trial_diff(self):
        value_range = range(min(self.diff) , max(self.diff)+2)
        # plt.hist(self.diff, bins = value_range, edgecolor = 'black', alpha = 0.7)
        # plt.xlabel("Differences")
        # plt.ylabel("Frequency")
        print(len(self.diff))
        counter = Counter(self.diff)

        #   Extract values for plotting


        plt.hist(self.diff, bins=range(min(self.diff), max(self.diff) + 2, 200), edgecolor='black', alpha=0.7)

        # Labels and title
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Numbers")
        plt.savefig("Distribution.png")
        #high_var_segment = df["values"].iloc[max_idx : max_idx + 200].values

    def make_dataset(self, subjects : List[int], fuse : bool): 
        '''
        Reads all the files and makes a numpy  array with all data
        '''
        self.data = defaultdict(list)
        self.fuse = fuse
        self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        count = 0 
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:       
                if self.task == 'fd': 
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                #self.data['labels'] = self.data.get('labels',[])
                trial_data = defaultdict(np.ndarray)
                executed  = True
                for modality, file_path in trial.files.items():
                    #here we need the processor class 
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    #processor = Processor(file_path, self.mode, self.max_length,label, key = key)
                    try: 
                        
                        unimodal_data = self.load_file(file_path)
                        trial_data[modality] = unimodal_data

                        if modality in ['accelerometer' , 'gyroscope']:
                            trial_data[modality] = butterworth_filter(unimodal_data, cutoff=7.5, fs=25)
                        # if modality in ['accelerometer', 'gyroscope'] and unimodal_data.shape[0]>250:
                        #     trial_data[modality] = self.select_subwindow_pandas(unimodal_data)                            
                        # if modality == 'skeleton':
                        #     print(unimodal_data.shape)
                        if not hasattr(unimodal_data, 'shape'):
                            raise AttributeError(f"Data loaded from {file_path} (modality: {modality}) does not have a 'shape' attribute")
                    except Exception as e :
                        executed = False
                        #print(e)
                # trial_difference = self.get_size_diff(trial_data)
                # self.store_trial_diff(trial_difference)
                
                if executed : 
                    try:
                        trial_data = align_sequence(trial_data)
                        trial_data = self.process(trial_data, label)
                        if self._len_check(trial_data):
                            self._add_trial_data(trial_data)
                    except Exception as e :
                        pass
        #self.viz_trial_diff()
        for key in self.data:
            #print(key)
            self.data[key] = np.concatenate(self.data[key], axis=0)
    
    def normalization(self) -> np.ndarray:
        '''
        Function to normalize  the data
        '''

        for key ,value  in self.data.items():        
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)
        return self.data
    
