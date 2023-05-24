import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset
import itertools
import glob

def pad_to_patch_size(x):
    assert x.ndim == 3
    x_pad = np.pad(x, ((0, 1), (0, 1), (0, 1)), 'constant')
    return x_pad


class Dataset_shared(Dataset):
    'Generates data'
    def __init__(self, list_IDs,  data_root,  
                num_subs = 9,  volume = (111, 127, 111), 
                delay = None, subject_id_root='/data/home/clip_mused/dataset/HCP/subject_IDs_train.txt',
                padding=False, 
                fea_root=None, fea_model=None, dim_rd=None, 
                hlv_fea_model=None, llv_fea_model=None, 
                hlv_dim_rd=None, llv_dim_rd=None,
                layer=None, llv_layer='0', hlv_layer='-1', split='train',
                sel_label=None, sel_label_path=None,
                fea_postproc=False, threshold=0.0):
        'Initialization'
        self.sel_label = sel_label
        self.sel_label_path = sel_label_path
        self.split = split
        self.subjects = np.genfromtxt(subject_id_root, delimiter = ',', dtype = str)[:num_subs]
        self.movie_names = ['MOVIE1_CC1', 'MOVIE2_HO1', 'MOVIE3_CC2', 'MOVIE4_HO2']

        self.data_root = data_root 
        self.vol_root = os.path.join(self.data_root, 'preprocessed', 'MinMax')
        self.wordnet_path = os.path.join(self.data_root, 'features', 'WordNetFeatures.hdf5')
        wordnet_file = h5py.File(self.wordnet_path, 'r')
    
        self.wordnet = []
        for movie_name in self.movie_names:
            self.wordnet.append(wordnet_file[movie_name][:])
        if self.sel_label:
            self.sel_label_idx = np.load(self.sel_label_path)
            self.class_num = self.sel_label_idx.shape[-1]
        else:
            self.class_num = wordnet_file[movie_name][:].shape[-1]
        wordnet_file.close()

        self.fea_root = fea_root
        if self.fea_root:
            if not layer:
                self.hlv_fea_path = os.path.join(self.fea_root, hlv_fea_model, hlv_dim_rd)
                hlv_files = glob.glob(os.path.join(self.hlv_fea_path, self.split+'_'+'*.npy'))
                hlv_files.sort()
                self.llv_fea_path = os.path.join(self.fea_root, llv_fea_model, llv_dim_rd)
                llv_files = glob.glob(os.path.join(self.llv_fea_path, self.split+'_'+'*.npy'))
                # print(os.path.join(self.fea_path, split_name+'_'+'*.npy'))
                llv_files.sort()
                print('high-level features,', hlv_files[int(hlv_layer)])
                print('low-level features,', llv_files[int(llv_layer)])
                features_hlv = np.load(hlv_files[int(hlv_layer)])
                features_llv = np.load(llv_files[int(llv_layer)])
                if fea_postproc:
                    features_hlv = self.post_proc(features_hlv, threshold)
                    features_llv = self.post_proc(features_llv, threshold)

                features = np.concatenate((features_hlv, features_llv), 1)
                self.hfdim = features_hlv.shape[-1]
                self.lfdim = features_llv.shape[-1]
            else:
                self.fea_path = os.path.join(self.fea_root, fea_model, dim_rd)
                files = glob.glob(os.path.join(self.fea_path, self.split+'_'+'*.npy'))
                files.sort()
                if layer == 'all':
                    features = []
                    for file in files:
                        features.append(np.load(file))
                    features = np.hstack(features)
                else:
                    file = files[int(layer)]
                    features = np.load(file)
            self.features = features
            self.fea_dim = self.features.shape[-1]
        self.num_subs = num_subs
        self.delay = delay

        self.volume = volume 
        self.list_IDs = list_IDs
        
        # for transformer patchify
        self.padding = padding
        

    def post_proc(self, fea, thr):
        ''' post-process of features (threshold and normalization) '''
        # thresholding values to [-thr, thr]. 
        if thr > 0:
            fea[np.where(fea>thr)] = thr
            fea[np.where(fea<-thr)] = -thr
        # l-2 norm.
        fea = fea / np.linalg.norm(fea, ord=2, axis=-1, keepdims=True)
        return fea

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
        sub_id, fea_idx, movie, frame = self.list_IDs[index]
        out = [sub_id]
        x = np.load(os.path.join(self.vol_root, sub_id, 'MOVIE'+ movie+'_MNI.npy'), 
                            mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]
        if self.padding:
            x = pad_to_patch_size(x)
        x = np.expand_dims(x, 0)
        out.append(x)
        if self.fea_root:
            y = self.features[int(fea_idx)]
            out.append(y)
        z = self.wordnet[int(movie)-1][int(int(frame)/24)]
        if self.sel_label:
            z = z[self.sel_label_idx]
        out.append(z)
        return out


class Dataset_individual_test(Dataset):
    'Generates test data'
    def __init__(self,  data_root, movie_idx = '4',
                subject = None, volume = (111, 127, 111), 
                delay = None, clips_root = '/data/home/clip_mused/dataset/HCP/clip_times_24.npy', 
                padding=False, 
                fea_root=None, fea_model=None, dim_rd=None, 
                hlv_fea_model=None, llv_fea_model=None, 
                hlv_dim_rd=None, llv_dim_rd=None,
                layer=None, llv_layer='0', hlv_layer='-1', split='train',
                sel_label=None, sel_label_path=None):
        'Initialization'
        self.sel_label = sel_label
        self.sel_label_path = sel_label_path
        self.split = split
        
        self.movie_names = ['MOVIE1_CC1', 'MOVIE2_HO1', 'MOVIE3_CC2', 'MOVIE4_HO2']

        self.data_root = data_root 
        self.vol_root = os.path.join(self.data_root, 'preprocessed', 'MinMax')
        self.wordnet_path = os.path.join(self.data_root, 'features', 'WordNetFeatures.hdf5')
        wordnet_file = h5py.File(self.wordnet_path, 'r')
        self.wordnet = []
        for movie_name in self.movie_names:
            self.wordnet.append(wordnet_file[movie_name][:])
        if self.sel_label:
            self.sel_label_idx = np.load(self.sel_label_path)
            self.class_num = self.sel_label_idx.shape[-1]
        else:
            self.class_num = wordnet_file[movie_name][:].shape[-1]
        wordnet_file.close()

        self.fea_root = fea_root
        if self.fea_root:
            if not layer:
                self.hlv_fea_path = os.path.join(self.fea_root, hlv_fea_model, hlv_dim_rd)
                hlv_files = glob.glob(os.path.join(self.hlv_fea_path, self.split+'_'+'*.npy'))
                hlv_files.sort()
                self.llv_fea_path = os.path.join(self.fea_root, llv_fea_model, llv_dim_rd)
                llv_files = glob.glob(os.path.join(self.llv_fea_path, self.split+'_'+'*.npy'))
                # print(os.path.join(self.fea_path, split_name+'_'+'*.npy'))
                llv_files.sort()
                print('high-level features,', hlv_files[int(hlv_layer)])
                print('low-level features,', llv_files[int(llv_layer)])
                features_hlv = np.load(hlv_files[int(hlv_layer)])
                features_llv = np.load(llv_files[int(llv_layer)])
                features = np.concatenate((features_hlv, features_llv), 1)
                self.hfdim = features_hlv.shape[-1]
                self.lfdim = features_llv.shape[-1]
            else:
                self.fea_path = os.path.join(self.fea_root, fea_model, dim_rd)
                files = glob.glob(os.path.join(self.fea_path, self.split+'_'+'*.npy'))
                files.sort()
                if layer == 'all':
                    features = []
                    for file in files:
                        features.append(np.load(file))
                    features = np.hstack(features)
                else:
                    file = files[int(layer)]
                    features = np.load(file)
            self.features = features
            self.fea_dim = self.features.shape[-1]

        self.subject = subject
        self.delay = delay
        self.volume = volume
        clips = np.load(clips_root, allow_pickle=True)
        idxs = clips.item().get(movie_idx)
        frame_idx = []
        for c in range(len(idxs)-1): ## Get rid of the last segment (it is repeated in the first 3 movies)
            frame_idx.append(np.arange(idxs[c,0]/24, idxs[c,1]/24).astype('int'))
        self.list_IDs = np.array(list(itertools.chain(*frame_idx)))*24  # sample a frame every 24 frames, sum up to 699
        self.movie_idx = movie_idx
        
        # for transformer patchify
        self.padding = padding

    def __len__(self):
        return len(self.list_IDs)   # 699

    def __getitem__(self, index):
        # Store sample
        frame = self.list_IDs[index]
        out = [self.subject]
        x = np.load(os.path.join(self.vol_root, self.subject, 'MOVIE'+ self.movie_idx +'_MNI.npy'), 
                                mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]   # BOLD volume
        if self.padding:
            x = pad_to_patch_size(x)
        x = np.expand_dims(x, 0)
        out.append(x)
        if self.fea_root:
            y = self.features[index]
            out.append(y)
        z = self.wordnet[int(self.movie_idx)-1][int(int(frame)/24)]
        if self.sel_label:
            z = z[self.sel_label_idx]
        out.append(z)
        return out
