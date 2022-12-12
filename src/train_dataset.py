import numpy as np
import os
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, dataloader
import torch.nn as nn
import torchaudio
from zalo_utils import load, g2p_vn, syms, phone2seq
from zalo_config import *
from sortedcontainers import SortedList
from audio_transform import aug



class LyricsTrainDataset(Dataset):
    def __init__(self, df, partition, sr, time_window, specConfig, hdf_dir, in_memory=False, transformType='mfcc', has_transform=True):
        super(LyricsTrainDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, partition+'.hdf5')
        
        self.input_sample = int(time_window*sr)
        self.sr = sr
        self.hop = (self.input_sample//2)
        self.specConfig = specConfig
        self.in_memory = in_memory
        self.transformType = transformType
        self.has_transform = has_transform
        if self.has_transform:
            self.transform = aug(partition)
        #check hdf file
        if not os.path.exists(self.hdf_file):
            os.makedirs(hdf_dir, exist_ok=True)
            #create hdf 
            with h5py.File(self.hdf_file, 'w') as f:
                f.attrs['sr'] = sr

                print(f'preprocessing...')
                for idx, example in tqdm(df.iterrows(), leave=True):
                    y = load(example.cleanPath, sr=self.sr, mono=True)[0]
                    grp = f.create_group(str(idx))
                    grp.create_dataset('inputs', shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["audio_name"] = example["fileName"]
                    grp.attrs["input_length"] = y.shape[1]

                    # word level
                    annot_num = len(example["listLabel"])
                    lyrics = [sample["d"].encode() for sample in example["listLabel"]]
                    times = np.array([(sample["s"], sample['e']) for sample in example["listLabel"]])

                    # phoneme
                    max_phone = np.max([len(sample['phoneme']) for sample in example['listLabel']])
                    phonemes_encode = [[phone.encode() for phone in sample['phoneme']] for sample in example['listLabel']]
                    
                    grp.attrs["annot_num"] = annot_num
                    
                    # words and corresponding times
                    grp.create_dataset("lyrics", shape=(annot_num, 1), dtype='S100', data=lyrics)
                    grp.create_dataset("times", shape=(annot_num, 2), dtype=times.dtype, data=times)

                    grp.create_dataset("phoneme", shape=(annot_num, max_phone), dtype='S10')

                    for i in range(annot_num):
                        phonemes_sample = phonemes_encode[i]
                        grp["phoneme"][i, :len(phonemes_sample)] = np.array(phonemes_sample)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r") as f:
            if f.attrs["sr"] != sr:
                raise ValueError("Tried to load existing HDF file, but sampling rate is not as expected.")


        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:
            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [( (l - self.input_sample) // self.hop) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):
        # open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)    

        while True:
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]
            # length of audio signal
            audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
            # number of words in this song
            annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]

            # determine where to start
            start_pos = index * self.hop
            end_pos = start_pos + self.input_sample

            # front padding
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # back padding
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0
            
            # read audio and zero padding
            audio = self.hdf_dataset[str(song_idx)]["inputs"][0, start_pos:end_pos].astype(np.float32)            
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)
            if self.has_transform:
                audio = self.transform(data=np.array(audio))['data']

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            try:
                first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                             if self.sr*val/1000 >= start_pos)
            except StopIteration:
                first_word_to_include = np.Inf

            try:
                last_word_to_include = annot_num - 1 - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                             if self.sr*val/1000 <= end_pos)
            except StopIteration:
                last_word_to_include = -np.Inf
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                index = np.random.randint(self.length)
                continue
            phoneme_list = []
            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
                phoneme = self.hdf_dataset[str(song_idx)]["phoneme"][first_word_to_include:last_word_to_include+1]
                phoneme_list = self.convert_phone_list(phoneme)
  
            phone_seq = phone2seq(phoneme_list)
            break

        # del self.hdf_dataset
        return audio, phone_seq
    
    def text2seq(self, text):
        seq = []
        for c in text.lower():
            idx = vn_string.find(c)
            seq.append(idx)
        return np.array(seq)


    def convert_phone_list(self, phoneme):
        ret = []
        for l in phoneme:
            l_decode = [' '] + [s.decode() for s in l if len(s) > 0]
            ret += l_decode
        if len(ret) > 1:
            return ret[1:]
        else:
            return []

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        spectrograms = []
        phones = []
        phone_lengths = []
        for audio, phone_seq in batch:
            audio = torch.Tensor(audio)
            if self.transformType=='spectrogram':
                spec = nn.Sequential(
                    torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, **self.specConfig)
                    )(audio).squeeze(0).transpose(0, 1) # time x n_mels
            elif self.transformType=='mfcc':
                spec = nn.Sequential(
                    torchaudio.transforms.MFCC(sample_rate=self.sr, **self.specConfig)
                    )(audio).squeeze(0).transpose(0, 1) # time x n_mels
            spectrograms.append(spec)

            # get phoneme list (mapped to integers)
            phone_seq = torch.Tensor(phone_seq)
            phones.append(phone_seq)

            phone_lengths.append(len(phone_seq))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        phones = nn.utils.rnn.pad_sequence(phones, batch_first=True)

        return spectrograms, phones, phone_lengths
