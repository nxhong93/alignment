import numpy as np
import os
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, dataloader
import torch.nn as nn
import torchaudio
from zalo_utils import load, g2p_vn
from zalo_config import *


class LyricsTestDataset(Dataset):
    def __init__(self, df, sr, specConfig, hdf_dir, in_memory=False, transformType='spectrogram'):
        super(LyricsTestDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, 'test.hdf5')
        
        self.sr = sr
        self.specConfig = specConfig
        self.in_memory = in_memory
        self.transformType = transformType

        #check hdf file
        if not os.path.exists(self.hdf_file):
            os.makedirs(hdf_dir, exist_ok=True)

            #create hdf 
            with h5py.File(self.hdf_file, 'w') as f:
                f.attrs['sr'] = sr

                print(f'preprocessing...')
                for idx, example in tqdm(df.iterrows(), leave=True):
                    audio_name = example.fileName
                    y = load(example.cleanPath, sr=self.sr, mono=True)[0]
                    words, raw_lines = self.load_lyrics(example)
                    lyrics_p, words_p, idx_in_full_p, idx_line_p = self.gen_phone_gt(words, raw_lines)
                    annot_num = len(words)
                    line_num = len(idx_line_p)
                    
                    grp = f.create_group(str(idx))
                    grp.create_dataset('inputs', shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["audio_name"] = audio_name
                    grp.attrs["input_length"] = y.shape[1]
                    grp.create_dataset("lyrics_p", shape=(len(lyrics_p), 1), dtype='S100', data=np.array([l_p.encode() for l_p in lyrics_p]))
                    grp.create_dataset("idx_p", shape=(annot_num, 2), dtype=np.int, data=idx_in_full_p)
                    grp.create_dataset("idx_line_p", shape=(line_num, 2), dtype=np.int, data=idx_line_p)
                    

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r") as f:
            if f.attrs["sr"] != sr:
                raise ValueError("Tried to load existing HDF file, but sampling rate is not as expected.")

        with h5py.File(self.hdf_file, "r") as f:
            self.length = len(f) # number of songs
    
    def __getitem__(self, index):
        
        # open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver) 
        audio_length = self.hdf_dataset[str(index)].attrs["input_length"]
        # read audio, name, and lyrics
        audio = self.hdf_dataset[str(index)]["inputs"][0, :].astype(np.float32)
        audio_name = self.hdf_dataset[str(index)].attrs["audio_name"]
        lyrics = self.hdf_dataset[str(index)]["lyrics_p"][:, 0]
        lyrics = [l.decode() for l in lyrics]
        word_idx = self.hdf_dataset[str(index)]["idx_p"][:]
        line_idx = self.hdf_dataset[str(index)]["idx_line_p"][:]

        # chunks = [audio]

        return audio, (word_idx, line_idx), (lyrics, audio_name, audio_length)

    def __len__(self):
        return self.length

    def load_lyrics(self, example):
        raws = example.line_list
        raw_lines = []
        for line in raws:
            lines = []
            for word in line.split(' '):
                words = ''
                for c in word:
                    if c.lower() in vn_string:
                        words += c
                if len(words) > 0:
                    lines.append(words.lower())
            lines = (' ').join(lines).lower()
            raw_lines.append(lines)
        words_lines = (" ").join(raw_lines).split(' ')

        return words_lines, raw_lines

    def gen_phone_gt(self, words, raw_lines):

        # helper function
        def getsubidx(x, y):  # find y in x
            l1, l2 = len(x), len(y)
            for i in range(l1 - l2 + 1):
                if x[i:i + l2] == y:
                    return i
        words_p = []
        lyrics_p = []
        for word in words:
            out = g2p_vn(word)
            words_p.append(out)
            if len(lyrics_p) > 0:
                lyrics_p.append(' ')
            lyrics_p += out
        len_words_p = [len(phones) for phones in words_p]
        idx_in_full_p = []
        s1 = 0
        s2 = s1
        for l in len_words_p:
            s2 = s1 + l
            idx_in_full_p.append([s1, s2])
            s1 = s2 + 1
            # beginning of a line
            idx_line_p = []
            last_end = 0
            for i in range(len(raw_lines)):
                line = g2p_vn(raw_lines[i])
                line = [' ' if i=='_' else i for i in line]
                offset = getsubidx(lyrics_p[last_end:], line)
                assert (offset >= 0)
                assert (line == lyrics_p[last_end + offset:last_end + offset + len(line)])
                idx_line_p.append([last_end + offset, last_end + offset + len(line)])
                last_end += offset + len(line)

        return lyrics_p, words_p, idx_in_full_p, idx_line_p
    
    def collate_fn(self, batch):
        audios, targets, seqs = zip(*batch)
        spectrograms = []
        for audio in audios:
            audio = torch.tensor(audio)
            if self.transformType=='spectrogram':
                spec = nn.Sequential(
                            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **self.specConfig)
                            )(audio).squeeze(0).transpose(0, 1) # time x n_mels
            elif self.transformType=='mfcc':
                spec = nn.Sequential(
                            torchaudio.transforms.MFCC(sample_rate=sample_rate, **self.specConfig)
                            )(audio).squeeze(0).transpose(0, 1) # time x n_mels
            spectrograms.append(spec)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        targets = list(targets)
        seqs = list(seqs)
        return spectrograms, targets, seqs