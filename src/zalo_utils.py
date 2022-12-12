import numpy as np
import pandas as pd
import json
import os
import warnings
from viphoneme import vi2IPA_split, syms
import librosa


def labelTransfer(json_path):
    delimit = '/'
    with open(json_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    fileList, list_length_text = [], []
    for row in file:
        list_word = row['l']
        for word in list_word:
            word['word_duration'] = word['e'] - word['s']
            word['phoneme'] = vi2IPA_split(word['d'], delimit=delimit).split(delimit)[1:-3]
            fileList.append(word)
            list_length_text.append(len(word['d'].split(' ')))
    
    return pd.Series([fileList, np.max(list_length_text)])


def load(path, sr, mono=True, offset=0., duration=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        y = y[np.newaxis, :] # (channel, sample)

    return y, curr_sr


def g2p_vn(word):
    delimit='/'
    return vi2IPA_split(word, delimit=delimit).split(delimit)[1:-3]


def phone2seq(text):
    phone2int = {v: i for i, v in enumerate(syms)}
    seq = []
    for c in text:
        if c in syms:
            idx = phone2int[c]
            seq.append(idx)
    return np.array(seq)


def durationCal(path, sr=16000):
    y = librosa.load(path, sr=sr)[0]
    return float(librosa.get_duration(y))


def seperateData(audio_path, save_folder, model, sample_rate):
    name = audio_path.split('\\')[-1]
    y = librosa.load(audio_path, sr=sample_rate, mono=False)[0]
    if len(y.shape)!=2:
        y = np.array([y, y])
    templateFile = os.path.join(save_folder, name)
    model.separate_track(y.T, templateFile)
    
    return templateFile


def readTxt(labelFile):
    with open(labelFile, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def RawSplit(raw):
    raw = raw[0].upper() + raw[1:]
    raw_list = raw.split(' ')
    first_index = [idx for idx, word in enumerate(raw_list) if word.istitle()] + [len(raw_list)]
    line_list = [' '.join(raw_list[first_index[i]:first_index[i+1]]) for i in range(len(first_index)-1)]
    return line_list

    
def textLine(labelFile):
    with open(labelFile, 'r', encoding='utf-8') as f:
        labelDict = json.load(f)
    row_line, line_list = [], []
    for line in labelDict:
        line_conten = line['l']
        word_line = [i['d'] for i in line_conten]
        row_line.extend(word_line)
        line_list.append(word_line)
    row_line = ' '.join(row_line)
    line_list = [' '.join(i) for i in line_list]

    return pd.Series([row_line, line_list])