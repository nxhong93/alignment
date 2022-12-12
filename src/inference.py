import __init__
import pandas as pd
import numpy as np
import os
import argparse
import json
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from zalo_utils import seperateData, readTxt, RawSplit, textLine
from ISMIR2020_U_Nets_SVS.source_separation.models.tfc_tdf_net import TFC_TDF_NET_Framework
from model import AcousticModel
from zalo_config import *
from test_dataset import LyricsTestDataset
from alignment import alignment



def AlignmentLysics(arg):
    audio_name = (arg.audio_path.split('\\')[-1]).split('.')[0]
    templateFolder = './template'
    os.makedirs(templateFolder, exist_ok=True)
    seperate_model = TFC_TDF_NET_Framework(**args)
    seperate_model = seperate_model.load_from_checkpoint(arg.seperate_path)
    templateFile = seperateData(arg.audio_path, templateFolder, seperate_model, sample_rate)

    alignment_model = AcousticModel(**modelConfig.model_params)
    state = torch.load(arg.alignment_path, map_location=lambda storage, loc: storage)
    alignment_model.load_state_dict(state)
    alignment_model.eval()
    resolution = 256/sample_rate*3

    df = pd.DataFrame({
        'audioPath': [arg.audio_path],
        'labelPath': [arg.label_path],
        'cleanPath': [templateFile],
        'fileName': [audio_name],
    })
    if '.txt' in arg.label_path:
        df['raw_lines'] = df.labelPath.apply(lambda x: readTxt(x))
        df['line_list'] = df.raw_lines.apply(lambda x: RawSplit(x))
    elif '.json' in arg.label_path:
        df[['raw_lines', 'line_list']] = df.labelPath.apply(lambda x: textLine(x))

    ds = LyricsTestDataset(df, sample_rate, hdf_dir=templateFolder, specConfig=melSpecConfig, transformType='spectrogram')
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_fn)

    audio, idx, seqs = next(iter(dl))
    idx = idx[0][0]
    lyrics, audio_names, audio_length = seqs[0]
    song_lyrics = df[df.fileName==audio_names]['raw_lines'].values[0].split()
    output = alignment_model(audio)
    output = F.log_softmax(output, dim=2)

    song_pred = output.reshape(-1, output.shape[-1]).detach().cpu().numpy()
    total_length = int(audio_length / sample_rate // resolution)
    song_pred = song_pred[:total_length, :]

    # smoothing
    P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
    song_pred = np.log(np.exp(song_pred) + P_noise)

    # dynamic programming alignment
    word_align = alignment(song_pred, lyrics, idx)
    all_result = []
    line_start = word_align[0][0]
    next_word = None
    line_list = []
    for index, (start, end) in enumerate(word_align):
        word = song_lyrics[index]
        start = int(1000*start*resolution)
        end = int(1000*end*resolution)
        word_dict = {
            "s": start,
            "e": end,
            "d": word
        }
        line_list.append(word_dict)
        if index!=len(word_align)-1:
            next_word = song_lyrics[index+1]
        if next_word.istitle() or index==len(word_align)-1:
            line_end = end
            line_dict = {
                "s": line_start,
                "e": line_end,
                "l": line_list
            }
            all_result.append(line_dict)
            line_list = []
            if index!=len(word_align)-1:
                line_start = int(1000*word_align[index+1][0]*resolution)

    with open(f'{templateFolder}/{audio_names}.json', 'w', encoding='utf-8') as f:
        json.dump(all_result, f, ensure_ascii=False)

    os.remove(templateFile)




if __name__=='__main__':
    name = '38303237345f3134'
    parser = argparse.ArgumentParser(description='Inference Config', add_help=False)
    parser.add_argument('--audio_path', default=f'.\private_test\songs\{name}.wav')
    parser.add_argument('--label_path', default=f'.\private_test\sample_labels\{name}.json')
    parser.add_argument('--seperate_path', default='./ISMIR2020_U_Nets_SVS/etc/checkpoints/tfc_tdf_net/debug_large/vocals_epoch=2007.ckpt')
    parser.add_argument('--alignment_path', default='./save/Fold1.pth')
    arg = parser.parse_args()

    AlignmentLysics(arg)
