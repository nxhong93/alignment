import numpy as np
from zalo_utils import phone2seq
from viphoneme import syms


def alignment(song_pred, lyrics, idx):
    audio_length, num_class = song_pred.shape
    lyrics_int = phone2seq(lyrics)
    lyrics_length = len(lyrics_int)

    s = np.zeros((audio_length, 2*lyrics_length+1)) - np.Inf
    opt = np.zeros((audio_length, 2*lyrics_length+1))

    blank = len(syms)+1

    # init
    s[0][0] = song_pred[0][blank]
    # insert eps
    for i in np.arange(1, audio_length):
        s[i][0] = s[i-1][0] + song_pred[i][blank]

    for j in np.arange(lyrics_length):
        if j == 0:
            s[j+1][2*j+1] = s[j][2*j] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 1  # 45 degree
        else:
            s[j+1][2*j+1] = s[j][2*j-1] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 2 # 28 degree

        s[j+2][2*j+2] = s[j+1][2*j+1] + song_pred[j+2][blank]
        opt[j+2][2*j+2] = 1  # 45 degree


    for audio_pos in np.arange(2, audio_length):

        for ch_pos in np.arange(1, 2*lyrics_length+1):

            if ch_pos % 2 == 1 and (ch_pos+1)/2 >= audio_pos:
                break
            if ch_pos % 2 == 0 and ch_pos/2 + 1 >= audio_pos:
                break

            if ch_pos % 2 == 1: # ch
                ch_idx = int((ch_pos-1)/2)
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # last ch -> ch
                b = s[audio_pos-1][ch_pos-2] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][lyrics_int[ch_idx]]
                if a > b and a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                elif b >= a and b >= c:
                    s[audio_pos][ch_pos] = b
                    opt[audio_pos][ch_pos] = 2
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

            if ch_pos % 2 == 0: # eps
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][blank]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][blank]
                if a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

    # retrive optimal path
    path = []
    x = audio_length-1
    y = 2*lyrics_length
    path.append([x, y])
    
    while x > 0 or y > 0:
        if opt[x][y] == 1:
            x -= 1
            y -= 1
        elif opt[x][y] == 2:
            x -= 1
            y -= 2
        else:
            x -= 1
        path.append([x, y])

    path = list(reversed(path))
    word_align = []
    path_i = 0

    word_i = 0
    while word_i < len(idx):
        # e.g. "happy day"
        # find the first time "h" appears
        if path[path_i][1] == 2*idx[word_i][0]+1:
            st = path[path_i][0]
            # find the first time " " appears after "h"
            while  path_i < len(path)-1 and (path[path_i][1] != 2*idx[word_i][1]+1):
                path_i += 1
            ed = path[path_i][0]
            # append
            word_align.append([st, ed])
            # move to next word
            word_i += 1
        else:
            # move to next audio frame
            path_i += 1

    return word_align