from viphoneme import syms
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

args = {}

# FFT params
args['target_name'] = 'vocals'
args['n_fft'] = 4096
args['hop_length'] = 1024
args['num_frame'] = 128

# SVS Framework
args['spec_type'] = 'complex'
args['spec_est_mode'] = 'mapping'

# Other Hyperparams
args['optimizer'] = 'adam'
args['lr'] = 0.001
args['dev_mode'] = False
args['train_loss'] = 'spec_mse'
args['val_loss'] = 'raw_l1'
args['layer_level_init_weight'] = False
args['unfreeze_stft_from'] = -1

# DenseNet Hyperparams

args ['n_blocks'] = 9
args ['input_channels'] = 4
args ['internal_channels'] = 24
args ['first_conv_activation'] = 'relu'
args ['last_activation'] = 'identity'
args ['t_down_layers'] = None
args ['f_down_layers'] = None
args ['tif_init_mode'] = None

# TFC_TDF Block's Hyperparams
args['n_internal_layers'] =5
args['kernel_size_t'] = 3
args['kernel_size_f'] = 3
args['tfc_tdf_activation'] = 'relu'
args['bn_factor'] = 16
args['min_bn_units'] = 16
args['tfc_tdf_bias'] = True


train_label_path = './train/labels/'
train_audio_path = './train/songs/'
train_clean_path = './train/song_clean/'
test_lyric_path = './public_test/lyrics/'
test_audio_path = './public_test/songs/'
test_clean_path = './test_clean/'
private_lyric_path = './private_test/sample_labels/'
private_audio_path = './private_test/songs/'
private_clean_path = './private_clean/'

sample_rate = 44100
phone2int = {v: i for i, v in enumerate(syms)}

vn_string = 'aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịlkmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ'
vn_dict =  {vn_string[i]: i for i in range(len(vn_string))}

melSpecConfig = {'n_mels': 128, 'n_fft': 512} 
mfccConfig = {"n_mfcc": 13, "melkwargs": {"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}}
delimit = '/'
kfold = 10
seed = 15



class modelConfig:
    sr = sample_rate
    fold_num = kfold
    batch_size = 8
    predict_batch_size = 2*batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    seed = 111
    model_params = {
        'n_cnn_layers': 1,
        'rnn_dim': 128,
        'n_class': len(syms)+2,
        'n_feats': 32
    }
    transformType = 'spectrogram'
    if transformType=='spectrogram':
        dataset_config = {
            'sr': sample_rate,
            'time_window': 5.4,
            'specConfig': {'n_mels': 128, 'n_fft': 512},
            'hdf_dir': 'hdf',
            'in_memory': False
        }
    elif transformType=='mfcc':
        dataset_config = {
            'sr': sample_rate,
            'time_window': 5,
            'specConfig': {"n_mfcc": 13, "melkwargs": {"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}},
            'hdf_dir': 'hdf',
            'in_memory': False
        }
    save_path = 'save'
    opt_type = 'adamw'
    scheduler_type = 'cosine'
    lr = 1e-4
    n_epochs = 200
    verbose = 1
    verbose_step = 1
    grad_max_norm = 3
    accumulation_step = 1
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerReduce = torch.optim.lr_scheduler.ReduceLROnPlateau
    reduce_params = dict(
        mode='min',
        factor=0.2,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-7
    )
    SchedulerCosine = CosineAnnealingLR
    cosine_params = {
        'T_max': 50,
        'eta_min': 1e-7,
        'verbose': False
    }
    has_warmup = True
    warmup_params = {
        'multiplier': 2,
        'total_epoch': 1,
    }
    apex = False
    number_check = 7