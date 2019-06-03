# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Model
word_vec_dim = 384
encoder_n_layer = 6
encoder_head = 2
encoder_conv1d_filter_size = 1536
max_sep_len = 2048
encoder_output_size = 384
decoder_n_layer = 6
decoder_head = 2
decoder_conv1d_filter_size = 1536
decoder_output_size = 384
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

# Train
pre_target = True
n_warm_up_step = 4000
batch_size = 2
epochs = 10000
dataset_path = "dataset"
logger_path = "logger"
alignment_target_path = "alignment_targets"
learning_rate = 1e-3
checkpoint_path = "./model_new"
grad_clip_thresh = 1.0
decay_step = [200000, 500000, 1000000]
save_step = 2000
log_step = 5
clear_Time = 20
