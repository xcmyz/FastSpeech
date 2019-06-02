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
max_sep_len = 2048
encoder_output_size = 384
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1


# Train
batch_size = 2
epochs = 10000
dataset_path = "dataset"
learning_rate = 1e-3
checkpoint_path = "./model_new"
grad_clip_thresh = 1.0
decay_step = [200000, 500000, 1000000]
save_step = 1200
log_step = 5
clear_Time = 20
