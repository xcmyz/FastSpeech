import torch
import torch.nn as nn

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from FastSpeech import FastSpeech
from loss import FastSpeechLoss
from data_utils import FastSpeechDataset, collate_fn, DataLoader
from optimizer import ScheduledOptim
from alignment import get_alignment, get_tacotron2
import hparams as hp


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(FastSpeech()).to(device)
    tacotron2 = get_tacotron2()
    print("FastSpeech and Tacotron2 Have Been Defined")
    num_param = sum(param.numel() for param in model.parameters())
    print('Number of FastSpeech Parameters:', num_param)

    # Get dataset
    dataset = FastSpeechDataset()

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduled_optim = ScheduledOptim(optimizer,
                                     hp.word_vec_dim,
                                     hp.n_warm_up_step,
                                     args.restore_step)
    fastspeech_loss = FastSpeechLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=cpu_count())

    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n------Model Restored at Step %d------\n" % args.restore_step)

    except:
        print("\n------Start New Training------\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    Time = np.array(list())
    Start = time.clock()

    for epoch in range(hp.epochs):
        for i, data_of_batch in enumerate(training_loader):
            start_time = time.clock()

            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            scheduled_optim.zero_grad()

            if not hp.pre_target:
                # Prepare Data
                src_seq = data_of_batch["texts"]
                src_pos = data_of_batch["pos"]
                mel_tgt = data_of_batch["mels"]

                src_seq = torch.from_numpy(src_seq).long().to(device)
                src_pos = torch.from_numpy(src_pos).long().to(device)
                mel_tgt = torch.from_numpy(mel_tgt).float().to(device)
                alignment_target = get_alignment(
                    src_seq, tacotron2).float().to(device)
                # For Data Parallel
                mel_max_len = mel_tgt.size(1)
            else:
                # Prepare Data
                src_seq = data_of_batch["texts"]
                src_pos = data_of_batch["pos"]
                mel_tgt = data_of_batch["mels"]
                alignment_target = data_of_batch["alignment"]

                src_seq = torch.from_numpy(src_seq).long().to(device)
                src_pos = torch.from_numpy(src_pos).long().to(device)
                mel_tgt = torch.from_numpy(mel_tgt).float().to(device)
                alignment_target = torch.from_numpy(
                    alignment_target).float().to(device)
                # For Data Parallel
                mel_max_len = mel_tgt.size(1)

            # Forward
            mel_output, mel_output_postnet, duration_predictor_output = model(
                src_seq, src_pos,
                mel_max_length=mel_max_len,
                length_target=alignment_target)

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_predictor_loss = fastspeech_loss(
                mel_output, mel_output_postnet, duration_predictor_output, mel_tgt, alignment_target)
            total_loss = mel_loss + mel_postnet_loss + duration_predictor_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            m_p_l = mel_postnet_loss.item()
            d_p_l = duration_predictor_loss.item()

            with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                f_total_loss.write(str(t_l)+"\n")

            with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                f_mel_loss.write(str(m_l)+"\n")

            with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                f_mel_postnet_loss.write(str(m_p_l)+"\n")

            with open(os.path.join("logger", "duration_predictor_loss.txt"), "a") as f_d_p_loss:
                f_d_p_loss.write(str(d_p_l)+"\n")

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

            # Update weights
            if args.frozen_learning_rate:
                scheduled_optim.step_and_update_lr_frozen(
                    args.learning_rate_frozen)
            else:
                scheduled_optim.step_and_update_lr()

            # Print
            if current_step % hp.log_step == 0:
                Now = time.clock()

                str1 = "Epoch [{}/{}], Step [{}/{}], Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f};".format(
                    epoch+1, hp.epochs, current_step, total_step, mel_loss.item(), mel_postnet_loss.item())
                str2 = "Duration Predictor Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    duration_predictor_loss.item(), total_loss.item())
                str3 = "Current Learning Rate is {:.6f}.".format(
                    scheduled_optim.get_learning_rate())
                str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))

                print("\n" + str1)
                print(str2)
                print(str3)
                print(str4)

                with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                    f_logger.write(str1 + "\n")
                    f_logger.write(str2 + "\n")
                    f_logger.write(str3 + "\n")
                    f_logger.write(str4 + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            end_time = time.clock()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


if __name__ == "__main__":
    # Main
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
