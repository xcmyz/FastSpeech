import torch
import numpy as np


def get_alignment(text, pos):
    """ You need a reliable model for alignment """
    """ For test, Multiply three there """

    test_out = torch.zeros(np.shape(pos)[0], np.shape(pos)[1])
    for i_batch in range(np.shape(pos)[0]):
        for i_ele in range(np.shape(pos)[1]):
            if pos[i_batch][i_ele] != 0:
                test_out[i_batch][i_ele] = 3.0

    return test_out
