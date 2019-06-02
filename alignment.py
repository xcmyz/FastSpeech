import torch


def get_alignment(text, pos):
    """ You need a reliable model for alignment """
    """ For test, Multiply three there """

    test_out = torch.zeros(pos.size(0), pos.size(1))
    for i_batch in range(pos.size(0)):
        for i_ele in range(pos.size(1)):
            if pos[i_batch][i_ele] != 0:
                test_out[i_batch][i_ele] = 3.0

    return test_out
