import numpy as np

loss_arr = np.array(list())
with open("total_loss.txt", "r") as f_loss:
    cnt = 0
    for loss in f_loss.readlines():
        cnt += 1
        # print(loss)
        loss_arr = np.append(loss_arr, float(loss))
        print(cnt)
