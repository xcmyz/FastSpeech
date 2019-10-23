import matplotlib.pyplot as plt
import numpy as np


def visualize(file_name, start, end=-1):
    plt.figure()

    loss_arr = np.array(list())

    with open(file_name, "r") as f_loss:
        for loss in f_loss.readlines():
            loss_arr = np.append(loss_arr, float(loss))

    loss_arr = loss_arr[start:end]

    x = np.array([i for i in range(np.shape(loss_arr)[0])])
    y = loss_arr

    plt.plot(x, y, color="y", lw=0.7)
    plt.xlabel("sequence number")
    plt.ylabel("loss item")
    plt.title("loss")
    plt.savefig("loss_one.jpg")


if __name__ == "__main__":
    # Test
    visualize("duration_loss.txt", 3000)
