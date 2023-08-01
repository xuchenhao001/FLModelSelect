import sys

from plot.utils import plot_time_bar

acc = [23.91, 0.24, 0.07, 34.52, 58.74]
rdm = [23.96, 0.23, 0.05, 34.55, 58.79]

data = {
    "rdm": rdm,
    "acc": acc,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_bar("", data, save_path=save_path, plot_size="2")
