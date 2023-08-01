import sys

from plot.utils import plot_node_bar

acc_rdm = [62.48, 67.36, 67.91, 68.47]
acc_dsz = [65.34, 65.01, 64.87, 64.78]
acc_g_max = [65.03, 64.78, 63.93, 63.24]
acc_e_max = [65.44, 66.59, 67.33, 68.96]

data = {
    "rdm": acc_rdm,
    "dsz": acc_dsz,
    "g_max": acc_g_max,
    "e_max": acc_e_max,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_node_bar("", data, legend_pos="out", save_path=save_path, plot_size="2")
