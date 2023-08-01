# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from cycler import cycler
import pylab

# input latex symbols in matplotlib
# https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"


# Plot number in a row: "2", "3", "4"
# 2: Two plots in a row (the smallest fonts)
# 3: Three plots in a row
# 4: Four plots in a row (the biggest fonts)
def get_font_settings(size):
    if size == "2":
        font_size_dict = {"l": 21, "m": 18, "s": 16}
        fig_width = 8  # by default is 6.4 x 4.8
        fig_height = 4
    elif size == "3":
        font_size_dict = {"l": 25, "m": 21, "s": 19}
        fig_width = 7
        fig_height = 4.8
    else:
        font_size_dict = {"l": 25, "m": 25, "s": 20}
        # fig_width = 6.4
        # fig_height = 4.8
        fig_width = 7.4
        fig_height = 3.7

    xy_label_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["l"])
    title_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["m"])
    legend_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["s"])
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', size=font_size_dict["s"])
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}
    cs_xy_ticks_font = {'fontproperties': ticks_font}
    font_factory = {
        'legend_font': legend_font,
        'cs_xy_label_font': cs_xy_label_font,
        'cs_title_font': cs_title_font,
        'cs_xy_ticks_font': cs_xy_ticks_font,
        'fig_width': fig_width,
        'fig_height': fig_height,
    }
    return font_factory


def plot_legend_head(axes, legend_column, width, height, save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    figlegend = pylab.figure(layout='constrained')
    figlegend.legend(axes.get_legend_handles_labels()[0], axes.get_legend_handles_labels()[1],
                     prop=font_settings.get("legend_font"), ncol=legend_column, loc='upper center')
    # figlegend.tight_layout()
    figlegend.set_size_inches(width, height)
    if save_path:
        save_path = save_path[:-4] + "-legend.pdf"
        figlegend.savefig(save_path, format="pdf")
    else:
        figlegend.show()


# data: {"scheme01": scheme01_data, "scheme02": scheme02_data}
# legend: legend position. Values: "in", "out", or nothing
def plot_round_acc(title, data, legend_pos="", save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    x = range(1, len(data["random_mean"]) + 1)

    my_colors = plt.get_cmap('tab10').colors
    my_cycler = cycler(color=my_colors)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(my_cycler)
    line_width = 2
    marker_size = 6
    range_alpha = 0.1

    axes.plot(x, data["accuracy_mean"], label="ACC", linewidth=line_width, marker='o', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["accuracy_min"], data["accuracy_max"], linewidth=0, alpha=range_alpha, color=my_colors[0])
    axes.plot(x, data["datasize_mean"], label="DSZ", linewidth=line_width, marker='D', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["datasize_min"], data["datasize_max"], linewidth=0, alpha=range_alpha, color=my_colors[1])
    axes.plot(x, data["entropy_max_mean"], label="ENT-MAX", linewidth=line_width, marker='v', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["entropy_max_min"], data["entropy_max_max"], linewidth=0, alpha=range_alpha, color=my_colors[2])
    axes.plot(x, data["entropy_min_mean"], label="ENT-MIN", linewidth=line_width, marker='>', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["entropy_min_min"], data["entropy_min_max"], linewidth=0, alpha=range_alpha, color=my_colors[3])
    axes.plot(x, data["gradiv_max_mean"], label="G-MAX", linewidth=line_width, marker='x', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["gradiv_max_min"], data["gradiv_max_max"], linewidth=0, alpha=range_alpha, color=my_colors[4])
    axes.plot(x, data["gradiv_min_mean"], label="G-MIN", linewidth=line_width, marker='|', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["gradiv_min_min"], data["gradiv_min_max"], linewidth=0, alpha=range_alpha, color=my_colors[5])
    axes.plot(x, data["loss_max_mean"], label="LOSS-MAX", linewidth=line_width, marker='<', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["loss_max_min"], data["loss_max_max"], linewidth=0, alpha=range_alpha, color=my_colors[6])
    axes.plot(x, data["loss_min_mean"], label="LOSS-MIN", linewidth=line_width, marker='s', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["loss_min_min"], data["loss_min_max"], linewidth=0, alpha=range_alpha, color=my_colors[7])
    axes.plot(x, data["random_mean"], label="RDM", linewidth=line_width, marker='*', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["random_min"], data["random_max"], linewidth=0, alpha=range_alpha, color=my_colors[8])

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right', ncol=2).set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 5, 9.3, 0.9, save_path, plot_size)


def plot_node_acc(title, data, legend_pos="", save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    x = range(1, len(data["rdm_03_mean"]) + 1)

    my_colors = plt.get_cmap('tab10').colors
    my_cycler = cycler(color=my_colors)

    fig, axes = plt.subplots(layout='constrained')
    axes.set_prop_cycle(my_cycler)
    line_width = 2
    marker_size = 4
    range_alpha = 0.17

    axes.plot(x, data["rdm_03_mean"], label="RDM-3", marker='o', linewidth=line_width, markevery=5, markersize=marker_size)
    axes.fill_between(x, data["rdm_03_min"], data["rdm_03_max"], linewidth=0, alpha=range_alpha, color=my_colors[0])
    axes.plot(x, data["rdm_05_mean"], label="RDM-5", marker='d', linewidth=line_width, markevery=5, markersize=marker_size)
    axes.fill_between(x, data["rdm_05_min"], data["rdm_05_max"], linewidth=0, alpha=range_alpha, color=my_colors[1])
    axes.plot(x, data["rdm_07_mean"], label="RDM-7", marker='v', linewidth=line_width, markevery=5, markersize=marker_size)
    axes.fill_between(x, data["rdm_07_min"], data["rdm_07_max"], linewidth=0, alpha=range_alpha, color=my_colors[2])

    axes.plot(x, data["acc_03_mean"], label="ACC-3", linewidth=line_width, linestyle="dashdot", marker='>', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["acc_03_min"], data["acc_03_max"], linewidth=0, alpha=range_alpha, color=my_colors[3])
    axes.plot(x, data["acc_05_mean"], label="ACC-5", linewidth=line_width, linestyle="dashdot", marker='*', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["acc_05_min"], data["acc_05_max"], linewidth=0, alpha=range_alpha, color=my_colors[4])
    axes.plot(x, data["acc_07_mean"], label="ACC-7", linewidth=line_width, linestyle="dashdot", marker='X', markevery=5, markersize=marker_size)
    axes.fill_between(x, data["acc_07_min"], data["acc_07_max"], linewidth=0, alpha=range_alpha, color=my_colors[5])

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.ylim(bottom=20)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right', ncol=2).set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 7, 12, 0.6, save_path, plot_size)


def plot_time_bar(title, data, save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)

    x = ["Training", "Evaluating", "Communication", "Waiting", "Total"]

    fig, axes = plt.subplots(layout='constrained')

    width = 0.42  # the width of the bars

    axes.bar([p - width/2 for p in range(len(x))], height=data["rdm"], width=width, label="RDM", hatch='x', alpha=.99,)
    axes.bar([p + width/2 for p in range(len(x))], height=data["acc"], width=width, label="ACC", hatch='o', alpha=.99,)

    # annotate
    axes.bar_label(axes.containers[0], label_type='edge', **font_settings.get("cs_xy_ticks_font"))
    axes.bar_label(axes.containers[1], label_type='edge', **font_settings.get("cs_xy_ticks_font"))

    plt.ylim(top=65)

    plt.xticks(range(len(x)), x)
    axes.set_xlabel("Stage in Training Rounds", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Average Time Cost (s)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.legend(prop=font_settings.get("legend_font"), loc='upper left').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_node_bar(title, data, legend_pos="", save_path=None, plot_size="2"):
    font_settings = get_font_settings(plot_size)

    x = ["10", "30", "50", "70"]

    fig, axes = plt.subplots(layout='constrained')

    width = 0.22  # the width of the bars

    axes.bar([p - 3 * width/2 for p in range(len(x))], height=data["rdm"], width=width, label="RDM", hatch='x', alpha=.99,)
    axes.bar([p - width/2 for p in range(len(x))], height=data["dsz"], width=width, label="DSZ", hatch='o', alpha=.99,)
    axes.bar([p + width/2 for p in range(len(x))], height=data["g_max"], width=width, label="G-MAX", hatch='.', alpha=.99,)
    axes.bar([p + 3 * width/2 for p in range(len(x))], height=data["e_max"], width=width, label="ENT-MAX", hatch='-', alpha=.99,)

    # plt.ylim(top=65)

    plt.xticks(range(len(x)), x)
    axes.set_xlabel("Number of Selected Models", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right', ncol=2).set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 4, 7, 0.6, save_path, plot_size)
