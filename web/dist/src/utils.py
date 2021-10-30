import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.figure import figaspect
import time
import os
import random
import threading

plt.rcdefaults()
pyplot.locator_params(nbins=5)


class PlotGenerator:
    lock = threading.Lock()

    def __init__(self):
        self.fig_n = int(time.time())

        self.colors = [(114 / 255, 147 / 255, 203 / 255, 1),
                       (225 / 255, 151 / 255, 76 / 255, 1),
                       (132 / 255, 186 / 255, 91 / 255, 1),
                       (211 / 255, 94 / 255, 96 / 255, 1),
                       (128 / 255, 133 / 255, 133 / 255, 1),
                       (144 / 255, 103 / 255, 167 / 255, 1),
                       (171 / 255, 104 / 255, 87 / 255, 1),
                       (204 / 255, 194 / 255, 16 / 255, 1)]

        self.edge_colors = [(57 / 255, 106 / 255, 177 / 255, 1),
                            (218 / 255, 124 / 255, 48 / 255, 1),
                            (62 / 255, 150 / 255, 81 / 255, 1),
                            (204 / 255, 37 / 255, 41 / 255, 1),
                            (83 / 255, 81 / 255, 84 / 255, 1),
                            (107 / 255, 76 / 255, 154 / 255, 1),
                            (146 / 255, 36 / 255, 40 / 255, 1),
                            (148 / 255, 139 / 255, 61 / 255, 1)]

        self.n = random.randint(0, 255)

    def gen_bar_plot(self, table, title, maximum):
        PlotGenerator.lock.acquire()
        maximum = max(maximum, 1)
        self.fig_n += 1
        self.n += 1
        plt.figure(figsize=(8, 6), dpi=300)
        plt.tight_layout()

        cols = table.get_active_cols()
        if cols:
            y_pos = np.arange(len(cols[0]))

            w, h = figaspect(4 / 3)
            # plt.axes().set_aspect('equal', 'datalim')

            col_n = self.n % len(self.colors)
            plt.bar(y_pos, cols[1], align='center', color=self.colors[col_n], edgecolor=self.edge_colors[col_n])
            plt.xticks(y_pos, cols[0], fontsize=14)
            plt.yticks(np.arange(0, (5.5/5)*maximum, step=maximum/5), fontsize=14)
            plt.ylim(bottom=0, top=(5.5/5)*maximum)

        plt.xlabel(table.col_names[0], fontsize=14, labelpad=15)
        plt.ylabel(table.col_names[1], fontsize=14, labelpad=15)
        plt.title(title)

        file_path = os.path.dirname(__file__) + "/../../media/{}.png".format(self.fig_n)
        plt.savefig(file_path)
        plt.figure(self.fig_n)
        plt.clf()
        PlotGenerator.lock.release()
        return "{}.png".format(self.fig_n)
