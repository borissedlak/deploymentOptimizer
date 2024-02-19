# Compare the SLO fulfillment when all on one
# Against when all on optimal
# Against when all on random
# Against equally distributed

import pandas as pd
from matplotlib import pyplot as plt

from detector import utils

fig, ax = plt.subplots()

# df = pd.read_csv("./performance/comparison_slo.csv")
df_all = pd.read_csv('../analysis/inference/n_n_assignments.csv')

boxplots = []
lines = []
boxplot_position = 0
constrained_thresh_O_X = 40 - (utils.get_latency_for_devices("Nano", "Orin") + utils.get_latency_for_devices("Orin", "Xavier"))
# constrained_thresh_O_L = 40 - (utils.get_latency_for_devices("Nano", "Orin") + utils.get_latency_for_devices("Orin", "Laptop"))
for index, col, thresh in [(0, 'dimgray', ("dashed", 45.75)), (2, 'firebrick', ("solid", constrained_thresh_O_X)),
                           (3, 'chocolate', ("solid", constrained_thresh_O_X)), (5, 'steelblue', ("solid", 31))]:  # , (1, 'steelblue')
    df_t = df_all[(df_all['t'] == index) & (df_all['service_name'] == "Processor") & (df_all['select'] == True)]

    pixel = df_t['pixel'].unique()[0]
    min_latency = df_t['min_latency'].unique()[0]  # Increase resolution a bit here
    fps = df_t['fps'].unique()[0]
    processor_host = df_t['host'].unique()[0]

    df_exp = pd.read_csv(f"../analysis/performance/{processor_host}.csv")
    df_exp = df_exp[(df_exp['pixel'] == pixel) & (df_exp['fps'] == fps)]

    boxplot = ax.boxplot(list(df_exp['delta']), positions=[boxplot_position], patch_artist=True, widths=0.45, whis=1.5)
    boxplot['boxes'][0].set_facecolor(col)
    boxplots.append(boxplot)

    line = plt.hlines(thresh[1], xmin=boxplot_position - 0.4, xmax=boxplot_position + 0.4,
                      color=col,
                      linestyle=thresh[0],
                      linewidth=1.8)  # , alpha=0.5)
    lines.append(line)
    boxplot_position += 1

ax.set_ylabel(r'$\mathit{Worker}$ processing delay (ms)')
ax.legend([box['boxes'][0] for box in boxplots] + lines,
          [r'Worker @ $\mathit{Xavier}$', r'Worker @ $\mathit{Orin}$', r'Worker @ $\mathit{Orin}$', r'Worker @ $\mathit{Xavier}$',
           'Virtual threshold', 'Delay threshold'])
ax.set_ylim(30.5, 46)
fig.set_size_inches(4.5, 4.8)
plt.xticks([0, 1, 2, 3], [r'$\mathit{t_0}$', r'$\mathit{t_2}$', r'$\mathit{t_3}$',
                          r'$\mathit{t_5}$'])
# Show the plot
plt.savefig("./plots/delta_distribution_high.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
plt.show()
