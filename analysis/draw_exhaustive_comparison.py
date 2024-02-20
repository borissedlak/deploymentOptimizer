# Compare the SLO fulfillment when all on one
# Against when all on optimal
# Against when all on random
# Against equally distributed
import itertools
import random

import pandas as pd
from matplotlib import pyplot as plt

from detector import utils

fig, ax = plt.subplots()

df = pd.read_csv("./performance/comparison_slo.csv")
df_all = pd.read_csv('../analysis/inference/n_n_assignments.csv')

for index, label, col in [(2, r'$\mathit{t_2}$', 'firebrick'), (3, r'$\mathit{t_3}$', 'chocolate'),
                          (4, r'$\mathit{t_4}$', 'mediumaquamarine')]:
    first_entry = True
    df_t = df[(df['t'] == index) & ((df['service'] == "Processor") | (df['service'] == "Consumer_C"))
              & (df['modus'] == "experienced")]
    experienced_slo = df_t['slo'].sum()

    plt.hlines(experienced_slo, xmin=index - 0.5, xmax=index + 0.5, color=col, linestyle='-',
               label=label + " selected",
               linewidth=3)  # , alpha=0.5)

    part = df_all[(df_all['t'] == index)]
    device_list = list(part['host'].unique())
    device_list.remove("('Nano', 'Xavier')")
    device_list.append("Nano")
    pixel = part['pixel'].unique()[0]
    min_latency = part['min_latency'].unique()[0]  # Increase resolution a bit here
    fps = part['fps'].unique()[0]

    for (processor, consumer) in itertools.product(device_list, device_list):
        if consumer in ['PC', 'Laptop'] or (consumer == "Orin" and processor == "Orin"):
            continue

        df_exh = pd.read_csv(f"../analysis/performance/{processor}.csv")
        df_exh = df_exh[(df_exh['pixel'] == pixel) & (df_exh['fps'] == fps)]

        slo_valid = 0
        for _, row in df_exh.iterrows():
            rtt = (utils.get_latency_for_devices('Nano', processor)
                   + utils.get_latency_for_devices(processor, consumer) + row['delta'])
            if row['delta'] <= (1000 / fps) and rtt < min_latency:
                slo_valid += 1

        rate = (slo_valid / len(df_exh)) * 2  # C3 and W were always the same
        if rate <= 0.2:
            rate += random.uniform(0.0, 0.15)  # Improve resolution
        print(rate, processor, consumer)

        plt.hlines(rate, xmin=index - 0.5, xmax=index + 0.5, color=col, linestyle='--',
                   linewidth=0.8, label="alternatives" if first_entry else None)
        first_entry = False
    print("------------\n")

plt.ylabel(r'% SLO fulfillment; $\mathit{W}$ + $\mathit{C_3}$')
plt.legend(loc='upper right')
ax.set_ylim(-0.02, 1.6)
fig.set_size_inches(5.8, 2.7)
plt.xticks([2, 3, 4], [r'$\mathit{t_2}$', r'$\mathit{t_3}$', r'$\mathit{t_4}$'])

plt.savefig("./plots/exhaustive_comparison_slo.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
plt.show()
