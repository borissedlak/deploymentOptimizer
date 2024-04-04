import os

import pandas as pd
from matplotlib import pyplot as plt

df_fulfill = pd.read_csv('./precision_fulfillment.csv')
# df_fulfill['']


df_traffic = df_fulfill[df_fulfill['service'] == 'traffic']
df_monitor = df_fulfill[df_fulfill['service'] == 'monitor']
df_routing = df_fulfill[df_fulfill['service'] == 'routing']

fig, ax = plt.subplots()

x = [i / 10.0 for i in range(1, 11)]
plt.plot(x, df_routing['hl_fulfill'], color='firebrick', label="VehicleRouting HL")
plt.scatter(x, df_routing['hl_fulfill'], color='firebrick', marker='+', s=30)
plt.plot(x, df_routing['ll_fulfill'], color='firebrick', linestyle='--', label="VehicleRouting LL")
plt.scatter(x, df_routing['ll_fulfill'], color='firebrick', marker='+', s=30)
plt.plot(x, df_traffic['hl_fulfill'], color='steelblue', label="TrafficPrediction HL")
plt.scatter(x, df_traffic['hl_fulfill'], color='steelblue', marker='+', s=30)
plt.plot(x, df_traffic['ll_fulfill'], color='steelblue', linestyle='--', label="TrafficPrediction LL")
plt.scatter(x, df_traffic['ll_fulfill'], color='steelblue', marker='+', s=30)
plt.plot(x, df_monitor['hl_fulfill'], color='mediumaquamarine', label="LiveMonitoring HL")
plt.scatter(x, df_monitor['hl_fulfill'], color='mediumaquamarine', marker='+', s=30)
plt.plot(x, df_monitor['ll_fulfill'], color='mediumaquamarine', linestyle='--', label="LiveMonitoring LL")
plt.scatter(x, df_monitor['ll_fulfill'], color='mediumaquamarine', marker='+', s=30)
ax.set_xlabel(r'State Acceptance Rate ($\lambda$)')
ax.set_ylabel('SLO Fulfillment Rate')

ax.set_xticks([i / 10.0 for i in range(1, 11)])
fig.set_size_inches(5.4, 3.0)
ax.set_xlim(0.09, 1.0)
ax.set_ylim(0.55, 1.00)
ax.legend()

# Show the plot
plt.savefig("slo_fulfillment_lambda.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
plt.show()