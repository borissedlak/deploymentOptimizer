import pandas as pd
from sklearn.utils import shuffle

from detector import utils
from detector.utils import get_latency_for_devices

df_anomaly = shuffle(pd.read_csv('../PW_Traffic/W_metrics_anomaly.csv'), random_state=35)

df_analysis = pd.read_csv('../PW_Street_Analysis/W_metrics_analysis.csv')
df_analysis = shuffle(pd.concat([df_analysis, df_analysis]), random_state=35)  # To get almost 10.000 samples

del df_analysis['cpu']
del df_analysis['gpu']
del df_analysis['memory']
del df_analysis['in_time']

def calculate_cumulative_net_delay(row):
    return (get_latency_for_devices('Nano', row['device_type'], ) +
            get_latency_for_devices(row['device_type'], 'Laptop') +
            row['delta_analysis'])


df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1)
print(df_analysis)

del df_anomaly['cpu']
del df_anomaly['gpu']
del df_anomaly['memory']
del df_anomaly['timestamp']

df = pd.merge(df_anomaly, df_analysis, left_index=True, right_index=True)  # Only joins to the max of one list

utils.train_to_BN(df_analysis, "Traffic Prediction")
