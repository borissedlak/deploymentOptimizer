import pandas as pd
from sklearn.utils import shuffle

from detector import utils
from detector.utils import get_latency_for_devices

df_anomaly = shuffle(pd.read_csv('../PW_Traffic/W_metrics_anomaly.csv'), random_state=35)

df_analysis = pd.read_csv('../PW_Street_Analysis/W_metrics_analysis.csv')
df_analysis = shuffle(pd.concat([df_analysis] * 2), random_state=35)  # To get almost 10.000 samples

# del df_analysis['cpu']
# del df_analysis['gpu']
# del df_analysis['memory']
del df_analysis['in_time']


# TODO: Do I put them all in one BN?
# TODO: In any case, I should create a loop for the following BNs


def calculate_cumulative_net_delay(row, src, dest):
    return (get_latency_for_devices(src, row['device_type'], ) +
            get_latency_for_devices(row['device_type'], dest) +
            row['delta'])


df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1, args=("Nano", "Laptop",))
df_analysis['cumm_net_delay_True'] = df_analysis['cumm_net_delay'] <= 50
# df_analysis['delta_True'] = df_analysis['delta'] <= 61
# print(df_analysis)

# utils.train_to_BN(df_analysis, "Traffic Prediction", export_file="./model_analysis.xml")

#########################################################

# del df_anomaly['cpu']
# del df_anomaly['gpu']
# del df_anomaly['memory']
del df_anomaly['timestamp']

df_anomaly['cumm_net_delay'] = df_anomaly.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
df_anomaly['cumm_net_delay_True'] = df_anomaly['cumm_net_delay'] <= 50
print(df_anomaly)

# df = pd.merge(df_anomaly, df_analysis, left_index=True, right_index=True)  # Only joins to the max of one list

# utils.train_to_BN(df_anomaly, "Traffic Prediction", export_file="./model_anomaly.xml")

#########################################################

df_weather = pd.read_csv('../PW_Weather_Transport/W_metrics_weather.csv')
# Idea: if I actually keep them separate, then the latter is obsolete, no?
df_weather = shuffle(pd.concat([df_weather] * 6), random_state=35)  # To get almost 10.000 samples

# del df_anomaly['cpu']
# del df_anomaly['gpu']
# del df_anomaly['memory']
del df_weather['timestamp']

df_weather['cumm_net_delay'] = df_weather.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
# df_weather['cumm_net_delay_True'] = df_weather['cumm_net_delay'] <= 50
print(df_weather)

# df = pd.merge(df_anomaly, df_analysis, left_index=True, right_index=True)  # Only joins to the max of one list

utils.train_to_BN(df_weather, "Traffic Prediction", export_file="./model_weather.xml")

#########################################################

# df_concat = pd.merge(df_weather, df_analysis, left_index=True, right_index=True)
# df_concat = pd.merge(df_concat, df_anomaly, left_index=True, right_index=True)
# utils.train_to_BN(df_concat, "Traffic Prediction", export_file="./model_master.xml")

