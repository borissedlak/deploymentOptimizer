import pandas as pd
from sklearn.utils import shuffle

from SOSE.C_Traffic_Prediction.tools import calculate_cumulative_net_delay, append_privacy_values

# Just a detail, but its idempotent, means can be repeated arbitrary many times, the result is the same

df_analysis = shuffle(pd.read_csv('../PW_Street_Analysis/W_metrics_Analysis.csv'), random_state=35)
df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1, args=("Nano", "Laptop",))
# DeviceMetricCollector only considers the cpu load, this balances this out
df_analysis['consumption_all'] = df_analysis['energy'] + (df_analysis['gpu'] / 15)
df_analysis['consumption_all'] = df_analysis['consumption_all'].astype(int)
df_analysis['energy'] = df_analysis['consumption_all']
df_analysis.to_csv('../PW_Street_Analysis/W_metrics_Analysis.csv', index=False)

df_privacy = shuffle(pd.read_csv('../W_Privacy_Transform/W_metrics_Privacy_raw.csv'), random_state=35)
del df_privacy['cpu']
del df_privacy['gpu']
del df_privacy['memory']
del df_privacy['device_type']
del df_privacy['consumption']
del df_privacy['timestamp']
# df_merge = df_privacy.join(df_analysis, lsuffix='_df1', rsuffix='_df2', how='inner')
# df_merge = pd.merge(df_analysis, df_privacy, on=['fps', 'pixel'], how="left")
# df_privacy = pd.concat([df_analysis, df_privacy])
df_analysis['viewer_satisfaction'] = (df_analysis['pixel']) / 150 + (df_analysis['fps']) / 5
df_analysis['viewer_satisfaction'] = df_analysis['viewer_satisfaction'].astype(int)
df_analysis['delta_privacy'] = df_analysis.apply(append_privacy_values, axis=1, args=(df_privacy,))
df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1, args=("Nano", "PC",))
df_analysis['cumm_net_delay'] = df_analysis['cumm_net_delay'] + df_analysis['delta_privacy']
df_analysis.to_csv('../W_Privacy_Transform/W_metrics_Privacy.csv', index=False)

df_anomaly = pd.read_csv('../PW_Traffic/W_metrics_Anomaly.csv')
df_anomaly['cumm_net_delay'] = df_anomaly.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
df_anomaly['energy'] = df_anomaly['consumption']
df_anomaly.to_csv('../PW_Traffic/W_metrics_Anomaly.csv', index=False)

df_weather = pd.read_csv('../PW_Weather/W_metrics_Weather.csv')
df_weather['cumm_net_delay'] = df_weather.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
df_weather['viewer_satisfaction'] = df_weather['fig_size']
df_weather['energy'] = df_weather['consumption']
df_weather.to_csv('../PW_Weather/W_metrics_Weather.csv', index=False)

df_cloud = pd.read_csv('../PW_Cloud_DB/W_metrics_CloudDB.csv')
df_cloud['cumm_net_delay'] = df_cloud.apply(calculate_cumulative_net_delay, axis=1, args=("PC", "Laptop",))
df_cloud['energy'] = df_cloud['consumption']
df_cloud.to_csv('../PW_Cloud_DB/W_metrics_CloudDB.csv', index=False)
