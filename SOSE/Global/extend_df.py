import pandas as pd

from SOSE.C_Traffic_Prediction.tools import calculate_cumulative_net_delay

# Just a detail, but its idempotent, means can be repeated arbitrary many times, the result is the same

df_privacy = pd.read_csv('../W_Privacy_Transform/W_metrics_Privacy.csv')
# df_privacy['cumm_net_delay'] = df_privacy.apply(calculate_cumulative_net_delay, axis=1, args=("PC", "Laptop",))
df_privacy['energy'] = df_privacy['consumption']
df_privacy.to_csv('../W_Privacy_Transform/W_metrics_Privacy.csv', index=False)

df_analysis = pd.read_csv('../PW_Street_Analysis/W_metrics_Analysis.csv')
df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1, args=("Nano", "Laptop",))
df_analysis['viewer_satisfaction'] = df_analysis['pixel']
df_analysis['energy'] = df_analysis['consumption']
df_analysis.to_csv('../PW_Street_Analysis/W_metrics_Analysis.csv', index=False)

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
