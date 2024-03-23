import pandas as pd
from pgmpy.base import DAG
from sklearn.utils import shuffle

from SOSE.C_Traffic_Precition.tools import calculate_cumulative_net_delay
from detector import utils

#########################################################

df_analysis = pd.read_csv('../PW_Street_Analysis/W_metrics_analysis.csv')
# df_analysis = shuffle(pd.concat([df_analysis] * 2), random_state=35)  # To get almost 10.000 samples

del df_analysis['in_time']

df_analysis['cumm_net_delay'] = df_analysis.apply(calculate_cumulative_net_delay, axis=1, args=("Nano", "Laptop",))
dag = DAG()
dag.add_nodes_from(["delta", "cumm_net_delay", "memory", "fps", "pixel", "cpu", "gpu", "device_type", "consumption"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("cpu", "consumption"), ("pixel", "cpu"), ("fps", "cpu"),
                    ("fps", "memory"), ("pixel", "delta"), ("gpu", "delta"), ("fps", "gpu")])

utils.train_to_BN(df_analysis, "Analysis", export_file="model_analysis.xml", dag=dag)

#########################################################

df_anomaly = shuffle(pd.read_csv('../PW_Traffic/W_metrics_anomaly.csv'), random_state=35)

del df_anomaly['timestamp']

df_anomaly['cumm_net_delay'] = df_anomaly.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
dag = DAG()
dag.add_nodes_from(["memory", "delta", "cumm_net_delay", "device_type", "gpu", "cpu", "batch_size", "consumption"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("batch_size", "delta"), ("batch_size", "cpu"),
                    ("cpu", "consumption"), ("batch_size", "memory"), ])

utils.train_to_BN(df_anomaly, "Anomaly", export_file="model_anomaly.xml", dag=dag)

#########################################################

df_weather = pd.read_csv('../PW_Weather/W_metrics_weather.csv')
# Idea: if I actually keep them separate, then the latter is obsolete, no?
# Idea: Well I should keep 20% for testing the assignments later, which I could shuffle with rs=35
# df_weather = shuffle(pd.concat([df_weather] * 6), random_state=35)  # To get almost 10.000 samples

del df_weather['timestamp']

df_weather['cumm_net_delay'] = df_weather.apply(calculate_cumulative_net_delay, axis=1, args=("Xavier", "Laptop",))
dag = DAG()
dag.add_nodes_from(["memory", "delta", "cumm_net_delay", "isentropic", "device_type", "gpu", "cpu", "fig_size",
                    "data_size", "consumption"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("fig_size", "delta"), ("isentropic", "delta"), ("data_size", "delta"),
                    ("cpu", "consumption"), ("isentropic", "cpu"), ("data_size", "cpu"), ("delta", "memory")])

utils.train_to_BN(df_weather, "Weather", export_file="model_weather.xml", dag=dag)

#########################################################

df_cloud = pd.read_csv('../PW_Cloud_DB/W_metrics_cloud.csv')
df_cloud = shuffle(pd.concat([df_cloud] * 5), random_state=35)  # To get almost 10.000 samples

del df_cloud['timestamp']

df_cloud['cumm_net_delay'] = df_cloud.apply(calculate_cumulative_net_delay, axis=1, args=("PC", "Laptop",))
dag = DAG()
dag.add_nodes_from(["threads", "limit", "cumm_net_delay", "device_type", "gpu", "cpu", "batch_size", "consumption",
                    "cached", "memory", "delta"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("batch_size", "delta"), ("cached", "cpu"), ("cpu", "consumption"),
                    ("threads", "delta"), ("limit", "delta"), ("cached", "delta")])

utils.train_to_BN(df_cloud, "CloudDB", export_file="model_cloud.xml", dag=dag)

#########################################################

# df_concat = pd.merge(df_weather, df_analysis, left_index=True, right_index=True)
# df_concat = pd.merge(df_concat, df_anomaly, left_index=True, right_index=True)
# utils.train_to_BN(df_concat, "Traffic Prediction", export_file="./model_master.xml")
