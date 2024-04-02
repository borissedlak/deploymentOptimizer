import pandas as pd
from pgmpy.base import DAG

from SOSE.C_Traffic_Prediction.tools import filter_training_data
from detector import utils

#########################################################

df_analysis = filter_training_data(pd.read_csv('../PW_Street_Analysis/W_metrics_Analysis.csv'))
del df_analysis['in_time']

dag = DAG()
dag.add_nodes_from(["delta", "cumm_net_delay", "memory", "fps", "pixel", "cpu", "gpu", "consumption_all",
                    "viewer_satisfaction", "energy"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("cpu", "consumption_all"), ("pixel", "cpu"), ("fps", "cpu"),
                    ("fps", "memory"), ("pixel", "delta"), ("gpu", "delta"), ("fps", "gpu"),
                    ("pixel", "viewer_satisfaction"), ("consumption_all", "energy"), ("gpu", "consumption_all")])

utils.train_to_BN(df_analysis, "Analysis", export_file="model_analysis.xml", dag=dag)

#########################################################

df_privacy = filter_training_data(pd.read_csv('../W_Privacy_Transform/W_metrics_Privacy.csv'))

dag = DAG()
dag.add_nodes_from(["delta", "cumm_net_delay", "memory", "fps", "pixel", "cpu", "gpu", "consumption_all",
                    "viewer_satisfaction", "energy", "delta_privacy"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("cpu", "consumption_all"), ("pixel", "cpu"), ("fps", "cpu"),
                    ("fps", "memory"), ("pixel", "delta"), ("gpu", "delta"), ("fps", "gpu"),
                    ("pixel", "viewer_satisfaction"), ("consumption_all", "energy"), ("delta_privacy", "cumm_net_delay"),
                    ("gpu", "consumption_all"), ("fps", "viewer_satisfaction"), ("pixel", "delta_privacy")])

utils.train_to_BN(df_privacy, "Privacy", export_file="model_privacy.xml", dag=dag)

#########################################################

df_anomaly = filter_training_data(pd.read_csv('../PW_Traffic/W_metrics_Anomaly.csv'))
del df_anomaly['timestamp']

dag = DAG()
dag.add_nodes_from(["memory", "delta", "cumm_net_delay", "gpu", "cpu", "batch_size", "consumption", "energy"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("batch_size", "delta"), ("batch_size", "cpu"),
                    ("cpu", "consumption"), ("batch_size", "memory"), ("consumption", "energy")])

utils.train_to_BN(df_anomaly, "Anomaly", export_file="model_anomaly.xml", dag=dag)

#########################################################

df_weather = filter_training_data(pd.read_csv('../PW_Weather/W_metrics_Weather.csv'))
del df_weather['timestamp']

dag = DAG()
dag.add_nodes_from(["memory", "delta", "cumm_net_delay", "isentropic", "gpu", "cpu", "fig_size",
                    "data_size", "consumption", "viewer_satisfaction", "energy"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("fig_size", "delta"), ("isentropic", "delta"), ("data_size", "delta"),
                    ("cpu", "consumption"), ("isentropic", "cpu"), ("data_size", "cpu"), ("delta", "memory"),
                    ("fig_size", "viewer_satisfaction"), ("consumption", "energy")])

utils.train_to_BN(df_weather, "Weather", export_file="model_weather.xml", dag=dag)

#########################################################

df_cloud = filter_training_data(pd.read_csv('../PW_Cloud_DB/W_metrics_CloudDB.csv'))
del df_cloud['timestamp']

dag = DAG()
dag.add_nodes_from(["threads", "limit", "cumm_net_delay", "gpu", "cpu", "batch_size", "consumption",
                    "cached", "memory", "delta", "energy"])
dag.add_edges_from([("delta", "cumm_net_delay"), ("batch_size", "delta"), ("cached", "cpu"), ("cpu", "consumption"),
                    ("threads", "delta"), ("limit", "delta"), ("cached", "delta"), ("consumption", "energy")])

utils.train_to_BN(df_cloud, "CloudDB", export_file="model_cloud.xml", dag=dag)

#########################################################

# df_concat = pd.merge(df_weather, df_analysis, left_index=True, right_index=True)
# df_concat = pd.merge(df_concat, df_anomaly, left_index=True, right_index=True)
# utils.train_to_BN(df_concat, "Traffic Prediction", export_file="./model_master.xml")
