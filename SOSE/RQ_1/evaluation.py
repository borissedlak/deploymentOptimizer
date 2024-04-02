import pandas as pd

from SOSE.C_Traffic_Prediction.tools import evaluate_all_services

df_traffic = pd.read_csv('./traffic_ll_slos.csv')
evaluate_all_services(slo_df=df_traffic, only_set_params=True)

df_monitor = pd.read_csv('./monitor_ll_slos.csv')
evaluate_all_services(slo_df=df_monitor, only_set_params=True)

df_routing = pd.read_csv('./routing_ll_slos.csv')
evaluate_all_services(slo_df=df_routing, only_set_params=True)
