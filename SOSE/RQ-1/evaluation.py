import pandas as pd

from SOSE.C_Traffic_Prediction.tools import evaluate_all_services, evaluate_all_permutations

slo_df = pd.read_csv('./routing_ll_slos.csv')
evaluate_all_services(slo_df=slo_df, only_set_params=True)
evaluate_all_permutations(slo_df=slo_df)
print()

slo_df = pd.read_csv('./traffic_ll_slos.csv')
evaluate_all_services(slo_df=slo_df, only_set_params=True)
evaluate_all_permutations(slo_df=slo_df)
print()

slo_df = pd.read_csv('./monitor_ll_slos.csv')
evaluate_all_services(slo_df=slo_df, only_set_params=True)
evaluate_all_permutations(slo_df=slo_df)
