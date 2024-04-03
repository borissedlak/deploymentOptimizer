import sys
from itertools import product

import pandas as pd

from SOSE.C_Traffic_Prediction.tools import evaluate_all_services, filter_test_data
from detector.utils import find_nested_files_with_suffix

slo_df = pd.read_csv('./routing_ll_slos.csv')
evaluate_all_services(slo_df=slo_df, only_set_params=True)

# TODO: (1) Filter all parameters
# TODO: (2) Filter their nodes from BN
# TODO: (3) Get all permutations
# TODO: (4) Evaluate all permutations and compare

service_list = slo_df['service'].unique()
for service in service_list:
    params_vars = slo_df[(slo_df['service'] == service) & (slo_df['root'])].iloc[:, 1].tolist()

    test_data_file = find_nested_files_with_suffix('../', f'W_metrics_{service}.csv')[0]
    test_df = filter_test_data(pd.read_csv(test_data_file))

    unique_values = [test_df[col].unique() for col in params_vars]
    permutations = product(*unique_values)

    for perm in permutations:
        filter_condition = test_df[params_vars[0]] == perm[0]
        for i in range(1, len(params_vars)):
            filter_condition &= test_df[params_vars[i]] == perm[i]

        filtered_df = test_df[filter_condition]
        if len(filtered_df) <= 0:
            continue

        # print(service, perm, len(filtered_df))
        # another_function(filtered_df)


        ll_slos = slo_df[(slo_df['service'] == service) & ~(slo_df['hl'])]  # .iloc[:2]
        ll_slos = ll_slos[ll_slos['root']]
        hl_slos = slo_df[(slo_df['service'] == service) & (slo_df['hl'])]
        tuples_list = list(ll_slos.itertuples(index=False))
        # cond_df = filtered_df[eval(" & ".join(["(test_df['{0}'].isin({1}))".format(col, cond)
        #                                    for _, col, cond, _, _ in tuples_list]))]


        for index, row in hl_slos.iterrows():
            # SLO fulfillment if the system is configured with the inferred ll SLOs
            fulfilled = filtered_df[filtered_df[row[1]].isin(row[2])]
            print(row[0], row[1], len(fulfilled) / len(filtered_df))

print()

df_traffic = pd.read_csv('./traffic_ll_slos.csv')
evaluate_all_services(slo_df=df_traffic, only_set_params=True)
print()

df_monitor = pd.read_csv('./monitor_ll_slos.csv')
evaluate_all_services(slo_df=df_monitor, only_set_params=True)
