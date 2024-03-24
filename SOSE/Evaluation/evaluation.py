import ast

import pandas as pd

from SOSE.C_Traffic_Prediction.tools import filter_test_data, convert_to_int_or_bool
from detector.utils import find_nested_files_with_suffix

# TODO
#  * configure the system with the desired states ==> filter test set
#  * calculate the slo fulfillment rate with the desired config
#  * calculate for alternative configurations
#  * compare results

# Group by application
slo_df = pd.read_csv('../C_Traffic_Prediction/ll_slos.csv')
service_list = slo_df['service'].unique()
slo_df['states'] = slo_df['states'].apply(ast.literal_eval)
slo_df['states'] = slo_df['states'].apply(convert_to_int_or_bool)  # TODO: This is dangerous when floats come

for service in service_list:
    test_data_file = find_nested_files_with_suffix('../', f'W_metrics_{service}.csv')[0]
    test_df = filter_test_data(pd.read_csv(test_data_file))
    ll_slos = slo_df[(slo_df['service'] == service) & ~(slo_df['hl'])]#.iloc[:2]
    hl_slos = slo_df[(slo_df['service'] == service) & (slo_df['hl'])]

    tuples_list = list(ll_slos.itertuples(index=False))
    conditioned_df = test_df[eval(" & ".join(["(test_df['{0}'].isin({1}))".format(col, cond)
                                              for _, col, cond, _ in tuples_list]))]

    print(conditioned_df)
