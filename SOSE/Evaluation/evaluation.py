import ast

import pandas as pd

from SOSE.C_Traffic_Prediction.tools import filter_test_data, convert_to_int_or_bool
from detector.utils import find_nested_files_with_suffix, print_in_red

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


def evaluate_all_services(only_set_params=False):
    for service in service_list:
        test_data_file = find_nested_files_with_suffix('../', f'W_metrics_{service}.csv')[0]
        test_df = filter_test_data(pd.read_csv(test_data_file))
        ll_slos = slo_df[(slo_df['service'] == service) & ~(slo_df['hl'])]  # .iloc[:2]
        hl_slos = slo_df[(slo_df['service'] == service) & (slo_df['hl'])]

        if only_set_params:
            ll_slos = ll_slos[ll_slos['root']]

        tuples_list = list(ll_slos.itertuples(index=False))
        cond_df = test_df[eval(" & ".join(["(test_df['{0}'].isin({1}))".format(col, cond)
                                           for _, col, cond, _, _ in tuples_list]))]
        if len(cond_df) == 0:
            print_in_red("No samples with desired characteristics found")

        # Idea: I think I should only set the parameters, although ensuring the ll_slo through them is the responsibility
        #  of the elasticity strategies locally. But the difference between them is a super important metrics because it
        #  allows to estimate how well the target outcome can be influenced by the parameters that are set

        for index, row in hl_slos.iterrows():
            fulfilled = cond_df[cond_df[row[1]].isin(row[2])]
            print(row[0], row[1], len(fulfilled) / len(cond_df))
            fulfilled_rand = test_df[test_df[row[1]].isin(row[2])]
            print(row[0], row[1], len(fulfilled_rand) / len(test_df))


# evaluate_all_services(only_set_params=False)
evaluate_all_services(only_set_params=True)
