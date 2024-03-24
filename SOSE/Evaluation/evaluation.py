import pandas as pd

from SOSE.C_Traffic_Prediction.tools import filter_test_data
from detector.utils import find_nested_files_with_suffix

# TODO
#  * configure the system with the desired states ==> filter test set
#  * calculate the slo fulfillment rate with the desired config
#  * calculate for alternative configurations
#  * compare results

# Group by application
slo_df = pd.read_csv('../C_Traffic_Prediction/ll_slos.csv')
service_list = slo_df['service'].unique()

for service in service_list:
    test_data_file = find_nested_files_with_suffix('../', f'W_metrics_{service}.csv')[0]
    test_df = filter_test_data(pd.read_csv(test_data_file))
