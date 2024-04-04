import pandas as pd
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, export_slos_csv, clear_precision_file, evaluate_all_services, \
    add_to_fulfillment_file
from detector.utils import print_in_red

########################################

model_privacy = XMLBIFReader("../Global/model_privacy.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

clear_precision_file()

monitor_ll = constrain_services_variables(
    [model_weather, model_privacy, model_cloud],
    [("cumm_net_delay", 110), ("viewer_satisfaction", 'max')])

for ll, name in [(monitor_ll, "monitor")]:

    non_conflicting_slos = filter_non_conflicting(ll)
    potential_conflicts = verify_slo_duplicates(ll)
    print(potential_conflicts)

    resolved_slos, conflicting_slos = find_compromise(potential_conflicts)

    all_ll_slos = non_conflicting_slos + resolved_slos
    export_slos_csv(all_ll_slos, service_name="buffer")

    # slo_df = pd.read_csv('./buffer_ll_slos.csv')
    # try:
    #     list_fulfill_hl = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=False)
    #     fulfillment_hl = sum(list_fulfill_hl) / len(list_fulfill_hl)
    # except SyntaxError:
    #     print_in_red(f"SyntaxError for {name} due to conflict")
    #     fulfillment_hl = 0
    # slo_df = pd.read_csv('./buffer_ll_slos.csv')
    # try:
    #     list_fulfill_ll = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=True)
    #     fulfillment_ll = sum(list_fulfill_ll) / len(list_fulfill_ll)
    # except SyntaxError:
    #     print_in_red(f"SyntaxError for {name} due to conflict")
    #     fulfillment_ll = 0

    # add_to_fulfillment_file(name, lamb, fulfillment_hl, fulfillment_ll)
