import pandas as pd
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, export_slos_csv, clear_precision_file, evaluate_all_services, \
    add_to_fulfillment_file
from detector.utils import print_in_red

########################################

model_analysis = XMLBIFReader("../Global/model_analysis.xml").get_model()
model_privacy = XMLBIFReader("../Global/model_privacy.xml").get_model()
model_anomaly = XMLBIFReader("../Global/model_anomaly.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

clear_precision_file()

for lamb in [i / 10.0 for i in range(1, 11)]:

    traffic_ll = constrain_services_variables(
        [model_analysis, model_weather, model_anomaly, model_cloud],
        [("cumm_net_delay", 40)], lamb=lamb)

    monitor_ll = constrain_services_variables(
        [model_weather, model_privacy, model_cloud],
        [("cumm_net_delay", 110), ("viewer_satisfaction", 'max')], lamb=lamb)

    routing_ll = constrain_services_variables(
        [model_analysis, model_weather],
        [("cumm_net_delay", 45), ("energy", 'min')], lamb=lamb)  # , ("viewer_satisfaction", 'max')])

    for ll, name in [(traffic_ll, "traffic"), (monitor_ll, "monitor"), (routing_ll, "routing")]:

        non_conflicting_slos = filter_non_conflicting(ll)
        potential_conflicts = verify_slo_duplicates(ll)
        print(potential_conflicts)

        resolved_slos = find_compromise(potential_conflicts)

        all_ll_slos = non_conflicting_slos + resolved_slos
        export_slos_csv(all_ll_slos, service_name="buffer")

        slo_df = pd.read_csv('./buffer_ll_slos.csv')
        try:
            list_fulfill_hl = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=False)
            fulfillment_hl = sum(list_fulfill_hl) / len(list_fulfill_hl)
        except SyntaxError:
            print_in_red(f"SyntaxError for {name} due to conflict")
            fulfillment_hl = 0
        slo_df = pd.read_csv('./buffer_ll_slos.csv')
        try:
            list_fulfill_ll = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=True)
            fulfillment_ll = sum(list_fulfill_ll) / len(list_fulfill_ll)
        except SyntaxError:
            print_in_red(f"SyntaxError for {name} due to conflict")
            fulfillment_ll = 0

        add_to_fulfillment_file(name, lamb, fulfillment_hl, fulfillment_ll)
