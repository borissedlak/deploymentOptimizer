import pandas as pd
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, export_slos_csv, clear_precision_file, evaluate_all_services, \
    add_to_fulfillment_file

########################################

model_analysis = XMLBIFReader("../Global/model_analysis.xml").get_model()
model_privacy = XMLBIFReader("../Global/model_privacy.xml").get_model()
model_anomaly = XMLBIFReader("../Global/model_anomaly.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

clear_precision_file()

for lamb in [i / 10.0 for i in range(1, 11)]:
    print("\n", "Now processing", lamb, "\n")
    traffic_ll = constrain_services_variables(
        [model_analysis, model_weather, model_anomaly, model_cloud],
        [("cumm_net_delay", 40)], lamb=lamb)

    non_conflicting_slos = filter_non_conflicting(traffic_ll)
    potential_conflicts = verify_slo_duplicates(traffic_ll)
    print(potential_conflicts)

    resolved_slos = find_compromise(potential_conflicts)

    all_ll_slos = non_conflicting_slos + resolved_slos
    export_slos_csv(all_ll_slos, service_name="buffer")

    # Analysis

    slo_df = pd.read_csv('./buffer_ll_slos.csv')
    list_fulfill_hl = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=False)
    fulfillment_hl = sum(list_fulfill_hl) / len(list_fulfill_hl)
    slo_df = pd.read_csv('./buffer_ll_slos.csv')
    list_fulfill_ll = evaluate_all_services(slo_df=slo_df, only_set_params=True, ll=True)
    fulfillment_ll = sum(list_fulfill_ll) / len(list_fulfill_ll)

    add_to_fulfillment_file(lamb, fulfillment_hl, fulfillment_ll)
