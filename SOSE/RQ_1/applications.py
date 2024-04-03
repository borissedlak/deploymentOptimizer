from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, export_slos_csv

########################################

model_analysis = XMLBIFReader("../Global/model_analysis.xml").get_model()
model_privacy = XMLBIFReader("../Global/model_privacy.xml").get_model()
model_anomaly = XMLBIFReader("../Global/model_anomaly.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

traffic_ll = constrain_services_variables(
    [model_analysis, model_weather, model_anomaly, model_cloud],
    [("cumm_net_delay", 40)])

monitor_ll = constrain_services_variables(
    [model_weather, model_privacy, model_cloud],
    [("cumm_net_delay", 110), ("viewer_satisfaction", 'max')])

routing_ll = constrain_services_variables(
    [model_analysis, model_weather],
    [("cumm_net_delay", 45), ("energy", 'min')])  # , ("viewer_satisfaction", 'max')])

for ll, name in [(traffic_ll, "traffic"), (monitor_ll, "monitor"), (routing_ll, "routing")]:
    non_conflicting_slos = filter_non_conflicting(ll)
    potential_conflicts = verify_slo_duplicates(ll)
    print(potential_conflicts)

    resolved_slos = find_compromise(potential_conflicts)

    all_ll_slos = non_conflicting_slos + resolved_slos
    export_slos_csv(all_ll_slos, service_name=name)
