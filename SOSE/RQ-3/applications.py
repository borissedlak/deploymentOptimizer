from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, clear_conflict_file, append_to_conflicts

########################################

model_privacy = XMLBIFReader("../Global/model_privacy.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

clear_conflict_file()

hl_slos_lib = [[("cumm_net_delay", 120), ("viewer_satisfaction", 'max')],
               [("cumm_net_delay", 100), ("viewer_satisfaction", 'max')],
               [("cumm_net_delay", 50), ("viewer_satisfaction", 'max')],
               [("cumm_net_delay", 40), ("viewer_satisfaction", 'max')],
               [("cumm_net_delay", 25), ("viewer_satisfaction", 'max')],
               [("cumm_net_delay", 120), ("energy", 'min')],
               [("cumm_net_delay", 100), ("energy", 'min')],
               [("cumm_net_delay", 50), ("energy", 'min')],
               [("cumm_net_delay", 40), ("energy", 'min')],
               [("cumm_net_delay", 25), ("energy", 'min')]]

for hl_slo_list in hl_slos_lib:
    monitor_ll = constrain_services_variables(
        [model_weather, model_privacy, model_cloud], hl_slo_list)

    non_conflicting_slos = filter_non_conflicting(monitor_ll)
    potential_conflicts = verify_slo_duplicates(monitor_ll)

    resolved_slos, conflicting_slos = find_compromise(potential_conflicts)
    # print(len(conflicting_slos))

    all_ll_slos = non_conflicting_slos + resolved_slos
    append_to_conflicts(hl_slo_list, conflicting_slos)
