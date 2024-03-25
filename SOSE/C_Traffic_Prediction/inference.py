from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Prediction.tools import verify_slo_duplicates, find_compromise, \
    filter_non_conflicting, constrain_services_variables, export_slos_csv
from detector import utils

########################################

model_analysis = XMLBIFReader("../Global/model_analysis.xml").get_model()
model_anomaly = XMLBIFReader("../Global/model_anomaly.xml").get_model()
model_cloud = XMLBIFReader("../Global/model_cloud.xml").get_model()
model_weather = XMLBIFReader("../Global/model_weather.xml").get_model()

# utils.export_BN_to_graph(model_analysis)
# utils.export_BN_to_graph(model_anomaly)
# utils.export_BN_to_graph(model_cloud)
# utils.export_BN_to_graph(model_weather)

ve = VariableElimination(model_analysis)

# Write 1: get all ll SLOs

ll_slos = constrain_services_variables([model_analysis, model_weather, model_anomaly, model_cloud],
                                       [("cumm_net_delay", 45), ("consumption", 'min')])

# Write 2: remove slos from intermediary nodes
# Does not occur in test cases, hence omitted for now

# Write 3: identify conflicts
non_conflicting_slos = filter_non_conflicting(ll_slos)
potential_conflicts = verify_slo_duplicates(ll_slos)
print(potential_conflicts)

# TODO: Some manual resolution for one use case that can be evaluated
# Write 4: resolve conflicts --> afterward parental nodes must be inferred again --> but they are all leaves so far...
resolved_slos = find_compromise(potential_conflicts)

# Write 5: summary with params
all_ll_slos = non_conflicting_slos + resolved_slos
export_slos_csv(all_ll_slos)

# TODO: Prepare some show cases
#  consumption min/max
#  delay 100, pixel max
#  delay 45, pixel max
#  delay 45, pixel max, consumption min
