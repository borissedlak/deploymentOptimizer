from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Precition.tools import get_target_distribution, verify_slo_duplicates, find_compromise

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


def constrain_services_variables(app_list, hl_slos):
    constraints = []

    for m in app_list:
        for (var, thresh) in hl_slos:
            if not m.has_node(var):
                continue  # some hl slo might not fit for some app

            hl_states = m.get_cpds(var).__getattribute__("state_names")[var]
            if thresh == "min":
                # TODO: I should not cast to int, what if its a float behind...
                hl_valid_states = [str(min(list(map(int, hl_states))))]
            elif thresh == "max":
                hl_valid_states = [str(max(list(map(int, hl_states))))]
            else:
                hl_valid_states = list(filter(lambda x: int(x) <= thresh, hl_states))

            constraints.append((m.name, var, hl_valid_states))  # add hl SLO

            # Traverse parents and constrain them
            for parent in m.get_parents(var):
                constraints_per_parent = get_target_distribution(m, var, hl_valid_states, parent, [])
                constraints.extend(constraints_per_parent)

    return constraints


# 1: get all ll SLOs
# ll_slos = constrain_services_variables([model_analysis, model_weather],
#                                        [("cumm_net_delay", 45), ("consumption", "min")])
ll_slos = constrain_services_variables([model_analysis, model_weather, model_anomaly, model_cloud],
                                       [("cumm_net_delay", 50), ("consumption", "min")])

# 2: remove slos from intermediary nodes
# Does not occur in test cases, hence omitted for now

# 3: identify conflicts
potential_conflicts = verify_slo_duplicates(ll_slos)
print(potential_conflicts)

# 4: resolve conflicts --> afterward the parental nodes must be inferred again --> but they are always leaves so far...
find_compromise(potential_conflicts)

# 5: summary with params

pass

# TODO: Prepare some show cases
#  consumption min/max
#  delay 100, pixel max
#  delay 45, pixel max
#  delay 45, pixel max, consumption min
