from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Precition.tools import get_target_distribution

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
            # Start with the high-level SLO
            for parent in m.get_parents(var):
                hl_states = m.get_cpds(var).__getattribute__("state_names")[var]
                if thresh == "min":
                    # TODO: I should not cast to int, what if its a float behind...
                    hl_valid_states = [str(min(list(map(int, hl_states))))]
                elif thresh == "max":
                    hl_valid_states = [str(max(list(map(int, hl_states))))]
                else:
                    hl_valid_states = list(filter(lambda x: int(x) <= thresh, hl_states))

                constraints_per_parent = get_target_distribution(m, var, hl_valid_states, parent, [])
                constraints.append(constraints_per_parent)

    return constraints


constrain_services_variables([model_analysis, model_weather],
                             [("cumm_net_delay", 45), ("consumption", "min")])
# constrain_services([model_analysis],
#                    [("consumption", "min")])
# constrain_services([model_analysis],
#                    [("consumption", "max")])

# TODO: Afterward, optimize by resolving conflicts and removing slos from some intermediary nodes, also identify params
