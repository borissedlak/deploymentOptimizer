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

low_level_slos = {}

ve = VariableElimination(model_analysis)


# high_level_thresh = 50
# high_level_var = "cumm_net_delay"

# high_level_thresh = 40
# high_level_var = "delta"


def constrain_services(app_list, hl_slos):
    constraints = []

    for (var, thresh) in hl_slos:
        for m in app_list:
            # Start with the high-level SLO
            for parent in m.get_parents(var):
                hl_states = m.get_cpds(var).__getattribute__("state_names")[var]
                if thresh == "min":
                    hl_valid_states = [min(list(map(int, hl_states)))]
                else:
                    hl_valid_states = list(filter(lambda x: int(x) <= thresh, hl_states))

                constraints_per_parent = get_target_distribution(m, var, hl_valid_states, parent, [])
                constraints.append(constraints_per_parent)


constrain_services([model_weather],  # model_analysis
                   [("cumm_net_delay", 100), ("consumption", "min")])

# TODO: Afterward, optimize by resolving conflicts and removing slos from som intermediary nodes
