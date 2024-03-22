from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from SOSE.C_Traffic_Precition.tools import get_target_distribution

# model = XMLBIFReader("./model_anomaly.xml").get_model()
# ve = VariableElimination(model)
# result = ve.query(variables=['cumm_net_delay_True'], evidence={'delta': '61'})  # 61 + 9 = 70...
# print(result)
#
# result = ve.query(variables=['batch_size'], evidence={'cumm_net_delay_True': 'True'})
# print(result)


########################################


########################################


model_analysis = XMLBIFReader("./model_analysis.xml").get_model()
model_anomaly = XMLBIFReader("./model_anomaly.xml").get_model()
model_cloud = XMLBIFReader("./model_cloud.xml").get_model()
model_weather = XMLBIFReader("./model_weather.xml").get_model()

# TODO: Must extend with "expert knowledge"
# utils.export_BN_to_graph(model_analysis)
# utils.export_BN_to_graph(model_anomaly)
# utils.export_BN_to_graph(model_cloud)
# utils.export_BN_to_graph(model_weather)

low_level_slos = {}

ve = VariableElimination(model_analysis)

high_level_thresh = 50
high_level_var = "cumm_net_delay"

# high_level_thresh = 40
# high_level_var = "delta"

constraints = []

# Start with the high-level SLO
for parent in model_analysis.get_parents(high_level_var):
    hl_states = model_analysis.get_cpds(high_level_var).__getattribute__("state_names")[high_level_var]
    hl_valid_states = list(filter(lambda x: int(x) <= high_level_thresh, hl_states))

    constraints_per_parent = get_target_distribution(model_analysis, high_level_var, hl_valid_states, parent, [])
    constraints.append(constraints_per_parent)

pass
