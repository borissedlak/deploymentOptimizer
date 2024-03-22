from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from detector import utils

# model = XMLBIFReader("./model_anomaly.xml").get_model()
# ve = VariableElimination(model)
# result = ve.query(variables=['cumm_net_delay_True'], evidence={'delta': '61'})  # 61 + 9 = 70...
# print(result)
#
# result = ve.query(variables=['batch_size'], evidence={'cumm_net_delay_True': 'True'})
# print(result)


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

# Start with the high-level SLO
for parent in model_analysis.get_parents("cumm_net_delay"):
    high_level_states_raw = model_analysis.get_cpds("cumm_net_delay").__getattribute__("state_names")["cumm_net_delay"]
    high_level_states = [int(x) for x in high_level_states_raw]
    valid_states = list(filter(lambda x: x <= 50, high_level_states))

    for state in valid_states:
        result = ve.query(variables=[parent], evidence={'cumm_net_delay': str(state)})
        print(result)

    # result = ve.query(variables=['delta'], evidence={'cumm_net_delay': 'True'})

# result = ve.query(variables=['cumm_net_delay_True'], evidence={'delta': '61'})  # 61 + 9 = 70...
# print(result)

# result = ve.query(variables=['pixel'], evidence={'cumm_net_delay_True': 'True'})
# print(result)

# result = ve.query(variables=['cpu'], evidence={'pixel': '1080', 'fps': '35'})
# print(result)
# result = ve.query(variables=['cpu'], evidence={'pixel': '480', 'fps': '15'})
# print(result)
# result = ve.query(variables=['cpu'], evidence={'consumption': '6'})
# print(result)
# result = ve.query(variables=['cpu'], evidence={'consumption': '7'})
# print(result)

# result = ve.query(variables=['cpu'], evidence={'pixel': '1080', 'fps': '35'})
# print(result)
