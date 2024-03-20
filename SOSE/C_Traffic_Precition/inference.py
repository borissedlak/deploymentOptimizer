from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

# model = XMLBIFReader("./model_anomaly.xml").get_model()
# ve = VariableElimination(model)
# result = ve.query(variables=['cumm_net_delay_True'], evidence={'delta': '61'})  # 61 + 9 = 70...
# print(result)
#
# result = ve.query(variables=['batch_size'], evidence={'cumm_net_delay_True': 'True'})
# print(result)


########################################


model = XMLBIFReader("./model_analysis.xml").get_model()
ve = VariableElimination(model)
# result = ve.query(variables=['cumm_net_delay_True'], evidence={'delta': '61'})  # 61 + 9 = 70...
# print(result)

# TODO: Actual Energy does not go up significantly, but cpu does

result = ve.query(variables=['pixel'], evidence={'cumm_net_delay_True': 'True'})
print(result)

result = ve.query(variables=['cpu'], evidence={'pixel': '1080', 'fps': '35'})
print(result)
result = ve.query(variables=['cpu'], evidence={'pixel': '480', 'fps': '15'})
print(result)
