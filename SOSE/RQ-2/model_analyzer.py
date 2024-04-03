from pgmpy.readwrite import XMLBIFReader

from detector import utils

model_analysis = XMLBIFReader("../Global/model_analysis.xml").get_model()

utils.export_BN_to_graph(model_analysis)

# Idea: Given that I want the lowest energy consumption and I set this through fps and pixel, the precision of the model is also measured
#  through the low-level SLO fulfillment, i.e., do the parameters move the low-level SLOs into the desired range.
#

pass
