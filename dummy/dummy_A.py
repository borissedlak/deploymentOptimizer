# 1) Consume what comes from the producer service --> for this I can use the last MongoDB entry
# 2) Evaluate whether the SLOs are fulfilled from that
# 3) Log the entry in a new collection that can be optimized
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
from pgmpy.sampling import BayesianModelSampling

from detector import utils

local_device = 'Xavier'
latency_thresh = 45

file_name = f'dummy_A_model.xml'


def create_MB():
    samples = pd.read_csv("samples.csv")

    model = BayesianModel()
    model.add_node('latency')
    model.add_node('size')

    latency = samples[samples['device_type'] == 'PC']['delta'] + 30

    # Use ParameterEstimator to estimate CPDs based on data (you can replace data with your own dataset)
    data = pd.DataFrame(data={'size': [480, 720, 1080] * 1204, # 1204, 1806
                              'latency': latency})
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    utils.export_BN_to_graph(model, vis_ls=['circo'], save=False, name="raw_model", show=True)

    writer = XMLBIFWriter(model)
    writer.write_xmlbif(filename=file_name)
    print(f"Model exported as '{file_name}'")


def check_dependencies():
    model_a = XMLBIFReader(f'{file_name}').get_model()
    model_main = XMLBIFReader(f'../inference/model.xml').get_model()
    sample_size = 100

    samples_lower_blanket = BayesianModelSampling(model_main).forward_sample(size=sample_size, seed=35)
    samples_higher_blanket = BayesianModelSampling(model_a).forward_sample(size=sample_size, seed=35)

    for lower_blanket_name in samples_lower_blanket.columns:
        if lower_blanket_name in ['in_time', 'device_type']:
            continue

        for higher_blanket_name in ['size', 'latency']:

            list1 = sorted(samples_lower_blanket[lower_blanket_name].astype(int))
            list2 = sorted(samples_higher_blanket[higher_blanket_name].astype(int))

            # TODO: Do with all columns
            correlation_coefficient = np.corrcoef(list1, list2)[0, 1]

            # Create a scatter plot
            plt.scatter(list1, list2)

            # Add labels and title
            plt.xlabel(f'{lower_blanket_name}')
            plt.ylabel(f'{higher_blanket_name}')
            plt.title(f'Correlation Coefficient for {higher_blanket_name} --> {lower_blanket_name}: {correlation_coefficient:.2f}')

            # Show the plot
            plt.show()


# TODO: Move to master class?
def evaluate_slo_fulfillment():
    samples = pd.read_csv("samples.csv")

    del samples['in_time']
    del samples['consumption']
    del samples['cpu']

    device_types = samples['device_type'].unique()
    slo_fulfillment_per_device = []

    for device_type in device_types:
        slo_valid = 0
        filtered = samples[samples['device_type'] == device_type]
        network_latency = utils.get_latency_for_devices(local_device, device_type)

        for index, row in filtered.iterrows():
            if row['delta'] + network_latency <= latency_thresh:
                slo_valid += 1

        rate = slo_valid / len(filtered)
        slo_fulfillment_per_device.append((device_type, rate))

    sorted_tuples = sorted(slo_fulfillment_per_device, key=lambda x: x[1], reverse=True)
    return sorted_tuples


if __name__ == '__main__':
    create_MB()
    check_dependencies()
    # inference.inference.load()
    # evaluate_slo_fulfillment()
