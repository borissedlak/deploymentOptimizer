# 1) Consume what comes from the producer service --> for this I can use the last MongoDB entry
# 2) Evaluate whether the SLOs are fulfilled from that
# 3) Log the entry in a new collection that can be optimized
import random

import pandas as pd
import pgmpy
from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, AICScore, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
from scipy.stats import wasserstein_distance

from detector import utils

local_device = 'Xavier'
latency_thresh = 45

file_name = f'dummy_A_model.xml'


def create_MB():
    raw_samples = pd.read_csv("samples.csv")
    # TODO: Does it mather which device its from?
    raw_samples = raw_samples[raw_samples['device_type'] == 'Laptop']

    size_higher_blanket = raw_samples['pixel']
    latency_higher_blanket = raw_samples['delta'] + 50
    # latency_higher_blanket = [value + random.randint(1, 500) for value in latency_higher_blanket]

    # Use ParameterEstimator to estimate CPDs based on data (you can replace data with your own dataset)
    higher_blanket_data = pd.DataFrame(data={'size': size_higher_blanket,  # 1204, 1806
                                             'latency': latency_higher_blanket})

    scoring_method = AICScore(data=higher_blanket_data)  # BDeuScore | AICScore
    estimator = HillClimbSearch(data=higher_blanket_data)

    dag: pgmpy.base.DAG = estimator.estimate(
        scoring_method=scoring_method, max_indegree=4, epsilon=1,
    )

    model = BayesianNetwork(ebunch=dag)
    model.fit(higher_blanket_data, estimator=MaximumLikelihoodEstimator)

    utils.export_BN_to_graph(model, vis_ls=['circo'], save=False, name="raw_model", show=True)

    writer = XMLBIFWriter(model)
    writer.write_xmlbif(filename=file_name)
    print(f"Model exported as '{file_name}'")


# TODO: Move to master class and do for all dummy services
def check_dependencies():
    model_lower_blanket = XMLBIFReader(f'../inference/model.xml').get_model()
    model_higher_blanket = XMLBIFReader(f'{file_name}').get_model()

    for lower_blanket_name in model_lower_blanket.nodes:
        if lower_blanket_name in ['in_time', 'device_type', 'consumption', 'cpu']:
            continue

        # 1 Check which variables could potentially match
        for higher_blanket_name in model_higher_blanket.nodes:
            wd = wasserstein_distance(model_higher_blanket.get_cpds(higher_blanket_name).values.flatten(),
                                      model_lower_blanket.get_cpds(lower_blanket_name).values.flatten())
            print(f"Wasserstein Distance ({higher_blanket_name} --> {lower_blanket_name}): ", wd)
        # 2 TODO: Missing conditional dependency linking


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
