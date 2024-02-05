import os

import numpy as np
import pandas as pd
import pgmpy
import pymongo
from pgmpy.base import DAG
from pgmpy.estimators import AICScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader

from detector import utils
from dummy import dummy_A, dummy_B

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

sample_file = "samples.csv"


def load():
    # Connect to MongoDB
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    laptop = pd.DataFrame(list(mongo_client['Laptop-Provider'].find()))
    # orin = pd.DataFrame(list(mongo_client['Orin-Provider'].find()))
    pc = pd.DataFrame(list(mongo_client['PC-Provider'].find()))
    merged_list = pd.concat([laptop, pc])
    # c2 = list(mongoClient['Provider'].find())
    # merged_list = utils.merge_single_dict(c1, c2)

    samples = pd.DataFrame(merged_list)
    samples["delta"] = samples["delta"].apply(np.floor).astype(int)
    samples["cpu"] = samples["cpu"].apply(np.floor).astype(int)
    samples["memory"] = samples["memory"].apply(np.floor).astype(int)
    samples['in_time'] = samples['delta'] <= (1000 / samples['fps'])

    del samples['_id']
    del samples['timestamp']
    del samples['memory']
    # del samples['device_type']

    del samples['cpu']
    del samples['consumption']
    del samples['device_type']

    samples.to_csv(sample_file, index=False)
    print(f"Loaded {sample_file} from MongoDB")


def train():
    samples = pd.read_csv("samples.csv")

    scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
    estimator = HillClimbSearch(data=samples)

    dag: pgmpy.base.DAG = estimator.estimate(
        scoring_method=scoring_method, max_indegree=4, epsilon=1,
    )

    utils.export_BN_to_graph(dag, vis_ls=['circo'], save=False, name="raw_model", show=True)
    model = BayesianNetwork(ebunch=dag)
    model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    writer = XMLBIFWriter(model)
    file_name = f'model.xml'
    writer.write_xmlbif(filename=file_name)
    print(f"Model exported as '{file_name}'")


# TODO: This needs some sort of abstraction here so that I can evaluate multiple services
def infer(device):
    model = XMLBIFReader(f'model.xml').get_model()
    samples = pd.read_csv(sample_file)

    median_fps = get_median_demand(samples)

    inference = VariableElimination(model)
    evidence = {'device_type': device, 'fps': f'{median_fps}'}
    return utils.get_true(inference.query(variables=["in_time"], evidence=evidence))


def rate_devices_for_internal():
    device_list = ['Orin', 'PC']
    internal_slo = []

    for device in device_list:
        slo_fulfillment = infer(device)
        internal_slo.append((device, slo_fulfillment))

    sorted_tuples = sorted(internal_slo, key=lambda x: x[1], reverse=True)
    return sorted_tuples


def rate_devices_for_interaction():
    print("Service A SLO fulfillment if Service ", dummy_A.evaluate_slo_fulfillment())
    print("Service B SLO fulfillment if Service ", dummy_B.evaluate_slo_fulfillment())
    # dummy_A.evaluate_slo_fulfillment()
    # dummy_A.evaluate_slo_fulfillment()


def get_median_demand(samples: pd.DataFrame) -> int:
    # filtered = samples[samples['device_type'] == device_name]
    median = samples['fps'].median().astype(int)
    return median  # or 20


if __name__ == "__main__":
    load()
    train()
    # print("Service P", rate_devices_for_internal())
    # rate_devices_for_interaction()
