import os

import pandas as pd
import pymongo
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

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

    laptop = pd.DataFrame(list(mongo_client['Processor-Laptop'].find()))
    orin = pd.DataFrame(list(mongo_client['Processor-Orin'].find()))
    pc = pd.DataFrame(list(mongo_client['Processor-PC'].find()))
    merged_list = pd.concat([laptop, pc, orin])

    utils.prepare_samples(merged_list, export_path=sample_file)


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
    utils.train_to_MB(None, samples_path=sample_file)
    print("Service P", rate_devices_for_internal())
    # rate_devices_for_interaction()
