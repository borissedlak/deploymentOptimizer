import os
import sys

import pandas as pd
import pymongo
from pgmpy.inference import VariableElimination

from detector import utils
from inference import footprint_extractor

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

sample_file = "samples.csv"


def load_processor_blanket():
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    laptop = pd.DataFrame(list(mongo_client['Processor-Laptop'].find()))
    orin = pd.DataFrame(list(mongo_client['Processor-Orin'].find()))
    pc = pd.DataFrame(list(mongo_client['Processor-PC'].find()))
    merged_list = pd.concat([laptop, pc, orin])

    samples = utils.prepare_samples(merged_list, export_path=sample_file)
    utils.train_to_MB(samples, 'Processor', export_file=f'Processor_model.xml')


# TODO: This needs some sort of abstraction here so that I can evaluate multiple services
def infer_slo_fulfillment(model, device, slos, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints | {'device_type': device}
    ve = VariableElimination(model)
    result = ve.query(variables=slos, evidence=evidence)

    return result


# Idea: Should also do all ratings for one service at one, but this requires the impacts in one model as well
def rate_devices_for_processor(model):
    device_list = ['Orin', 'PC']
    internal_slo = []

    for device in device_list:
        slo_fulfillment = infer_slo_fulfillment(model, device, )
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
    # 1) Provider
    # Skipped!

    # 2) Processor
    # load_processor_blanket()
    Processor_SLOs = ["in_time"]
    constraints_from_upper_blankets = {'pixel': '1080', 'fps': '25', 'gpu': 'None'}
    # Processor = footprint_extractor.extract_footprint("Processor", "Orin")
    # print(utils.get_true(infer_slo_fulfillment(Processor, "Orin", Processor_SLOs,
    #                                            constraints=constraints_from_upper_blankets)))

    # TODO: Error here
    # TODO: Extract utilization for each setup (incl. energy consumption)
    # TODO: Rank deployment options

    Processor = footprint_extractor.extract_footprint("Processor", "Laptop")
    print(utils.get_true(infer_slo_fulfillment(Processor, "Laptop", Processor_SLOs,
                                               constraints=constraints_from_upper_blankets)))

    sys.exit()
    # 3) Consumers
    Consumer_A_SLOs = ["latency_slo", "size_slo"]
    Consumer_A_Orin = footprint_extractor.extract_footprint("Consumer_A", "Orin")
    print(utils.get_true(infer_slo_fulfillment(Consumer_A_Orin, "Orin", Consumer_A_SLOs)))

    Consumer_A_PC = footprint_extractor.extract_footprint("Consumer_A", "PC")
    print(utils.get_true(infer_slo_fulfillment(Consumer_A_PC, "PC", Consumer_A_SLOs)))

    Consumer_A_Nano = footprint_extractor.extract_footprint("Consumer_A", "Xavier")
    print(utils.get_true(infer_slo_fulfillment(Consumer_A_Nano, "Orin", Consumer_A_SLOs)))

    Consumer_B_SLOs = ["latency_slo", "rate_slo"]
    Consumer_B_Orin = footprint_extractor.extract_footprint("Consumer_B", "Orin")
    print(utils.get_true(infer_slo_fulfillment(Consumer_B_Orin, "Orin", Consumer_B_SLOs)))

    # print("Service P", rate_devices_for_internal())
    # rate_devices_for_interaction()
