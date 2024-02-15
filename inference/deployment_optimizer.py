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
cpd_max_sum = 0.95


def load_processor_blanket(latency_slo=None):
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    laptop = pd.DataFrame(list(mongo_client['Processor-Laptop'].find()))
    orin = pd.DataFrame(list(mongo_client['Processor-Orin'].find()))
    pc = pd.DataFrame(list(mongo_client['Processor-PC'].find()))
    merged_list = pd.concat([laptop, pc, orin])

    samples = utils.prepare_samples(merged_list, export_path=sample_file, latency_slo=latency_slo)
    utils.train_to_MB(samples, 'Processor', export_file=f'Processor_model.xml')


def infer_slo_fulfillment(model, device_type, slos, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints | {'device_type': device_type}
    ve = VariableElimination(model)
    result = ve.query(variables=slos, evidence=evidence)

    return result


# Idea: Summarize from min to top and stop at e.g. 95% of utilization's cpd, to avoid peaks claiming too much space
def infer_device_utilization(model, device_type, hw_variable, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints | {'device_type': device_type}
    ve = VariableElimination(model)
    result = ve.query(variables=[hw_variable], evidence=evidence)

    return result


# Idea: Should also do all ratings for one service at one, but this requires the impacts in one model as well
# def rate_devices_for_processor(model):
#     device_list = ['Orin', 'PC']
#     internal_slo = []
#
#     for device in device_list:
#         slo_fulfillment = infer_slo_fulfillment(model, device)
#         internal_slo.append((device, slo_fulfillment))
#
#     sorted_tuples = sorted(internal_slo, key=lambda x: x[1], reverse=True)
#     return sorted_tuples


def get_median_demand(samples: pd.DataFrame) -> int:
    # filtered = samples[samples['device_type'] == device_name]
    median = samples['fps'].median().astype(int)
    return median  # or 20


if __name__ == "__main__":

    # 1) Provider
    # Skipped! Assumed at Nano
    # Consumes 30% CPU, 15% Memory, No GPU

    # 2) Processor
    # load_processor_blanket(latency_slo=50)  # Takes most restrictive from the consumer SLOs
    Processor_SLOs = ["in_time"]
    Consumer_SLOs = ["latency_slo"]
    lower_blanket_constraints = {'pixel': '480', 'fps': '25'}  # | {'consumer_location': 'PC'}

    for device in ['PC', 'Orin', 'Laptop']:
        print('\n' + device)
        Processor = footprint_extractor.extract_footprint("Processor", device)
        print(utils.get_true(infer_slo_fulfillment(Processor, device, Processor_SLOs + Consumer_SLOs,
                                                   constraints=lower_blanket_constraints)))
        for metric, unit in [('cpu', '%'), ('gpu', '%'), ('memory', '%'), ('consumption', 'W')]:
            cpd = infer_device_utilization(Processor, device, metric, constraints=lower_blanket_constraints)
            print(metric, utils.get_sum_up_to_x(cpd, metric, cpd_max_sum), unit)

    print('------------------------------------------')

    # 3) Consumers
    Consumer_A_SLOs = ["latency_slo", "size_slo"]
    for device in ['PC', 'Orin', 'Laptop', ('Xavier', 'Orin'), ('Nano', 'Orin')]:
        print('\n' + (device[0] if isinstance(device, tuple) else device))
        Consumer_A = footprint_extractor.extract_footprint("Consumer_A",
                                                           device[0] if isinstance(device, tuple) else device)
        print(utils.get_true(
            infer_slo_fulfillment(Consumer_A, device[1] if isinstance(device, tuple) else device, Consumer_A_SLOs)))
        for metric, unit in [('cpu', '%'), ('memory', '%'), ('consumption', 'W')]:
            cpd = infer_device_utilization(Consumer_A, device[1] if isinstance(device, tuple) else device, metric)
            print(metric, utils.get_sum_up_to_x(cpd, metric, cpd_max_sum), unit)

    sys.exit()
    print('------------------------------------------')

    Consumer_B_SLOs = ["latency_slo", "rate_slo"]
    for device in ['PC', 'Orin', 'Laptop', ('Nano', 'Orin')]:
        Consumer_B = footprint_extractor.extract_footprint("Consumer_B",
                                                           device[0] if isinstance(device, tuple) else device)
        print(utils.get_true(
            infer_slo_fulfillment(Consumer_B, device[1] if isinstance(device, tuple) else device, Consumer_B_SLOs)))

    print('------------------------------------------')

    # print("Service P", rate_devices_for_internal())
    # rate_devices_for_interaction()
