import os

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
    xavier = pd.DataFrame(list(mongo_client['Processor-Xavier'].find()))  # Or commented out to call case 3.1
    merged_list = pd.concat([laptop, pc, orin, xavier])

    samples = utils.prepare_samples(merged_list, export_path=sample_file, latency_slo=latency_slo)
    utils.train_to_MB(samples, 'Processor', export_file=f'Processor_model.xml')


def infer_slo_fulfillment(model, device_type, slos, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints | {'device_type': device_type}
    ve = VariableElimination(model)
    result = ve.query(variables=slos, evidence=evidence)

    return result


# Write: Summarize from min to top and stop at e.g. 95% of utilization's cpd, to avoid peaks claiming too much space
def infer_device_utilization(model, device_type, hw_variable, constraints=None):
    if constraints is None:
        constraints = {}
    evidence = constraints | {'device_type': device_type}
    ve = VariableElimination(model)
    result = ve.query(variables=[hw_variable], evidence=evidence)

    return result


if __name__ == "__main__":
    device_list = ['PC', 'Laptop', 'Orin', 'Xavier', ('Nano', 'Xavier')]
    Consumer_to_Worker_SLOs = ["latency_slo"]
    service_list = ['Consumer_A', 'Consumer_B', 'Consumer_C']
    consumer_SLOs = {'Consumer_A': ["latency_slo", "size_slo"], 'Consumer_B': ["latency_slo", "rate_slo"],
                     'Consumer_C': ["latency_slo"]}

    # Idea: Remember, that I must retrain if I modify the list of consumers
    Consumer_to_Worker_constraints = {'pixel': '480', 'fps': '15'}
    if "Consumer_A" in service_list:
        Consumer_to_Worker_constraints['pixel'] = '720'
        most_restrictive_consumer_latency = 1000
    if "Consumer_B" in service_list:
        Consumer_to_Worker_constraints['fps'] = '25'
        most_restrictive_consumer_latency = 70
    if "Consumer_C" in service_list:
        most_restrictive_consumer_latency = 40

    consumer_location_fixed = {}  # | {'consumer_location': 'PC'}
    processor_location_fixed = {}  # | {'processor_location': 'Orin'}

    # 1) Provider
    # Skipped! Assumed at Nano
    # Utilizes 30% CPU, 15% Memory, No GPU, Consumption depending on fps

    # 2) Processor
    # load_processor_blanket(latency_slo=most_restrictive_consumer_latency)
    Processor_SLOs = ["in_time"]

    for device in device_list:
        variable_dict = {}
        print('\n' + (device[0] if isinstance(device, tuple) else device))
        Processor = footprint_extractor.extract_footprint("Processor", device[0] if isinstance(device, tuple) else device)
        slo = utils.get_true(infer_slo_fulfillment(Processor, device[1] if isinstance(device, tuple) else device,
                                                   Processor_SLOs + Consumer_to_Worker_SLOs,
                                                   constraints=Consumer_to_Worker_constraints | consumer_location_fixed))
        variable_dict['slo_fulfillment'] = slo
        for metric, unit in [('cpu', '%'), ('memory', '%'), ('consumption', 'W'), ('gpu', '%')]:
            cpd = infer_device_utilization(Processor, device[1] if isinstance(device, tuple) else device, metric,
                                           constraints=Consumer_to_Worker_constraints)
            # print(metric, utils.get_sum_up_to_x(cpd, metric, cpd_max_sum), unit)
            variable_dict[metric] = utils.get_sum_up_to_x(cpd, metric, cpd_max_sum)

        utils.log_dict("Processor", device, variable_dict, Consumer_to_Worker_constraints, most_restrictive_consumer_latency)

    print('------------------------------------------')

    # 3) Consumers

    for cons in service_list:
        variable_dict = {}
        for device in device_list:
            print('\n' + (device[0] if isinstance(device, tuple) else device))
            Consumer = footprint_extractor.extract_footprint(cons, device[0] if isinstance(device, tuple) else device)

            slo = utils.get_true(infer_slo_fulfillment(Consumer, device[1] if isinstance(device, tuple) else device, consumer_SLOs[cons],
                                                       constraints=Consumer_to_Worker_constraints | processor_location_fixed))
            variable_dict['slo_fulfillment'] = slo
            for metric, unit in [('cpu', '%'), ('memory', '%'), ('consumption', 'W')]:
                cpd = infer_device_utilization(Consumer, device[1] if isinstance(device, tuple) else device, metric,
                                               constraints=Consumer_to_Worker_constraints | processor_location_fixed)
                # print(metric, utils.get_sum_up_to_x(cpd, metric, cpd_max_sum), unit)
                variable_dict[metric] = utils.get_sum_up_to_x(cpd, metric, cpd_max_sum)
            variable_dict["gpu"] = 0

            utils.log_dict(cons, device, variable_dict, Consumer_to_Worker_constraints, most_restrictive_consumer_latency)
        print('------------------------------------------')
