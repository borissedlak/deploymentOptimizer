import random

import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
from scipy.stats import wasserstein_distance

from detector import utils


class Consumer:
    def __init__(self, name, slos):
        self.service_name = name
        self.file_name = name + "_model.xml"
        self.wd_thresh = 0.1
        self.js_thresh = 0.9
        self.slos = slos

    def create_service_MB(self):
        raw_samples = pd.read_csv("samples.csv")
        # raw_samples = raw_samples[raw_samples['device_type'] == 'PC']

        data = {}
        for (var, rel, val) in self.slos:
            if 'size' == var:
                size_higher_blanket = raw_samples['pixel']
                data.update({'size': size_higher_blanket})
                data.update({'size_slo': raw_samples['pixel'] >= val})
            elif 'latency' == var:
                # Write: That I introduced this fluctuation
                latency_higher_blanket = raw_samples['delta']
                latency_higher_blanket = [value + random.randint(1, 25) for value in latency_higher_blanket]
                data.update({'latency': latency_higher_blanket})
                data.update({'latency_slo': raw_samples['delta'] < val})
            elif 'rate' == var:
                rate_higher_blanket = raw_samples['fps']
                data.update({'rate': rate_higher_blanket})
                data.update({'rate_slo': raw_samples['fps'] > val})

        higher_blanket_data = pd.DataFrame(data=data)
        utils.train_to_MB(higher_blanket_data, self.service_name, export_file=self.file_name)

    def check_dependencies(self):
        model_higher_blanket = XMLBIFReader(f'{self.file_name}').get_model()
        model_lower_blanket = XMLBIFReader(f'../inference/Processor_model.xml').get_model()

        promising_combinations = []
        for lower_blanket_variable_name in model_lower_blanket.nodes:
            if lower_blanket_variable_name in ['in_time', 'device_type', 'consumption', 'cpu', 'gpu']:
                continue

            # 1 Check which variables could potentially match
            for higher_blanket_variable_name in model_higher_blanket.nodes:
                if higher_blanket_variable_name.endswith('_slo'):
                    continue

                # Write: About this marginalization
                p = VariableElimination(model_higher_blanket).query(variables=[higher_blanket_variable_name]).values
                q = VariableElimination(model_lower_blanket).query(variables=[lower_blanket_variable_name]).values

                # Idea: Compare (all) combination of MB [device_type = x] between the models. Some device should match
                # Idea: OR do this repeatedly for multiple combinations and compare the results
                wd = wasserstein_distance(p, q)

                if wd <= self.wd_thresh:
                    print(f"High WD ({higher_blanket_variable_name} --> {lower_blanket_variable_name}): ", wd,
                          ", indicating a (potentially confounded) dependency")
                    promising_combinations.append((higher_blanket_variable_name, lower_blanket_variable_name))

        print("---------------")

        # 2 Check if they match in terms of values, if not, it's confounded by a constant factor
        for (hb_v, lb_v) in promising_combinations:
            p = VariableElimination(model_higher_blanket).query(variables=[hb_v]).state_names[hb_v]
            q = VariableElimination(model_lower_blanket).query(variables=[lb_v]).state_names[lb_v]

            # # vrite: Must normalize length of distributions
            # if len(p) is not len(q):
            #     min_len = min(len(p), len(q))
            #     p, bin_centers_p = utils.normalize_to_pods(p, min_len)
            #     q, bin_centers_q = utils.normalize_to_pods(q, min_len)
            #
            #     # KL Divergence is asymmetric!
            # print(entropy(p, q))
            # print(utils.JSD(p, q))

            # Low similarity indicates a constant shift within the distribution due to a confounding var
            similarity = utils.jaccard_similarity(p, q)
            if similarity < self.js_thresh:
                print(f"Low JS ({hb_v} --> {lb_v}): ", similarity, ", flagging as confounded")

    def add_footprint_MB(self, no_laptop=False):
        current_blanket = XMLBIFReader(f'{self.file_name}').get_model()
        current_blanket.add_node("cpu")
        current_blanket.add_node("device_type")
        current_blanket.add_node("memory")
        current_blanket.add_node("consumption")

        current_blanket.add_edge("device_type", "cpu")
        current_blanket.add_edge("device_type", "memory")
        current_blanket.add_edge("device_type", "consumption")

        if no_laptop:
            cpd_device_type = TabularCPD(variable='device_type', variable_card=2, values=[[0.33], [0.33]],
                                         state_names={'device_type': ['Orin', 'PC']})
            cpd_cpu = TabularCPD(variable='cpu', variable_card=3,
                                 values=[[0.0, 1.0],
                                         [0.0, 0.0],
                                         [1.0, 0.0]],
                                 evidence=['device_type'],
                                 evidence_card=[2],
                                 state_names={'cpu': ['15', '20', '25'],
                                              'device_type': ['Orin', 'PC']})
            cpd_memory = TabularCPD(variable='memory', variable_card=3,
                                    values=[[0.0, 1.0],
                                            [0.0, 0.0],
                                            [1.0, 0.0]],
                                    evidence=['device_type'],
                                    evidence_card=[2],
                                    state_names={'memory': ['20', '30', '45'],
                                                 'device_type': ['Orin', 'PC']})

            cpd_consumption = TabularCPD(variable='consumption', variable_card=3,
                                         values=[[1.0, 0.0],
                                                 [0.0, 0.0],
                                                 [0.0, 1.0]],
                                         evidence=['device_type'],
                                         evidence_card=[2],
                                         state_names={'consumption': ['7', '22', '88'],
                                                      'device_type': ['Orin', 'PC']})
        else:
            cpd_device_type = TabularCPD(variable='device_type', variable_card=3, values=[[0.33], [0.33], [0.33]],
                                         state_names={'device_type': ['Orin', 'Laptop', 'PC']})
            cpd_cpu = TabularCPD(variable='cpu', variable_card=3,
                                 values=[[0.0, 0.0, 1.0],
                                         [0.0, 1.0, 0.0],
                                         [1.0, 0.0, 0.0]],
                                 evidence=['device_type'],
                                 evidence_card=[3],
                                 state_names={'cpu': ['15', '20', '25'],
                                              'device_type': ['Orin', 'Laptop', 'PC']})
            cpd_memory = TabularCPD(variable='memory', variable_card=3,
                                    values=[[0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0],
                                            [1.0, 0.0, 0.0]],
                                    evidence=['device_type'],
                                    evidence_card=[3],
                                    state_names={'memory': ['20', '30', '45'],
                                                 'device_type': ['Orin', 'Laptop', 'PC']})

            cpd_consumption = TabularCPD(variable='consumption', variable_card=3,
                                         values=[[1.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 1.0]],
                                         evidence=['device_type'],
                                         evidence_card=[3],
                                         state_names={'consumption': ['7', '22', '88'],
                                                      'device_type': ['Orin', 'Laptop', 'PC']})

        current_blanket.add_cpds(cpd_device_type, cpd_cpu, cpd_memory, cpd_consumption)
        utils.export_model_to_path(current_blanket, self.file_name)

    def evaluate_slo_fulfillment(self):

        local_device = 'Xavier'
        latency_thresh = 45

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
