import random

import pandas as pd
import pgmpy
from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, AICScore, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
from scipy.stats import wasserstein_distance, entropy

from detector import utils

# TODO: Refactor
local_device = 'Xavier'
latency_thresh = 45


class DummyMaster:
    def __init__(self, name, slos):
        self.service_name = name
        self.file_name = name + "_model.xml"
        self.wd_thresh = 0.1
        self.slos = slos

    def create_MB(self):
        raw_samples = pd.read_csv("samples.csv")
        # TODO: Does it mather which device its from?
        # raw_samples = raw_samples[raw_samples['device_type'] == 'PC']

        data = {}
        for (var, rel, val) in self.slos:
            if 'size' == var:
                size_higher_blanket = raw_samples['pixel']
                data.update({'size': size_higher_blanket})
            elif 'latency' == var:
                latency_higher_blanket = raw_samples['delta'] + random.randint(20, 30)
                # latency_higher_blanket = [value + random.randint(1, 50000) for value in latency_higher_blanket]
                data.update({'latency': latency_higher_blanket})
            elif 'rate' == var:
                rate_higher_blanket = raw_samples['fps']
                data.update({'rate': rate_higher_blanket})

        # Use ParameterEstimator to estimate CPDs based on data (you can replace data with your own dataset)
        higher_blanket_data = pd.DataFrame(data=data)

        scoring_method = AICScore(data=higher_blanket_data)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=higher_blanket_data)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        model = BayesianNetwork(ebunch=dag)
        model.fit(higher_blanket_data, estimator=MaximumLikelihoodEstimator)

        utils.export_BN_to_graph(model, vis_ls=['circo'], save=False, name="raw_model", show=True)

        writer = XMLBIFWriter(model)
        writer.write_xmlbif(filename=self.file_name)
        print(f"Model exported as '{self.file_name}'")

    def check_dependencies(self):
        # TODO: Move this to master class
        model_lower_blanket = XMLBIFReader(f'../inference/model.xml').get_model()
        model_higher_blanket = XMLBIFReader(f'{self.file_name}').get_model()

        # TODO: Limit two two devices, one from each side
        promising_combinations = []
        for lower_blanket_variable_name in model_lower_blanket.nodes:
            if lower_blanket_variable_name in ['in_time', 'device_type', 'consumption', 'cpu']:
                continue

            # 1 Check which variables could potentially match
            for higher_blanket_variable_name in model_higher_blanket.nodes:
                p = model_higher_blanket.get_cpds(higher_blanket_variable_name).values.flatten()
                q = model_lower_blanket.get_cpds(lower_blanket_variable_name).values.flatten()
                wd = wasserstein_distance(p, q)

                if wd <= self.wd_thresh:
                    print(f"High WD ({higher_blanket_variable_name} --> {lower_blanket_variable_name}): ", wd)
                    promising_combinations.append((higher_blanket_variable_name, lower_blanket_variable_name))

        # 2 TODO: Missing conditional dependency linking, this must include device_type!
        # TODO: Simply sort from top to bottom (make sure same number of samples) and create linear interpolation
        for (hb_v, lb_v) in promising_combinations:
            p = model_higher_blanket.get_cpds(hb_v).values.flatten()
            q = model_lower_blanket.get_cpds(lb_v).values.flatten()

            if len(p) is not len(q):
                min_len = min(len(p), len(q))
                p = utils.normalize_to_pods(p, min_len)
                q = utils.normalize_to_pods(q, min_len)

            print(entropy(p, q))

    # TODO: Move to master class?
    def evaluate_slo_fulfillment(self):
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
