import random

import pandas as pd
import pgmpy
from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, AICScore, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
from scipy.stats import wasserstein_distance

from detector import utils

# TODO: Refactor
local_device = 'Xavier'
latency_thresh = 45


class DummyMaster:
    def __init__(self, name, slos):
        self.service_name = name
        self.file_name = name + "_model.xml"
        self.wd_thresh = 0.05
        self.js_thresh = 0.9
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
                # Write: That I introduced this fluctuation
                latency_higher_blanket = raw_samples['delta'] + 20
                latency_higher_blanket = [value + random.randint(1, 5) for value in latency_higher_blanket]
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

                # Write: About this marginalization
                p = VariableElimination(model_higher_blanket).query(variables=[higher_blanket_variable_name]).values
                q = VariableElimination(model_lower_blanket).query(variables=[lower_blanket_variable_name]).values

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

            # # Write: Must normalize length of distributions
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
