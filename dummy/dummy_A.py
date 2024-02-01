# 1) Consume what comes from the producer service --> for this I can use the last MongoDB entry
# 2) Evaluate whether the SLOs are fulfilled from that
# 3) Log the entry in a new collection that can be optimized

import pandas as pd

import inference
from detector import utils

local_device = 'Xavier'
latency_thresh = 45


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
    inference.inference.load()
    evaluate_slo_fulfillment()
