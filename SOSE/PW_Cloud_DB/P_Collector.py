import csv
import itertools
import os
import threading
import time

import numpy as np
import pandas as pd
import pymongo

from detector import utils
from detector.DeviceMetricReporter import DeviceMetricReporter

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]


@utils.print_execution_time
def provide_data(limit=100, batch=1000, number_threads=1, cache_df=None):
    if cache_df is not None:
        df = cache_df
    else:
        df = pd.DataFrame(list(mongo_client['Processor-Laptop'].find({}, batch_size=batch).limit(limit)))

    t_list = []
    for _ in range(number_threads):
        iterations = int(100 / number_threads)
        thread = threading.Thread(target=process_data, args=(df, iterations))
        t_list.append(thread)
        thread.start()
    for thread in t_list:
        thread.join()

    return df

    pass


def process_data(df, repeat):
    for r in ["delta", "fps", "pixel", "cpu", "memory"]:
        for _ in range(repeat):
            np.sum(df[r] ** 2)


limit_list = [100, 500, 1000, 2000]
batch_size_list = [10000, 1000, 500, 100, 50]
thread_list = [1, 5, 10, 20]
cache_list = [True, False]

device_reporter = DeviceMetricReporter(gpu_available=False)
device_metrics = device_reporter.create_metrics(source_fps=None)
all_permutations = list(itertools.product(limit_list, batch_size_list, thread_list, cache_list))

with open("W_metrics_CloudDB.csv", 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["delta", "limit", "batch_size", "threads", "cached"] + list(device_metrics['metrics'].keys()))

for (lim, bs, th, ca) in all_permutations:
    cached = None
    for _ in range(10):
        start_time = time.time()

        df = provide_data(lim, bs, th, cached)
        if ca:
            cached = df

        delta = int((time.time() - start_time) * 1000)
        device_metrics = device_reporter.create_metrics(source_fps=None)

        with open("W_metrics_CloudDB.csv", 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([delta, lim, bs, th, ca] + list(device_metrics['metrics'].values()))

print("done")
