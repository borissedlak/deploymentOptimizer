import os
import threading
from datetime import datetime

import psutil
import pymongo

from consumption.ConsRegression import ConsRegression
from detector import utils

# This might actually run as a detached thread, but I think it facilitates the linking of entries
# class Reporter:
#     def create_metrics(self) -> dict[str, float]:
#         pass
#
#     def report_metrics(self) -> None:
#         pass
DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")


class DeviceMetricReporter:
    def __init__(self, gpu_available=0):
        self.target = DEVICE_NAME
        self.consumption_regression = ConsRegression(self.target)
        self.mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
        self.gpu_available = gpu_available

        # if clear_collection:
        #     self.mongoClient.drop_collection(target)
        #     print(f"Dropping collection {target}")

    def create_metrics(self):
        # TODO: This might also include network traffic information
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, self.gpu_available)

        return {"target": self.target,
                "metrics": {"device_type": self.target, "cpu": cpu, "memory": mem, "consumption": cons,
                            "timestamp": datetime.now()}}

    # @utils.print_execution_time
    def report_metrics(self, target, record):
        insert_thread = threading.Thread(target=self.run_detached, args=(target, record))
        insert_thread.start()

    def run_detached(self, target, record):
        # mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
        self.mongo_client[target].insert_one(record)
