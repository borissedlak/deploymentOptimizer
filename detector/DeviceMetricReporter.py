import os
from datetime import datetime

import psutil
import pymongo

from consumption.ConsRegression import ConsRegression

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
    def __init__(self, clear_collection=False):
        # TODO: Get this from env variables
        self.target = DEVICE_NAME
        self.consumption_regression = ConsRegression(self.target)
        self.mongoClient = pymongo.MongoClient(MONGO_HOST)["metrics"]

        if clear_collection:
            self.mongoClient.drop_collection(self.target)

    def create_metrics(self):
        # TODO: This might also include network traffic information
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, 0)

        return {"target": self.target,
                "metrics": {"device_type": self.target, "cpu": cpu, "memory": mem, "consumption": cons,
                            "timestamp": datetime.now()}}

    # def report_metrics(self):
    #     record = self.create_metrics()
    #     self.mongoClient[self.target].insert_one(record)

    def report_metrics(self, target, record):
        self.mongoClient[target].insert_one(record)
