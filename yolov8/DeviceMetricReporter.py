from datetime import datetime

import psutil
import pymongo

from consumption.ConsRegression import ConsRegression

# TODO: This might actually run as a detached thread, but I think it facilitates the linking of entries
class DeviceMetricReporter:
    def __init__(self, target, clear_collection=False):
        self.target = target
        self.consumption_regression = ConsRegression(self.target)
        self.mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")["metrics"]

        if clear_collection:
            self.mongoClient.drop_collection(target)

    def report_now(self):
        # TODO: This might also include network traffic information
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()
        cons = self.consumption_regression.predict(cpu, 0)

        record = {"cpu": cpu, "memory": mem, "consumption": cons, "timestamp": datetime.now()}
        self.mongoClient[self.target].insert_one(record)
