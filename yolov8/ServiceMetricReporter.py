from datetime import datetime

import pymongo


# This might actually run as a detached thread, but I think it facilitates the linking of entries
class ServiceMetricReporter:
    def __init__(self, target, clear_collection=False):
        self.target = target
        self.mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")["metrics"]

        if clear_collection:
            self.mongoClient.drop_collection(target)

    def create_metrics(self, time, fps, pixel):
        return {"target": self.target,
                "metrics": {"delta": time, "fps": fps, "pixel": pixel, "timestamp": datetime.now()}}

    def report_metrics(self, time, fps, pixel):
        record = self.create_metrics(time, fps, pixel)
        self.mongoClient[self.target].insert_one(record)
