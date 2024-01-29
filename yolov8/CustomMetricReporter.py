from datetime import datetime

import pymongo


# This might actually run as a detached thread, but I think it facilitates the linking of entries
class CustomMetricReporter:
    def __init__(self, target, clear_collection=False):
        self.target = target
        self.mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")["metrics"]

        if clear_collection:
            self.mongoClient.drop_collection(target)

    def report_this(self, time, fps, pixel):
        record = {"delta": time, "fps": fps, "pixel": pixel, "timestamp": datetime.now()}
        self.mongoClient[self.target].insert_one(record)

