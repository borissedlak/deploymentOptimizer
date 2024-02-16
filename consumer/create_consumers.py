# 1) Consume what comes from the producer service --> for this I can use the last MongoDB entry
# 2) Evaluate whether the SLOs are fulfilled from that
# 3) Log the entry in a new collection that can be optimized

from consumer.Consumer import Consumer

if __name__ == '__main__':
    consumer_A = Consumer("Consumer_A", [("latency", "<", 1000), ("size", ">=", 720), ("rate", ">=", 15)])
    consumer_A.create_service_MB()
    consumer_A.check_dependencies()
    consumer_A.add_footprint_MB()

    consumer_B = Consumer("Consumer_B", [("latency", "<", 70), ("size", ">=", 480), ("rate", ">=", 25)])
    consumer_B.create_service_MB()
    consumer_B.check_dependencies()
    consumer_B.add_footprint_MB()

    consumer_C = Consumer("Consumer_C", [("latency", "<", 40), ("size", ">=", 480), ("rate", ">=", 15)])
    consumer_C.create_service_MB()
    consumer_C.check_dependencies()
    consumer_C.add_footprint_MB()  # no_laptop=True)
