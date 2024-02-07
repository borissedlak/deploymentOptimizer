# 1) Consume what comes from the producer service --> for this I can use the last MongoDB entry
# 2) Evaluate whether the SLOs are fulfilled from that
# 3) Log the entry in a new collection that can be optimized

from dummy.DummyMaster import DummyMaster


class DummyA(DummyMaster):
    pass


if __name__ == '__main__':
    dummy_A = DummyA("dummy_A", [("latency", "<", 100), ("size", ">=", 720)])
    # dummy_A.create_service_MB()
    # dummy_A.check_dependencies()
    dummy_A.add_footprint_MB()
    # evaluate_slo_fulfillment()
