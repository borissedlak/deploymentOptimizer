from dummy.DummyMaster import DummyMaster


class DummyB(DummyMaster):
    pass


if __name__ == '__main__':
    dummy_B = DummyB("dummy_B", [("latency", "<", 30), ("rate", ">=", 10)])
    dummy_B.create_service_MB()
    dummy_B.check_dependencies()
    dummy_B.add_footprint_MB()
    # evaluate_slo_fulfillment()
