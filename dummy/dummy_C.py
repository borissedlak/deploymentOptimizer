from dummy.DummyMaster import DummyMaster


class DummyC(DummyMaster):
    pass


if __name__ == '__main__':
    dummy_C = DummyC("dummy_C", [("latency", "<", 10)])
    dummy_C.create_service_MB()
    dummy_C.check_dependencies()
    # dummy_C.add_footprint_MB()
    # evaluate_slo_fulfillment()
