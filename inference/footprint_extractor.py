import os

from pgmpy.readwrite import XMLBIFReader

from detector import utils

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

sample_file = "samples.csv"


# footprint = Service-Host Deployment implications as MB
def extract_footprint(service, host):
    # mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]
    # list_of_collections = mongo_client.list_collection_names()

    # if utils.get_mb_name(service, host) in list_of_collections:
    #     raw_samples = pd.DataFrame(list(mongo_client[utils.get_mb_name(service, host)].find()))
    #     samples = utils.prepare_samples(raw_samples)
    #     return utils.train_to_MB(samples)

    # Case 1: Exact match evaluated empirically
    model_xml_files = utils.find_nested_files_with_suffix('../', f'{service}_model.xml')
    if len(model_xml_files) > 0:
        service_mb = XMLBIFReader(model_xml_files[0]).get_model()
        if utils.check_device_present_in_mb(service_mb, host):
            return service_mb

    # Case 2.1: Comparable service evaluated at the target device type [Metadata]
    # Idea: Should also check the online footprints, not only locally from the dummies
    similar_services = utils.check_similar_services_same_host(host)  # Limitation: Only works with Consumer
    service_mb = XMLBIFReader(model_xml_files[0]).get_model()  # Limitation: Should never be empty
    if len(similar_services) > 0:
        for potential_host_mb in similar_services:

            # Only when no edges between blankets
            if utils.check_edges_with_service(potential_host_mb):
                # Just take the first, no comparison between them
                return utils.plug_in_service_variables(service_mb, potential_host_mb)

    # Case 2.2: Comparable service evaluated at the target device type [Footprint]

    # Write: Shouldn't this be stronger device? No! If it worked on a weaker device, it will also work on the target
    # Write: Whereas, if it worked on a stronger device, you don't know anything about the weaker one --> Case 4
    # Case 3.1: Same service evaluated at a comparable (=same or weaker) device type --> Case 1
    comparable_devices = utils.check_same_services_similar_host(service, host)
    if len(comparable_devices) > 0:
        return comparable_devices[0]

    # Case 3.2: Comparable service evaluated at a comparable device type --> Case 2.1

    # Case 4: No comparable service for the device type existing
    closest_devices = utils.check_same_services_similar_host(service, host, any_host=True)
    # Write: Interpolation according to the known devices, but requires at least two
    return utils.penalize_device_mb(closest_devices[0], 1)

    # raise RuntimeError("Should not happen :(")


if __name__ == "__main__":
    # Case 1
    extract_footprint("Processor", "Laptop")
    extract_footprint("Consumer_A", "Laptop")  # Case 1 implementation incomplete, but return value matches still

    # Case 2.1
    extract_footprint("Consumer_C", "Laptop")

    # Case 3.1
    extract_footprint("Consumer_A", "Xavier")
    extract_footprint("Processor", "Xavier")

    # Case 4
    extract_footprint("Consumer_B", "Nano")
    extract_footprint("Processor", "Nano")
