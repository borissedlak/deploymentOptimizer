import csv

import pandas as pd

from detector import utils

slo_csv_list = []

for i in range(0, 8):
    # 1 Get the expectation

    df = pd.read_csv('../analysis/inference/n_n_assignments.csv')

    part = df[(df['t'] == i) & df['select']]
    service_list = part['service_name'].unique()
    device_list = part['host'].unique()
    pixel = part['pixel'].unique()[0]
    fps = part['fps'].unique()[0]

    cpu_utilization = {key: 0 for key in device_list}
    memory_utilization = {key: 0 for key in device_list}
    gpu_utilization = {key: 0 for key in device_list}

    for service in service_list:
        s_part = part[(part['service_name'] == service)]
        row_as_dict = s_part.iloc[0].to_dict()
        cpu_utilization[row_as_dict['host']] += row_as_dict['cpu']
        memory_utilization[row_as_dict['host']] += row_as_dict['memory']
        gpu_utilization[row_as_dict['host']] += row_as_dict['gpu']
        print(service, row_as_dict['host'], "| SLO", row_as_dict['slo_fulfillment'])
        slo_csv_list.append([service, row_as_dict['host'], row_as_dict['slo_fulfillment'], "estimated", i])

    for device in device_list:
        print(device, "CPU", cpu_utilization[device], "MEM", memory_utilization[device], "GPU", gpu_utilization[device])
    print("")

    # 2 Get the actual performance

    for service in service_list:
        s_part = part[(part['service_name'] == service)]
        row_as_dict = s_part.iloc[0].to_dict()

        if service == 'Processor':
            df = pd.read_csv(f"../analysis/performance/{row_as_dict['host']}.csv")
            # Don't include the higher samples, e.g., fps 35, since they are more likely to violate the slos and
            # at the same time, the system would not operate with that fps because it wants to save energy
            df = df[(df['pixel'] == pixel) & (df['fps'] == fps)]

            slo_valid = 0
            for index, row in df.iterrows():
                rtt = utils.get_latency_for_devices(row_as_dict['host'], 'Nano') + utils.get_latency_for_devices(
                    row_as_dict['host'], row_as_dict['fixed_location']) + row['delta']
                if row['delta'] <= (1000 / fps) and rtt < row_as_dict['min_latency']:
                    slo_valid += 1

            rate = slo_valid / len(df)
            print(service, rate)

            slo_csv_list.append([service, row_as_dict['host'], rate, "experienced", i])
        else:
            df = pd.read_csv(f"../analysis/performance/{row_as_dict['fixed_location']}.csv")
            df = df[(df['pixel'] == pixel) & (df['fps'] == fps)]

            slo_valid = 0
            for index, row in df.iterrows():
                rtt = utils.get_latency_for_devices(row_as_dict['fixed_location'], 'Nano') + utils.get_latency_for_devices(
                    row_as_dict['fixed_location'], row_as_dict['host']) + row['delta']
                if rtt < row_as_dict['min_latency']:
                    slo_valid += 1

            rate = slo_valid / len(df)
            print(service, rate)
            slo_csv_list.append([service, row_as_dict['host'], rate, "experienced", i])

    # TODO: Missing resource evaluation, which in turn requires the processing data with increased resource consumption
    print('-------------------')

with open("./performance/comparison_slo.csv", 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["service", "host", "slo", "modus", "t"])
    csv_writer.writerows(slo_csv_list)
