import pandas as pd

from detector import utils

for i in range(0, 7):
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

    for device in device_list:
        print(device, "CPU", cpu_utilization[device], "MEM", memory_utilization[device], "GPU", gpu_utilization[device])
    print("")

    # 2 Get the actual performance

    for service in service_list:
        s_part = part[(part['service_name'] == service)]
        row_as_dict = s_part.iloc[0].to_dict()

        if service == 'Processor':
            df = pd.read_csv(f"../analysis/performance/{row_as_dict['host']}.csv")
            df = df[(df['pixel'] == pixel) & (df['fps'] == fps)]

            slo_valid = 0
            for index, row in df.iterrows():
                rtt = utils.get_latency_for_devices(row_as_dict['host'], 'Nano') + utils.get_latency_for_devices(
                    row_as_dict['host'], row_as_dict['fixed_location']) + row['delta']
                if row['delta'] <= (1000 / fps) and rtt < row_as_dict['min_latency']:
                    slo_valid += 1

            rate = slo_valid / len(df)
            print(service, rate)
        else:
            df = pd.read_csv(f"../analysis/performance/{row_as_dict['fixed_location']}.csv")
            df = df[(df['pixel'] == pixel) & (df['fps'] == fps)]

            slo_valid = 0
            for index, row in df.iterrows():
                rtt = utils.get_latency_for_devices(row_as_dict['host'], 'Nano') + utils.get_latency_for_devices(
                    row_as_dict['host'], row_as_dict['fixed_location']) + row['delta']
                if row['delta'] <= rtt < row_as_dict['min_latency']:
                    slo_valid += 1

            rate = slo_valid / len(df)
            print(service, rate)

    print('-------------------')
