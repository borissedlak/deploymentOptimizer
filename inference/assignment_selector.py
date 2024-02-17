import pandas as pd

df = pd.read_csv('../analysis/inference/n_n_assignments.csv')

for i in range(0, 7):
    part = df[df['t'] == i]
    service_list = part['service_name'].unique()
    device_list = part['host'].unique()

    print(service_list)
    print(device_list)

    answer_dict = {}
    for service in service_list:
        for l1 in device_list:
            for l2 in device_list:
                s_part = part[(part['service_name'] == service) & (part['host'] == l1) & (part['fixed_location'] == l2)]
                row_as_dict = s_part.iloc[0].to_dict()

                if service == "Processor":
                    key = f"{l1}-{l2}"
                else:
                    key = f"{l2}-{l1}"
                if key in answer_dict:
                    old_values = answer_dict[key]
                    slo = old_values['slo_fulfillment'] + row_as_dict['slo_fulfillment']
                    cpu = old_values['cpu'] + row_as_dict['cpu']
                    memory = old_values['memory'] + row_as_dict['memory']
                    gpu = old_values['gpu'] + row_as_dict['gpu']

                    answer_dict[key] = {'slo_fulfillment': slo, 'cpu': cpu, 'memory': memory, 'gpu': gpu}
                else:
                    slo = row_as_dict['slo_fulfillment']
                    cpu = row_as_dict['cpu']
                    memory = row_as_dict['memory']
                    gpu = row_as_dict['gpu']

                    answer_dict[key] = {'slo_fulfillment': slo, 'cpu': cpu, 'memory': memory, 'gpu': gpu}

    answer_list = []
    for key, value in answer_dict.items():
        processor = key.split("-")[0]
        consumer = key.split("-")[1]

        value.update({'processor': processor, 'consumer': consumer})
        answer_list.append(value)

    answer = pd.DataFrame(answer_list)
    # answer = answer[(answer['cpu'] < 100) & (answer['memory'] < 100)]
    pass

