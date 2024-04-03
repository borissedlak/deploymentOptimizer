import ast
import csv

import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.utils import shuffle

from detector.utils import get_latency_for_devices, print_in_red, find_nested_files_with_suffix


def constrain_services_variables(app_list, hl_slos):
    constraints = []

    for m in app_list:
        for (var, thresh) in hl_slos:
            if not m.has_node(var):
                continue  # some hl slo might not fit for some app

            hl_states = m.get_cpds(var).__getattribute__("state_names")[var]
            if thresh == "min":
                # TODO: I should not cast to int, what if its a float behind...
                hl_valid_states = [str(min(list(map(int, hl_states))))]
            elif thresh == "max":
                hl_valid_states = [str(max(list(map(int, hl_states))))]
            else:
                hl_valid_states = list(filter(lambda x: int(x) <= thresh, hl_states))

            is_root = len(m.get_parents(var)) == 0  # Usually this is never a root, but you never know
            constraints.append((m.name, var, hl_valid_states, True, is_root))  # add hl SLO

            # Traverse parents and constrain them
            for parent in m.get_parents(var):
                constraints_per_parent = get_target_distribution(m, var, hl_valid_states, parent, [])
                constraints.extend(constraints_per_parent)

    return constraints


def get_target_distribution(model: BayesianNetwork, hl_target_var, hl_desired_states, ll_parent_node, constraints):
    print(f"{model.name}: Constraining {ll_parent_node} --> {hl_target_var}")
    ve = VariableElimination(model)

    # Write: The desired logic is as follows: I should for each ll_state ask whats the probability of resulting
    # Write: in the desired hl states. And in case its unlikely (< 0.7 * max) then its not advisable

    ll_states = model.get_cpds(ll_parent_node).__getattribute__("state_names")[ll_parent_node]
    acceptance_matrix = []

    for ll_state in ll_states:
        result = ve.query(variables=[hl_target_var], evidence={ll_parent_node: str(ll_state)})

        chance_desired = 0  # Get chance for each ll state to be in desired range
        for index in range(len(result.values)):
            # Check if state is among the desired ones
            if result.state_names[hl_target_var][index] in hl_desired_states:
                chance_desired += result.values[index]

        acceptance_matrix.append(chance_desired)

    # Filter out states that with a probability of < 60% (compared to best state) produce desired outcomes
    max_value = max(acceptance_matrix)
    # TODO: Here I might require something more sophisticated than just some %
    acceptance_thresh = max_value * 0.70

    # TODO: Actually, all the probabilities to fulfill the ll SLOs are contained here below ...
    ll_valid_states = []
    for i in range(len(ll_states)):
        if acceptance_matrix[i] >= acceptance_thresh and acceptance_thresh > 0:
            ll_valid_states.append(ll_states[i])

    print(f"{ll_parent_node} should only take {ll_valid_states}\n")

    # TODO: Check if node has parents, if not, were at a parametrizable root
    is_root = len(model.get_parents(ll_parent_node)) == 0
    constraints.append((model.name, ll_parent_node, ll_valid_states, False, is_root))

    for par in model.get_parents(ll_parent_node):
        get_target_distribution(model, ll_parent_node, ll_valid_states, par, constraints)

    return constraints


def calculate_cumulative_net_delay(row, src, dest):
    return (get_latency_for_devices(src, row['device_type'], ) +
            get_latency_for_devices(row['device_type'], dest) +
            row['delta'])


def append_privacy_values(row, df_privacy):
    df_filter = shuffle(df_privacy[(df_privacy['fps'] == row['fps']) & (df_privacy['pixel'] == row['pixel'])])
    # TODO: for each row add a random value from df which fulfills the same fps and pixel
    return df_filter.iloc[0]['delta_privacy']


def filter_training_data(df):
    shuffled_set = shuffle(df, random_state=35)
    boundary = int(len(shuffled_set) * 0.85)
    return shuffled_set.iloc[:boundary]


def filter_test_data(df):
    shuffled_set = shuffle(df, random_state=35)
    boundary = int(len(shuffled_set) * 0.85)
    return shuffled_set.iloc[boundary:]


def verify_slo_duplicates(tuples_list):
    grouped_dict = {}
    for tup in tuples_list:
        first_two = tup[:2]
        if first_two in grouped_dict:
            grouped_dict[first_two].append(tup[2])
        else:
            grouped_dict[first_two] = [tup[2]]

    result = []
    for key, value in grouped_dict.items():
        if len(value) > 1:
            result.append((key, value))

    return result


def filter_non_conflicting(tuples_list):
    counts = {}
    result = []
    for tup in tuples_list:
        key = (tup[0], tup[1])
        if key not in counts:
            counts[key] = 1
        else:
            counts[key] += 1

    for tup in tuples_list:
        key = (tup[0], tup[1])
        if counts[key] == 1:
            result.append(tup)

    return result


def find_compromise(conflict_dict):
    resolved_slos = []
    for (ID, values) in conflict_dict:

        # All values are equal (might be even one theoretically)
        if all(x == values[0] for x in values):
            print(ID, "easy, direct match", values[0])
            # Write: Is this natural due to the conditional independence of the configurable params?
            resolved_slos.append((ID[0], ID[1], values[0], False, True))  # TODO: Is it a rule that they are roots?
            continue

        intersection = set.intersection(*map(set, values))
        if len(intersection) != 0:
            print(ID, "still good, some intersection", list(intersection))
            resolved_slos.append((ID[0], ID[1], list(intersection), False, True))
        else:
            print_in_red(f"{ID}, not good, sets disjoint, we have a conflict")

            # TODO: some options to resolve the conflict
            #  1: analyze all possible states and take the values that are between the disjoint sets
            #  2: same as (1) + both disjoint sets
            #  3: just merge the sets, regardless of missing intersections

    return resolved_slos


def export_slos_csv(slos_list, service_name=""):
    with open(f"{service_name}_ll_slos.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["service", "variable", "states", "hl", "root"])
        for row in slos_list:
            csv_writer.writerow(row)


def convert_to_int_or_bool(lst):
    converted_list = []
    for elem in lst:
        if elem.isdigit():
            converted_list.append(int(elem))
        elif elem.lower() == 'true':
            converted_list.append(True)
        elif elem.lower() == 'false':
            converted_list.append(False)
        else:
            converted_list.append(elem)
    return converted_list


def evaluate_all_services(slo_df, only_set_params=False):
    service_list = slo_df['service'].unique()
    slo_df['states'] = slo_df['states'].apply(ast.literal_eval)
    slo_df['states'] = slo_df['states'].apply(convert_to_int_or_bool)  # TODO: This is dangerous when floats come

    for service in service_list:
        test_data_file = find_nested_files_with_suffix('../', f'W_metrics_{service}.csv')[0]
        test_df = filter_test_data(pd.read_csv(test_data_file))
        ll_slos = slo_df[(slo_df['service'] == service) & ~(slo_df['hl'])]  # .iloc[:2]
        hl_slos = slo_df[(slo_df['service'] == service) & (slo_df['hl'])]

        if only_set_params:
            ll_slos = ll_slos[ll_slos['root']]

        tuples_list = list(ll_slos.itertuples(index=False))
        cond_df = test_df[eval(" & ".join(["(test_df['{0}'].isin({1}))".format(col, cond)
                                           for _, col, cond, _, _ in tuples_list]))]
        if len(cond_df) == 0:
            print_in_red("No samples with desired characteristics found")

        # Write: I think I should only set parameters, although ensuring the ll_slo through them is the responsibility
        #  of the elasticity strategies locally. But the difference between them is a super important metrics because it
        #  allows to estimate how well the target outcome can be influenced by the parameters that are set

        for index, row in hl_slos.iterrows():
            # SLO fulfillment if the system is configured with the inferred ll SLOs
            fulfilled = cond_df[cond_df[row[1]].isin(row[2])]
            print(row[0], row[1], len(fulfilled) / len(cond_df))

            # SLO fulfillment for the system under an arbitrary (i.e., random) configuration
            fulfilled_rand = test_df[test_df[row[1]].isin(row[2])]
            # print(row[0], row[1], len(fulfilled_rand) / len(test_df))
