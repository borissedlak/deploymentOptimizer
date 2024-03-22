import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from detector.utils import get_latency_for_devices


def get_target_distribution(model: BayesianNetwork, hl_target_var, hl_valid_states, ll_parent_node, constraints):
    print(f"{model.name}: Constraining {ll_parent_node} --> {hl_target_var}")
    ve = VariableElimination(model)

    value_matrix, ll_state_names = None, None
    for state in hl_valid_states:
        result = ve.query(variables=[ll_parent_node], evidence={hl_target_var: str(state)})
        # print(result)

        if ll_state_names is None:
            ll_state_names = result.state_names[ll_parent_node]

        if value_matrix is None:
            value_matrix = result.values
        else:
            value_matrix = value_matrix + result.values

    max_value = np.max(value_matrix)
    acceptance_thresh = max_value * 0.75

    ll_valid_states = []
    for i in range(len(ll_state_names)):
        if value_matrix[i] >= acceptance_thresh:
            ll_valid_states.append(ll_state_names[i])

    print(f"{ll_parent_node} should only take {ll_valid_states}\n")
    constraints.append((model.name, ll_parent_node, ll_valid_states))

    for par in model.get_parents(ll_parent_node):
        get_target_distribution(model, ll_parent_node, ll_valid_states, par, constraints)

    return constraints



def calculate_cumulative_net_delay(row, src, dest):
    return (get_latency_for_devices(src, row['device_type'], ) +
            get_latency_for_devices(row['device_type'], dest) +
            row['delta'])