def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    n_states = len(values)
    new_values = [0.0] * n_states

    for s in range(n_states):
        action_values = []
        for a in range(len(transitions[s])):
            q = rewards[s][a]
            for s_next in range(n_states):
                q += gamma * transitions[s][a][s_next] * values[s_next]
            action_values.append(q)
        new_values[s] = max(action_values) if action_values else 0.0
    return new_values