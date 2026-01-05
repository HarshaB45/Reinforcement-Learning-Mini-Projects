import numpy as np
from Assignment2Tools import prob_vector_generator, markov_matrix_generator

def compute_Q(i_phi, i_z, i_b, i_m, a, V, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta):
    
    phi = Swind[i_phi]
    z = Swind[i_z]
    m = i_m
    #n_phi = len(Swind)
    d_arr = np.arange(Delta + 1)  # Possible solar power outcomes
    Q_val = 0.0 

    #passive phase: 
        
    if m == 0:
        immediate = - (phi - z)**2
        b_next = np.minimum(i_b + d_arr, B)  
        T = np.array([gamma * V[:, i_z, b_val, tau] + (1 - gamma) * V[:, i_z, b_val, 0] 
                      for b_val in b_next]) 
        p_phi = P[i_phi, :]  
        expected_next = np.dot(p_phi, np.sum(T * alpha.reshape(-1, 1), axis=0))
        Q_val = immediate + beta * expected_next

    else:
        # Active phase
        m_next = m - 1 if m > 1 else 0
        if a == -1:
            # No transmission action.
            immediate = - (phi - z)**2
            b_next = np.minimum(i_b + d_arr, B)
            T = np.array([V[:, i_z, b_val, m_next] for b_val in b_next])  # shape: (Delta+1, n_phi)
            p_phi = P[i_phi, :]
            expected_next = np.dot(p_phi, np.sum(T * alpha.reshape(-1, 1), axis=0))
            Q_val = immediate + beta * expected_next
        else:
            # Transmission action.
            a_val = Swind[a]
            immediate = - ( lmbda * (phi - a_val)*2 + (1 - lmbda) * (phi - z)*2 )
            b_next = np.minimum(i_b - eta + d_arr, B)
            # Find index corresponding to a_val in Swind.
            i_z_new = np.where(np.isclose(Swind, a_val))[0][0]
            T = np.array([lmbda * V[:, i_z_new, b_val, m_next] + (1 - lmbda) * V[:, i_z, b_val, m_next]
                          for b_val in b_next])  # shape: (Delta+1, n_phi)
            p_phi = P[i_phi, :]
            expected_next = np.dot(p_phi, np.sum(T * alpha.reshape(-1, 1), axis=0))
            Q_val = immediate + beta * expected_next
    return Q_val

def policy_evaluation(V, policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    
    n_phi = len(Swind)
    n_z = len(Swind)
    n_b = B + 1
    n_m = tau + 1

    iteration = 0
    while True:
        delta_val = 0.0
        V_new = np.copy(V)
        for i_phi in range(n_phi):
            for i_z in range(n_z):
                for i_b in range(n_b):
                    for i_m in range(n_m):
                        a = policy[i_phi, i_z, i_b, i_m]
                        Q_val = compute_Q(i_phi, i_z, i_b, i_m, a, V, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
                        V_new[i_phi, i_z, i_b, i_m] = Q_val
                        delta_val = max(delta_val, abs(Q_val - V[i_phi, i_z, i_b, i_m]))
        V = V_new
        iteration += 1
        print(f"Policy Evaluation Iteration {iteration}, delta: {delta_val:.6f}")
        if iteration >= Kmin and delta_val < theta:
            break
    return V

def policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    n_phi = len(Swind)
    n_z = len(Swind)
    n_b = B + 1
    n_m = tau + 1


    V = np.zeros((n_phi, n_z, n_b, n_m))
    policy = -np.ones((n_phi, n_z, n_b, n_m), dtype=int)

    pi_iteration = 0
    policy_stable = False
    while not policy_stable:
        pi_iteration += 1
        print(f"\nPolicy Iteration {pi_iteration} Start")
        #Policy Evaluation
        V = policy_evaluation(V, policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
        policy_stable = True

        #Policy Improvement
        changes = 0  # count the number of state policy changes
        for i_phi in range(n_phi):
            for i_z in range(n_z):
                for i_b in range(n_b):
                    for i_m in range(n_m):
                        old_action = policy[i_phi, i_z, i_b, i_m]
                        feasible_actions = []
                        if i_m == 0:
                            feasible_actions = [-1]
                        else:
                            if i_b < eta:
                                feasible_actions = [-1]
                            else:
                                feasible_actions = [-1] + list(range(n_phi))
                        Q_vals = []
                        for a in feasible_actions:
                            Q_val = compute_Q(i_phi, i_z, i_b, i_m, a, V, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
                            Q_vals.append(Q_val)
                        best_action = feasible_actions[np.argmax(Q_vals)]
                        if best_action != old_action:
                            changes += 1
                            policy_stable = False
                            policy[i_phi, i_z, i_b, i_m] = best_action
        print(f"Policy Iteration {pi_iteration} completed, number of policy changes: {changes}")
    return V, policy



# System parameters (set to default values)
Swind = np.linspace(0, 1, 21)                      # The set of all possible normalized wind speed.
mu_wind = 0.3                                      # Mean wind speed. You can vary this between 0.2 to 0.8.
z_wind = 0.5                                       # Z-factor of the wind speed. You can vary this between 0.25 to 0.75.
                                                   # Z-factor = Standard deviation divided by mean.
                                                   # Higher the Z-factor, the more is the fluctuation in wind speed.
stddev_wind = z_wind*np.sqrt(mu_wind*(1-mu_wind))  # Standard deviation of the wind speed.
retention_prob = 0.9                               # Retention probability is the probability that the wind speed in the current and the next time slot is the same.
                                                   # You can vary the retention probability between 0.05 to 0.95.
                                                   # Higher retention probability implies lower fluctuation in wind speed.
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)  # Markovian probability matrix governing wind speed.

lmbda = 0.7  # Probability of successful transmission.

B = 10         # Maximum battery capacity.
eta = 2        # Battery power required for one transmission.
Delta = 3      # Maximum solar power in one time slot.
mu_delta = 2   # Mean of the solar power in one time slot.
z_delta = 0.5  # Z-factor of the slower power in one time slot. You can vary this between 0.25 to 0.75.                  
stddev_delta = z_delta*np.sqrt(Delta*(Delta-mu_delta))  # Standard deviation of the solar power in one time slot.
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.

tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.

beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.


# Call policy iteration function.
V_optimal_pi, policy_optimal_pi = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)