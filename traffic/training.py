import numpy as np
import gymnasium as gym
from GymTraffic import GymTrafficEnv

def SARSA(env, beta, Nepisodes, alpha):
    Qmax = 20
    Q = np.zeros((Qmax+1, Qmax+1, 2, 11, 2))
    nu = 1e-3
    start_epsilon, end_epsilon, decay = 1.0, 0.05, 0.995

    for ep in range(Nepisodes):
        epsilon = max(end_epsilon, start_epsilon * (decay**ep))
        state, _ = env.reset()
        truncated = False

        # clip initial state
        r1, r2, g, c = state
        r1, r2 = min(r1, Qmax), min(r2, Qmax)

        # initial action
        if c < env.max_red:
            a = 0
        else:
            if np.random.rand() < epsilon:
                qv = Q[r1, r2, g, c, :]
                denom = nu + np.sum(np.abs(qv))
                w = np.exp(qv/denom); p = w/w.sum()
                a = np.random.choice([0,1], p=p)
            else:
                a = int(np.argmax(Q[r1, r2, g, c, :]))

        while not truncated:
            (r1p, r2p, gp, cp), reward, _, truncated, _ = env.step(a)
            r1p, r2p = min(r1p, Qmax), min(r2p, Qmax)

            # next action
            if cp < env.max_red:
                a_p = 0
            else:
                if np.random.rand() < epsilon:
                    qv_p = Q[r1p, r2p, gp, cp, :]
                    denom = nu + np.sum(np.abs(qv_p))
                    w_p = np.exp(qv_p/denom); p_p = w_p/w_p.sum()
                    a_p = np.random.choice([0,1], p=p_p)
                else:
                    a_p = int(np.argmax(Q[r1p, r2p, gp, cp, :]))

            # SARSA update
            Q[r1, r2, g, c, a] += alpha * (
                reward + beta * Q[r1p, r2p, gp, cp, a_p]
                - Q[r1, r2, g, c, a]
            )

            r1, r2, g, c, a = r1p, r2p, gp, cp, a_p

    # extract policy
    policy = np.zeros((Qmax+1, Qmax+1, 2, 11), dtype=int)
    for i1 in range(Qmax+1):
        for i2 in range(Qmax+1):
            for g in (0,1):
                for c in range(11):
                    if c < env.max_red:
                        policy[i1,i2,g,c] = 0
                    else:
                        policy[i1,i2,g,c] = int(np.argmax(Q[i1,i2,g,c,:]))
    return policy

def ExpectedSARSA(env, beta, Nepisodes, alpha):
    Qmax = 20
    Q = np.zeros((Qmax+1, Qmax+1, 2, 11, 2))
    nu = 1e-3
    start_epsilon, end_epsilon, decay = 1.0, 0.05, 0.995

    for ep in range(Nepisodes):
        epsilon = max(end_epsilon, start_epsilon * (decay**ep))
        state, _ = env.reset()
        truncated = False

        r1, r2, g, c = state
        r1, r2 = min(r1, Qmax), min(r2, Qmax)

        while not truncated:
            # choose action a
            if c < env.max_red:
                a = 0
            else:
                if np.random.rand() < epsilon:
                    qv = Q[r1, r2, g, c, :]
                    denom = nu + np.sum(np.abs(qv))
                    w = np.exp(qv/denom); p = w/w.sum()
                    a = np.random.choice([0,1], p=p)
                else:
                    a = int(np.argmax(Q[r1, r2, g, c, :]))

            (r1p, r2p, gp, cp), reward, _, truncated, _ = env.step(a)
            r1p, r2p = min(r1p, Qmax), min(r2p, Qmax)

            # compute expected Q at next state
            qv_p = Q[r1p, r2p, gp, cp, :]
            denom = nu + np.sum(np.abs(qv_p))
            w_p = np.exp(qv_p/denom); p_p = w_p/w_p.sum()
            expected_q = np.dot(p_p, qv_p)

            # Expected SARSA update
            Q[r1, r2, g, c, a] += alpha * (
                reward + beta * expected_q
                - Q[r1, r2, g, c, a]
            )

            r1, r2, g, c = r1p, r2p, gp, cp

    # extract policy
    policy = np.zeros((Qmax+1, Qmax+1, 2, 11), dtype=int)
    for i1 in range(Qmax+1):
        for i2 in range(Qmax+1):
            for g in (0,1):
                for c in range(11):
                    if c < env.max_red:
                        policy[i1,i2,g,c] = 0
                    else:
                        policy[i1,i2,g,c] = int(np.argmax(Q[i1,i2,g,c,:]))
    return policy

def ValueFunctionSARSA(env, beta, Nepisodes, alpha):
    V = np.zeros((21, 21, 2, 11))
    nu = 1e-3

    for ep in range(Nepisodes):
        if ep % 100 == 0:
            print(f"Episode {ep}/{Nepisodes}")
        epsilon = max(0.1, 1.0 - ep / Nepisodes)

        state, _ = env.reset()
        truncated = False

        while not truncated:
            r1, r2, g, c = state
            i1, i2 = min(r1,20), min(r2,20)

            # one-step lookahead
            Q = np.zeros(2)
            for a in (0,1):
                if a==1 and c<env.max_red:
                    g_new, c_new = g, min(c+1,env.max_red)
                elif a==1:
                    g_new, c_new = 1-g, 0
                else:
                    g_new, c_new = g, min(c+1,env.max_red)

                r_sa = -(r1+r2)
                backup = 0.0
                for a1 in (0,1):
                    p_a1 = env.alpha[0] if a1 else 1-env.alpha[0]
                    for a2 in (0,1):
                        p_a2 = env.alpha[1] if a2 else 1-env.alpha[1]
                        if g_new==0:
                            p_d1 = env.green_depart
                            p_d2 = env.green_depart*(1-c_new**2/100)
                        else:
                            p_d2 = env.green_depart
                            p_d1 = env.green_depart*(1-c_new**2/100)
                        for d1 in (0,1):
                            p_d1_eff = p_d1 if d1 else 1-p_d1
                            for d2 in (0,1):
                                p_d2_eff = p_d2 if d2 else 1-p_d2
                                p = p_a1*p_a2*p_d1_eff*p_d2_eff
                                r1p = np.clip(r1-d1+a1,0,env.max_queue)
                                r2p = np.clip(r2-d2+a2,0,env.max_queue)
                                backup += p * V[min(r1p,20),min(r2p,20),g_new,c_new]
                Q[a] = r_sa + beta*backup

            # action selection
            if c < env.max_red:
                action = 0
            else:
                if np.random.rand() < epsilon:
                    denom = nu + np.sum(np.abs(Q))
                    w = np.exp(Q/denom); p = w/w.sum()
                    action = np.random.choice([0,1], p=p)
                else:
                    action = int(np.argmax(Q))

            (r1p, r2p, gp, cp), reward, _, truncated, _ = env.step(action)
            i1p, i2p = min(r1p,20), min(r2p,20)
            delta = reward + beta*V[i1p,i2p,gp,cp] - V[i1,i2,g,c]
            V[i1,i2,g,c] += alpha*delta
            state = (r1p, r2p, gp, cp)

    policy = np.zeros((21,21,2,11), dtype=int)
    for i1 in range(21):
        for i2 in range(21):
            for g in (0,1):
                for c in range(11):
                    Q = np.zeros(2)
                    for a in (0,1):
                        if a==1 and c<env.max_red:
                            g_new,c_new = g, min(c+1,env.max_red)
                        elif a==1:
                            g_new,c_new = 1-g,0
                        else:
                            g_new,c_new = g, min(c+1,env.max_red)

                        r_sa = -(i1+i2)
                        backup = 0.0
                        for a1 in (0,1):
                            p_a1 = env.alpha[0] if a1 else 1-env.alpha[0]
                            for a2 in (0,1):
                                p_a2 = env.alpha[1] if a2 else 1-env.alpha[1]
                                if g_new==0:
                                    p_d1 = env.green_depart
                                    p_d2 = env.green_depart*(1-c_new**2/100)
                                else:
                                    p_d2 = env.green_depart
                                    p_d1 = env.green_depart*(1-c_new**2/100)
                                for d1 in (0,1):
                                    p_d1_eff = p_d1 if d1 else 1-p_d1
                                    for d2 in (0,1):
                                        p_d2_eff = p_d2 if d2 else 1-p_d2
                                        p = p_a1*p_a2*p_d1_eff*p_d2_eff
                                        r1pp = min(max(i1-d1+a1,0),20)
                                        r2pp = min(max(i2-d2+a2,0),20)
                                        backup += p * V[r1pp,r2pp,g_new,c_new]
                    if c<env.max_red:
                        policy[i1,i2,g,c] = 0
                    else:
                        policy[i1,i2,g,c] = int(np.argmax(Q))
    return policy

# main training
env = GymTrafficEnv()
Nepisodes, alpha, beta = 2000, 0.1, 0.997

policy1 = SARSA(env, beta, Nepisodes, alpha)
policy2 = ExpectedSARSA(env, beta, Nepisodes, alpha)
policy3 = ValueFunctionSARSA(env, beta, Nepisodes, alpha)

np.save('policy1.npy', policy1)
np.save('policy2.npy', policy2)
np.save('policy3.npy', policy3)

env.close()