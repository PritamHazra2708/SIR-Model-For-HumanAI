import numpy as np

def simulate_stochastic(N, I0, beta, gamma, days):
    S = N - I0
    I = I0
    R = 0
    
    history = {"S": [S], "I": [I], "R": [R]}
    
    for _ in range(days):
        # Stochastic transitions using Binomial distribution (simple approximation)
        # Probability of infection and recovery
        p_infect = min(beta * I / N, 1.0)
        p_recover = min(gamma, 1.0)
        
        new_infections = np.random.binomial(S, p_infect) if S > 0 else 0
        new_recoveries = np.random.binomial(I, p_recover) if I > 0 else 0
        
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries
        
        history["S"].append(float(S))
        history["I"].append(float(I))
        history["R"].append(float(R))
        
    return history