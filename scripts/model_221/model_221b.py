#!/usr/bin/env python3

import sys
import os
import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt


def gaussin_probability(value):
    '''
    take in value and return random value based on gaussian distribution
    with mean as the initial value and the std as value/4
    '''
    #np.random.seed(np.random.choice(range(1000), size=1))
    return np.random.normal(loc=value, scale=(value*4), size=None)

def model_212(initial_states, hosts_parameters, vector_parameters, natural_parameters):
    '''single iteration of all states in 212 model'''

    hosts_parameters = [gaussin_probability(x) for x in hosts_parameters]
    vector_parameters = [gaussin_probability(x) for x in vector_parameters]

    '''stating variables'''
    N, V, W, Y1, Y2, Y12, X1, X2, X12, Z1, Z2, Z12 = initial_states
    A1, A2, A12, A21, r1, r2, r12, r21 = hosts_parameters
    Ahat1, Ahat2, Ahat12, Ahat21, gv1, gv2, gv12, gv21, gw1, gw2, gw12, gw21, uv1, uv2, uv12, uv21, uw1, uw2, uw12, uw21 = vector_parameters
    B, b, Bv, bv, Bw, bw, K, Mv, Mw = natural_parameters

    '''Change in Hosts and Vector (Ticks V and W) population'''
    dN = B * ((K-N)/K)*N - b*N
    dV = Bv*V * (((Mv*N-V)/(Mv*N))) - bv*V
    dW = Bw*W * (((Mw*N-W)/(Mw*N))) - bw*V

    '''Change in Host populations infected with pathogen 1'''
    dY1 = (0
    + (A1 * ((N - Y1 - Y2 - Y12) / N) * (X1 + X12)) # growth from X infection
    + (A1 * ((N - Y1 - Y2 - Y12) / N) * (Z1 + Z12)) # growth from W infection
    + (r12 * Y12) # growth from recovery from coinfection
    - (A12 * (Y1 / N) * (X2 + X12)) # dec V from coinfection
    - (A12 * (Y1 / N) * (Z2 + Z12)) # dec V from coinfection
    - B * ((N*Y1)/(K)) # dec from growth
    - (b + r1) * Y1 # dec from recovery
    )

    '''Change in Host populations infected with pathogen 2'''
    dY2 = (0
    + (A2 * ((N - Y1 - Y2 - Y12) / N) * (X2 + X12)) # growth from X infection
    + (A2 * ((N - Y1 - Y2 - Y12) / N) * (Z2 + Z12)) # growth from W infection
    + (r21 * Y12) # growth from recovery from coinfection
    - (A21 * (Y1 / N) * (X1 + X12)) # dec V from coinfection
    - (A21 * (Y1 / N) * (Z1 + Z12)) # dec V from coinfection
    - B * ((N*Y2)/(K)) # dec from growth
    - (b + r2) * Y2 # dec from recovery
    )

    '''Change in Host coinfected'''
    dY12 = (0 
    + (A12 * (Y1 / N) * (X2 + X12)) # increase in coinfection from 1 via X
    + (A21 * (Y2 / N) * (X1 + X12)) # increase in coinfection from 2 via X
    + (A12 * (Y1 / N) * (Z2 + Z12)) # increase in coinfection from 1 via W
    + (A21 * (Y2 / N) * (Z1 + Z12)) # increase in coinfection from 2 via W
    - (B * (N * Y12 / K) # dec due to population factors
    - (b + r12 + r21) * Y12) # dec due to host death and recovery
    )

    '''Change in Vector population X (Ticks) infected with pathogen 1'''
    dX1 = (0
    + (Ahat1 * ((Y1 + Y12)/(N)) * (V - X1 - X2 - X12)) # increase via contraction from host
    + (Bv * (gv1 * X1 + gv12 * X12)) # increase via transovarial/transstadial contraction
    + (uv1 * (V - X1 - X2 - X12) * (X1 + X12) / V) # cofeeding from V ticks
    + (uv1 * (V - X1 - X2 - X12) * (Z1 + Z12) / W) # cofeeding from W ticks
    - (Ahat12 * ((Y2 + Y12) / N) * X1) # host to tick coinfection
    - (uv12 * (X2 + X12) * X1 / V) # decrement due to coinfection from cofeeding with infected V ticks
    - (uw12 * (Z2 + Z12) * X1 / W) # decrement due to coinfection from cofeeding with infected W ticks 
    - ((Bv * V * X1) / (Mv * N)) # decrement due to carrying capacity
    - (bv * X1) # decrement due to death
    )

    '''Change in Vector population X (Ticks) infected with pathogen 2'''
    dX2 = (0
    + (Ahat2 * ((Y2 + Y12)/(N)) * (V - X1 - X2 - X12)) # increase via contraction from host
    + (Bv * (gv2 * X2 + gv21 * X12)) # increase via transovarial/transstadial contraction
    + (uv2 * (V - X1 - X2 - X12) * (X2 + X12) / V) # cofeeding from V ticks
    + (uv2 * (V - X1 - X2 - X12) * (Z2 + Z12) / W) # cofeeding from W ticks
    - (Ahat21 * ((Y1 + Y12) / N) * X2) # host to tick coinfection
    - (uv21 * (X1 + X12) * X2 / V) # decrement due to coinfection from cofeeding with infected V ticks
    - (uw21 * (Z1 + Z12) * X2 / W) # decrement due to coinfection from cofeeding with infected W ticks 
    - ((Bv * V * X2) / (Mv * N)) # decrement due to carrying capacity
    - (bv * X2) # decrement due to death
    )

    '''Change in Vector population X (Ticks) coinfected'''
    dX12 = (Ahat12 * ((Y2 + Y12) / N) * X1) \
           + (Ahat21 * ((Y1 + Y12) / N) * X2) \
           + (uv12 * ((X2 + X12) * X1) / V) \
           + (uv21 * ((X1 + X12) * X2) / V) \
           + (uw12 * ((Z2 + Z12) * X1) / W) \
           + (uw21 * ((Z1 + Z12) * X2) / W) \
           - ((Bv * V * X12) / (Mv * N)) \
           - (bv * X12)

    '''Change in Vector population W (Ticks) infected with pathogen 1'''
    dZ1 = (0
    + (Ahat1 * ((Y1 + Y12)/(N)) * (W - Z1 - Z2 - Z12)) # increase via contraction from host
    + (Bw * (gw1 * Z1 + gw12 * Z12)) # increase via transovarial/transstadial contraction
    + (uw1 * (W - Z1 - Z2 - Z12) * (X1 + X12) / V) # cofeeding from V ticks
    + (uw1 * (W - Z1 - Z2 - Z12) * (Z1 + Z12) / W) # cofeeding from W ticks
    - (Ahat12 * ((Y2 + Y12) / N) * Z1) # host to tick coinfection
    - (uv12 * (X2 + X12) * Z1 / V) # decrement due to coinfection from cofeeding with infected V ticks
    - (uw12 * (Z2 + Z12) * Z1 / W) # decrement due to coinfection from cofeeding with infected W ticks 
    - ((Bw * W * Z1) / (Mw * N)) # decrement due to carrying capacity
    - (bv * Z1) # decrement due to death
    )

    '''Change in Vector population W (Ticks) infected with pathogen 2'''
    dZ2 = (0
    + (Ahat2 * ((Y2 + Y12)/(N)) * (W - Z1 - Z2 - Z12)) # increase via contraction from host
    + (Bw * (gw2 * Z2 + gw21 * Z12)) # increase via transovarial/transstadial contraction
    + (uw2 * (W - Z1 - Z2 - Z12) * (X2 + X12) / V) # cofeeding from V ticks
    + (uw2 * (W - Z1 - Z2 - Z12) * (Z2 + Z12) / W) # cofeeding from W ticks
    - (Ahat21 * ((Y1 + Y12) / N) * Z2) # host to tick coinfection
    - (uv21 * (X1 + X12) * Z2 / V) # decrement due to coinfection from cofeeding with infected V ticks
    - (uw21 * (Z1 + Z12) * Z2 / W) # decrement due to coinfection from cofeeding with infected W ticks 
    - ((Bw * W * Z2) / (Mw * N)) # decrement due to carrying capacity
    - (bw * Z2) # decrement due to death
    )

    '''Change in Vector population W (Ticks) coinfected'''
    dZ12 = ( 0 
    + (Ahat12 * ((Y2 + Y12) / N) * Z1) # host to tick 1 -> 12 coinfection
    + (Ahat21 * ((Y1 + Y12) / N) * Z2) # host to tick 2 -> 12 coinfection
    + (uw12 * ((X2 + X12) * Z1) / V) # cofeeding with V, 1 to 12 coinfection
    + (uv21 * ((X1 + X12) * Z2) / V) # cofeeding with V, 2 to 12 coinfection
    + (uw12 * ((Z2 + Z12) * Z1) / W) # cofeeding with W, 1 to 12 coinfection
    + (uw21 * ((Z1 + Z12) * Z2) / W) # cofeeding with W, 2 to 12 coinfection
    - ((Bw * W * Z12) / (Mw * N)) # decrement due to carrying capacity 
    - (bw * Z12)) # decrement due to death

    return [dN, dV, dW, dY1, dY2, dY12, dX1, dX2, dX12, dZ1, dZ2, dZ12]

def main():

    times = np.linspace(start=0, stop=10000, num=10000, endpoint=True, retstep=False)

    initial_states = [100, 4000, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    '''N, V, W, Y1, Y2, Y12, X1, X2, X12, Z1, Z2, Z12'''
    hosts_parameters = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
    '''A1, A2, A12, A21, r1, r2, r12, r21'''
    vector_parameters = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05]
    '''Ahat1, Ahat2, Ahat12, Ahat21, g1, g2, g12, g21, u1, u2, u12, u21'''
    natural_parameters = [0.01, 0.0001, 0.01, 0.0001, 0.01, 0.01, 10, 10, 20]
    '''Ba, ba, Bb, bb, Bv, bv, Ka, Kb, M'''

    SIR_solution = solve_ivp(lambda t, y: model_212(initial_states,
                                                    hosts_parameters,
                                                    vector_parameters,
                                                    natural_parameters),
                             t_span=[min(times), max(times)],
                             y0=initial_states,
                             t_eval=times)

    # plt.plot(SIR_solution.t, SIR_solution.y[0], label="Na", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[1], label="Nb", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[2], label="V", linestyle="-")
    plt.plot(SIR_solution.t[:2000], SIR_solution.y[3][:2000], label="Ya1", linestyle="-", alpha=0.5)
    plt.plot(SIR_solution.t[:2000], SIR_solution.y[4][:2000], label="Ya2", linestyle="-", alpha=0.5)
    # plt.plot(SIR_solution.t, SIR_solution.y[5], label="Ya12", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[6], label="Yb1", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[7], label="Yb2", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[8], label="Yb12", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[9], label="X1", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[9], label="X2", linestyle="-")
    # plt.plot(SIR_solution.t, SIR_solution.y[9], label="X12", linestyle="-")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()