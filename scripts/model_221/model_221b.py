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

def model_221(times, initial_states, concatenated_params):
    '''single iteration of all states in 221 model'''
    hosts_parameters = concatenated_params[0:8]
    vector_parameters = concatenated_params[8:28]
    natural_parameters = concatenated_params[28:]
    # print(times)
    print(initial_states)
    '''stating variables'''
    N, V, W, Y1, Y2, Y12, X1, X2, X12, Z1, Z2, Z12, prior_time = initial_states
    # N = max(N, 0)
    # V = max(V, 0)
    # W = max(W,0)
    # Y1 = max(Y1, 0)
    # Y2 = max(Y2, 0)
    # Y12 = max(Y12, 0)
    # X1 = max(X1, 0)
    # X2 = max(X2, 0)
    # X12 = max(X12, 0)
    # Z1 = max(Z1, 0)
    # Z2 = max(Z2, 0)
    # Z12 = max(Z12, 0)
    delta = times - prior_time
    norm_hosts_parameters = [i * delta for i in hosts_parameters]
    norm_vector_parameters = [i * delta for i in vector_parameters]
    A1, A2, A12, A21, r1, r2, r12, r21 = norm_hosts_parameters
    Ahat1, Ahat2, Ahat12, Ahat21, gv1, gv2, gv12, gv21, gw1, gw2, gw12, gw21, uv1, uv2, uv12, uv21, uw1, uw2, uw12, uw21 = norm_vector_parameters
    B, b, Bv, bv, Bw, bw, K, Mv, Mw = natural_parameters
    B = B * delta 
    b = b * delta
    Bv = Bv * delta 
    bv = bv * delta 
    Bw = Bw * delta 
    bw = bw * delta 
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
    + (uv12 * ((X2 + X12) * Z1) / V) # cofeeding with V, 1 to 12 coinfection
    + (uv21 * ((X1 + X12) * Z2) / V) # cofeeding with V, 2 to 12 coinfection
    + (uw12 * ((Z2 + Z12) * Z1) / W) # cofeeding with W, 1 to 12 coinfection
    + (uw21 * ((Z1 + Z12) * Z2) / W) # cofeeding with W, 2 to 12 coinfection
    - ((Bw * W * Z12) / (Mw * N)) # decrement due to carrying capacity 
    - (bw * Z12)) # decrement due to death
    # print([N, V, W, Y1, Y2, Y12, X1, X2, X12, Z1, Z2, Z12])
    # print([dN, dV, dW, dY1, dY2, dY12, dX1, dX2, dX12, dZ1, dZ2, dZ12])
    return [dN, dV, dW, dY1, dY2, dY12, dX1, dX2, dX12, dZ1, dZ2, dZ12, delta]
