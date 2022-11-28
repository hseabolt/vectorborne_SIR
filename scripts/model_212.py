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
    Na, Nb, V, Ya1, Ya2, Ya12, Yb1, Yb2, Yb12, X1, X2, X12 = initial_states
    A1, A2, A12, A21, r1, r2, r12, r21 = hosts_parameters
    Ahat1, Ahat2, Ahat12, Ahat21, g1, g2, g12, g21, u1, u2, u12, u21 = vector_parameters
    Ba, ba, Bb, bb, Bv, bv, Ka, Kb, M = natural_parameters

    '''Change in Hosts (A and B) and Vector (Tick) population'''
    dNa = Ba * ((Ka-Na)/Ka)*Na - ba*Na
    dNb = Bb * ((Kb-Nb)/Kb)*Nb - bb*Nb
    dV = Bv*V * (((M*Na-V)/(M*Na)) + ((M*Nb-V)/(M*Nb))) - bv*V

    '''Change in Host A or Host B populations sinfected with pathogen 1'''
    dYa1 = (A1 * ((Na - Ya1 - Ya2 - Ya12) / Na) * (X1 + X12)) + (r12 * Ya12) - (A12 * (Ya1 / Na) * (X2 + X12)) - (
                Ba * (Na * Ya1 / Ka) + (ba + r1) * Ya1)
    dYb1 = (A1 * ((Nb - Yb1 - Yb2 - Yb12) / Nb) * (X1 + X12)) + (r12 * Yb12) - (A12 * (Yb1 / Nb) * (X2 + X12)) - (
                Bb * (Nb * Yb1 / Kb) + (bb + r1) * Yb1)

    '''Change in Host A or Host B populations infected with pathogen 2'''
    dYa2 = (A2 * ((Na - Ya1 - Ya2 - Ya12) / Na) * (X2 + X12)) + (r21 * Ya12) - (A21 * (Ya2 / Na) * (X1 + X12)) - (
            Ba * (Na * Ya2 / Ka) + (ba + r2) * Ya2)
    dYb2 = (A2 * ((Nb - Yb1 - Yb2 - Yb12) / Nb) * (X2 + X12)) + (r21 * Yb12) - (A21 * (Yb2 / Nb) * (X1 + X12)) - (
            Bb * (Nb * Yb2 / Kb) + (bb + r2) * Yb2)

    '''Change in Host A or Host B coinfected'''
    dYa12 = (A12 * (Ya1 / Na) * (X2 + X12)) + (A21 * (Ya2 / Na) * (X1 + X12)) - (
                Ba * (Na * Ya12 / Ka) - (ba + r12 + r21) * Ya12)
    dYb12 = (A12 * (Yb1 / Nb) * (X2 + X12)) + (A21 * (Yb2 / Nb) * (X1 + X12)) - (
                Bb * (Nb * Yb12 / Kb) - (bb + r12 + r21) * Yb12)

    '''Change in Vector population (Ticks) infected with pathogen 1'''
    dX1 = (Ahat1 * (V - X1 - X2 - X12) * ((Ya1 + Ya12) / Na + (Yb1 + Yb12) / Nb)) \
          + (Bv * (g1 * X1 + g12 * X12)) \
          + (u1 * (V - X1 - X2 - X12) * (X1 + X12) / V) \
          - (Ahat12 * ((Ya1 + Ya12) / Na) * X1) \
          - (Ahat12 * ((Yb1 + Yb12) / Nb) * X1) \
          - (u12 * (X2 + X12) * X1 / V) \
          - ((Bv * V * X1) / M * (1 / Na) + (1 / Nb)) \
          - (bv * X1)

    '''Change in Vector population (Ticks) infected with pathogen 2'''
    dX2 = (Ahat2 * (V - X1 - X2 - X12) * ((Ya2 + Ya12) / Na + (Yb2 + Yb12) / Nb)) \
          + (Bv * (g2 * X2 + g21 * X12)) \
          + (u2 * (V - X1 - X2 - X12) * (X2 + X12) / V) \
          - (Ahat21 * ((Ya1 + Ya12) / Na) * X2) \
          - (Ahat21 * ((Yb1 + Yb12) / Nb) * X2) \
          - (u21 * (X1 + X12) * X2 / V) \
          - ((Bv * V * X2) / M * (1 / Na) + (1 / Nb)) \
          - (bv * X2)

    '''Change in Vector population (Ticks) coinfected'''
    dX12 = (Ahat12 * ((Ya2 + Ya12) / Na) * X1) \
           + (Ahat12 * ((Yb2 + Yb12) / Nb) * X1) \
           + (Ahat21 * ((Ya1 + Ya12) / Na) * X2) \
           + (Ahat21 * ((Yb1 + Yb12) / Nb) * X2) \
           + (u12 * ((X2 + X12) * X1) / V) \
           + (u21 * ((X1 + X12) * X2) / V) \
           - ((Bv * V * X12) / M * (1 / Na) + (1 / Nb)) \
           - (bv * X12)

    return [dNa, dNb, dV, dYa1, dYb1, dYa2, dYb2, dYa12, dYb12, dX1, dX2, dX12]

def main():

    times = np.linspace(start=0, stop=10000, num=10000, endpoint=True, retstep=False)

    initial_states = [100, 100, 4000, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    '''Na, Nb, V, Ya1, Ya2, Ya12, Yb1, Yb2, Yb12, X1, X2, X12'''
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