#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def gaussin_probability(value):
    '''
    take in value and return random value based on gaussian distribution
    with mean as the initial value and the std as value/4
    '''
    #np.random.seed(np.random.choice(range(1000), size=1))
    #return np.random.normal(loc=value, scale=(value*7), size=None)
    return np.random.uniform(low=0, high=(value/2 + value), size=None)

def model_212(initial_states, hosts_parameters, vector_parameters, natural_parameters):
    '''single iteration of all states in 212 model'''

    # hosts_parameters_new = [gaussin_probability(x) for x in hosts_parameters]
    # vector_parameters = [gaussin_probability(x) for x in vector_parameters]

    '''stating variables'''
    Na, Nb, V, Ya1, Ya2, Ya12, Yb1, Yb2, Yb12, X1, X2, X12 = initial_states
    A1, A2, A12, A21, r1, r2, r12, r21 = hosts_parameters
    Ahat1, Ahat2, Ahat12, Ahat21, g1, g2, g12, g21, u1, u2, u12, u21 = vector_parameters
    Ba, ba, Bb, bb, Bv, bv, Ka, Kb, M = natural_parameters
    print(Na)
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
          - (Ahat12 * ((Ya2 + Ya12) / Na) * X1) \
          - (Ahat12 * ((Yb2 + Yb12) / Nb) * X1) \
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

def model_212_odeint(y, t, hosts_parameters, vector_parameters, natural_parameters):
    '''single iteration of all states in 212 model'''

    # hosts_parameters_new = [gaussin_probability(x) for x in hosts_parameters]
    # vector_parameters = [gaussin_probability(x) for x in vector_parameters]

    '''stating variables'''
    Na, Nb, V, Ya1, Ya2, Ya12, Yb1, Yb2, Yb12, X1, X2, X12 = y
    A1a, A2a, A12a, A21a, A1b, A2b, A12b, A21b, r1a, r2a, r12a, r21a, r1b, r2b, r12b, r21b = hosts_parameters
    Ahat1, Ahat2, Ahat12, Ahat21, g1, g2, g12, g21, u1, u2, u12, u21 = vector_parameters
    Ba, ba, Bb, bb, Bv, bv, Ka, Kb, M = natural_parameters

    '''Change in Hosts (A and B) and Vector (Tick) population'''
    dNa = Ba * ((Ka-Na)/Ka)*Na - ba*Na

    dNb = Bb * ((Kb-Nb)/Kb)*Nb - bb*Nb
    dV = Bv*V * (((M*Na-V)/(M*Na)) + ((M*Nb-V)/(M*Nb))) - bv*V
    '''Change in Host A or Host B populations sinfected with pathogen 1'''
    dYa1 = (A1a * ((Na - Ya1 - Ya2 - Ya12) / Na) * (X1 + X12)) + (r12a * Ya12) - (A12a * (Ya1 / Na) * (X2 + X12)) - (
                Ba * (Na * Ya1 / Ka) + (ba + r1a) * Ya1)

    dYb1 = (A1b * ((Nb - Yb1 - Yb2 - Yb12) / Nb) * (X1 + X12)) + (r12b * Yb12) - (A12b * (Yb1 / Nb) * (X2 + X12)) - (
                Bb * (Nb * Yb1 / Kb) + (bb + r1b) * Yb1)

    '''Change in Host A or Host B populations infected with pathogen 2'''
    dYa2 = (A2a * ((Na - Ya1 - Ya2 - Ya12) / Na) * (X2 + X12)) + (r2a * Ya12) - (A21a * (Ya2 / Na) * (X1 + X12)) - (
            Ba * (Na * Ya2 / Ka) + (ba + r2a) * Ya2)
    dYb2 = (A2b * ((Nb - Yb1 - Yb2 - Yb12) / Nb) * (X2 + X12)) + (r2b * Yb12) - (A21b * (Yb2 / Nb) * (X1 + X12)) - (
            Bb * (Nb * Yb2 / Kb) + (bb + r2b) * Yb2)

    '''Change in Host A or Host B coinfected'''
    dYa12 = (A12a * (Ya1 / Na) * (X2 + X12)) + (A21a * (Ya2 / Na) * (X1 + X12)) - (
                Ba * (Na * Ya12 / Ka) - (ba + r12a + r21a) * Ya12)
    dYb12 = (A12b * (Yb1 / Nb) * (X2 + X12)) + (A21b * (Yb2 / Nb) * (X1 + X12)) - (
                Bb * (Nb * Yb12 / Kb) - (bb + r12b + r21b) * Yb12)

    '''Change in Vector population (Ticks) infected with pathogen 1'''
    dX1 = (Ahat1 * (V - X1 - X2 - X12) * ((Ya1 + Ya12) / Na + (Yb1 + Yb12) / Nb)) \
          + (Bv * (g1 * X1 + g12 * X12)) \
          + (u1 * (V - X1 - X2 - X12) * (X1 + X12) / V) \
          - (Ahat12 * ((Ya2 + Ya12) / Na) * X1) \
          - (Ahat12 * ((Yb2 + Yb12) / Nb) * X1) \
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


    data = [dNa, dNb, dV, dYa1, dYa2, dYa12, dYb1, dYb2, dYb12, dX1, dX2, dX12]
    #output = [0 if x < 0 else x for x in data]
    return data

def main():

    end_time = 30
    times = np.linspace(start=0, stop=end_time, num=end_time*10, endpoint=False, retstep=False)
    #################################################################################
    population_starting_sizes = [20, 20, 4000]
    '''Na, Nb, V'''
    infected_hosts = [0, 0, 0, 0, 0, 0]
    '''Ya1, Ya2, Ya12, Yb1, Yb2, Yb12'''
    infected_ticks = [200, 200, 0]
    '''X1, X2, X12'''
    #################################################################################
    #tick_to_hots_transmission_rate = [0.1, 0.02, 0.01, 0.01]
    tick_to_hots_transmission_rate = [0.1, 0.1, 0.005, 0.25, 0.05, 0.05, 0.025, 0.025] #p1 more infectios to host1, p2
    #tick_to_hots_transmission_rate = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05]
    '''A1a, A2a, A12a, A21a, A1b, A2b, A12b, A21b'''
    host_recovery_rate = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
    '''r1a, r2a, r12a, r21a, r1b, r2b, r12b, r21b'''
    #################################################################################
    host_to_tick_transmission_rate = [0.07, 0.07, 0.035, 0.035]
    '''Ahat1, Ahat2, Ahat12, Ahat21'''
    tick_transovarial_transstadial_transmission = [0.4, 0.4, 0.2, 0.2]
    '''g1, g2, g12, g21'''
    tick_cofeeding_transmission_rate = [0.01, 0.01, 0.005, 0.005]
    '''u1, u2, u12, u21'''
    #################################################################################
    hosts_growth_mortality_rates = [0.2, 0.0, 0.2, 0.0]
    '''Ba, ba, Bb, bb'''
    tick_growth_mortality_rates = [0.75, 0.001]
    '''Bv, bv'''
    density_dependent_constants = [20, 20, 200]
    '''Ka, Kb, M'''
    #################################################################################
    initial_states = population_starting_sizes + infected_hosts + infected_ticks
    hosts_parameters = tick_to_hots_transmission_rate + host_recovery_rate
    vector_parameters = host_to_tick_transmission_rate + tick_transovarial_transstadial_transmission + tick_cofeeding_transmission_rate
    natural_parameters = hosts_growth_mortality_rates + tick_growth_mortality_rates + density_dependent_constants

    ### ODE Model ###
    SIR_solution = odeint(model_212_odeint, t=times, y0=initial_states, args=(hosts_parameters, vector_parameters, natural_parameters))

    ### calculate susceptible hosts and vector populations ###
    Ia = np.sum(SIR_solution.T[3:6], axis=0)
    Ib = np.sum(SIR_solution.T[6:9], axis=0)
    Iv = np.sum(SIR_solution.T[9:], axis=0)

    Sa = SIR_solution.T[0] - Ia
    Sb = SIR_solution.T[1] - Ib
    Sv = np.array(SIR_solution.T[2] - Iv, dtype=object)


    ### plots ###
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("212 Model")
    ax1.plot(times, Sa, label="Ya0", linestyle="-")
    ax1.plot(times, SIR_solution.T[3], label="Ya1", linestyle="-")
    ax1.plot(times, SIR_solution.T[4], label="Ya2", linestyle="-")
    ax1.plot(times, SIR_solution.T[5], label="Ya12", linestyle="-")
    ax1.set_title("Host A population dynamics", size=8)
    ax1.set_ylabel("Population")
    ax1.legend()

    ax2.plot(times, Sb, label="Yb0", linestyle="--")
    ax2.plot(times, SIR_solution.T[6], label="Yb1", linestyle="--")
    ax2.plot(times, SIR_solution.T[7], label="Yb2", linestyle="--")
    ax2.plot(times, SIR_solution.T[8], label="Yb12", linestyle="--")
    ax2.set_xlabel("Months")
    ax2.set_title("Host B population dynamics", size=8)

    ax2.legend()

    ax3.plot(times, Sv, label="X0", linestyle="-.")
    ax3.plot(times, SIR_solution.T[9], label="X1", linestyle="-.")
    ax3.plot(times, SIR_solution.T[10], label="X2", linestyle="-.")
    ax3.plot(times, SIR_solution.T[11], label="X12", linestyle="-.")
    ax3.set_title("Vector population dynamics", size=8)
    ax3.legend()
    fig.tight_layout()
    plt.savefig("212_model_plot.png", dpi=1200)
    plt.show()



if __name__ == "__main__":
    main()