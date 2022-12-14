#!/usr/bin/env python3
'''

:.:.:.:.: 3-1-1 model (3-pathogen/1-vector/1-host compartment model) :.:.:.:.:

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def three_one_one_model_basic(
        y,
        t,
        transmission_parameters_host,
        recovery_parameters_host,
        transmission_parameters_vector,
        transovarial_transstadial_patameters_vector,
        cofeeding_trasmission_vector,
        natural_parameters,
):
    '''function will take in several lists with values for host/tick susceptible/infected/coninfected population
      plus constant parameters like transmission/recovery/Transovarial/transstadial rates'''

    Y_1, Y_2, Y_3, Y_12, Y_13, Y_23, Y_123, X_1, X_2, X_3, X_12, X_13, X_23, X_123, N, V = y
    A_1, A_2, A_3, A_12, A_21, A_13, A_31, A_23, A_32, A_123, A_132, A_231 = transmission_parameters_host
    r_1, r_2, r_3, r_12, r_21, r_13, r_31, r_23, r_32, r_123, r_132, r_231 = recovery_parameters_host
    Ahat_1, Ahat_2, Ahat_3, Ahat_12, Ahat_21, Ahat_13, Ahat_31, Ahat_23, Ahat_32, Ahat_123, Ahat_132, Ahat_231 = transmission_parameters_vector
    g_1, g_2, g_3, g_12, g_21, g_13, g_31, g_23, g_32, g_123, g_213, g_312 = transovarial_transstadial_patameters_vector
    u_1, u_2, u_3, u_12, u_21, u_13, u_31, u_23, u_32, u_123, u_132, u_231 = cofeeding_trasmission_vector
    Bh, bh, Bv, bv, K, M = natural_parameters

    ## Vector to Host equations ##
    # summation of tick populations infected with pathogen
    X_1_sum = sum([X_1, X_12, X_13, X_123])
    X_2_sum = sum([X_2, X_12, X_23, X_123])
    X_3_sum = sum([X_3, X_13, X_23, X_123])

    # susceptible host population total
    Sh = (N - sum([Y_1, Y_2, Y_3, Y_12, Y_13, Y_23, Y_123])) / N

    # single pathogen transmission
    dY1 = (A_1 * Sh * X_1_sum) + ((r_12 * Y_12) + (r_13 * Y_13)) - \
          ((A_12 * (Y_1 / N) * X_2_sum) + (A_13 * (Y_1 / N) * X_3_sum)) - (Bh * (N * Y_1 / K) + ((bh + r_1) * Y_1))
    dY2 = (A_2 * Sh * X_2_sum) + ((r_21 * Y_12) + (r_23 * Y_23)) - \
          ((A_21 * (Y_2 / N) * X_1_sum) + (A_23 * (Y_2 / N) * X_3_sum)) - (Bh * (N * Y_2 / K) + (bh + r_2) * Y_2)
    dY3 = (A_3 * Sh * X_2_sum) + ((r_31 * Y_13) + (r_32 * Y_23)) - \
          ((A_31 * (Y_3 / N) * X_1_sum) + (A_32 * (Y_3 / N) * X_2_sum)) - (Bh * (N * Y_3 / K) + (bh + r_3) * Y_3)

    # coinfect1on w1th two pathogens
    dY12 = (A_12 * (Y_1 / N) * X_2_sum) + (A_21 * (Y_2 / N) * X_1_sum) + (r_123 + Y_123) - \
           (A_123 * (Y_12 / N) * X_3_sum) - (Bh * (N * Y_12 / K) + (bh + r_12 + r_21) * Y_12)
    dY13 = (A_13 * (Y_1 / N) * X_3_sum) + (A_31 * (Y_3 / N) * X_1_sum) + (r_132 + Y_123) - \
           (A_132 * (Y_13 / N) * X_2_sum) - (Bh * (N * Y_13 / K) + (bh + r_13 + r_31) * Y_13)
    dY23 = (A_23 * (Y_2 / N) * X_3_sum) + (A_32 * (Y_3 / N) * X_2_sum) + (r_231 + Y_123) - \
           (A_231 * (Y_23 / N) * X_1_sum) - (Bh * (N * Y_23 / K) + (bh + r_23 + r_32) * Y_23)
    # coinfect1on with three pathogens
    dY123 = (A_123 * (Y_12 / N) * X_3_sum) + (A_132 * (Y_13 / N) * X_2_sum) + (A_231 * (Y_23 / N) * X_1_sum) - \
            (Bh * (N * Y_123 / K)) + (bh + r_123 + r_132 + r_231) * Y_123

    ## Host to Vector ##
    # summation of Host populations infected with pathogen
    Y_1_sum = sum([Y_1, Y_12, Y_13, Y_123])
    Y_2_sum = sum([Y_2, Y_12, Y_23, Y_123])
    Y_3_sum = sum([Y_3, Y_13, Y_23, Y_123])

    ### Vector Population Below ###
    # susceptible vector population total
    Vh = V - sum([X_1, X_2, X_3, X_12, X_13, X_23, X_123])

    # single pathogen transmission for ticks
    dX1 = (Ahat_1 * (Y_1_sum / N) * Vh) \
          + (Bv * (g_1 * X_1 + g_12 * X_12 + g_13 * X_13 + g_123 * X_123)) \
          + (u_1 * ((Vh * X_1_sum) / V)) \
          - ((Ahat_12 * (Y_2_sum / N) * X_1) + (Ahat_13 * (Y_3_sum / N) * X_1)) \
          - (u_12 * ((X_2_sum * X_1) / V) + u_13 * ((X_3_sum * X_1) / V)) \
          - (Bv * ((V * X_1)/(M * N)) + (bv * X_1))

    dX2 = (Ahat_2 * (Y_2_sum / N) * Vh) \
          + (Bv * ((g_2 * X_2) + (g_21 * X_12) + (g_23 * X_23) + (g_213 * X_123))) \
          + (u_2 * ((Vh * X_2_sum) / V)) \
          - ((Ahat_21 * (Y_1_sum / N) * X_2) + (Ahat_23 * (Y_3_sum / N) * X_2)) \
          - (u_21 * ((X_1_sum * X_2) / V) + u_23 * ((X_3_sum * X_2) / V)) \
          - (Bv * ((V * X_2)/(M * N)) + (bv * X_2))

    dX3 = (Ahat_3 * (Y_3_sum / N) * Vh) \
          + (Bv * ((g_3 * X_3) + (g_31 * X_13) + (g_32 * X_23) + (g_312 * X_123))) \
          + (u_3 * ((Vh * X_3_sum) / V)) \
          - ((Ahat_31 * (Y_1_sum / N) * X_3) + (Ahat_32 * (Y_2_sum / N) * X_3)) \
          - (u_31 * ((X_1_sum * X_3) / V) + u_32 * ((X_2_sum * X_3) / V)) \
          - (Bv * ((V * X_3)/(M * N)) + (bv * X_3))

    # coinfection of two pathogens for ticks

    dX12 = ((Ahat_12 * (Y_2_sum / N) * X_1) + (Ahat_21 * (Y_1_sum / N) * X_2)) \
           + ((u_12 * (X_2_sum * X_1)/V) + (u_21 * (X_1_sum * X_2)/V)) \
           - (Ahat_123 * (Y_3_sum / N) * X_12) \
           - (Bh * ((V*X_12/M*N)) + (bv * X_12))

    dX13 = ((Ahat_13 * (Y_3_sum / N) * X_1) + (Ahat_31 * (Y_1_sum / N) * X_3)) \
           + ((u_13 * (X_3_sum * X_1) / V) + (u_31 * (X_1_sum * X_3) / V)) \
           - (Ahat_132 * (Y_2_sum / N) * X_13) \
           - (Bh * ((V * X_13 / M * N)) + (bv * X_13))

    dX23 = ((Ahat_23 * (Y_3_sum / N) * X_2) + (Ahat_32 * (Y_2_sum / N) * X_3)) \
           + ((u_23 * (X_3_sum * X_2) / V) + (u_32 * (X_2_sum * X_3) / V)) \
           - (Ahat_231 * (Y_1_sum / N) * X_23) \
           - (Bh * ((V * X_23 / M * N)) + (bv * X_23))

    # coinfection of three pathogens for ticks

    dX123 = ((Ahat_123 * (Y_3_sum) * X_12) + (Ahat_132 * (Y_2_sum) * X_13) + (Ahat_231 * (Y_1_sum) * X_23)) \
            + ((u_123 * (X_3_sum * X_12)/V) + (u_132 * (X_2_sum * X_13)/V) + (u_231 * (X_1_sum * X_23)/V)) \
            - (Bv * (V*X_123/M*N) + bv*X_123)

    dN = (Bh * ((K - N)/K) * N) - bh*N
    dV = (Bv * (((M*N)-V)/(M*N)) * V) - (bv*V)

    return [dY1, dY2, dY3, dY12, dY13, dY23, dY123, dX1, dX2, dX3, dX12, dX13, dX23, dX123, dN, dV]

def uniform_random():
    return np.random.uniform(low=0.0, high=0.4)

def main():

    end_time = 30
    times = np.linspace(start=1, stop=end_time, num=end_time*10, endpoint=True, retstep=False)

    # A_1 = uniform_random()
    # A_2 = uniform_random()
    # A_3 = uniform_random()
    # Ahat_1 = uniform_random()
    # Ahat_2 = uniform_random()
    # Ahat_3 = uniform_random()

    initial_states = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 4000]
    '''Y_1, Y_2, Y_3, Y_12, Y_13, Y_23, Y_123, X_1, X_2, X_3, X_12, X_13, X_23, X_123, N, V'''
    #transmission_parameters_host = [0.5, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
    transmission_parameters_host = [0.5, 0.1, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
    '''A_1, A_2, A_3, A_12, A_21, A_13, A_31, A_23, A_32, A_123, A_132, A_231'''
    recovery_parameters_host = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
    #transmission_parameters_vector = [Ahat_1, Ahat_2, Ahat_3, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.0175, 0.0175, 0.0175]
    transmission_parameters_vector = [0.07, 0.07, 0.07, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.0175, 0.0175, 0.0175]
    '''Ahat_1, Ahat_2, Ahat_3, Ahat_12, Ahat_21, Ahat_13, Ahat_31, Ahat_23, Ahat_32, Ahat_123, Ahat_132, Ahat_231'''
    transovarial_transstadial_patameters_vector = [0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
    cofeeding_trasmission_vector = [0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.0025, 0.0025]
    natural_parameters = [0.02, 0, 0.75, 0.001, 10, 200]


    SIR_solution = odeint(three_one_one_model_basic, t=times, y0=initial_states,
                          args=(transmission_parameters_host,
                                recovery_parameters_host,
                                transmission_parameters_vector,
                                transovarial_transstadial_patameters_vector,
                                cofeeding_trasmission_vector,
                                natural_parameters
                                )
                          )


    #fig, (ax1, ax2) = plt.subplots(1,2)

    # ax1.plot(times, SIR_solution.T[14], label="Y0", linestyle="-")
    # ax1.plot(times, SIR_solution.T[0], label="Y1", linestyle="-")
    # ax1.plot(times, SIR_solution.T[1], label="Y2", linestyle="-")
    # ax1.plot(times, SIR_solution.T[2], label="Y3", linestyle="-")
    # ax1.plot(times, SIR_solution.T[3], label="Y12", linestyle="-")
    # ax1.plot(times, SIR_solution.T[4], label="Y13", linestyle="-")
    # ax1.plot(times, SIR_solution.T[5], label="Y23", linestyle="-")
    # ax1.plot(times, SIR_solution.T[6], label="Y123", linestyle="-")
    # ax1.legend()

    plt.plot(times, SIR_solution.T[15], label="X0", linestyle="-")
    plt.plot(times, SIR_solution.T[7], label="X1", linestyle="--")
    plt.plot(times, SIR_solution.T[8], label="X2", linestyle="--")
    plt.plot(times, SIR_solution.T[9], label="X3", linestyle="--")
    plt.plot(times, SIR_solution.T[10], label="X12", linestyle="-.")
    plt.plot(times, SIR_solution.T[11], label="X13", linestyle="-.")
    plt.plot(times, SIR_solution.T[12], label="X23", linestyle="-.")
    plt.plot(times, SIR_solution.T[13], label="X123", linestyle=":")
    plt.legend()
    plt.title("311 model: Vector population dynamics over time")
    plt.xlabel("Months")
    plt.ylabel("Population")

    # ax3.plot(times, SIR_solution.T[14], label="N", linestyle="-.")
    # ax3.plot(times, SIR_solution.T[15], label="V", linestyle="-.")
    # ax3.legend()
    plt.savefig("model_311_plot.png", dpi=1200)
    plt.show()


if __name__ == "__main__":
    main()
