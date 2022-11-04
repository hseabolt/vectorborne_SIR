#!/usr/bin/env python3

#!/usr/bin/env python3
'''

:.:.:.:.: 3-1-1 model (3-pathogen/1-vector/1-host compartment model) :.:.:.:.:


'''
import sys
import numpy as np
import pandas as pd
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt


def three_one_one_model_basic(
        host_states,
        vector_states,
        transmission_parameters,
        recovery_parameters,
        natural_parameters,
        N, K
):
    Y_i, Y_j, Y_k, Y_ij, Y_ik, Y_jk, Y_ijk = host_states
    X_i, X_j, X_k, X_ij, X_ik, X_jk, X_ijk = vector_states
    A_i, A_j, A_k, A_ij, A_ji, A_ik, A_ki, A_jk, A_kj, A_ijk, A_ikj, A_jki = transmission_parameters
    r_i, r_j, r_k, r_ij, r_ji, r_ik, r_ki, r_jk, r_kj, r_ijk, r_ikj, r_jki = recovery_parameters
    Bh, bh, Bv, bv = natural_parameters

    X_i_sum = sum([X_i, X_ij, X_ik, X_ijk])
    X_j_sum = sum([X_j, X_ij, X_jk, X_ijk])
    X_k_sum = sum([X_k, X_ik, X_jk, X_ijk])
    Sh = (N - sum(host_states))/N


    dYi = (A_i * Sh * X_i_sum) + (r_ij*Y_ij + r_ik*Y_ik) - \
          ((A_ij * (Y_i/N) * X_j_sum) + (A_ik * (Y_i/N) * X_k_sum)) - (Bh*(N*Y_i/K) + (bh + r_i)*Y_i)
    dYj = (A_j * Sh * X_j_sum) + (r_ji*Y_ij + r_jk*Y_jk) - \
          ((A_ji * (Y_j/N) * X_i_sum) + (A_jk * (Y_j/N) * X_k_sum)) - (Bh*(N*Y_j/K) + (bh + r_j)*Y_j)
    dYk = (A_k * Sh * X_j_sum) + (r_ki*Y_ik + r_kj*Y_jk) - \
          ((A_ki * (Y_k/N) * X_i_sum) + (A_kj * (Y_k/N) * X_j_sum)) - (Bh*(N*Y_k/K) + (bh + r_k)*Y_k)

    dYij = (A_ij * (Y_i/N) * X_j_sum) + (A_ji * (Y_j/N) * X_i_sum) + (r_ijk +Y_ijk) - \
           (A_ijk(Y_ij)*X_k_sum) - (Bh(N*Y_ij/K) + (bh + r_ij + r_ji)*Y_ij)
    dYik = (A_ik * (Y_i/N) * X_k_sum) + (A_ki * (Y_k/N) * X_i_sum) + (r_ikj + Y_ijk) - \
           (A_ikj(Y_ik) * X_j_sum) - (Bh(N * Y_ik/K) + (bh + r_ik + r_ki) * Y_ik)
    dYjk = (A_jk * (Y_j / N) * X_k_sum) + (A_kj * (Y_k / N) * X_j_sum) + (r_jki + Y_ijk) - \
           (A_jki(Y_jk) * X_i_sum) - (Bh(N * Y_jk / K) + (bh + r_jk + r_kj) * Y_jk)

    dYijk = (A_ijk * (Y_ij/N) * X_k_sum) + (A_ikj * (Y_ik/N) * X_j_sum) + (A_jki * (Y_jk/N) * X_i_sum) - \
            (Bh * (N*Y_ijk / K)) + (bh + r_ijk + r_ikj + r_jki)*Y_ijk

    return [dYi, dYj, dYk]


def main():


    #positional args
    S = float(sys.argv[1])
    beta = float(sys.argv[2])
    gamma = float(sys.argv[3])

    I = 1 - S
    initial = [S, 1-S, (1-(S+I))]  # S, I, R initial starting values
    parameters = [beta, gamma] # beta, gamma for 1.3.1 first part
    #parameters = [0.05, 0.1] # beta, gamma for 1.3.1 second part
    times = np.linspace(0, 200, 200) # times, start, stop, observations

    SIR_solution = solve_ivp(lambda t, y: SIR_model_basic(y, parameters),
                        t_span=[min(times), max(times)], y0=initial, t_eval=times)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    # document for solve_ivp function
     #returns an array with y as key and list of list as value with each element in list being proporiton at time t

    #create dataset using pandas for each time point with t, S, I, R as column names
    SIR_output = pd.DataFrame({"time": SIR_solution["t"],
                               "S": SIR_solution["y"][0],
                               "I": SIR_solution["y"][1],
                               "R": SIR_solution["y"][2]})



if __name__ == "__main__":
    main()