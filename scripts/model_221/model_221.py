
from itertools import combinations

class PVH_Model_211:
    def __init__(self, pathogens: set, vectors: set, hosts: set, 
    # transmission constants
    vector_to_host_trx_consts: dict, 
    host_to_vector_trx_consts: dict, 
    # recovery constants
    host_recovery: dict,
    vector_recovery: dict,
    # host population dynamics constants
    host_growth_rates: dict, 
    host_mortality_rates: dict, 
    host_population_sizes: dict, 
    host_carrying_capacities: dict,
    # vector popuation dynamics constants
    vector_growth_rates: dict,
    vector_mortality_rates: dict,
    vector_population_sizes: dict,
    vector_transStadialOvarial: dict,
    vector_cofeeding: dict,
    max_vectors_per_host: dict):
        """Initialize the PVH_Model and verify that the inputs are valid"""
        self.pathogens = pathogens
        self.vectors = vectors
        self.hosts = hosts
        # trx constants
        self.vector_to_host_trx_consts = vector_to_host_trx_consts
        self.host_to_vector_trx_consts = host_to_vector_trx_consts
        # recovery constants
        self.host_recovery = host_recovery
        self.vector_recovery = vector_recovery
        # host pop dynamics
        self.host_growth_rates = host_growth_rates
        self.host_mortality_rates = host_mortality_rates
        self.host_population_sizes = host_population_sizes
        self.host_carrying_capacities = host_carrying_capacities
        # vector population dynamics
        self.vector_growth_rates = vector_growth_rates
        self.vector_mortality_rates = vector_mortality_rates
        self.vector_population_sizes = vector_population_sizes
        self.max_vectors_per_hosts = max_vectors_per_host
        self.vector_transStadialOvarial = vector_transStadialOvarial
        self.vector_cofeeding = vector_cofeeding
        # validate inputs
        self.validate()
        # generate infection states
        self.infection_states = self.generate_infection_states(self.pathogens)

    def generate_model(self):
        """
        Generate the 2-2-1 model function for computation here

        each time the ode step is run the following input vector will be provided
        [N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12]
        """
        def dNdt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            beta = self.host_growth_rates["Y"]
            K = self.host_carrying_capacities["Y"]
            b = self.host_mortality_rates["Y"]
            return  beta * ((K - N)/ (K)) * N - b*N
        def dY1dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            A1V = self.vector_to_host_trx_consts["V"]["1"]["Y"]
            A1W = self.vector_to_host_trx_consts["W"]["1"]["Y"]
            v12_1 = self.host_recovery["Y"]["12"]["1"]
            A12V = self.vector_to_host_trx_consts["V"]["2"]["Y"]
            A12W = self.vector_to_host_trx_consts["W"]["2"]["Y"]
            Y_growth = self.host_growth_rates["Y"]
            K = self.host_carrying_capacities["Y"]
            b = self.host_mortality_rates["Y"]
            v1_0 = self.host_recovery["Y"]["1"][""]
            # components
            X1_toY = A1V * ((N - Y1 - Y2 - Y12)/(N)) (X1 + X12)
            W1_toY = A1W * ((N - Y1 - Y2 - Y12)/(N)) (Z1 + Z12)
            recov_from_coinf = v12_1 * Y12 
            coinfX = - A12V * (Y1/N) * (X2 + X12)
            coinfW = - A12W * (Y1/N) * (Z2 + Z12)
            growth = - Y_growth * ((N*Y1)/(K))
            deathAndRecovery = - (b + v1_0) * Y1
            return sum([X1_toY, W1_toY, recov_from_coinf, coinfX, coinfW, 
                        growth, deathAndRecovery])
        def dY2dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            A2V = self.vector_to_host_trx_consts["V"]["2"]["Y"]
            A2W = self.vector_to_host_trx_consts["W"]["2"]["Y"]
            v12_2 = self.host_recovery["Y"]["12"]["2"]
            A12V = self.vector_to_host_trx_consts["V"]["1"]["Y"]
            A12W = self.vector_to_host_trx_consts["W"]["1"]["Y"]
            Y_growth = self.host_growth_rates["Y"]
            K = self.host_carrying_capacities["Y"]
            b = self.host_mortality_rates["Y"]
            v2_0 = self.host_recovery["Y"]["2"][""]
            # components
            X2_toY = A2V * ((N - Y1 - Y2 - Y12)/(N)) (X2 + X12)
            W2_toY = A2W * ((N - Y1 - Y2 - Y12)/(N)) (Z2 + Z12)
            recov_from_coinf = v12_2 * Y12 
            coinfX = - A12V * (Y1/N) * (X1 + X12)
            coinfW = - A12W * (Y1/N) * (Z1 + Z12)
            growth = - Y_growth * ((N*Y2)/(K))
            deathAndRecovery = - (b + v2_0) * Y2
            return sum([X2_toY, W2_toY, recov_from_coinf, coinfX, coinfW, 
                        growth, deathAndRecovery])
        def dY12dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            A1V = self.vector_to_host_trx_consts["V"]["1"]["Y"]
            A1W = self.vector_to_host_trx_consts["W"]["1"]["Y"]
            A2V = self.vector_to_host_trx_consts["V"]["2"]["Y"]
            A2W = self.vector_to_host_trx_consts["W"]["2"]["Y"]
            v12_2 = self.host_recovery["Y"]["12"]["2"]
            Y_growth = self.host_growth_rates["Y"]
            K = self.host_carrying_capacities["Y"]
            b = self.host_mortality_rates["Y"]
            v12_1 = self.host_recovery["Y"]["12"]["1"]
            v12_1 = self.host_recovery["Y"]["12"]["2"]
            # components
            X2_toY = A1V * ((Y1)/(N)) (X2 + X12)
            W2_toY = A1W * ((Y1)/(N)) (Z2 + Z12)
            X1_toY = A2V * ((Y2)/(N)) (X2 + X12)
            W1_toY = A2W * ((Y2)/(N)) (Z2 + Z12)
            growth = - Y_growth * ((N*Y12)/(K))
            deathAndRecovery = - (b + v12_1 + v12_2) * Y12
            return sum([X2_toY, W2_toY, X1_toY, W1_toY, growth, deathAndRecovery])
        def dVdt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            betaHat = self.vector_growth_rates["V"]
            M = self.max_vectors_per_hosts["V"]
            b_hat = self.vector_mortality_rates["V"]
            return  betaHat * ((M*N - V)/(M*N)) * V - b_hat * V
        def dX1dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat1V = self.host_to_vector_trx_consts["Y"][""]["1"]["V"]
            betaHatV = self.vector_growth_rates["V"]
            gamma1 = self.vector_transStadialOvarial["V"]["1"]
            gamma12 = self.vector_transStadialOvarial["V"]["12"]
            mu_1V = self.vector_cofeeding["V"]["V"][""]["1"]
            mu_1W = self.vector_cofeeding["W"]["V"][""]["1"]
            AHat1to12V = self.host_to_vector_trx_consts["Y"]["1"]["2"]["V"]
            mu_2to12V = self.vector_cofeeding["V"]["V"]["1"]["2"]
            mu_2to12W = self.vector_cofeeding["W"]["V"]["1"]["2"]
            v_growth = self.vector_growth_rates["V"]
            M_V = self.max_vectors_per_hosts["V"]["Y"]
            bHat = self.vector_mortality_rates["V"]
            # components
            host2TickInf = AHat1V * ((Y1 + Y12)/(N)) * (V - X1 - X2 - X12)
            trans = betaHatV * (gamma1 * X1 + gamma12 * X12)
            cofeedingXInc = mu_1V * ((X1 + X12)/(V)) * (V - X1 - X2 - X12)
            cofeedingWInc = mu_1W * ((Z1 + Z12)/(W)) * (V - X1 - X2 - X12)
            host2TickCoinf = - AHat1to12V * ((Y2 + Y12)/(N)) * X1 
            cofeedingXDec = - mu_2to12V * ((X2 + X12)/(V)) * X1
            cofeedingWDec = - mu_2to12W * ((Z2 + Z12)/(W)) * X1 
            growthDec = v_growth * ((V * X1)/(M_V * N))
            death = -bHat * X1 
            return sum([host2TickInf, trans, cofeedingXInc, cofeedingWInc, 
                host2TickCoinf, cofeedingXDec, cofeedingWDec, growthDec, death])
        def dX2dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat2V = self.host_to_vector_trx_consts["Y"][""]["2"]["V"]
            betaHatV = self.vector_growth_rates["V"]
            gamma2 = self.vector_transStadialOvarial["V"]["2"]
            gamma12 = self.vector_transStadialOvarial["V"]["12"]
            mu_2V = self.vector_cofeeding["V"]["V"][""]["2"]
            mu_2W = self.vector_cofeeding["W"]["V"][""]["2"]
            AHat2to12V = self.host_to_vector_trx_consts["Y"]["2"]["1"]["V"]
            mu_2to12V = self.vector_cofeeding["V"]["V"]["2"]["1"]
            mu_2to12W = self.vector_cofeeding["W"]["V"]["2"]["1"]
            v_growth = self.vector_growth_rates["V"]
            M_V = self.max_vectors_per_hosts["V"]["Y"]
            bHat = self.vector_mortality_rates["V"]
            # components
            host2TickInf = AHat2V * ((Y2 + Y12)/(N)) * (V - X1 - X2 - X12)
            trans = betaHatV * (gamma2 * X2 + gamma12 * X12)
            cofeedingXInc = mu_2V * ((X2 + X12)/(V)) * (V - X1 - X2 - X12)
            cofeedingWInc = mu_2W * ((Z2 + Z12)/(W)) * (V - X1 - X2 - X12)
            host2TickCoinf = - AHat2to12V * ((Y2 + Y12)/(N)) * X2  
            cofeedingXDec = - mu_2to12V * ((X2 + X12)/(V)) * X2
            cofeedingWDec = - mu_2to12W * ((Z2 + Z12)/(W)) * X2 
            growthDec = v_growth * ((V * X2)/(M_V * N))
            death = -bHat * X2 
            return sum([host2TickInf, trans, cofeedingXInc, cofeedingWInc, 
                host2TickCoinf, cofeedingXDec, cofeedingWDec, growthDec, death])
        def dX12dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat1to12V = self.host_to_vector_trx_consts["Y"]["1"]["2"]["V"]
            AHat2to12V = self.host_to_vector_trx_consts["Y"]["2"]["1"]["V"]
            mu_1to12V = self.vector_cofeeding["V"]["V"]["1"]["2"]
            mu_2to12V = self.vector_cofeeding["V"]["V"]["2"]["1"]
            mu_1to12W = self.vector_cofeeding["W"]["V"]["1"]["2"]
            mu_2to12W = self.vector_cofeeding["W"]["V"]["2"]["1"]
            v_growth = self.vector_growth_rates["V"]
            M_V = self.max_vectors_per_hosts["V"]["Y"]
            bHat = self.vector_mortality_rates["V"]
            # components
            host2TickInf1 = AHat1to12V * ((Y2 + Y12)/(N)) * X1
            host2TickInf2 = AHat2to12V * ((Y1 + Y12)/(N)) * X2
            cofeeding1to12X = mu_1to12V * ((X2 + X12)/V) * X1 
            cofeeding2to12X = mu_2to12V * ((X1 + X12)/V) * X2
            cofeeding1to12Z = mu_1to12W * ((Z2 + Z12)/W) * X1 
            cofeeding2to12Z = mu_2to12W * ((Z1 + Z12)/W) * X2
            growthDec = - v_growth * ((V * X12)/(M_V/N))
            death = bHat * X12 
            return sum([host2TickInf1, host2TickInf2,
                     cofeeding1to12X, cofeeding2to12X, cofeeding1to12Z, cofeeding2to12Z,
                     growthDec, death])
        def dWdt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            betaHat = self.vector_growth_rates["W"]
            M = self.max_vectors_per_hosts["W"]
            b_hat = self.vector_mortality_rates["W"]
            return  betaHat * ((M*N - W)/(M*N)) * W - b_hat * W
        def dZ1dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat1W = self.host_to_vector_trx_consts["Y"][""]["1"]["W"]
            betaHatW = self.vector_growth_rates["W"]
            gamma1 = self.vector_transStadialOvarial["W"]["1"]
            gamma12 = self.vector_transStadialOvarial["W"]["12"]
            mu_1V = self.vector_cofeeding["V"]["W"][""]["1"]
            mu_1W = self.vector_cofeeding["W"]["W"][""]["1"]
            AHat1to12W = self.host_to_vector_trx_consts["Y"]["1"]["2"]["W"]
            mu_2to12V = self.vector_cofeeding["V"]["W"]["1"]["2"]
            mu_2to12W = self.vector_cofeeding["W"]["W"]["1"]["2"]
            W_growth = self.vector_growth_rates["W"]
            M_W = self.max_vectors_per_hosts["W"]["Y"]
            bHat = self.vector_mortality_rates["W"]
            # components
            host2TickInf = AHat1W * ((Y1 + Y12)/(N)) * (W - Z1 - Z2 - Z12)
            trans = betaHatW * (gamma1 * Z1 + gamma12 * Z12)
            cofeedingXInc = mu_1V * ((X1 + X12)/(V)) * (W - Z1 - Z2 - Z12)
            cofeedingWInc = mu_1W * ((Z1 + Z12)/(W)) * (W - Z1 - Z2 - Z12)
            host2TickCoinf = - AHat1to12W * ((Y2 + Y12)/(N)) * Z1 
            cofeedingXDec = - mu_2to12V * ((X2 + X12)/(V)) * Z1
            cofeedingWDec = - mu_2to12W * ((Z2 + Z12)/(W)) * Z1 
            growthDec = W_growth * ((W * Z1)/(M_W * N))
            death = -bHat * Z1 
            return sum([host2TickInf, trans, cofeedingXInc, cofeedingWInc, 
                host2TickCoinf, cofeedingXDec, cofeedingWDec, growthDec, death])
        def dZ2dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat2W = self.host_to_vector_trx_consts["Y"][""]["2"]["W"]
            betaHatW = self.vector_growth_rates["W"]
            gamma2 = self.vector_transStadialOvarial["W"]["2"]
            gamma12 = self.vector_transStadialOvarial["W"]["12"]
            mu_2V = self.vector_cofeeding["V"]["W"][""]["2"]
            mu_2W = self.vector_cofeeding["W"]["W"][""]["2"]
            AHat2to12W = self.host_to_vector_trx_consts["Y"]["2"]["1"]["W"]
            mu_2to12V = self.vector_cofeeding["V"]["W"]["2"]["1"]
            mu_2to12W = self.vector_cofeeding["W"]["W"]["2"]["1"]
            w_growth = self.vector_growth_rates["W"]
            M_W = self.max_vectors_per_hosts["W"]["Y"]
            bHat = self.vector_mortality_rates["W"]
            # components
            host2TickInf = AHat2W * ((Y2 + Y12)/(N)) * (W - Z1 - Z2 - Z12)
            trans = betaHatW * (gamma2 * Z2 + gamma12 * Z12)
            cofeedingXInc = mu_2V * ((X2 + X12)/(V)) * (W - Z1 - Z2 - Z12)
            cofeedingWInc = mu_2W * ((Z2 + Z12)/(W)) * (W - Z1 - Z2 - Z12)
            host2TickCoinf = - AHat2to12W * ((Y1 + Y12)/(N)) * Z2 
            cofeedingXDec = - mu_2to12V * ((X1 + X12)/(V)) * Z2 
            cofeedingWDec = - mu_2to12W * ((Z1 + Z12)/(W)) * Z2  
            growthDec = w_growth * ((V * X2)/(M_W * N))
            death = -bHat * X2 
            return sum([host2TickInf, trans, cofeedingXInc, cofeedingWInc, 
                host2TickCoinf, cofeedingXDec, cofeedingWDec, growthDec, death])
        def dZ12dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            AHat1to12W = self.host_to_vector_trx_consts["Y"]["1"]["2"]["W"]
            AHat2to12 = self.host_to_vector_trx_consts["Y"]["2"]["1"]["W"]
            mu_1to12V = self.vector_cofeeding["V"]["W"]["1"]["2"]
            mu_2to12V = self.vector_cofeeding["V"]["W"]["2"]["1"]
            mu_1to12W = self.vector_cofeeding["W"]["W"]["1"]["2"]
            mu_2to12W = self.vector_cofeeding["W"]["W"]["2"]["1"]
            growth = self.vector_growth_rates["W"]
            M = self.max_vectors_per_hosts["W"]["Y"]
            bHat = self.vector_mortality_rates["W"]
            # components
            host2TickInf1 = AHat1to12W * ((Y2 + Y12)/(N)) * Z1
            host2TickInf2 = AHat2to12 * ((Y1 + Y12)/(N)) * Z2
            cofeeding1to12X = mu_1to12V * ((X2 + X12)/V) * Z1 
            cofeeding2to12X = mu_2to12V * ((X1 + X12)/V) * Z2
            cofeeding1to12Z = mu_1to12W * ((Z2 + Z12)/W) * Z1 
            cofeeding2to12Z = mu_2to12W * ((Z1 + Z12)/W) * Z2
            growthDec = - growth * ((W * Z12)/(M/N))
            death = bHat * Z12 
            return sum([host2TickInf1, host2TickInf2,
                     cofeeding1to12X, cofeeding2to12X, cofeeding1to12Z, cofeeding2to12Z,
                     growthDec, death])
        def model_ode(times, init: List[float], params: List[float]) -> List[float]:
            """
            This function represents the model
            times and params are completely ignored as each step is time dependent and the params are also assumed to be constant over time
            """
            eqns = [dNdt, 
                    dY1dt, dY2dt, dY12dt, 
                    dVdt,
                    dX1dt, dX2dt, dX12dt,
                    dWdt,
                    dZ1dt, dZ2dt, dZ12dt
                    ]
            return list(map(lambda f: f(init), eqns))
        return model_ode
        
    def validate(self):
        # implement sanity checks and data validation
        if not self.pathogens:
            raise DataValidationException("Pathogens can not be empty")
        if not self.vectors:
            raise DataValidationException("Vectors can not be empty")
        if not self.hosts:
            raise DataValidationException("Hosts can not be empty")
        if not self.vector_to_host_trx_consts:
            raise DataValidationException("Vector to Host Trx cosntants can not be empty")
        else:
            if set(self.vectors).difference(set(self.vector_to_host_trx_consts.keys())):
                raise DataValidationException(f"vectors from vector_to_host_trx_consts did not match vectors {self.vectors}")
        if not self.host_to_vector_trx_consts:
            raise DataValidationException("Host to Vector Trx cosntants can not be empty")
        else:
            if set(self.hosts).difference(set(self.host_to_vector_trx_consts.keys())):
                raise DataValidationException(f"hosts from host_to_vector_trx_consts did not match hosts {self.hosts}")
    
    def generate_infection_states(pathogens: set) -> set:
        infection_states = []
        for i in range(1, len(pathogens) + 1):
            combos = list(combinations(pathogens, i))
            infection_states.append(combos)
        return set(infection_states)


class DataValidationException(Exception):
    pass