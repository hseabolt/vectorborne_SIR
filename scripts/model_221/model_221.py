
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
            return sum(X1_toY, W1_toY, recov_from_coinf, coinfX, coinfW, growth, deathAndRecovery)
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
            return sum(X2_toY, W2_toY, recov_from_coinf, coinfX, coinfW, growth, deathAndRecovery)
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
            return sum(X2_toY, W2_toY, X1_toY, W1_toY, recov, coinfX, coinfW, growth, deathAndRecovery)
        def dX1dt(input):
            N, Y1, Y2, Y12, V, X1, X2, X12, W, Z1, Z2, Z12 = input
            # constants
            
            # components
            host2TickInf = 0
            trans = 0
            cofeedingXInc = 0
            cofeedingWInc = 0
            host2TickCoinf = 0
            cofeedingXDec = 0
            cofeedingWDec = 0
            growthDec = 0
            death = 0
            return sum()
        def model_ode(times, init: List[float], params: List[float]) -> List[float]:
            """
            This function represents the model
            times and params are completely ignored as each step is time dependent and the params are also assumed to be constant over time
            """
            eqns = [dY1dt, dY2dt, dY12dt]
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