
class PVH_Model:
    def __init__(self, pathogens: list, vectors: list, hosts: list, 
    # transmission constants
    vector_to_host_trx_consts: dict, 
    host_to_vector_trx_consts: dict, 
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

    def generate_model(self):
        def model_ode(times, init: List[float], params: List[float]) -> List[float]:
            pass
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
        



class DataValidationException(Exception):
    pass