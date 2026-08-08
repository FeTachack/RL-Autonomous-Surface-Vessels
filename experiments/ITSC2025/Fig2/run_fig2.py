# importing dependencies
import os
import time
from Pipeline.SimulationIO import *
from rules.common.helper import load_yaml

# loading configuration yaml
base_path = os.path.dirname(os.path.abspath(__name__))
master_configuration = load_yaml(base_path + "/configuration_scenario1.yaml")

# creating a simulation factory
simfac = SimulatorFactory(1.0)

# creating a helper IO (this IO is useful for importing/exporting scenarios)
sim_io = SimulationIO(simfac) # Linking the factory to the IO. With the IO we will now configure the linked factory

sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()

# starting the simulator
sim.display_run()
# Exporting the simulator
sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)
