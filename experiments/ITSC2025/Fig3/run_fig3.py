# run simulations for scenarios 2 -5 after each other
# scenario 2) - ZAM_...-2.xml
# scenario 3) - USA...xml
# scenario 4) - ZAM_...-4.xml
# scenario 5) - ZAM_...-5.xml

import os
import time
from Pipeline.SimulationIO import *
from rules.common.helper import load_yaml

# loading configuration yaml
base_path = os.path.dirname(os.path.abspath(__name__))

# creating a simulation factory
simfac = SimulatorFactory(1.0)

# creating a helper IO (this IO is useful for importing/exporting scenarios)
sim_io = SimulationIO(simfac) # Linking the factory to the IO. With the IO we will now configure the linked factory


# SCENARIO 2
master_configuration = load_yaml(base_path + "/configs/configuration_scenario2.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()

# starting the simulator
sim.display_run()
# Exporting the simulator
sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


# SCENARIO 3
master_configuration = load_yaml(base_path + "/configs/configuration_scenario3.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()

# starting the simulator
sim.display_run()
# Exporting the simulator
sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


# SCENARIO 4
master_configuration = load_yaml(base_path + "/configs/configuration_scenario4.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()

# starting the simulator
sim.display_run()
# Exporting the simulator
sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


# SCENARIO 5
master_configuration = load_yaml(base_path + "/configs/configuration_scenario5.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()

# starting the simulator
sim.display_run()
# Exporting the simulator
sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)
