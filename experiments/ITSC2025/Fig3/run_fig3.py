

import os
import time
from Pipeline.SimulationIO import *
from rules.common.helper import load_yaml


base_path = os.path.dirname(os.path.abspath(__name__))


simfac = SimulatorFactory(1.0)


sim_io = SimulationIO(simfac)


master_configuration = load_yaml(base_path + "/configs/configuration_scenario2.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()


sim.display_run()

sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


master_configuration = load_yaml(base_path + "/configs/configuration_scenario3.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()


sim.display_run()

sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


master_configuration = load_yaml(base_path + "/configs/configuration_scenario4.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()


sim.display_run()

sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)


master_configuration = load_yaml(base_path + "/configs/configuration_scenario5.yaml")
sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()


sim.display_run()

sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)
