
import os
import time
from Pipeline.SimulationIO import *
from rules.common.helper import load_yaml


base_path = os.path.dirname(os.path.abspath(__name__))
master_configuration = load_yaml(base_path + "/configuration_scenario1.yaml")


simfac = SimulatorFactory(1.0)


sim_io = SimulationIO(simfac)

sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

configured = simfac.is_configured()
assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"

sim = simfac.generate_scenario()


sim.display_run()

sim_io.load_simulator(sim)
sim_io.export_scenario(current_configuration=master_configuration)
