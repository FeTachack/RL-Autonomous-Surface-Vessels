

testingvessels = [22,24,26,28,30,32]
nround = 10
nvessels = 6


assert nvessels <= len(testingvessels), "You can not draw k from n with k>n"
assert nvessels > 0, "You have to draw at least one vessel"


import os
import time
from CollisionHandling.CollisionAvoiderManager import CollisionAvoider
from CollisionHandling.CollisionDetector import CollisionDetector
from Pipeline.SimulationIO import *
from rules.common.helper import load_yaml
import random


base_path = os.path.dirname(os.path.abspath(__name__))
master_configuration = load_yaml(base_path + "/configuration_scenario5.yaml")


simfac = SimulatorFactory(1.0)


sim_io = SimulationIO(simfac)

total_control_times = []
total_colav_times = []
total_coldet_times = []
total_display_times = []
postprocessingtimes = []
update_waypointtimes = []
total_simulation_times = []
total_eventlist_times = []


def dosimulation(tobetestedvessels):

    sim_io.configure_simfac_from_config_dict(current_configuration_input = master_configuration)

    configured = simfac.is_configured()
    assert configured, "Not configured. You have probably changed the configuration.yaml to an invalid state"


    allvesselmodels = []
    for i in simfac.models:
        allvesselmodels.append(i.id)


    for keeps in allvesselmodels:
        if not (keeps in tobetestedvessels):


            for model in simfac.models:
                if model.id == keeps:

                    simfac.models.remove(model)

    def measuretime(simref):

        total_display_time = 0
        total_colav_time = 0
        total_coldet_time = 0
        for listener in simref.listeners:
            if simref.displayer is not None:
                from Simulator.Displayer import Displayer
                if isinstance(listener, Displayer):
                    from Simulator.Displayer import Displayer
                    total_display_time = listener.total_display_time
            elif isinstance(listener, CollisionAvoider):
                total_colav_time = listener.total_colav_time
            elif isinstance(listener, CollisionDetector):
                total_coldet_time = listener.total_coldet_time

        total_control_times.append((simref.total_control_time /  simref.n) / (10**9))
        total_colav_times.append((total_colav_time / simref.n) / (10**9))
        total_coldet_times.append((total_coldet_time / simref.n) / (10**9))
        total_display_times.append((total_display_time / simref.n) / (10**9))
        postprocessingtimes.append((simref.postprocessingtime / simref.n) / (10**9))
        update_waypointtimes.append((simref.set_current_statetime / simref.n) / (10**9))
        total_simulation_times.append((simref.total_simulation_time / simref.n) / (10**9))
        total_eventlist_times.append((simref.total_listener_time / simref.n) / (10**9))

    simfac.runner.append(measuretime)

    sim = simfac.generate_scenario()


    try:
        sim.display_run()
    except ValueError as ve:
        if "exceeding" in str(ve):
            print("OK")
        else:
            raise


for round in range(nround):

    tobetestedvessels = random.sample(testingvessels, nvessels)
    dosimulation(tobetestedvessels)


print("---- Result of Runtime Benchmarking ----")
print("The configuration was: testingvessels: " +str(testingvessels) + " , nround: "+str(nround) + " , nvessels:" +str(nvessels))
print("The yaml was: " +str(master_configuration))
print("\t-------------------------------------")
print("Component \t\t Computation time (s) averaged over all simulation rounds and all timesteps:")
print("------ \t\t\t ----------------------")
print("Total:\t\t\t " + str(np.array(total_simulation_times).mean()) + " , std: " + str(np.array(total_simulation_times).std()))
print("Controller:\t\t "+str(np.array(total_control_times).mean()) + " , std: " + str(np.array(total_control_times).std()))
print("Displayer:\t\t " + str(np.array(total_display_times).mean()) + " , std: " + str(np.array(total_display_times).std()))
print("Detector: \t\t " + str(np.array(total_coldet_times).mean()) + " , std: " + str(np.array(total_coldet_times).std()))
print("Avoider:\t\t " + str(np.array(total_colav_times).mean()) + " , std: " + str(np.array(total_colav_times).std()))
print("Check state/signal:\t" + str(np.array(postprocessingtimes).mean()) + " , std: " + str(np.array(postprocessingtimes).std()))
print("Update state:\t\t " + str(np.array(update_waypointtimes).mean()) + " , std: " + str(np.array(update_waypointtimes).std()))


sumstack = np.stack([total_simulation_times,total_control_times,total_display_times,total_coldet_times,total_colav_times,postprocessingtimes,update_waypointtimes]).T
np.savetxt("runtime_nvessels_" +str(nvessels)+".csv", sumstack, delimiter=',')
