import numpy as np
import time
from kernel_tuner.observers.observer import BenchmarkObserver, ContinuousObserver

try:
    import amdsmi
    print("DEBUG: imported amdsmi")
except ImportError:
    amdsmi = None
    print("DEBUG: not imported amdsmi")

class AMDSMIWrapper:
    """Class that gathers the amdsmi functionality for one device."""

    def __init__(self, device_id=0):
        amdsmi.amdsmi_init()
        self.devices = amdsmi.amdsmi_get_processor_handles()
        self.dev = self.devices[device_id]

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.reset_clocks()
        amdsmi.amdsmi_shut_down()

    def set_clocks(self, mem_clock, gr_clock):
        amdsmi.amdsmi_set_gpu_clk_range(self.dev, gr_clock, gr_clock, amdsmi.AmdSmiClkType.AMDSMI_CLK_TYPE_SYS)
        amdsmi.amdsmi_set_gpu_clk_range(self.dev, mem_clock, mem_clock, amdsmi.AmdSmiClkType.AMDSMI_CLK_TYPE_MEM)

    def reset_clocks(self):
        amdsmi.amdsmi_set_clk_freq(self.dev, amdsmi.AmdSmiClkType.GFX, 0)
        amdsmi.amdsmi_set_clk_freq(self.dev, amdsmi.AmdSmiClkType.MEM, 0)

    @property
    def gr_clock(self):
        return amdsmi.amdsmi_get_clk_freq(self.dev, amdsmi.AmdSmiClkType.GFX)

    @gr_clock.setter
    def gr_clock(self, new_clock):
        if new_clock != self.gr_clock:
            self.set_clocks(self.mem_clock, new_clock)

    @property
    def mem_clock(self):
        return amdsmi.amdsmi_get_clk_freq(self.dev, amdsmi.AmdSmiClkType.MEM)

    @mem_clock.setter
    def mem_clock(self, new_clock):
        if new_clock != self.mem_clock:
            self.set_clocks(new_clock, self.gr_clock)


class AMDSMIObserver(BenchmarkObserver):

    def __init__(
        self,
        observables,
        device=0,
        save_all=False,
        continous_duration=1,
    ):
        self.amdsmi = AMDSMIWrapper(device_id=device)
        supported = [
            "core_freq",
            "mem_freq",
        ]
        for obs in observables:
            if obs not in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        self.during_obs = [obs for obs in observables if obs in ["core_freq", "mem_freq", "temperature"]]
        self.iteration = {obs: [] for obs in self.during_obs}

    def before_start(self):
        # clear results of the observables for next measurement
        self.iteration = {obs: [] for obs in self.during_obs}

    def after_start(self):
        self.t0 = time.perf_counter()
        # ensure during is called at least once
        self.during()

    def during(self):
        if "core_freq" in self.observables:
            self.iteration["core_freq"].append(self.amdsmi.gr_clock)
        if "mem_freq" in self.observables:
            self.iteration["mem_freq"].append(self.amdsmi.mem_clock)

    def after_finish(self):
        if "core_freq" in self.observables:
            self.results["core_freqs"].append(np.average(self.iteration["core_freq"]))
        if "mem_freq" in self.observables:
            self.results["mem_freqs"].append(np.average(self.iteration["mem_freq"]))

    def get_results(self):
        averaged_results = {}

        # return averaged results, except when save_all is True
        for obs in self.observables:
            # save all information, if the user requested
            if self.save_all:
                averaged_results[obs + "s"] = self.results[obs + "s"]
            # save averaged results, default
            averaged_results[obs] = np.average(self.results[obs + "s"])

        # clear results for next round
        for obs in self.observables:
            self.results[obs + "s"] = []

        return averaged_results