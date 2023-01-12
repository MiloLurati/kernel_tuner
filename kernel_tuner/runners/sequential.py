""" The default runner for sequentially tuning the parameter space """
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from time import perf_counter

from kernel_tuner.core import DeviceInterface
from kernel_tuner.util import (ErrorConfig, print_config_output,
                               process_metrics, store_cache)


class SequentialRunner:
    """ SequentialRunner is used for tuning with a single process/thread """

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        """ Instantiate the SequentialRunner

        :param kernel_source: The kernel source
        :type kernel_source: kernel_tuner.core.KernelSource

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param device_options: A dictionary with all options for the device
            on which the kernel should be tuned.
        :type device_options: kernel_tuner.interface.Options

        :param iterations: The number of iterations used for benchmarking
            each kernel instance.
        :type iterations: int
        """

        #detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)

        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False
        self.simulation_mode = False
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0

        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)

    def run(self, parameter_space, kernel_options, tuning_options):
        """ Iterate through the entire parameter space using a single Python process

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.iterface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times. And a dictionary that contains information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()

        """
        logging.debug('sequential runner started for ' + kernel_options.kernel_name)

        results = []

        # iterate over parameter space
        for element in parameter_space:
            params = OrderedDict(zip(tuning_options.tune_params.keys(), element))

            result = None
            warmup_time = 0

            # check if configuration is in the cache
            x_int = ",".join([str(i) for i in element])
            if tuning_options.cache and x_int in tuning_options.cache:
                params.update(tuning_options.cache[x_int])
                params['compile_time'] = 0
                params['verification_time'] = 0
                params['benchmark_time'] = 0
            else:

                # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
                if not self.warmed_up:
                    warmup_time = perf_counter()
                    self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, kernel_options, tuning_options)
                    self.warmed_up = True
                    warmup_time = 1e3 * (perf_counter() - warmup_time)

                result = self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, kernel_options, tuning_options)

                params.update(result)

                # only compute metrics on configs that have not errored
                if isinstance(result[tuning_options.objective], ErrorConfig):
                    logging.debug('kernel configuration was skipped silently due to compile or runtime failure')
                elif tuning_options.metrics:
                    params = process_metrics(params, tuning_options.metrics)

                # print configuration to the console
                print_config_output(tuning_options.tune_params, params, self.quiet, tuning_options.metrics, self.units)

            # get the framework time by estimating based on other times
            total_time = 1000 * (perf_counter() - self.start_time) - warmup_time
            params['strategy_time'] = self.last_strategy_time
            params['framework_time'] = max(total_time - (params['compile_time'] + params['verification_time'] + params['benchmark_time'] + params['strategy_time']), 0)
            params['timestamp'] = str(datetime.now(timezone.utc))
            self.start_time = perf_counter()

            if result:
                store_cache(x_int, params, tuning_options)

            # all visited configurations are added to results to provide a trace for optimization strategies
            results.append(params)

        return results, self.dev.get_environment()
