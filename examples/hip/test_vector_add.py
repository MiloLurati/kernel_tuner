#!/usr/bin/env python
"""Minimal example for a HIP Kernel unit test with the Kernel Tuner"""

import numpy
from kernel_tuner import run_kernel
import pytest

#Check pyhip is installed and if a HIP capable device is present, if not skip the test
try:
    from pyhip import hip, hiprtc
except ImportError:
    print("PyHIP not installed or no HIP device detected or PYTHONPATH does not includes PyHIP")
    hip = None
    hiprtc = None

try:
    import pyhip as hip
    #device = hip.hipGetDevice()
    hipProps = hip.hipGetDeviceProperties(0)
    name = hipProps._name.decode('utf-8')
    max_threads = hipProps.maxThreadsPerBlock
    print(f'{name} with {max_threads} max threads per block')
except ImportError:
    pytest.skip("PyHIP not installed or no HIP device detected")

def test_vector_add():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    params = {"block_size_x": 512}

    answer = run_kernel("vector_add", kernel_string, problem_size, args, params, lang="HIP")

    assert numpy.allclose(answer[0], a+b, atol=1e-8)