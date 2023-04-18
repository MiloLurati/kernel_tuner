import numpy as np
import ctypes
from .context import skip_if_no_pyhip

import pytest
import kernel_tuner
from kernel_tuner.backends import hip as kt_hip
from kernel_tuner.core import KernelSource, KernelInstance

try: 
    from pyhip import hip, hiprtc
    hip_present = True
except ImportError:
    pass

@skip_if_no_pyhip
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.bool_(True)
    d = np.zeros_like(b)

    arguments = [d, a, b, c]

    dev = kt_hip.HipFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    assert(gpu_args, ctypes.Structure)

@skip_if_no_pyhip
def test_compile():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kernel_name = "vector_add"
    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    dev = kt_hip.HipFunctions(0)
    try:
        dev.compile(kernel_instance)
    except Exception as e:
        pytest.fail("Did not expect any exception:" + str(e))

@skip_if_no_pyhip
def test_memset_and_memcpy_dtoh():
    a = [23, 23, 23, 23]
    x = np.array(a).astype(np.float32)
    x_d = hip.hipMalloc(x.nbytes)
    output = np.empty(4, dtype=np.float32)

    Hipfunc = kt_hip.HipFunctions()
    Hipfunc.memset(x_d, 23, x.nbytes)
    Hipfunc.memcpy_dtoh(output, x_d)

    print(a)
    print(output)

    assert all(output == a)
    assert all(x == a)


@skip_if_no_pyhip
def test_memcpy_htod():
    a = [1, 2, 3, 4]
    src = np.array(a).astype(np.float32)
    x = np.zeros_like(src)
    x_c = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    Hipfunc = kt_hip.HipFunctions()
    Hipfunc.memcpy_htod(x_c, src)

    assert all(x_c.numpy == a)

@skip_if_no_pyhip
def test_benchmark(env):
    results, _ = kernel_tuner.tune_kernel(*env, block_size_names=["nthreads"])
    assert len(results) == 3
    assert all(["nthreads" in result for result in results])
    assert all(["time" in result for result in results])
    assert all([result["time"] > 0.0 for result in results])

def dummy_func(a, b, block=0, grid=0, stream=None, shared=0, texrefs=None):
    pass

