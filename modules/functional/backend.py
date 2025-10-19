import os
import sys
import importlib.util

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

# Try to load pre-compiled backend first
_backend = None
try:
    # Try to load from build directory
    build_path = os.path.join(_src_path, 'build', '_pvcnn_backend.pyd')
    if os.path.exists(build_path):
        spec = importlib.util.spec_from_file_location("_pvcnn_backend", build_path)
        if spec and spec.loader:
            backend_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend_module)
            _backend = backend_module
            print(f"Successfully loaded pre-compiled backend from {build_path}")
except Exception as e:
    print(f"Failed to load pre-compiled backend: {e}")
    _backend = None

# If pre-compiled version failed, try JIT compilation
if _backend is None:
    try:
        _backend = load(name='_pvcnn_backend',
                        extra_cflags=['-O3', '-std=c++17'],
                        sources=[os.path.join(_src_path,'src', f) for f in [
                            'ball_query/ball_query.cpp',
                            'ball_query/ball_query.cu',
                            'grouping/grouping.cpp',
                            'grouping/grouping.cu',
                            'interpolate/neighbor_interpolate.cpp',
                            'interpolate/neighbor_interpolate.cu',
                            'interpolate/trilinear_devox.cpp',
                            'interpolate/trilinear_devox.cu',
                            'sampling/sampling.cpp',
                            'sampling/sampling.cu',
                            'voxelization/vox.cpp',
                            'voxelization/vox.cu',
                            'bindings.cpp',
                        ]]
                        )
        print("Successfully compiled backend from source")
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        raise RuntimeError("Failed to load PVCNN backend. Please install Visual Studio Build Tools or use pre-compiled version.")

__all__ = ['_backend']
