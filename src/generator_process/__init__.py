"""
Generator process for 3DGS Painter.

This module provides subprocess-based computation for PyTorch/CUDA operations.
PyTorch is loaded ONLY in the subprocess to avoid TBB DLL conflicts with Blender.

Reference: Dream Textures generator_process/__init__.py
"""

from typing import Callable
from multiprocessing import current_process

from .actor import Actor, is_actor_process
from .future import Future


class RunInSubprocess(Exception):
    """
    Decorators to support running functions that are not defined under the Generator class in its subprocess.
    This is to reduce what would otherwise be duplicate function definitions that logically don't belong to
    the Generator, but require something in its subprocess (such as access to installed dependencies).
    
    Usage:
        @RunInSubprocess
        def my_function():
            import torch  # Only available in subprocess
            return torch.cuda.is_available()
    """

    def __new__(cls, func=None):
        if func is None:
            # support `raise RunInSubprocess`
            return super().__new__(cls)
        return cls.always(func)

    @staticmethod
    def always(func):
        """Always run in subprocess."""
        if is_actor_process:
            return func
        
        def wrapper(*args, **kwargs):
            return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
        
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def when(condition: bool | Callable[..., bool]):
        """Run in subprocess when condition is true."""
        if not isinstance(condition, Callable):
            if condition:
                return RunInSubprocess.always
            return lambda x: x
        
        def decorator(func):
            if is_actor_process:
                return func
            
            def wrapper(*args, **kwargs):
                if condition(*args, **kwargs):
                    return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
                return func(*args, **kwargs)
            
            RunInSubprocess._copy_attributes(func, wrapper)
            return wrapper
        
        return decorator

    @staticmethod
    def when_raised(func):
        """Run in subprocess if RunInSubprocess exception is raised."""
        if is_actor_process:
            return func
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RunInSubprocess:
                return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
        
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def _copy_attributes(src, dst):
        """Copy function attributes from source to destination."""
        for n in ["__annotations__", "__doc__", "__name__", "__module__", "__qualname__"]:
            if hasattr(src, n):
                setattr(dst, n, getattr(src, n))


class NPRGenerator(Actor):
    """
    The actor used for all background PyTorch/CUDA processes.
    
    All methods defined here (except protected ones) will be executed in the subprocess.
    The subprocess has access to PyTorch, CUDA, gsplat, etc.
    """

    # Import actions from separate modules (executed in subprocess)
    # These will be added as the project progresses
    
    @staticmethod
    def call(func, *args, **kwargs):
        """
        Call a function in the subprocess.
        Used by RunInSubprocess decorator.
        """
        return func(*args, **kwargs)
    
    # ==========================================================================
    # System Info Actions
    # ==========================================================================
    
    def get_torch_info(self) -> dict:
        """
        Get PyTorch and CUDA information.
        
        Returns:
            dict with torch version, cuda availability, device info
            All values are pure Python types (str, bool, int, float, list, dict)
        """
        try:
            import torch
            
            cuda_available = bool(torch.cuda.is_available())
            
            info = {
                "torch_version": str(torch.__version__),
                "cuda_available": cuda_available,
                "cuda_version": str(torch.version.cuda) if cuda_available else None,
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "device_name": None,
                "devices": [],
            }
            
            if cuda_available:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info["devices"].append({
                        "index": int(i),
                        "name": str(props.name),
                        "total_memory_gb": float(props.total_memory / (1024**3)),
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
                if info["devices"]:
                    info["device_name"] = info["devices"][0]["name"]
            
            return info
            
        except ImportError as e:
            return {"error": f"PyTorch not available: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_dependencies(self) -> dict:
        """
        Check if all required dependencies are available in subprocess.
        
        Returns:
            dict with package availability status
        """
        result = {}
        
        packages = [
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("pillow", "PIL"),
            ("pyyaml", "yaml"),
        ]
        
        for name, import_name in packages:
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "unknown")
                result[name] = {"available": True, "version": version}
            except ImportError as e:
                result[name] = {"available": False, "error": str(e)}
        
        # Check gsplat separately (may not be installed)
        try:
            import gsplat
            result["gsplat"] = {"available": True, "version": getattr(gsplat, "__version__", "unknown")}
        except ImportError:
            result["gsplat"] = {"available": False, "error": "Not installed"}
        
        return result
    
    # ==========================================================================
    # Computation Actions (to be expanded)
    # ==========================================================================
    
    def test_cuda_computation(self, size: int = 1000) -> dict:
        """
        Run a simple CUDA computation test.
        
        Args:
            size: Size of test tensor
            
        Returns:
            dict with test results (all pure Python types)
        """
        try:
            import torch
            import time
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create tensors
            start = time.time()
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Synchronize if CUDA
            if device == "cuda":
                torch.cuda.synchronize()
            
            compute_elapsed = time.time() - start
            
            # Transfer to CPU and convert to Python float
            transfer_start = time.time()
            result_sum = float(c.sum().cpu().item())
            transfer_elapsed = time.time() - transfer_start
            
            return {
                "success": True,
                "device": str(device),
                "size": int(size),
                "compute_time_ms": float(compute_elapsed * 1000),
                "transfer_time_ms": float(transfer_elapsed * 1000),
                "result_sum": result_sum,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Utility Functions
# =============================================================================

def kill_generator():
    """Kill the generator subprocess if running."""
    NPRGenerator.shared_close()


def get_generator() -> NPRGenerator:
    """Get or create the shared NPRGenerator instance."""
    return NPRGenerator.shared()
