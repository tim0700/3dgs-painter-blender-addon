"""
NPR Core: Framework-independent gaussian painting engine

This module contains all core logic for 3D Gaussian Splatting painting,
independent of Blender. It can be tested and used standalone.

Core modules:
- gaussian: Gaussian2D data structure
- scene_data: High-performance array-based scene representation
- brush: BrushStamp and painting logic
- brush_manager: Brush library management
- spline: StrokeSpline for path interpolation
- deformation: Non-rigid deformation (CPU)
- deformation_gpu: GPU-accelerated deformation (CUDA)
- api: Synchronous API for integration
- gpu_context: GPU context management for Blender integration

Dependency modules (always available, no external deps):
- dependencies: Package dependency checking
- installer: Package installation utilities
"""

__version__ = "0.1.0"

# =============================================================================
# Dependency management modules - always importable (no external dependencies)
# =============================================================================
from .dependencies import (
    DependencyInfo,
    REQUIRED_PACKAGES,
    get_missing_packages,
    get_installed_packages,
    check_all_dependencies,
    get_package_version,
    is_package_installed,
)
from .installer import PackageInstaller, PlatformInfo, CUDAVersion


# =============================================================================
# Core modules - require external dependencies (numpy, scipy, torch, etc.)
# =============================================================================

def _try_import_core():
    """
    Try to import core modules that require external dependencies.
    Returns True if all imports succeeded, False otherwise.
    """
    try:
        from .gaussian import Gaussian2D, create_test_gaussian
        from .scene_data import SceneData
        from .brush import BrushStamp, StrokePainter
        from .brush_manager import BrushManager, get_brush_manager
        from .api import NPRCoreAPI
        
        # Add to module globals so they can be accessed via npr_core.X
        globals().update({
            "Gaussian2D": Gaussian2D,
            "create_test_gaussian": create_test_gaussian,
            "SceneData": SceneData,
            "BrushStamp": BrushStamp,
            "StrokePainter": StrokePainter,
            "BrushManager": BrushManager,
            "get_brush_manager": get_brush_manager,
            "NPRCoreAPI": NPRCoreAPI,
        })
        return True
    except ImportError as e:
        # Dependencies not installed yet - this is expected on first run
        print(f"[NPR Core] Core modules not available: {e}")
        return False


# Try to import core modules (will fail gracefully if deps not installed)
_core_available = _try_import_core()


def is_core_available() -> bool:
    """
    Check if core modules are available (dependencies installed).
    
    Returns:
        True if all core modules can be imported
    """
    return _core_available


def reload_core():
    """
    Attempt to reload core modules after dependencies are installed.
    Call this after installing packages and before using core functionality.
    
    Returns:
        True if core modules are now available
    """
    global _core_available
    _core_available = _try_import_core()
    return _core_available


__all__ = [
    # Always available - dependency management
    "DependencyInfo",
    "REQUIRED_PACKAGES",
    "get_missing_packages",
    "get_installed_packages",
    "check_all_dependencies",
    "get_package_version",
    "is_package_installed",
    "PackageInstaller",
    "PlatformInfo",
    "CUDAVersion",
    "is_core_available",
    "reload_core",
    # Conditionally available - core modules (when dependencies installed)
    "Gaussian2D",
    "create_test_gaussian",
    "SceneData",
    "BrushStamp",
    "StrokePainter",
    "BrushManager",
    "get_brush_manager",
    "NPRCoreAPI",
]
