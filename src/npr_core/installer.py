# installer.py
# Package installer for 3DGS Painter addon
# Reference: Dream Textures addon implementation

import subprocess
import sys
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from enum import Enum


class CUDAVersion(Enum):
    """Supported CUDA versions for PyTorch."""
    CUDA_118 = "cu118"  # CUDA 11.8
    CUDA_121 = "cu121"  # CUDA 12.1
    CUDA_124 = "cu124"  # CUDA 12.4
    CPU = "cpu"         # CPU only
    MPS = "mps"         # Apple Metal (macOS)


class PlatformInfo:
    """Detected platform information."""
    
    def __init__(self):
        self.system = platform.system()  # Windows, Darwin, Linux
        self.machine = platform.machine()  # x86_64, arm64, etc.
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.cuda_version: Optional[str] = None
        self.cuda_available = False
        
        self._detect_cuda()
    
    def _detect_cuda(self):
        """Detect CUDA version using nvidia-smi or nvcc."""
        if self.system == "Darwin":
            # macOS doesn't have CUDA, uses Metal/MPS
            return
        
        # Method 1: Try nvidia-smi (most reliable for driver version)
        cuda_version = self._get_cuda_from_nvidia_smi()
        if cuda_version:
            self.cuda_version = cuda_version
            self.cuda_available = True
            return
        
        # Method 2: Try nvcc --version (CUDA toolkit version)
        cuda_version = self._get_cuda_from_nvcc()
        if cuda_version:
            self.cuda_version = cuda_version
            self.cuda_available = True
            return
        
        # Method 3: Check environment variable
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            # Try to extract version from path (e.g., "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1")
            match = re.search(r'v?(\d+\.\d+)', cuda_path)
            if match:
                self.cuda_version = match.group(1)
                self.cuda_available = True
                return
    
    def _get_cuda_from_nvidia_smi(self) -> Optional[str]:
        """Get CUDA version from nvidia-smi output."""
        try:
            # nvidia-smi shows the driver's CUDA version (maximum supported)
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if self.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                # Parse output: "CUDA Version: 12.1"
                match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None
    
    def _get_cuda_from_nvcc(self) -> Optional[str]:
        """Get CUDA version from nvcc compiler."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if self.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                # Parse output: "release 12.1, V12.1.105"
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None
    
    def get_recommended_cuda_version(self) -> CUDAVersion:
        """
        Get recommended PyTorch CUDA version based on detected CUDA.
        
        Returns:
            CUDAVersion enum for PyTorch installation
        """
        if self.system == "Darwin":
            return CUDAVersion.MPS
        
        if not self.cuda_available or not self.cuda_version:
            return CUDAVersion.CPU
        
        try:
            major, minor = map(int, self.cuda_version.split('.')[:2])
            cuda_numeric = major + minor / 10.0
            
            # PyTorch CUDA version mapping:
            # - CUDA 12.4+ -> cu124
            # - CUDA 12.1-12.3 -> cu121
            # - CUDA 11.8-12.0 -> cu118
            # - CUDA < 11.8 -> cu118 (may have compatibility issues)
            
            if cuda_numeric >= 12.4:
                return CUDAVersion.CUDA_124
            elif cuda_numeric >= 12.1:
                return CUDAVersion.CUDA_121
            else:
                return CUDAVersion.CUDA_118
                
        except (ValueError, AttributeError):
            return CUDAVersion.CPU
    
    def __str__(self) -> str:
        cuda_info = f"CUDA {self.cuda_version}" if self.cuda_available else "No CUDA"
        return f"{self.system} {self.machine} Python {self.python_version} ({cuda_info})"


class PackageInstaller:
    """
    Install Python packages inside Blender.
    Reference: Dream Textures addon implementation.
    """
    
    def __init__(self, addon_path: Optional[Path] = None):
        """
        Initialize the package installer.
        
        Args:
            addon_path: Path to the addon directory. If None, uses the directory containing this file.
        """
        if addon_path is None:
            addon_path = Path(__file__).parent.parent
        
        self.addon_path = Path(addon_path)
        self.dependencies_path = self.addon_path / ".python_dependencies"
        self.requirements_path = self.addon_path / "requirements"
        self.platform_info = PlatformInfo()
        self.python_exe = self._get_python_executable()
    
    def _get_python_executable(self) -> Path:
        """
        Get path to Blender's Python executable.
        
        Returns:
            Path to python executable
        """
        # sys.executable in Blender points to the Python interpreter
        return Path(sys.executable)
    
    def ensure_pip(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Ensure pip is installed in Blender's Python.
        
        Returns:
            True if pip is available
        """
        try:
            import pip
            return True
        except ImportError:
            if progress_callback:
                progress_callback("Installing pip...")
            
            try:
                subprocess.check_call(
                    [str(self.python_exe), "-m", "ensurepip", "--default-pip"],
                    creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
                )
                return True
            except subprocess.CalledProcessError as e:
                if progress_callback:
                    progress_callback(f"Failed to install pip: {e}")
                return False
    
    def get_pytorch_install_args(self, cuda_version: Optional[CUDAVersion] = None) -> List[str]:
        """
        Get pip install arguments for PyTorch based on platform and CUDA version.
        
        IMPORTANT: Use exact versions with +cuXXX suffix to prevent pip from
        installing a newer CPU-only version from PyPI.
        
        Args:
            cuda_version: Override CUDA version selection. If None, auto-detect.
        
        Returns:
            List of pip install arguments
        """
        if cuda_version is None:
            cuda_version = self.platform_info.get_recommended_cuda_version()
        
        # Use exact versions with CUDA suffix to prevent CPU version overwrite
        # PyTorch wheel naming: torch-2.6.0+cu124
        if cuda_version == CUDAVersion.CPU:
            return [
                "torch==2.6.0+cpu",
                "torchvision==0.21.0+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        elif cuda_version == CUDAVersion.CUDA_118:
            return [
                "torch==2.6.0+cu118",
                "torchvision==0.21.0+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        elif cuda_version == CUDAVersion.CUDA_121:
            return [
                "torch==2.6.0+cu121",
                "torchvision==0.21.0+cu121",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
        elif cuda_version == CUDAVersion.CUDA_124:
            return [
                "torch==2.6.0+cu124",
                "torchvision==0.21.0+cu124",
                "--index-url", "https://download.pytorch.org/whl/cu124"
            ]
        elif cuda_version == CUDAVersion.MPS:
            # macOS - use default PyPI (has MPS support built-in)
            return ["torch==2.6.0", "torchvision==0.21.0"]
        else:
            # Fallback to CPU
            return [
                "torch==2.6.0+cpu",
                "torchvision==0.21.0+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
    
    def get_requirements_file(self) -> Path:
        """
        Get the appropriate requirements file for the current platform.
        
        Returns:
            Path to requirements file
        """
        system = self.platform_info.system
        
        if system == "Windows":
            if self.platform_info.cuda_available:
                return self.requirements_path / "win_cuda.txt"
            else:
                return self.requirements_path / "win_cpu.txt"
        elif system == "Darwin":
            return self.requirements_path / "mac_mps.txt"
        else:  # Linux
            if self.platform_info.cuda_available:
                return self.requirements_path / "linux_cuda.txt"
            else:
                return self.requirements_path / "win_cpu.txt"  # Use CPU version for Linux without CUDA
    
    def install_requirements(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Install packages from requirements file to target directory.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            Tuple of (success, failed_packages)
        """
        if not self.ensure_pip(progress_callback):
            return False, ["pip"]
        
        # Create dependencies directory
        self.dependencies_path.mkdir(parents=True, exist_ok=True)
        
        requirements_file = self.get_requirements_file()
        if not requirements_file.exists():
            if progress_callback:
                progress_callback(f"Requirements file not found: {requirements_file}")
            return False, [str(requirements_file)]
        
        if progress_callback:
            progress_callback(f"Installing from {requirements_file.name}...")
        
        try:
            result = subprocess.run(
                [
                    str(self.python_exe),
                    "-m", "pip", "install",
                    "-r", str(requirements_file),
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--upgrade-strategy", "only-if-needed",  # Don't upgrade torch if already installed
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback("✓ Base packages installed successfully")
                return True, []
            else:
                if progress_callback:
                    progress_callback(f"✗ Failed to install packages: {result.stderr[:500]}")
                return False, ["requirements"]
                
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback("✗ Installation timed out (10 minutes)")
            return False, ["timeout"]
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Error: {str(e)}")
            return False, [str(e)]
    
    def install_pytorch(
        self,
        cuda_version: Optional[CUDAVersion] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Install PyTorch with appropriate CUDA support.
        
        Args:
            cuda_version: Override CUDA version. If None, auto-detect.
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        """
        if not self.ensure_pip(progress_callback):
            return False
        
        # Create dependencies directory
        self.dependencies_path.mkdir(parents=True, exist_ok=True)
        
        if cuda_version is None:
            cuda_version = self.platform_info.get_recommended_cuda_version()
        
        if progress_callback:
            progress_callback(f"Installing PyTorch ({cuda_version.value})... This may take several minutes.")
        
        args = self.get_pytorch_install_args(cuda_version)
        
        try:
            result = subprocess.run(
                [
                    str(self.python_exe),
                    "-m", "pip", "install",
                    *args,
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--force-reinstall",  # Force reinstall to ensure correct CUDA version
                    "--no-deps",  # Don't install dependencies (handled separately)
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minutes for PyTorch
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback(f"✓ PyTorch ({cuda_version.value}) installed successfully")
                return True
            else:
                if progress_callback:
                    progress_callback(f"✗ Failed to install PyTorch: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback("✗ PyTorch installation timed out (20 minutes)")
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Error installing PyTorch: {str(e)}")
            return False
    
    def install_gsplat(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Install gsplat package (requires CUDA).
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        """
        if not self.platform_info.cuda_available:
            if progress_callback:
                progress_callback("⚠ gsplat requires CUDA - skipping on non-CUDA system")
            return True  # Not a failure, just skipped
        
        if progress_callback:
            progress_callback("Installing gsplat...")
        
        try:
            result = subprocess.run(
                [
                    str(self.python_exe),
                    "-m", "pip", "install",
                    "gsplat>=0.1.0",
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--no-deps",  # Don't reinstall torch
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback("✓ gsplat installed successfully")
                return True
            else:
                if progress_callback:
                    progress_callback(f"⚠ gsplat installation failed (optional): {result.stderr[:200]}")
                return False
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠ gsplat installation error (optional): {str(e)}")
            return False
    
    def install_all(
        self,
        cuda_version: Optional[CUDAVersion] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Install all required packages.
        
        Installation order:
        1. PyTorch + torchvision (with exact CUDA version)
        2. Base requirements (with --upgrade-strategy only-if-needed)
        3. gsplat (optional)
        
        Args:
            cuda_version: Override CUDA version for PyTorch
            progress_callback: Optional callback for progress messages
        
        Returns:
            Tuple of (success, failed_packages)
        """
        failed = []
        
        if progress_callback:
            progress_callback(f"Platform: {self.platform_info}")
            progress_callback(f"Target directory: {self.dependencies_path}")
        
        # Step 1: Install PyTorch FIRST with exact CUDA version
        if not self.install_pytorch(cuda_version, progress_callback):
            failed.append("torch")
        
        # Step 2: Install base requirements (numpy, scipy, etc.)
        # Uses --upgrade-strategy only-if-needed to not touch already-installed torch
        success, failed_reqs = self.install_requirements(progress_callback)
        if not success:
            failed.extend(failed_reqs)
        
        # Step 3: Install gsplat (optional, requires CUDA)
        self.install_gsplat(progress_callback)
        
        all_success = len(failed) == 0
        
        if progress_callback:
            if all_success:
                progress_callback("=" * 40)
                progress_callback("✓ All dependencies installed successfully!")
                progress_callback("Please restart Blender to load the packages.")
            else:
                progress_callback("=" * 40)
                progress_callback(f"✗ Some packages failed: {', '.join(failed)}")
        
        return all_success, failed
    
    def uninstall_all(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Remove all installed dependencies.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        """
        if self.dependencies_path.exists():
            if progress_callback:
                progress_callback(f"Removing {self.dependencies_path}...")
            
            try:
                shutil.rmtree(self.dependencies_path)
                if progress_callback:
                    progress_callback("✓ Dependencies removed successfully")
                return True
            except Exception as e:
                if progress_callback:
                    progress_callback(f"✗ Failed to remove dependencies: {e}")
                return False
        else:
            if progress_callback:
                progress_callback("No dependencies directory found")
            return True
    
    def is_installed(self) -> bool:
        """
        Check if dependencies are installed.
        
        Returns:
            True if dependencies directory exists and is not empty
        """
        if not self.dependencies_path.exists():
            return False
        
        # Check if directory has content (more than just metadata)
        contents = list(self.dependencies_path.iterdir())
        return len(contents) > 2  # Allow for .dist-info directories
