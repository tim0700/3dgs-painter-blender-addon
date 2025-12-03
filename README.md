# 3DGS Painter for Blender

A Blender addon for non-photorealistic 3D Gaussian Splatting painting, based on the SIGGRAPH 2025 paper "Painting with 3D Gaussian Splat Brushes".

![Blender](https://img.shields.io/badge/Blender-5.0+-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

-   🎨 **NPR Gaussian Painting** - Paint with 3D Gaussian splat brushes
-   🖌️ **Brush System** - Customizable brushes with stamps and strokes
-   ⚡ **GPU Accelerated** - CUDA support for real-time performance
-   🔧 **Auto Dependency Management** - One-click installation of PyTorch and other packages

## Requirements

-   **Blender 5.0+** (tested on Blender 5.0)
-   **Python 3.11** (bundled with Blender)
-   **NVIDIA GPU** (recommended, for CUDA acceleration)
    -   CUDA 11.8, 12.1, or 12.4
    -   6GB+ VRAM recommended
-   **Internet connection** (for dependency installation)

## Installation

### Method 1: ZIP Installation (Recommended for Users)

1. **Download** the latest release from [Releases](https://github.com/nagi0101/3dgs-painter-blender-addon/releases)

2. **Install in Blender**:

    - Open Blender
    - Go to `Edit` → `Preferences` → `Add-ons`
    - Click `Install...` (top right)
    - Select the downloaded `.zip` file
    - Enable the addon by checking the box next to **"3DGS Painter"**

3. **Install Dependencies**:
    - In the addon preferences panel, you'll see "Missing Dependencies"
    - Select your CUDA version (or leave on Auto-detect)
    - Click **"Install Dependencies"**
    - Wait for installation to complete (15-20 minutes required)
    - **Restart Blender** after installation

### Method 2: Development Installation (For Developers)

Use a symbolic link to edit the addon without reinstalling:

1. **Clone the repository**:

    ```powershell
    git clone https://github.com/nagi0101/3dgs-painter-blender-addon.git
    cd 3dgs-painter-blender-addon
    ```

2. **Create a junction/symlink** (PowerShell as Administrator):

    ```powershell
    # Windows (Junction - no admin required)
    cmd /c mklink /J "$env:APPDATA\Blender Foundation\Blender\5.0\scripts\addons\threegds_painter" "$(Get-Location)\src"

    # Or with symbolic link (requires admin)
    New-Item -ItemType SymbolicLink -Path "$env:APPDATA\Blender Foundation\Blender\5.0\scripts\addons\threegds_painter" -Target "$(Get-Location)\src"
    ```

    For **macOS/Linux**:

    ```bash
    ln -s "$(pwd)/src" ~/.config/blender/5.0/scripts/addons/threegds_painter
    ```

3. **Enable in Blender**:

    - Open Blender
    - Go to `Edit` → `Preferences` → `Add-ons`
    - Search for "3DGS Painter"
    - Enable the addon

4. **Install Dependencies** (same as Method 1, step 3)

## Usage

_Coming soon - painting tools will be available in View3D > Sidebar > 3DGS Paint_

## Project Structure

```
3dgs-painter-blender-addon/
├── src/                          # Addon source code
│   ├── __init__.py               # Addon entry point
│   ├── operators.py              # Blender operators
│   ├── preferences.py            # Addon preferences UI
│   ├── blender_manifest.toml     # Blender extension manifest
│   ├── requirements/             # Platform-specific dependencies
│   │   ├── win_cuda.txt
│   │   ├── win_cpu.txt
│   │   ├── mac_mps.txt
│   │   └── linux_cuda.txt
│   └── npr_core/                 # Core painting engine (bpy-independent)
│       ├── dependencies.py       # Dependency checking
│       ├── installer.py          # Package installer
│       ├── gaussian.py           # Gaussian data structures
│       ├── brush.py              # Brush system
│       └── ...
├── docs/                         # Development documentation
├── tests/                        # Unit tests
└── README.md
```

## Development

### Running Tests

```powershell
# Run tests (requires dependencies installed)
python -m pytest tests/
```

### Reloading After Changes

In Blender, press `F3` and search for "Reload Scripts", or restart Blender.

## Troubleshooting

### "Missing Dependencies" after installation

-   Make sure to **restart Blender** after installing dependencies
-   Check the Installation Log in addon preferences for errors

### CUDA not detected

-   Ensure NVIDIA drivers are installed
-   Run `nvidia-smi` in terminal to verify CUDA is available
-   Try selecting a specific CUDA version instead of Auto-detect

### Installation times out

-   Check your internet connection
-   Try using CPU version first (smaller download)
-   PyTorch CUDA version is ~2-3GB

### Permission errors on Windows

-   Run Blender as Administrator for initial installation
-   Or use a portable Blender installation

## References

-   [Painting with 3D Gaussian Splat Brushes](https://arxiv.org/abs/xxx) - SIGGRAPH 2025
-   [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - SIGGRAPH 2023
-   [gsplat](https://github.com/nerfstudio-project/gsplat) - Differentiable Gaussian rasterization
-   [Dream Textures](https://github.com/carson-katri/dream-textures) - Dependency management reference

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

-   KIRI Innovation for [3dgs-render-blender-addon](https://github.com/Kiri-Innovation/3dgs-render-blender-addon) (rendering reference)
-   Dream Textures addon for dependency installation patterns
