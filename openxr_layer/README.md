# Gaussian OpenXR API Layer

OpenXR API Layer for rendering 3D Gaussian Splatting in VR headsets.

## Overview

This layer intercepts OpenXR calls from Blender VR and injects custom Gaussian rendering on top of Blender's scene.

## Architecture

```
Blender (Python) → Shared Memory → OpenXR API Layer (C++) → Quest 3 HMD
```

## Building

### Prerequisites

- Visual Studio 2022 with C++ Desktop Development
- CMake 3.20+
- OpenXR SDK

### Build Steps

```powershell
# Clone OpenXR SDK (if not installed via vcpkg)
git clone https://github.com/KhronosGroup/OpenXR-SDK.git external/OpenXR-SDK

# Configure
cmake -B build -S . -G "Visual Studio 17 2022"

# Build
cmake --build build --config Release
```

## Installation

```powershell
# Run as Administrator
cd build/Release
..\..\manifest\install.bat
```

## Usage

1. Start Blender
2. Enable VR Scene Inspection
3. Start VR session
4. The layer will automatically load and log to `%TEMP%\gaussian_layer.log`

## Development Status

- [x] Phase 0: Project setup
- [ ] Phase 1: xrEndFrame hook + test quad
- [ ] Phase 2: Shared memory communication
- [ ] Phase 3: Gaussian rendering
- [ ] Phase 4: Optimization (72+ FPS)

## Files

| File                    | Description                        |
| ----------------------- | ---------------------------------- |
| `main.cpp`              | DLL entry point, layer negotiation |
| `xr_dispatch.cpp`       | OpenXR function hooks              |
| `shared_memory.cpp`     | Blender ↔ DLL communication        |
| `composition_layer.cpp` | OpenXR layer creation              |
| `gaussian_renderer.cpp` | GPU rendering pipeline             |
| `gpu_context.cpp`       | D3D11 device management            |

## License

Part of 3DGS Painter Blender Addon
