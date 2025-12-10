# VR Gaussian Position Bug Fix

**Date:** 2025-12-09  
**Status:** ✅ RESOLVED

## Problem

Gaussians painted in VR appeared at incorrect locations in the VR headset, while displaying correctly in the PC Blender viewport.

## Root Cause

The C++ OpenXR layer was applying transformations (camera offset, rotation compensation) that conflicted with the Python-provided view matrix.

**Key insight:** The Python view matrix is already in Blender WORLD space and handles all world-to-view transformation. Gaussians must remain in raw Blender world coordinates.

## Solution

Remove all coordinate transformations in `gaussian_renderer.cpp`:

```cpp
// Use raw Blender world coordinates - NO transformations!
instanceData[base + 0] = g.position[0];
instanceData[base + 1] = g.position[1];
instanceData[base + 2] = g.position[2];
```

## Lessons Learned

1. **Coordinate system consistency** - All data must match the view matrix's coordinate system
2. **Don't duplicate transformations** - View matrix already handles position/rotation
3. **Debug with isolation** - Remove transformations one by one to find issues

## Files Modified

- `openxr_layer/src/gaussian_renderer.cpp` - Removed all position transformations
- `openxr_layer/include/gaussian_data.h` - Added camera_position field (unused)
- `src/vr/vr_shared_memory.py` - Added camera_position support
- `src/vr/vr_operators.py` - Sends camera.matrix_world.translation
