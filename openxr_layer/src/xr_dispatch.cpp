/**
 * OpenXR Function Dispatch
 * 
 * Implements the hooked OpenXR functions that intercept
 * calls from Blender and inject Gaussian rendering.
 */

#include "xr_dispatch.h"
#include "shared_memory.h"
#include "gaussian_data.h"

#include <Windows.h>
#include <iostream>
#include <vector>

namespace gaussian {

// ============================================
// Global State
// ============================================
static LayerState g_layerState;
static SharedMemoryReader g_sharedMemory;

LayerState& GetLayerState() {
    return g_layerState;
}

// ============================================
// Logging Helper
// ============================================
static void LogXr(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    OutputDebugStringA("[GaussianXR] ");
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
}

// ============================================
// Initialize Dispatch Table
// ============================================
bool InitializeDispatch(XrInstance instance, PFN_xrGetInstanceProcAddr getProcAddr) {
    LogXr("Initializing dispatch table");
    
    g_layerState.next_xrGetInstanceProcAddr = getProcAddr;
    
    // Get original function pointers
    XrResult result;
    
    result = getProcAddr(instance, "xrEndFrame", 
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrEndFrame));
    if (XR_FAILED(result)) {
        LogXr("ERROR: Failed to get xrEndFrame");
        return false;
    }
    
    result = getProcAddr(instance, "xrBeginFrame",
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrBeginFrame));
    if (XR_FAILED(result)) {
        LogXr("ERROR: Failed to get xrBeginFrame");
        return false;
    }
    
    result = getProcAddr(instance, "xrWaitFrame",
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrWaitFrame));
    if (XR_FAILED(result)) {
        LogXr("ERROR: Failed to get xrWaitFrame");
        return false;
    }
    
    // Initialize shared memory
    if (g_sharedMemory.Open()) {
        LogXr("Shared memory opened successfully");
    } else {
        LogXr("WARNING: Shared memory not available (Blender may not be running)");
    }
    
    LogXr("Dispatch table initialized successfully");
    return true;
}

void ShutdownDispatch() {
    LogXr("Shutting down dispatch");
    g_sharedMemory.Close();
    g_layerState = LayerState{};
}

// ============================================
// xrWaitFrame Hook
// ============================================
XrResult XRAPI_CALL gaussian_xrWaitFrame(
    XrSession session,
    const XrFrameWaitInfo* frameWaitInfo,
    XrFrameState* frameState)
{
    // Call original
    XrResult result = g_layerState.next_xrWaitFrame(session, frameWaitInfo, frameState);
    
    if (XR_SUCCEEDED(result)) {
        g_layerState.predicted_display_time = frameState->predictedDisplayTime;
    }
    
    return result;
}

// ============================================
// xrBeginFrame Hook
// ============================================
XrResult XRAPI_CALL gaussian_xrBeginFrame(
    XrSession session,
    const XrFrameBeginInfo* frameBeginInfo)
{
    g_layerState.session = session;
    g_layerState.frame_active = true;
    
    return g_layerState.next_xrBeginFrame(session, frameBeginInfo);
}

// ============================================
// xrEndFrame Hook - MAIN GAUSSIAN INJECTION POINT
// ============================================
XrResult XRAPI_CALL gaussian_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo)
{
    g_layerState.frame_count++;
    
    // Log every 60 frames
    if (g_layerState.frame_count % 60 == 0) {
        LogXr("Frame %llu - Layers submitted: %d", 
            g_layerState.frame_count, 
            frameEndInfo->layerCount);
    }
    
    // Try to read Gaussian data from shared memory
    if (!g_sharedMemory.IsOpen()) {
        g_sharedMemory.Open();
    }
    
    bool hasGaussianData = false;
    if (g_sharedMemory.IsOpen()) {
        auto header = g_sharedMemory.ReadHeader();
        if (header.has_value() && header->gaussian_count > 0) {
            hasGaussianData = true;
            
            // Log occasionally
            if (g_layerState.frame_count % 120 == 0) {
                LogXr("Received %d gaussians from Blender", header->gaussian_count);
            }
        }
    }
    
    // ==== PHASE 1: Just pass through for now ====
    // TODO (Phase 2): Create composition layer with Gaussians
    // TODO (Phase 3): Render Gaussians to texture
    
    /*
    // PHASE 2+ CODE (uncomment when ready):
    if (hasGaussianData) {
        // Create new layer array with our Gaussian layer
        std::vector<const XrCompositionLayerBaseHeader*> newLayers;
        
        // Copy original layers
        for (uint32_t i = 0; i < frameEndInfo->layerCount; ++i) {
            newLayers.push_back(frameEndInfo->layers[i]);
        }
        
        // Add Gaussian layer
        // XrCompositionLayerQuad gaussianQuad = CreateGaussianLayer();
        // newLayers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&gaussianQuad));
        
        // Submit modified frame
        XrFrameEndInfo modifiedInfo = *frameEndInfo;
        modifiedInfo.layerCount = static_cast<uint32_t>(newLayers.size());
        modifiedInfo.layers = newLayers.data();
        
        return g_layerState.next_xrEndFrame(session, &modifiedInfo);
    }
    */
    
    // Pass through to runtime
    g_layerState.frame_active = false;
    return g_layerState.next_xrEndFrame(session, frameEndInfo);
}

}  // namespace gaussian
