#pragma once

#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D11

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <functional>
#include <unordered_map>
#include <string>
#include <mutex>

namespace gaussian {

// ============================================
// Function Pointer Types
// ============================================
using PFN_xrEndFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameEndInfo*);
using PFN_xrBeginFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameBeginInfo*);
using PFN_xrWaitFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameWaitInfo*, XrFrameState*);

// ============================================
// Layer State
// ============================================
struct LayerState {
    XrInstance instance = XR_NULL_HANDLE;
    XrSession session = XR_NULL_HANDLE;
    
    // Original function pointers (to call next in chain)
    PFN_xrEndFrame next_xrEndFrame = nullptr;
    PFN_xrBeginFrame next_xrBeginFrame = nullptr;
    PFN_xrWaitFrame next_xrWaitFrame = nullptr;
    PFN_xrGetInstanceProcAddr next_xrGetInstanceProcAddr = nullptr;
    
    // Frame timing
    XrTime predicted_display_time = 0;
    bool frame_active = false;
    
    // Statistics
    uint64_t frame_count = 0;
    uint64_t gaussian_render_count = 0;
};

// ============================================
// Global State Access
// ============================================
LayerState& GetLayerState();

// ============================================
// Hooked Functions
// ============================================
XrResult XRAPI_CALL gaussian_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo);

XrResult XRAPI_CALL gaussian_xrBeginFrame(
    XrSession session,
    const XrFrameBeginInfo* frameBeginInfo);

XrResult XRAPI_CALL gaussian_xrWaitFrame(
    XrSession session,
    const XrFrameWaitInfo* frameWaitInfo,
    XrFrameState* frameState);

// ============================================
// Initialization
// ============================================
bool InitializeDispatch(XrInstance instance, PFN_xrGetInstanceProcAddr getProcAddr);
void ShutdownDispatch();

}  // namespace gaussian
