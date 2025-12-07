/**
 * OpenXR API Layer - DLL Entry Point
 * 
 * This is the main entry point for the Gaussian Splatting OpenXR API Layer.
 * It implements xrNegotiateLoaderApiLayerInterface which is called by the
 * OpenXR loader to initialize the layer.
 */

#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D11

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "xr_dispatch.h"
#include "shared_memory.h"

#include <Windows.h>
#include <iostream>
#include <fstream>

// ============================================
// Logging
// ============================================
namespace {
    std::ofstream g_logFile;
    
    void Log(const char* format, ...) {
        if (!g_logFile.is_open()) {
            char path[MAX_PATH];
            GetTempPathA(MAX_PATH, path);
            std::string logPath = std::string(path) + "gaussian_layer.log";
            g_logFile.open(logPath, std::ios::app);
        }
        
        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        
        g_logFile << "[GaussianLayer] " << buffer << std::endl;
        g_logFile.flush();
        
        // Also output to debug console
        OutputDebugStringA("[GaussianLayer] ");
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }
}

// ============================================
// Layer Negotiation
// ============================================

// Forward declarations for hooked functions
XrResult XRAPI_CALL layer_xrGetInstanceProcAddr(
    XrInstance instance,
    const char* name,
    PFN_xrVoidFunction* function);

XrResult XRAPI_CALL layer_xrCreateApiLayerInstance(
    const XrInstanceCreateInfo* createInfo,
    const struct XrApiLayerCreateInfo* layerInfo,
    XrInstance* instance);

// ============================================
// Exported: Loader calls this to initialize layer
// ============================================
extern "C" __declspec(dllexport) XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo,
    const char* layerName,
    XrNegotiateApiLayerRequest* layerRequest)
{
    Log("xrNegotiateLoaderApiLayerInterface called");
    Log("  Layer name: %s", layerName);
    Log("  Loader min version: %d.%d", 
        XR_VERSION_MAJOR(loaderInfo->minInterfaceVersion),
        XR_VERSION_MINOR(loaderInfo->minInterfaceVersion));
    
    // Validate loader info
    if (loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
        loaderInfo->structVersion != XR_LOADER_INFO_STRUCT_VERSION ||
        loaderInfo->structSize != sizeof(XrNegotiateLoaderInfo)) {
        Log("ERROR: Invalid loader info structure");
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    
    // Validate layer request
    if (layerRequest->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
        layerRequest->structVersion != XR_API_LAYER_INFO_STRUCT_VERSION ||
        layerRequest->structSize != sizeof(XrNegotiateApiLayerRequest)) {
        Log("ERROR: Invalid layer request structure");
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    
    // Check version compatibility
    if (loaderInfo->minInterfaceVersion > XR_CURRENT_LOADER_API_LAYER_VERSION ||
        loaderInfo->maxInterfaceVersion < XR_CURRENT_LOADER_API_LAYER_VERSION) {
        Log("ERROR: Incompatible interface version");
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    
    // Fill in layer response
    layerRequest->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    layerRequest->layerApiVersion = XR_CURRENT_API_VERSION;
    layerRequest->getInstanceProcAddr = layer_xrGetInstanceProcAddr;
    layerRequest->createApiLayerInstance = layer_xrCreateApiLayerInstance;
    
    Log("Layer negotiation successful!");
    return XR_SUCCESS;
}

// ============================================
// Instance Creation
// ============================================
XrResult XRAPI_CALL layer_xrCreateApiLayerInstance(
    const XrInstanceCreateInfo* createInfo,
    const struct XrApiLayerCreateInfo* layerInfo,
    XrInstance* instance)
{
    Log("layer_xrCreateApiLayerInstance called");
    Log("  App name: %s", createInfo->applicationInfo.applicationName);
    
    // Get next layer's create function
    PFN_xrCreateApiLayerInstance nextCreate = layerInfo->nextInfo->nextCreateApiLayerInstance;
    
    // Create modified layer info for next layer
    XrApiLayerCreateInfo modifiedLayerInfo = *layerInfo;
    modifiedLayerInfo.nextInfo = layerInfo->nextInfo->next;
    
    // Call next layer
    XrResult result = nextCreate(createInfo, &modifiedLayerInfo, instance);
    if (XR_FAILED(result)) {
        Log("ERROR: Next layer instance creation failed: %d", result);
        return result;
    }
    
    Log("Instance created successfully");
    
    // Store instance and get function addresses
    gaussian::GetLayerState().instance = *instance;
    gaussian::InitializeDispatch(*instance, layerInfo->nextInfo->nextGetInstanceProcAddr);
    
    return XR_SUCCESS;
}

// ============================================
// Function Address Resolution
// ============================================
XrResult XRAPI_CALL layer_xrGetInstanceProcAddr(
    XrInstance instance,
    const char* name,
    PFN_xrVoidFunction* function)
{
    // Check for functions we want to intercept
    if (strcmp(name, "xrEndFrame") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(gaussian::gaussian_xrEndFrame);
        return XR_SUCCESS;
    }
    
    if (strcmp(name, "xrBeginFrame") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(gaussian::gaussian_xrBeginFrame);
        return XR_SUCCESS;
    }
    
    if (strcmp(name, "xrWaitFrame") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(gaussian::gaussian_xrWaitFrame);
        return XR_SUCCESS;
    }
    
    // For other functions, pass through to next layer
    auto& state = gaussian::GetLayerState();
    if (state.next_xrGetInstanceProcAddr) {
        return state.next_xrGetInstanceProcAddr(instance, name, function);
    }
    
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}

// ============================================
// DLL Entry Point
// ============================================
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved)
{
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            Log("DLL_PROCESS_ATTACH - Gaussian OpenXR Layer loaded");
            DisableThreadLibraryCalls(hModule);
            break;
            
        case DLL_PROCESS_DETACH:
            Log("DLL_PROCESS_DETACH - Gaussian OpenXR Layer unloaded");
            gaussian::ShutdownDispatch();
            if (g_logFile.is_open()) {
                g_logFile.close();
            }
            break;
    }
    return TRUE;
}
