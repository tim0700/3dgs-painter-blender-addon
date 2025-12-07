cmake_minimum_required(VERSION 3.20)
project(GaussianOpenXRLayer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# OpenXR SDK (Local)
# ============================================
set(OPENXR_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenXR-SDK)

# Build OpenXR loader - DISABLE ALL GRAPHICS APIs to avoid platform header issues
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_CONFORMANCE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_API_LAYERS OFF CACHE BOOL "" FORCE)

# Disable D3D11/12 detection in OpenXR SDK
set(BUILD_WITH_SYSTEM_JSONCPP OFF CACHE BOOL "" FORCE)
set(OPENXR_DEBUG_POSTFIX "" CACHE STRING "" FORCE)

add_subdirectory(${OPENXR_SDK_DIR} openxr_sdk_build)

# Remove D3D11/D3D12 definitions from our target (we'll add them later in Phase 3)
remove_definitions(-DXR_USE_GRAPHICS_API_D3D11)
remove_definitions(-DXR_USE_GRAPHICS_API_D3D12)

# ============================================
# Windows SDK (DirectX)
# ============================================
if(WIN32)
    set(D3D11_LIBRARIES d3d11 dxgi d3dcompiler)
endif()

# ============================================
# Source Files
# ============================================
set(SOURCES
    src/main.cpp
    src/xr_dispatch.cpp
    src/composition_layer.cpp
    src/shared_memory.cpp
    src/gaussian_renderer.cpp
    src/gpu_context.cpp
)

set(HEADERS
    include/gaussian_data.h
    include/xr_dispatch.h
    include/shared_memory.h
)

# ============================================
# Create DLL
# ============================================
add_library(gaussian_layer SHARED ${SOURCES} ${HEADERS})

target_include_directories(gaussian_layer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENXR_SDK_DIR}/include
)

target_link_libraries(gaussian_layer PRIVATE
    openxr_loader
    ${D3D11_LIBRARIES}
)

# ============================================
# Export Functions
# ============================================
set_target_properties(gaussian_layer PROPERTIES
    PREFIX ""  # No "lib" prefix
    OUTPUT_NAME "gaussian_layer"
)

# ============================================
# Copy Manifest
# ============================================
add_custom_command(TARGET gaussian_layer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/manifest/gaussian_layer.json
        $<TARGET_FILE_DIR:gaussian_layer>/gaussian_layer.json
)

# ============================================
# Install
# ============================================
install(TARGETS gaussian_layer RUNTIME DESTINATION bin)
install(FILES manifest/gaussian_layer.json DESTINATION bin)
