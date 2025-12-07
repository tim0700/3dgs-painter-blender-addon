cmake_minimum_required(VERSION 3.20)
project(GaussianOpenXRLayer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# OpenXR SDK
# ============================================
find_package(OpenXR REQUIRED)

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
    ${OpenXR_INCLUDE_DIRS}
)

target_link_libraries(gaussian_layer PRIVATE
    OpenXR::openxr_loader
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
