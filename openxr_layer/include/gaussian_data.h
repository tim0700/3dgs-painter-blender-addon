#pragma once

#include <cstdint>
#include <array>

namespace gaussian {

// ============================================
// Shared Memory Constants
// ============================================
constexpr const wchar_t* SHARED_MEMORY_NAME = L"Local\\3DGS_Gaussian_Data";
constexpr size_t MAX_GAUSSIANS = 100000;
constexpr size_t SHARED_MEMORY_SIZE = sizeof(uint32_t) + MAX_GAUSSIANS * 56;  // count + N * data

// ============================================
// Single Gaussian Data (56 bytes)
// ============================================
struct alignas(4) GaussianPrimitive {
    // Position (12 bytes)
    float position[3];
    
    // Color RGBA (16 bytes)
    float color[4];
    
    // Scale (12 bytes)
    float scale[3];
    
    // Rotation quaternion (16 bytes)
    float rotation[4];
};

static_assert(sizeof(GaussianPrimitive) == 56, "GaussianPrimitive size mismatch");

// ============================================
// Shared Memory Header
// ============================================
struct SharedMemoryHeader {
    uint32_t magic;              // 0x3DGS
    uint32_t version;            // 1
    uint32_t frame_id;           // Incremental frame counter
    uint32_t gaussian_count;     // Number of valid gaussians
    uint32_t flags;              // Reserved
    float view_matrix[16];       // Optional: Blender's current view
    float proj_matrix[16];       // Optional: Blender's current projection
};

// ============================================
// Complete Shared Memory Layout
// ============================================
struct SharedMemoryBuffer {
    SharedMemoryHeader header;
    GaussianPrimitive gaussians[MAX_GAUSSIANS];
};

// Magic number for validation
constexpr uint32_t MAGIC_NUMBER = 0x33444753;  // "3DGS" in little-endian

}  // namespace gaussian
