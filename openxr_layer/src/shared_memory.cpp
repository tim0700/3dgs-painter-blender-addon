/**
 * Shared Memory Implementation
 * 
 * Handles reading Gaussian data from Blender's shared memory buffer.
 */

#include "shared_memory.h"
#include <iostream>

namespace gaussian {

// ============================================
// Logging
// ============================================
static void LogShm(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    OutputDebugStringA("[GaussianSHM] ");
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
}

// ============================================
// SharedMemoryReader
// ============================================

SharedMemoryReader::SharedMemoryReader() = default;

SharedMemoryReader::~SharedMemoryReader() {
    Close();
}

bool SharedMemoryReader::Open() {
    if (m_handle != nullptr) {
        return true;  // Already open
    }
    
    // Try to open existing shared memory
    m_handle = OpenFileMappingW(
        FILE_MAP_READ,
        FALSE,
        SHARED_MEMORY_NAME
    );
    
    if (m_handle == nullptr) {
        DWORD error = GetLastError();
        if (error != ERROR_FILE_NOT_FOUND) {
            LogShm("OpenFileMapping failed: %lu", error);
        }
        return false;
    }
    
    // Map view
    m_buffer = static_cast<SharedMemoryBuffer*>(
        MapViewOfFile(
            m_handle,
            FILE_MAP_READ,
            0, 0,
            sizeof(SharedMemoryBuffer)
        )
    );
    
    if (m_buffer == nullptr) {
        LogShm("MapViewOfFile failed: %lu", GetLastError());
        CloseHandle(m_handle);
        m_handle = nullptr;
        return false;
    }
    
    LogShm("Shared memory opened successfully");
    return true;
}

void SharedMemoryReader::Close() {
    if (m_buffer) {
        UnmapViewOfFile(m_buffer);
        m_buffer = nullptr;
    }
    
    if (m_handle) {
        CloseHandle(m_handle);
        m_handle = nullptr;
    }
    
    m_lastFrameId = 0;
}

std::optional<SharedMemoryHeader> SharedMemoryReader::ReadHeader() {
    if (!m_buffer) {
        return std::nullopt;
    }
    
    // Validate magic number
    if (m_buffer->header.magic != MAGIC_NUMBER) {
        return std::nullopt;
    }
    
    // Check for new data
    if (m_buffer->header.frame_id == m_lastFrameId) {
        // Same frame, no new data
        return std::nullopt;
    }
    
    m_lastFrameId = m_buffer->header.frame_id;
    return m_buffer->header;
}

bool SharedMemoryReader::ReadGaussians(
    GaussianPrimitive* outBuffer, 
    uint32_t maxCount, 
    uint32_t* actualCount)
{
    if (!m_buffer || !outBuffer || !actualCount) {
        return false;
    }
    
    uint32_t count = std::min(m_buffer->header.gaussian_count, maxCount);
    count = std::min(count, static_cast<uint32_t>(MAX_GAUSSIANS));
    
    memcpy(outBuffer, m_buffer->gaussians, count * sizeof(GaussianPrimitive));
    *actualCount = count;
    
    return true;
}

// ============================================
// SharedMemoryWriter (for testing)
// ============================================

SharedMemoryWriter::SharedMemoryWriter() = default;

SharedMemoryWriter::~SharedMemoryWriter() {
    Close();
}

bool SharedMemoryWriter::Create() {
    if (m_handle != nullptr) {
        return true;
    }
    
    // Create shared memory
    m_handle = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        sizeof(SharedMemoryBuffer),
        SHARED_MEMORY_NAME
    );
    
    if (m_handle == nullptr) {
        LogShm("CreateFileMapping failed: %lu", GetLastError());
        return false;
    }
    
    // Map view
    m_buffer = static_cast<SharedMemoryBuffer*>(
        MapViewOfFile(
            m_handle,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            sizeof(SharedMemoryBuffer)
        )
    );
    
    if (m_buffer == nullptr) {
        LogShm("MapViewOfFile failed: %lu", GetLastError());
        CloseHandle(m_handle);
        m_handle = nullptr;
        return false;
    }
    
    // Initialize header
    memset(m_buffer, 0, sizeof(SharedMemoryBuffer));
    m_buffer->header.magic = MAGIC_NUMBER;
    m_buffer->header.version = 1;
    
    LogShm("Shared memory created successfully");
    return true;
}

void SharedMemoryWriter::Close() {
    if (m_buffer) {
        UnmapViewOfFile(m_buffer);
        m_buffer = nullptr;
    }
    
    if (m_handle) {
        CloseHandle(m_handle);
        m_handle = nullptr;
    }
}

bool SharedMemoryWriter::WriteHeader(const SharedMemoryHeader& header) {
    if (!m_buffer) {
        return false;
    }
    
    m_buffer->header = header;
    m_buffer->header.magic = MAGIC_NUMBER;  // Ensure magic is correct
    return true;
}

bool SharedMemoryWriter::WriteGaussians(const GaussianPrimitive* data, uint32_t count) {
    if (!m_buffer || !data) {
        return false;
    }
    
    count = std::min(count, static_cast<uint32_t>(MAX_GAUSSIANS));
    memcpy(m_buffer->gaussians, data, count * sizeof(GaussianPrimitive));
    m_buffer->header.gaussian_count = count;
    m_buffer->header.frame_id++;
    
    return true;
}

}  // namespace gaussian
