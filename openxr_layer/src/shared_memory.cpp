/**
 * Shared Memory (SIMPLIFIED)
 */

#include <Windows.h>
#include "shared_memory.h"

namespace gaussian {

// ============================================
// Logging
// ============================================
static void LogShm(const char* msg) {
    OutputDebugStringA("[GaussianSHM] ");
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
}

// ============================================
// SharedMemoryReader
// ============================================

SharedMemoryReader::SharedMemoryReader() = default;
SharedMemoryReader::~SharedMemoryReader() { Close(); }

bool SharedMemoryReader::Open() {
    if (m_handle) return true;
    
    m_handle = OpenFileMappingW(FILE_MAP_READ, FALSE, SHARED_MEMORY_NAME);
    if (!m_handle) return false;
    
    m_buffer = static_cast<SharedMemoryBuffer*>(
        MapViewOfFile(m_handle, FILE_MAP_READ, 0, 0, sizeof(SharedMemoryBuffer)));
    
    if (!m_buffer) {
        CloseHandle(m_handle);
        m_handle = nullptr;
        return false;
    }
    
    LogShm("Opened");
    return true;
}

void SharedMemoryReader::Close() {
    if (m_buffer) { UnmapViewOfFile(m_buffer); m_buffer = nullptr; }
    if (m_handle) { CloseHandle(m_handle); m_handle = nullptr; }
    m_lastFrameId = 0;
}

std::optional<SharedMemoryHeader> SharedMemoryReader::ReadHeader() {
    if (!m_buffer || m_buffer->header.magic != MAGIC_NUMBER) return std::nullopt;
    if (m_buffer->header.frame_id == m_lastFrameId) return std::nullopt;
    m_lastFrameId = m_buffer->header.frame_id;
    return m_buffer->header;
}

bool SharedMemoryReader::ReadGaussians(GaussianPrimitive* out, uint32_t max, uint32_t* count) {
    if (!m_buffer || !out || !count) return false;
    uint32_t n = (std::min)(m_buffer->header.gaussian_count, max);
    n = (std::min)(n, static_cast<uint32_t>(MAX_GAUSSIANS));
    memcpy(out, m_buffer->gaussians, n * sizeof(GaussianPrimitive));
    *count = n;
    return true;
}

// ============================================
// SharedMemoryWriter
// ============================================

SharedMemoryWriter::SharedMemoryWriter() = default;
SharedMemoryWriter::~SharedMemoryWriter() { Close(); }

bool SharedMemoryWriter::Create() {
    if (m_handle) return true;
    
    m_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(SharedMemoryBuffer), SHARED_MEMORY_NAME);
    if (!m_handle) return false;
    
    m_buffer = static_cast<SharedMemoryBuffer*>(
        MapViewOfFile(m_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryBuffer)));
    
    if (!m_buffer) {
        CloseHandle(m_handle);
        m_handle = nullptr;
        return false;
    }
    
    memset(m_buffer, 0, sizeof(SharedMemoryBuffer));
    m_buffer->header.magic = MAGIC_NUMBER;
    m_buffer->header.version = 1;
    return true;
}

void SharedMemoryWriter::Close() {
    if (m_buffer) { UnmapViewOfFile(m_buffer); m_buffer = nullptr; }
    if (m_handle) { CloseHandle(m_handle); m_handle = nullptr; }
}

bool SharedMemoryWriter::WriteHeader(const SharedMemoryHeader& header) {
    if (!m_buffer) return false;
    m_buffer->header = header;
    m_buffer->header.magic = MAGIC_NUMBER;
    return true;
}

bool SharedMemoryWriter::WriteGaussians(const GaussianPrimitive* data, uint32_t count) {
    if (!m_buffer || !data) return false;
    count = (std::min)(count, static_cast<uint32_t>(MAX_GAUSSIANS));
    memcpy(m_buffer->gaussians, data, count * sizeof(GaussianPrimitive));
    m_buffer->header.gaussian_count = count;
    m_buffer->header.frame_id++;
    return true;
}

}  // namespace gaussian
