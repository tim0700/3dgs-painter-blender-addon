#pragma once

#include "gaussian_data.h"
#include <Windows.h>
#include <memory>
#include <optional>

namespace gaussian {

// ============================================
// Shared Memory Reader (DLL Side)
// ============================================
class SharedMemoryReader {
public:
    SharedMemoryReader();
    ~SharedMemoryReader();

    // Non-copyable
    SharedMemoryReader(const SharedMemoryReader&) = delete;
    SharedMemoryReader& operator=(const SharedMemoryReader&) = delete;

    // Open/Close
    bool Open();
    void Close();
    bool IsOpen() const { return m_handle != nullptr; }

    // Read data
    std::optional<SharedMemoryHeader> ReadHeader();
    bool ReadGaussians(GaussianPrimitive* outBuffer, uint32_t maxCount, uint32_t* actualCount);
    
    // Get raw pointer (for direct access)
    const SharedMemoryBuffer* GetBuffer() const { return m_buffer; }

private:
    HANDLE m_handle = nullptr;
    SharedMemoryBuffer* m_buffer = nullptr;
    uint32_t m_lastFrameId = 0;
};

// ============================================
// Shared Memory Writer (Python Side - for reference)
// ============================================
// This is implemented in Python (vr_offscreen_renderer.py)
// The C++ version is for testing only

class SharedMemoryWriter {
public:
    SharedMemoryWriter();
    ~SharedMemoryWriter();

    bool Create();
    void Close();
    
    bool WriteHeader(const SharedMemoryHeader& header);
    bool WriteGaussians(const GaussianPrimitive* data, uint32_t count);

private:
    HANDLE m_handle = nullptr;
    SharedMemoryBuffer* m_buffer = nullptr;
};

}  // namespace gaussian
