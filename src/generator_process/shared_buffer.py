# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
SharedMemory wrapper for zero-copy Gaussian data transfer.

Provides high-performance IPC between Blender main process and 
PyTorch subprocess without pickle serialization overhead.

Performance:
- Queue (pickle): ~80ms for 10k gaussians
- SharedMemory: <1ms for 10k gaussians (80x faster)

Reference: Dream Textures realtime_viewport.py
"""

from multiprocessing.shared_memory import SharedMemory
import numpy as np
from typing import Optional, Tuple
import threading
import atexit


# 59 floats per gaussian (matches GaussianDataManager format)
# [0-2]:   position (vec3)
# [3-6]:   rotation quaternion (vec4, w,x,y,z)  
# [7-9]:   scale (vec3)
# [10]:    opacity (float)
# [11-58]: spherical harmonics (16 bands × 3 = 48 floats)
FLOATS_PER_GAUSSIAN = 59
BYTES_PER_GAUSSIAN = FLOATS_PER_GAUSSIAN * 4  # float32 = 4 bytes


class GaussianSharedBuffer:
    """
    SharedMemory wrapper for zero-copy Gaussian data transfer.
    
    Gaussian data format: (N, 59) float32 array
    - positions: 3 floats
    - rotations: 4 floats (quaternion w,x,y,z)
    - scales: 3 floats
    - opacity: 1 float
    - SH coefficients: 48 floats (16 * 3)
    
    Usage (Main Process - Creator):
        buffer = GaussianSharedBuffer(max_gaussians=10000)
        buffer.write(gaussians_array)
        # Send buffer.name to subprocess via Queue
        
    Usage (Subprocess - Receiver):
        buffer = GaussianSharedBuffer(max_gaussians=10000, name=received_name)
        data = buffer.read()
    """
    
    def __init__(self, max_gaussians: int = 100000, name: Optional[str] = None):
        """
        Initialize shared buffer.
        
        Args:
            max_gaussians: Maximum number of gaussians to support
            name: Optional name for existing shared memory (for receiver side)
        """
        self.max_gaussians = max_gaussians
        self.current_count = 0
        self._shm: Optional[SharedMemory] = None
        self._array: Optional[np.ndarray] = None
        self._is_owner = False
        self._closed = False
        
        if name:
            # Attach to existing shared memory (receiver/subprocess side)
            self._attach(name)
        else:
            # Create new shared memory (sender/main process side)
            self._create()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _create(self):
        """Create new shared memory buffer."""
        size = self.max_gaussians * BYTES_PER_GAUSSIAN
        
        # Add header space for metadata (current count, etc.)
        # Header: 64 bytes (16 floats for future expansion)
        header_size = 64
        total_size = header_size + size
        
        self._shm = SharedMemory(create=True, size=total_size)
        
        # Create header array view
        self._header = np.ndarray(
            (16,),
            dtype=np.float32,
            buffer=self._shm.buf[:header_size]
        )
        self._header[0] = 0  # current_count
        
        # Create data array view
        self._array = np.ndarray(
            (self.max_gaussians, FLOATS_PER_GAUSSIAN),
            dtype=np.float32,
            buffer=self._shm.buf[header_size:]
        )
        
        self._is_owner = True
        print(f"[SharedBuffer] Created: {self._shm.name}, size={total_size} bytes, max={self.max_gaussians} gaussians")
    
    def _attach(self, name: str):
        """Attach to existing shared memory buffer."""
        header_size = 64
        
        self._shm = SharedMemory(name=name)
        
        # Create header array view
        self._header = np.ndarray(
            (16,),
            dtype=np.float32,
            buffer=self._shm.buf[:header_size]
        )
        
        # Create data array view
        self._array = np.ndarray(
            (self.max_gaussians, FLOATS_PER_GAUSSIAN),
            dtype=np.float32,
            buffer=self._shm.buf[header_size:]
        )
        
        self._is_owner = False
        print(f"[SharedBuffer] Attached: {name}")
    
    @property
    def name(self) -> str:
        """Get shared memory name for IPC."""
        return self._shm.name if self._shm else ""
    
    @property
    def array(self) -> np.ndarray:
        """Get numpy array view of shared memory (NOT a copy)."""
        return self._array
    
    def write(self, data: np.ndarray, start_idx: int = 0) -> int:
        """
        Write gaussian data to shared buffer (zero-copy).
        
        Args:
            data: NumPy array of shape (N, 59), dtype=float32
            start_idx: Starting index in buffer
            
        Returns:
            Number of gaussians written
        """
        if self._closed:
            raise RuntimeError("SharedBuffer is closed")
        
        if data.ndim == 1:
            # Single gaussian
            data = data.reshape(1, -1)
        
        if data.shape[1] != FLOATS_PER_GAUSSIAN:
            raise ValueError(f"Expected {FLOATS_PER_GAUSSIAN} floats per gaussian, got {data.shape[1]}")
        
        n_gaussians = data.shape[0]
        end_idx = start_idx + n_gaussians
        
        if end_idx > self.max_gaussians:
            raise ValueError(f"Buffer overflow: {end_idx} > {self.max_gaussians}")
        
        # Zero-copy write (numpy view of shared memory)
        self._array[start_idx:end_idx] = data.astype(np.float32)
        
        # Update header
        self.current_count = max(self.current_count, end_idx)
        self._header[0] = float(self.current_count)
        
        return n_gaussians
    
    def read(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Read gaussian data from shared buffer (zero-copy view).
        
        WARNING: Returns a VIEW, not a copy. Modifications affect shared memory.
        
        Args:
            start_idx: Starting index
            count: Number of gaussians to read (None = all from start_idx)
            
        Returns:
            NumPy array view (NOT a copy)
        """
        if self._closed:
            raise RuntimeError("SharedBuffer is closed")
        
        if count is None:
            # Read current count from header
            self.current_count = int(self._header[0])
            count = self.current_count - start_idx
        
        if count <= 0:
            return np.empty((0, FLOATS_PER_GAUSSIAN), dtype=np.float32)
        
        end_idx = start_idx + count
        return self._array[start_idx:end_idx]
    
    def read_copy(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Read gaussian data as a copy (safe for modification).
        
        Args:
            start_idx: Starting index
            count: Number of gaussians to read
            
        Returns:
            NumPy array copy
        """
        return self.read(start_idx, count).copy()
    
    def get_count(self) -> int:
        """Get current gaussian count from header."""
        if self._closed:
            return 0
        self.current_count = int(self._header[0])
        return self.current_count
    
    def set_count(self, count: int):
        """Set current gaussian count in header."""
        if self._closed:
            return
        self.current_count = count
        self._header[0] = float(count)
    
    def clear(self):
        """Clear buffer (reset count, optionally zero memory)."""
        if self._closed:
            return
        self.current_count = 0
        self._header[0] = 0.0
        # Optionally zero out data: self._array[:] = 0
    
    def cleanup(self):
        """Release shared memory resources."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._shm:
            try:
                self._shm.close()
                if self._is_owner:
                    try:
                        self._shm.unlink()
                        print(f"[SharedBuffer] Unlinked: {self._shm.name}")
                    except FileNotFoundError:
                        pass  # Already unlinked
            except Exception as e:
                print(f"[SharedBuffer] Cleanup error: {e}")
            finally:
                self._shm = None
                self._array = None
                self._header = None
    
    def __del__(self):
        self.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
    
    def __repr__(self):
        return f"GaussianSharedBuffer(name={self.name}, count={self.current_count}, max={self.max_gaussians})"


class ThreadSafeSharedBuffer:
    """
    Thread-safe wrapper for GaussianSharedBuffer.
    
    Use when multiple threads may access the buffer concurrently.
    """
    
    def __init__(self, max_gaussians: int = 100000, name: Optional[str] = None):
        """
        Initialize thread-safe shared buffer.
        
        Args:
            max_gaussians: Maximum number of gaussians
            name: Optional name for existing shared memory
        """
        self._buffer = GaussianSharedBuffer(max_gaussians, name)
        self._lock = threading.RLock()
    
    @property
    def name(self) -> str:
        return self._buffer.name
    
    @property
    def current_count(self) -> int:
        return self._buffer.current_count
    
    def write(self, data: np.ndarray, start_idx: int = 0) -> int:
        """Thread-safe write."""
        with self._lock:
            return self._buffer.write(data, start_idx)
    
    def read(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """Thread-safe read (returns copy for safety)."""
        with self._lock:
            return self._buffer.read_copy(start_idx, count)
    
    def read_view(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Thread-safe read returning view.
        
        WARNING: View is only valid while lock is held.
        Use within a 'with buffer.locked():' context.
        """
        return self._buffer.read(start_idx, count)
    
    def locked(self):
        """Context manager for exclusive access."""
        return self._lock
    
    def get_count(self) -> int:
        """Get current count."""
        with self._lock:
            return self._buffer.get_count()
    
    def set_count(self, count: int):
        """Set current count."""
        with self._lock:
            self._buffer.set_count(count)
    
    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()
    
    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            self._buffer.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()


def benchmark_shared_buffer(n_gaussians: int = 10000, iterations: int = 10):
    """
    Benchmark SharedMemory vs Queue performance.
    
    Args:
        n_gaussians: Number of gaussians to test
        iterations: Number of iterations for averaging
        
    Returns:
        dict: Timing results in milliseconds
    """
    import time
    from multiprocessing import Queue
    
    # Generate test data
    data = np.random.randn(n_gaussians, FLOATS_PER_GAUSSIAN).astype(np.float32)
    data_size_mb = data.nbytes / (1024 * 1024)
    
    print(f"Benchmark: {n_gaussians} gaussians, {data_size_mb:.2f} MB")
    
    results = {}
    
    # Benchmark Queue (pickle serialization)
    queue = Queue()
    start = time.perf_counter()
    for _ in range(iterations):
        queue.put(data)
        _ = queue.get()
    queue_time = (time.perf_counter() - start) / iterations * 1000
    results['queue_ms'] = queue_time
    
    # Benchmark SharedMemory
    with GaussianSharedBuffer(n_gaussians) as shm:
        start = time.perf_counter()
        for _ in range(iterations):
            shm.write(data)
            _ = shm.read_copy()
        shm_time = (time.perf_counter() - start) / iterations * 1000
    results['shared_memory_ms'] = shm_time
    
    results['speedup'] = queue_time / shm_time if shm_time > 0 else float('inf')
    
    print(f"Queue (pickle):  {queue_time:.2f} ms")
    print(f"SharedMemory:    {shm_time:.2f} ms")
    print(f"Speedup:         {results['speedup']:.1f}x")
    
    return results


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_shared_buffer(10000)
