# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
HybridDataSync: Manages data synchronization between NumPy, PyTorch, and GLSL.

This module provides utilities for the Hybrid rendering architecture:
- NumPy (npr_core scene data)
- PyTorch tensors (gsplat computation in subprocess)
- GLSL textures (viewport rendering)

The synchronization is designed for real-time painting performance.
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..viewport.gaussian_data import GaussianDataManager


# Constants matching GaussianDataManager
FLOATS_PER_GAUSSIAN = 59
SH_C0 = 0.28209479177387814


class HybridDataSync:
    """
    Manages data synchronization between NumPy, PyTorch, and GLSL.
    
    Data Flow:
    1. Painting: NumPy (SceneData) → GLSL (immediate viewport update)
    2. Computation: NumPy → PyTorch (subprocess) → NumPy → GLSL
    
    Attributes:
        numpy_buffer: Cached NumPy array for incremental updates
        texture_manager: Reference to GaussianDataManager
    """
    
    def __init__(self, texture_manager: Optional["GaussianDataManager"] = None):
        """
        Initialize HybridDataSync.
        
        Args:
            texture_manager: GaussianDataManager for GLSL texture updates
        """
        self.numpy_buffer: Optional[np.ndarray] = None
        self.texture_manager = texture_manager
        self._buffer_count = 0
    
    def set_texture_manager(self, texture_manager: "GaussianDataManager"):
        """Set the texture manager for GLSL updates."""
        self.texture_manager = texture_manager
    
    # =========================================================================
    # SceneData ↔ Packed 59-float format conversions
    # =========================================================================
    
    def pack_scene_data(self, scene_data) -> np.ndarray:
        """
        Pack SceneData into 59-float stride format for GPU.
        
        Args:
            scene_data: SceneData with arrays (positions, rotations, scales, colors, opacities)
            
        Returns:
            np.ndarray: Shape (N, 59), dtype float32
        """
        N = scene_data.count
        if N == 0:
            return np.empty((0, FLOATS_PER_GAUSSIAN), dtype=np.float32)
        
        data = np.zeros((N, FLOATS_PER_GAUSSIAN), dtype=np.float32)
        
        # Position [0-2]
        data[:, 0:3] = scene_data.positions
        
        # Rotation [3-6] (quaternion w, x, y, z)
        # SceneData stores as (x, y, z, w), need to reorder to (w, x, y, z)
        data[:, 3] = scene_data.rotations[:, 3]  # w
        data[:, 4:7] = scene_data.rotations[:, 0:3]  # x, y, z
        
        # Scale [7-9]
        data[:, 7:10] = scene_data.scales
        
        # Opacity [10]
        data[:, 10] = scene_data.opacities
        
        # Spherical Harmonics [11-58]
        # For degree 0: Use color as base SH coefficient
        # color ≈ SH_C0 * sh_coeff + 0.5
        # Inverse: sh_coeff = (color - 0.5) / SH_C0
        data[:, 11:14] = (scene_data.colors - 0.5) / SH_C0
        # Remaining SH coefficients [14-58] are zero
        
        return data
    
    def unpack_to_scene_data(self, packed_data: np.ndarray, scene_data):
        """
        Unpack 59-float format back to SceneData.
        
        Args:
            packed_data: np.ndarray shape (N, 59)
            scene_data: SceneData to update (modified in place)
        """
        N = packed_data.shape[0]
        if N == 0:
            scene_data.clear()
            return
        
        # Position [0-2]
        scene_data.positions = packed_data[:, 0:3].copy()
        
        # Rotation [3-6] (w, x, y, z) → (x, y, z, w)
        scene_data.rotations = np.zeros((N, 4), dtype=np.float32)
        scene_data.rotations[:, 0:3] = packed_data[:, 4:7]  # x, y, z
        scene_data.rotations[:, 3] = packed_data[:, 3]  # w
        
        # Scale [7-9]
        scene_data.scales = packed_data[:, 7:10].copy()
        
        # Opacity [10]
        scene_data.opacities = packed_data[:, 10].copy()
        
        # Color from SH [11-14]
        scene_data.colors = (packed_data[:, 11:14] * SH_C0 + 0.5).clip(0, 1)
        
        scene_data.count = N
    
    # =========================================================================
    # GLSL Texture Updates
    # =========================================================================
    
    def sync_to_glsl(self, scene_data, full_update: bool = True):
        """
        Sync SceneData to GLSL texture.
        
        Args:
            scene_data: SceneData to sync
            full_update: If True, rebuild entire texture. If False, incremental.
        """
        if self.texture_manager is None:
            return
        
        self.texture_manager.update_from_scene_data(scene_data)
    
    def sync_range_to_glsl(self, packed_data: np.ndarray, start_idx: int, end_idx: int):
        """
        Sync a range of packed data to GLSL texture (incremental update).
        
        Args:
            packed_data: np.ndarray shape (count, 59) for the range
            start_idx: Starting gaussian index
            end_idx: Ending gaussian index (exclusive)
        """
        if self.texture_manager is None:
            return
        
        # Update numpy buffer cache
        if self.numpy_buffer is None or self.numpy_buffer.shape[0] < end_idx:
            # Grow buffer if needed
            new_size = max(end_idx, self._buffer_count * 2, 1000)
            new_buffer = np.zeros((new_size, FLOATS_PER_GAUSSIAN), dtype=np.float32)
            if self.numpy_buffer is not None:
                new_buffer[:self.numpy_buffer.shape[0]] = self.numpy_buffer
            self.numpy_buffer = new_buffer
        
        # Update range
        count = end_idx - start_idx
        self.numpy_buffer[start_idx:end_idx] = packed_data[:count]
        self._buffer_count = max(self._buffer_count, end_idx)
        
        # Update texture (partial update not directly supported, so full re-upload)
        # In a more optimized version, we would use glTexSubImage2D
        self.texture_manager._upload_to_texture(self.numpy_buffer[:self._buffer_count])
    
    # =========================================================================
    # SharedMemory Integration
    # =========================================================================
    
    def prepare_for_subprocess(self, scene_data) -> np.ndarray:
        """
        Prepare SceneData for transfer to subprocess via SharedMemory.
        
        Args:
            scene_data: SceneData to prepare
            
        Returns:
            np.ndarray: Packed 59-float format ready for SharedMemory.write()
        """
        return self.pack_scene_data(scene_data)
    
    def receive_from_subprocess(self, packed_data: np.ndarray, scene_data):
        """
        Receive processed data from subprocess and update SceneData.
        
        Args:
            packed_data: np.ndarray from SharedMemory.read()
            scene_data: SceneData to update
        """
        self.unpack_to_scene_data(packed_data, scene_data)
    
    # =========================================================================
    # Benchmarking
    # =========================================================================
    
    def benchmark_sync(self, n_gaussians: int = 10000) -> dict:
        """
        Benchmark synchronization overhead.
        
        Args:
            n_gaussians: Number of gaussians to test
            
        Returns:
            dict: Timing results in milliseconds
        """
        import time
        
        results = {}
        
        # Create mock scene data
        from ..npr_core.scene_data import SceneData
        scene_data = SceneData()
        
        # Generate random data
        positions = np.random.randn(n_gaussians, 3).astype(np.float32)
        rotations = np.random.randn(n_gaussians, 4).astype(np.float32)
        rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)  # Normalize quaternions
        scales = np.abs(np.random.randn(n_gaussians, 3).astype(np.float32)) * 0.1
        colors = np.random.rand(n_gaussians, 3).astype(np.float32)
        opacities = np.random.rand(n_gaussians).astype(np.float32)
        
        scene_data.add_gaussians_batch(positions, rotations, scales, colors, opacities)
        
        # Benchmark packing
        start = time.perf_counter()
        packed = self.pack_scene_data(scene_data)
        results['pack_ms'] = (time.perf_counter() - start) * 1000
        
        # Benchmark unpacking
        scene_data_copy = SceneData()
        start = time.perf_counter()
        self.unpack_to_scene_data(packed, scene_data_copy)
        results['unpack_ms'] = (time.perf_counter() - start) * 1000
        
        results['n_gaussians'] = n_gaussians
        results['data_size_mb'] = packed.nbytes / (1024 * 1024)
        
        print(f"HybridDataSync Benchmark ({n_gaussians} gaussians, {results['data_size_mb']:.2f} MB):")
        print(f"  Pack:   {results['pack_ms']:.2f} ms")
        print(f"  Unpack: {results['unpack_ms']:.2f} ms")
        
        return results


class HybridIPCManager:
    """
    Manages Hybrid Queue + SharedMemory IPC with automatic fallback.
    
    Usage:
        manager = HybridIPCManager()
        future = manager.send_gaussians(scene_data)
        result = future.result()
    """
    
    def __init__(self):
        """Initialize IPC manager."""
        self._shared_buffer = None
        self._subprocess_ready = False
        self._use_shared_memory = True
        self._data_sync = HybridDataSync()
    
    def setup(self, max_gaussians: int = 100000):
        """
        Setup SharedMemory buffer and subprocess.
        
        Args:
            max_gaussians: Maximum gaussians to support
        """
        from .shared_buffer import GaussianSharedBuffer
        from . import NPRGenerator
        
        try:
            # Create shared buffer in main process
            self._shared_buffer = GaussianSharedBuffer(max_gaussians=max_gaussians)
            
            # Tell subprocess to attach to buffer
            generator = NPRGenerator.shared()
            future = generator.setup_shared_buffer(
                buffer_name=self._shared_buffer.name,
                max_gaussians=max_gaussians
            )
            
            # Wait for subprocess to attach
            result = future.result()
            
            if result.get('success'):
                self._subprocess_ready = True
                print(f"[HybridIPC] SharedMemory setup complete: {self._shared_buffer.name}")
            else:
                print(f"[HybridIPC] Subprocess setup failed: {result.get('error')}")
                self._use_shared_memory = False
                
        except Exception as e:
            print(f"[HybridIPC] SharedMemory setup failed: {e}, using Queue fallback")
            self._use_shared_memory = False
    
    def send_gaussians(self, scene_data, start_idx: int = 0):
        """
        Send gaussian data to subprocess.
        
        Uses SharedMemory if available, falls back to Queue.
        
        Args:
            scene_data: SceneData to send
            start_idx: Starting index in buffer
            
        Returns:
            Future that resolves when subprocess acknowledges
        """
        from . import NPRGenerator
        
        if self._use_shared_memory and self._shared_buffer and self._subprocess_ready:
            try:
                # Pack and write to shared memory (zero-copy)
                packed = self._data_sync.pack_scene_data(scene_data)
                count = self._shared_buffer.write(packed, start_idx)
                
                # Notify subprocess (just metadata via Queue)
                generator = NPRGenerator.shared()
                return generator.sync_gaussians_from_shared(start_idx, count)
                
            except Exception as e:
                print(f"[HybridIPC] SharedMemory send failed: {e}, falling back to Queue")
                self._use_shared_memory = False
        
        # Fallback: Send via Queue (pickle serialization)
        # This is slower but more reliable
        packed = self._data_sync.pack_scene_data(scene_data)
        # Queue fallback would need a separate method in NPRGenerator
        # For now, just return a dummy future
        from .future import Future
        future = Future()
        future.add_response({"warning": "Queue fallback not implemented"})
        future.set_done()
        return future
    
    def receive_gaussians(self, scene_data, start_idx: int = 0, count: int = None):
        """
        Receive gaussian data from subprocess.
        
        Args:
            scene_data: SceneData to update
            start_idx: Starting index in buffer
            count: Number of gaussians (None = all)
        """
        if self._use_shared_memory and self._shared_buffer:
            # Read from shared memory
            packed = self._shared_buffer.read_copy(start_idx, count)
            self._data_sync.unpack_to_scene_data(packed, scene_data)
        else:
            # Queue fallback would need implementation
            pass
    
    def cleanup(self):
        """Cleanup resources."""
        from . import NPRGenerator
        
        if self._subprocess_ready:
            try:
                generator = NPRGenerator.shared()
                generator.cleanup_shared_buffer()
            except:
                pass
        
        if self._shared_buffer:
            self._shared_buffer.cleanup()
            self._shared_buffer = None
        
        self._subprocess_ready = False


# Convenience function for quick benchmarking
def benchmark_hybrid_sync(n_gaussians: int = 10000):
    """Run HybridDataSync benchmark."""
    sync = HybridDataSync()
    return sync.benchmark_sync(n_gaussians)
