# operators.py
# Operators for 3DGS Painter addon

import bpy
from bpy.types import Operator
from bpy.props import FloatProperty, FloatVectorProperty, BoolProperty, IntProperty
import threading
import numpy as np
from mathutils import Vector


# =============================================================================
# Raycasting and Input Helpers (Phase 4)
# =============================================================================

def raycast_mouse_to_surface(context, event):
    """
    Convert mouse coordinates to 3D surface position.
    
    Args:
        context: bpy.context
        event: Modal operator event
        
    Returns:
        tuple: (location: Vector, normal: Vector, hit: bool)
    """
    from bpy_extras import view3d_utils
    
    region = context.region
    rv3d = context.region_data
    
    if region is None or rv3d is None:
        return Vector((0, 0, 0)), Vector((0, 0, 1)), False
    
    coord = (event.mouse_region_x, event.mouse_region_y)
    
    # Get ray direction
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
    # Normalize view_vector for proper distance calculation
    view_vector_normalized = view_vector.normalized()
    
    # Raycast against scene objects with distance limit
    # Use view_layer.depsgraph for proper object evaluation
    depsgraph = context.view_layer.depsgraph
    result, location, normal, index, obj, matrix = context.scene.ray_cast(
        depsgraph,
        ray_origin,
        view_vector_normalized,
        distance=10000.0  # Maximum ray distance
    )
    
    if result:
        return location.copy(), normal.normalized(), True
    else:
        # No fallback - return None to indicate no valid surface hit
        return None, None, False


def get_tablet_pressure(event):
    """
    Get tablet pressure (0-1 range).
    
    Returns:
        float: pressure value, 1.0 if not using tablet
    """
    if hasattr(event, 'pressure') and event.pressure > 0:
        return event.pressure
    return 1.0


# =============================================================================
# Dependency Installation Operators
# =============================================================================

class THREEGDS_OT_install_dependencies(Operator):
    """Install missing Python packages for 3DGS Painter"""
    bl_idname = "threegds.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required Python packages (PyTorch, NumPy, etc.)"
    bl_options = {'REGISTER'}
    
    _timer = None
    _thread = None
    _installer = None
    _log_lines = []
    _finished = False
    _success = False
    
    @classmethod
    def poll(cls, context):
        # Check if not already installing
        try:
            prefs = context.preferences.addons[__package__].preferences
            return not prefs.is_installing
        except (KeyError, AttributeError):
            return True
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            # Update UI
            context.area.tag_redraw()
            
            # Check if installation finished
            if self._finished:
                # Update preferences with log
                try:
                    prefs = context.preferences.addons[__package__].preferences
                    prefs.install_log = '\n'.join(self._log_lines)
                    prefs.is_installing = False
                except (KeyError, AttributeError):
                    pass
                
                # Cleanup timer
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                
                if self._success:
                    self.report({'INFO'}, "Dependencies installed successfully! Please restart Blender.")
                    return {'FINISHED'}
                else:
                    self.report({'ERROR'}, "Some dependencies failed to install. Check the log.")
                    return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        from .npr_core.dependencies import get_missing_packages
        from .npr_core.installer import PackageInstaller, CUDAVersion
        
        # Check if there are missing packages
        missing = get_missing_packages()
        if not missing:
            self.report({'INFO'}, "All dependencies are already installed")
            return {'FINISHED'}
        
        # Get CUDA version preference
        try:
            prefs = context.preferences.addons[__package__].preferences
            cuda_pref = prefs.cuda_version
            prefs.is_installing = True
            prefs.install_log = ""
        except (KeyError, AttributeError):
            cuda_pref = 'AUTO'
        
        # Map preference to CUDAVersion enum
        cuda_version = None
        if cuda_pref != 'AUTO':
            cuda_map = {
                'cu124': CUDAVersion.CUDA_124,
                'cu121': CUDAVersion.CUDA_121,
                'cu118': CUDAVersion.CUDA_118,
                'cpu': CUDAVersion.CPU,
            }
            cuda_version = cuda_map.get(cuda_pref)
        
        # Initialize
        self._installer = PackageInstaller()
        self._log_lines = []
        self._finished = False
        self._success = False
        
        # Progress callback (thread-safe)
        def progress_callback(message):
            self._log_lines.append(message)
            print(f"[3DGS Painter] {message}")
        
        # Install in background thread
        def install_thread():
            self._success, failed = self._installer.install_all(
                cuda_version=cuda_version,
                progress_callback=progress_callback
            )
            self._finished = True
        
        self._thread = threading.Thread(target=install_thread, daemon=True)
        self._thread.start()
        
        # Setup timer for modal updates
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, "Installing dependencies... This may take several minutes.")
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        # Cleanup if cancelled
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
        
        try:
            prefs = context.preferences.addons[__package__].preferences
            prefs.is_installing = False
        except (KeyError, AttributeError):
            pass


class THREEGDS_OT_uninstall_dependencies(Operator):
    """Remove installed Python packages"""
    bl_idname = "threegds.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    bl_description = "Remove all installed Python packages for 3DGS Painter"
    bl_options = {'REGISTER'}
    
    def invoke(self, context, event):
        # Show confirmation dialog
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        from .npr_core.installer import PackageInstaller
        
        installer = PackageInstaller()
        
        log_lines = []
        def progress_callback(message):
            log_lines.append(message)
            print(f"[3DGS Painter] {message}")
        
        success = installer.uninstall_all(progress_callback)
        
        # Update log
        try:
            prefs = context.preferences.addons[__package__].preferences
            prefs.install_log = '\n'.join(log_lines)
        except (KeyError, AttributeError):
            pass
        
        if success:
            self.report({'INFO'}, "Dependencies uninstalled. Please restart Blender.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to uninstall dependencies")
            return {'CANCELLED'}


class THREEGDS_OT_check_dependencies(Operator):
    """Check for missing dependencies"""
    bl_idname = "threegds.check_dependencies"
    bl_label = "Check Dependencies"
    bl_description = "Check which Python packages are missing"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from .npr_core.dependencies import get_missing_packages, check_all_dependencies
        
        all_installed, missing, installed = check_all_dependencies()
        
        if all_installed:
            self.report({'INFO'}, f"All dependencies installed ({len(installed)} packages)")
        else:
            missing_names = [dep.name for dep in missing]
            self.report({'WARNING'}, f"Missing: {', '.join(missing_names)}")
        
        return {'FINISHED'}


class THREEGDS_OT_test_subprocess(Operator):
    """Test PyTorch loading in subprocess (bypasses TBB DLL conflict)"""
    bl_idname = "threegds.test_subprocess"
    bl_label = "Test Subprocess PyTorch"
    bl_description = "Test if PyTorch can be loaded in subprocess worker"
    bl_options = {'REGISTER'}
    
    _timer = None
    _future = None
    _test_type = 'torch_info'  # 'torch_info', 'cuda_test', 'dependencies'
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._future is None:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                self.report({'ERROR'}, "Future was lost")
                return {'CANCELLED'}
            
            if self._future.done:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                
                try:
                    result = self._future.result()
                    
                    if self._test_type == 'torch_info':
                        # Display torch info
                        torch_version = result.get('torch_version', 'N/A')
                        cuda_available = result.get('cuda_available', False)
                        cuda_version = result.get('cuda_version', 'N/A')
                        device_name = result.get('device_name', 'N/A')
                        device_count = result.get('device_count', 0)
                        
                        msg = f"PyTorch {torch_version}"
                        if cuda_available:
                            msg += f" | CUDA {cuda_version} | {device_name} ({device_count} GPU(s))"
                        else:
                            msg += " | CPU only"
                        
                        self.report({'INFO'}, msg)
                        print(f"[3DGS Painter] Subprocess PyTorch test SUCCESS:")
                        for key, value in result.items():
                            print(f"  {key}: {value}")
                    
                    elif self._test_type == 'cuda_test':
                        success = result.get('success', False)
                        compute_time = result.get('compute_time_ms', 0)
                        transfer_time = result.get('transfer_time_ms', 0)
                        
                        if success:
                            self.report({'INFO'}, f"CUDA compute: {compute_time:.2f}ms | Transfer: {transfer_time:.2f}ms")
                        else:
                            error = result.get('error', 'Unknown error')
                            self.report({'WARNING'}, f"CUDA test failed: {error}")
                    
                    elif self._test_type == 'dependencies':
                        available = result.get('available', {})
                        missing = result.get('missing', [])
                        
                        if missing:
                            self.report({'WARNING'}, f"Missing in subprocess: {', '.join(missing)}")
                        else:
                            available_list = [f"{k}={v}" for k, v in available.items() if v]
                            self.report({'INFO'}, f"All available: {', '.join(available_list[:5])}...")
                    
                    return {'FINISHED'}
                    
                except Exception as e:
                    self.report({'ERROR'}, f"Subprocess error: {e}")
                    print(f"[3DGS Painter] Subprocess error: {e}")
                    import traceback
                    traceback.print_exc()
                    return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        from .generator_process import NPRGenerator
        
        try:
            generator = NPRGenerator.shared()
            
            # Test PyTorch info (most comprehensive test)
            self._test_type = 'torch_info'
            self._future = generator.get_torch_info()
            
            # Setup timer for polling
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.1, window=context.window)
            wm.modal_handler_add(self)
            
            self.report({'INFO'}, "Testing subprocess PyTorch loading...")
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start subprocess: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)


class THREEGDS_OT_test_subprocess_cuda(Operator):
    """Test CUDA computation in subprocess"""
    bl_idname = "threegds.test_subprocess_cuda"
    bl_label = "Test Subprocess CUDA"
    bl_description = "Run a CUDA computation test in subprocess"
    bl_options = {'REGISTER'}
    
    _timer = None
    _future = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._future is None or self._future.done:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                
                if self._future is None:
                    self.report({'ERROR'}, "Future was lost")
                    return {'CANCELLED'}
                
                try:
                    result = self._future.result()
                    success = result.get('success', False)
                    
                    if success:
                        compute_time = result.get('compute_time_ms', 0)
                        transfer_time = result.get('transfer_time_ms', 0)
                        device = result.get('device', 'unknown')
                        self.report({'INFO'}, f"CUDA OK on {device} | Compute: {compute_time:.2f}ms | Transfer: {transfer_time:.2f}ms")
                    else:
                        error = result.get('error', 'Unknown error')
                        self.report({'WARNING'}, f"CUDA test failed: {error}")
                    
                    return {'FINISHED'}
                    
                except Exception as e:
                    self.report({'ERROR'}, f"Subprocess error: {e}")
                    return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        from .generator_process import NPRGenerator
        
        try:
            generator = NPRGenerator.shared()
            self._future = generator.test_cuda_computation()
            
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.1, window=context.window)
            wm.modal_handler_add(self)
            
            self.report({'INFO'}, "Running CUDA computation test...")
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start subprocess: {e}")
            return {'CANCELLED'}
    
    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)


class THREEGDS_OT_kill_subprocess(Operator):
    """Kill the subprocess worker"""
    bl_idname = "threegds.kill_subprocess"
    bl_label = "Kill Subprocess"
    bl_description = "Terminate the background PyTorch subprocess"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from .generator_process import kill_generator
        
        kill_generator()
        self.report({'INFO'}, "Subprocess terminated")
        return {'FINISHED'}


# =============================================================================
# Gaussian Painting Operators (Phase 4) - WorkSpaceTool Architecture
# =============================================================================

# Global scene data storage for painting session persistence
_paint_session_data = {
    'scene_data': None,
    'stroke_painter': None,
    'brush': None,
}


def get_or_create_paint_session(context):
    """Get or create painting session data (persists across strokes)."""
    from .npr_core.scene_data import SceneData
    from .npr_core.brush import BrushStamp, StrokePainter
    
    global _paint_session_data
    
    if _paint_session_data['scene_data'] is None:
        _paint_session_data['scene_data'] = SceneData()
    
    scene = context.scene
    
    # Create/update brush based on current settings
    brush = BrushStamp()
    brush_size = scene.npr_brush_size
    brush_opacity = scene.npr_brush_opacity
    brush_spacing = scene.npr_brush_spacing
    brush_color = scene.npr_brush_color
    brush_pattern = scene.npr_brush_pattern
    num_gaussians = scene.npr_brush_num_gaussians
    
    if brush_pattern == 'CIRCULAR':
        brush.create_circular_pattern(
            num_gaussians=num_gaussians,
            radius=0.1 * brush_size,
            gaussian_scale=0.02 * brush_size,
            opacity=brush_opacity
        )
    elif brush_pattern == 'LINE':
        brush.create_line_pattern(
            num_gaussians=num_gaussians,
            length=0.2 * brush_size,
            thickness=0.02 * brush_size,
            opacity=brush_opacity
        )
    elif brush_pattern == 'GRID':
        grid_size = int(np.sqrt(num_gaussians))
        brush.create_grid_pattern(
            grid_size=max(2, grid_size),
            spacing=0.04 * brush_size,
            gaussian_scale=0.015 * brush_size,
            opacity=brush_opacity
        )
    
    brush.apply_parameters(
        color=np.array(brush_color, dtype=np.float32),
        spacing=brush_spacing * brush_size * 0.2
    )
    
    _paint_session_data['brush'] = brush
    _paint_session_data['stroke_painter'] = StrokePainter(
        brush=brush,
        scene_gaussians=_paint_session_data['scene_data']
    )
    
    return _paint_session_data


def clear_paint_session():
    """Clear the paint session data."""
    global _paint_session_data
    _paint_session_data = {
        'scene_data': None,
        'stroke_painter': None,
        'brush': None,
    }


class THREEGDS_OT_GaussianPaintStroke(Operator):
    """Paint a single stroke with Gaussian Splats (invoked by WorkSpaceTool keymap)"""
    bl_idname = "threegds.gaussian_paint_stroke"
    bl_label = "Gaussian Paint Stroke"
    bl_description = "Paint a stroke with Gaussian splats"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Stroke mode property (for potential future erase functionality)
    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('ADD', "Add", "Add gaussians"),
            ('REMOVE', "Remove", "Remove gaussians (not implemented)"),
        ],
        default='ADD'
    )
    
    # Instance variables (initialized in invoke)
    # painting: bool
    # stroke_painter: StrokePainter
    # scene_data: SceneData  
    # brush: BrushStamp
    # viewport_renderer: GaussianViewportRenderer
    # _draw_handler: draw handler for brush preview
    
    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'
    
    def invoke(self, context, event):
        from .viewport.viewport_renderer import GaussianViewportRenderer
        
        # Get or create paint session
        session = get_or_create_paint_session(context)
        self.scene_data = session['scene_data']
        self.stroke_painter = session['stroke_painter']
        self.brush = session['brush']
        
        self.painting = False
        self._draw_handler = None
        
        # Get viewport renderer
        self.viewport_renderer = GaussianViewportRenderer.get_instance()
        if not self.viewport_renderer.enabled:
            self.viewport_renderer.register()
        
        # Register brush preview draw handler
        self._register_brush_preview(context)
        
        # Store initial mouse position for brush preview
        self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        
        # Start stroke immediately on invoke (tool keymap triggers on PRESS)
        location, normal, hit = raycast_mouse_to_surface(context, event)
        pressure = get_tablet_pressure(event)
        
        # Only start stroke if we hit a surface
        if not hit or location is None:
            # No surface hit - still enter modal but don't start painting yet
            self.painting = False
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        
        scene = context.scene
        self.painting = True
        
        # Scale brush by pressure
        effective_size = scene.npr_brush_size * (0.5 + 0.5 * pressure)
        self.brush.apply_parameters(
            color=np.array(scene.npr_brush_color, dtype=np.float32),
            size_multiplier=effective_size,
            global_opacity=scene.npr_brush_opacity * pressure
        )
        
        # Start stroke
        self.stroke_painter.start_stroke(
            position=np.array(location, dtype=np.float32),
            normal=np.array(normal, dtype=np.float32),
            enable_deformation=scene.npr_enable_deformation
        )
        
        # Update viewport
        self._sync_viewport()
        context.area.tag_redraw()
        
        # Set up modal handler for drag
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        scene = context.scene
        
        # Update mouse position for brush preview
        self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        context.area.tag_redraw()
        
        # Continue stroke on mouse move
        if event.type == 'MOUSEMOVE':
            location, normal, hit = raycast_mouse_to_surface(context, event)
            
            # If no surface hit, finish current stroke segment (if painting)
            if not hit or location is None:
                if self.painting:
                    # Finish the current stroke segment
                    self.stroke_painter.finish_stroke(
                        enable_deformation=scene.npr_enable_deformation,
                        enable_inpainting=False
                    )
                    self.painting = False
                return {'RUNNING_MODAL'}
            
            pressure = get_tablet_pressure(event)
            
            # If not painting yet but we hit a surface, start a new stroke
            if not self.painting:
                self.painting = True
                effective_size = scene.npr_brush_size * (0.5 + 0.5 * pressure)
                self.brush.apply_parameters(
                    color=np.array(scene.npr_brush_color, dtype=np.float32),
                    size_multiplier=effective_size,
                    global_opacity=scene.npr_brush_opacity * pressure
                )
                self.stroke_painter.start_stroke(
                    position=np.array(location, dtype=np.float32),
                    normal=np.array(normal, dtype=np.float32),
                    enable_deformation=scene.npr_enable_deformation
                )
                self._sync_viewport()
                return {'RUNNING_MODAL'}
            
            # Update brush parameters based on pressure
            effective_size = scene.npr_brush_size * (0.5 + 0.5 * pressure)
            self.brush.apply_parameters(
                size_multiplier=effective_size,
                global_opacity=scene.npr_brush_opacity * pressure
            )
            
            # Update stroke
            self.stroke_painter.update_stroke(
                position=np.array(location, dtype=np.float32),
                normal=np.array(normal, dtype=np.float32)
            )
            
            # Update viewport (incremental)
            self._sync_viewport()
            return {'RUNNING_MODAL'}
        
        # Finish stroke on left mouse release
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._finish_stroke(context)
            return {'FINISHED'}
        
        # Cancel on escape or right-click
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish_stroke(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _register_brush_preview(self, context):
        """Register draw handler for brush cursor preview."""
        import gpu
        from gpu_extras.presets import draw_circle_2d
        
        def draw_brush_cursor():
            if not hasattr(self, '_mouse_pos'):
                return
            
            scene = context.scene
            # Draw circle at mouse position
            # Size is approximate - would need proper 3D→2D projection for accuracy
            radius = scene.npr_brush_size * 20  # Rough pixel approximation
            color = (*scene.npr_brush_color, 0.5)  # RGBA with alpha
            
            gpu.state.blend_set('ALPHA')
            draw_circle_2d(self._mouse_pos, color, radius, segments=32)
            gpu.state.blend_set('NONE')
        
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_brush_cursor, (), 'WINDOW', 'POST_PIXEL'
        )
    
    def _unregister_brush_preview(self):
        """Unregister brush preview draw handler."""
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
    
    def _sync_viewport(self):
        """Sync scene data to viewport renderer."""
        if self.viewport_renderer and self.scene_data.count > 0:
            self.viewport_renderer.update_gaussians(scene_data=self.scene_data)
    
    def _finish_stroke(self, context):
        """Finish the current stroke."""
        scene = context.scene
        self.painting = False
        
        # Finish stroke (optionally with deformation)
        self.stroke_painter.finish_stroke(
            enable_deformation=scene.npr_enable_deformation,
            enable_inpainting=False
        )
        
        # Full viewport sync after deformation
        self._sync_viewport()
        
        # Unregister brush preview
        self._unregister_brush_preview()
        
        context.area.tag_redraw()
        self.report({'INFO'}, f"Stroke complete. Total gaussians: {self.scene_data.count}")
    
    def cancel(self, context):
        self._unregister_brush_preview()


class THREEGDS_OT_ClearPaintedGaussians(Operator):
    """Clear all painted gaussians"""
    bl_idname = "threegds.clear_painted_gaussians"
    bl_label = "Clear Painted Gaussians"
    bl_description = "Remove all painted gaussians from viewport"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .viewport.viewport_renderer import GaussianViewportRenderer
        
        # Clear viewport
        renderer = GaussianViewportRenderer.get_instance()
        renderer.clear()
        renderer.request_redraw()
        
        # Clear paint session
        clear_paint_session()
        
        self.report({'INFO'}, "All painted gaussians cleared")
        return {'FINISHED'}


class THREEGDS_OT_ApplyDeformation(Operator):
    """Apply deformation to painted strokes via subprocess"""
    bl_idname = "threegds.apply_deformation"
    bl_label = "Apply Deformation"
    bl_description = "Apply spline-based deformation to painted strokes using GPU"
    bl_options = {'REGISTER'}
    
    batch_size: IntProperty(
        name="Batch Size",
        description="Number of stamps to process per timer tick",
        default=10,
        min=1,
        max=100
    )
    
    _timer = None
    _future = None
    _progress = 0
    
    def invoke(self, context, event):
        # TODO: Implement timer-based deformation in subprocess
        # This will be implemented in Phase 4 Week 2-3
        self.report({'WARNING'}, "Deformation operator not yet implemented")
        return {'CANCELLED'}
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._future and self._future.done:
                self._cleanup(context)
                return {'FINISHED'}
            
            context.area.tag_redraw()
        
        if event.type == 'ESC':
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


# Registration
classes = [
    THREEGDS_OT_install_dependencies,
    THREEGDS_OT_uninstall_dependencies,
    THREEGDS_OT_check_dependencies,
    THREEGDS_OT_test_subprocess,
    THREEGDS_OT_test_subprocess_cuda,
    THREEGDS_OT_kill_subprocess,
    # Phase 4 painting operators (WorkSpaceTool architecture)
    THREEGDS_OT_GaussianPaintStroke,
    THREEGDS_OT_ClearPaintedGaussians,
    THREEGDS_OT_ApplyDeformation,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
