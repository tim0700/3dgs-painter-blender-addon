# operators.py
# Operators for 3DGS Painter addon

import bpy
from bpy.types import Operator
import threading


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


# Registration
classes = [
    THREEGDS_OT_install_dependencies,
    THREEGDS_OT_uninstall_dependencies,
    THREEGDS_OT_check_dependencies,
    THREEGDS_OT_test_subprocess,
    THREEGDS_OT_test_subprocess_cuda,
    THREEGDS_OT_kill_subprocess,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
