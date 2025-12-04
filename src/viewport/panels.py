# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
UI panels for 3DGS Painter viewport.
"""

import bpy
from bpy.types import Panel


class NPR_PT_ViewportPanel(Panel):
    """Main viewport panel for 3DGS Painter"""
    bl_label = "3DGS Painter"
    bl_idname = "NPR_PT_viewport_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    
    def draw(self, context):
        layout = self.layout
        
        # Get renderer instance
        from .viewport_renderer import GaussianViewportRenderer
        renderer = GaussianViewportRenderer.get_instance()
        
        # Viewport rendering controls
        box = layout.box()
        box.label(text="Viewport Rendering", icon='SHADING_RENDERED')
        
        row = box.row(align=True)
        if renderer.enabled:
            row.operator("npr.disable_viewport_rendering", text="Disable", icon='PAUSE')
        else:
            row.operator("npr.enable_viewport_rendering", text="Enable", icon='PLAY')
        
        # Stats
        if renderer.enabled and renderer.data_manager.is_valid:
            col = box.column(align=True)
            col.label(text=f"Gaussians: {renderer.gaussian_count:,}")
            tex_info = renderer.data_manager.get_texture_info()
            col.label(text=f"Texture: {tex_info['texture_width']}×{tex_info['texture_height']}")
        
        # Rendering settings
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        
        col = box.column(align=True)
        col.prop(context.scene, "npr_use_depth_test", text="Depth Test")
        col.prop(context.scene, "npr_depth_bias", text="Depth Bias")
        
        # Test controls
        box = layout.box()
        box.label(text="Testing", icon='EXPERIMENTAL')
        
        row = box.row(align=True)
        row.operator("npr.generate_test_gaussians", text="Generate Test", icon='MESH_UVSPHERE')
        row.operator("npr.clear_gaussians", text="Clear", icon='TRASH')
        
        col = box.column(align=True)
        col.operator("npr.run_benchmark", text="Run Benchmark", icon='TIME')


class NPR_PT_PaintingPanel(Panel):
    """Painting controls panel for 3DGS Painter"""
    bl_label = "Painting"
    bl_idname = "NPR_PT_painting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Tool activation info
        box = layout.box()
        box.label(text="Gaussian Paint Tool", icon='BRUSH_DATA')
        
        col = box.column(align=True)
        col.label(text="Select 'Gaussian Paint' tool", icon='INFO')
        col.label(text="from the toolbar (T) to paint", icon='BLANK1')
        
        col.separator()
        col.label(text="LMB drag: Paint stroke", icon='DOT')
        col.label(text="Change tool: Exit paint", icon='LOOP_BACK')
        
        # Brush settings
        box = layout.box()
        box.label(text="Brush Settings", icon='BRUSH_DATA')
        
        col = box.column(align=True)
        col.prop(scene, "npr_brush_size", text="Size")
        col.prop(scene, "npr_brush_opacity", text="Opacity")
        col.prop(scene, "npr_brush_spacing", text="Spacing")
        
        col.separator()
        col.prop(scene, "npr_brush_color", text="Color")
        
        # Brush pattern
        col.separator()
        col.prop(scene, "npr_brush_pattern", text="Pattern")
        col.prop(scene, "npr_brush_num_gaussians", text="Gaussians per Stamp")
        
        # Deformation settings
        box = layout.box()
        box.label(text="Deformation", icon='MOD_SIMPLEDEFORM')
        
        col = box.column(align=True)
        col.prop(scene, "npr_enable_deformation", text="Enable Deformation")
        
        # Actions
        box = layout.box()
        box.label(text="Actions", icon='ACTION')
        
        row = box.row(align=True)
        row.operator("threegds.clear_painted_gaussians", text="Clear All", icon='TRASH')


class NPR_PT_DependenciesPanel(Panel):
    """Dependencies panel for 3DGS Painter"""
    bl_label = "Dependencies"
    bl_idname = "NPR_PT_dependencies_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Dependency status
        box = layout.box()
        box.label(text="Package Management", icon='PACKAGE')
        
        col = box.column(align=True)
        col.operator("threegds.check_dependencies", text="Check Status", icon='FILE_REFRESH')
        col.operator("threegds.install_dependencies", text="Install Packages", icon='IMPORT')
        
        col.separator()
        col.operator("threegds.test_subprocess", text="Test PyTorch", icon='GHOST_ENABLED')
        col.operator("threegds.test_subprocess_cuda", text="Test CUDA", icon='OUTLINER_DATA_LIGHTPROBE')


# Scene properties for rendering settings
def _register_scene_props():
    """Register scene-level properties for viewport rendering."""
    # Viewport settings
    bpy.types.Scene.npr_use_depth_test = bpy.props.BoolProperty(
        name="Use Depth Test",
        description="Test gaussian depth against Blender scene geometry",
        default=True,
        update=_on_depth_test_changed
    )
    
    bpy.types.Scene.npr_depth_bias = bpy.props.FloatProperty(
        name="Depth Bias",
        description="Small offset to prevent z-fighting with scene geometry",
        default=0.0001,
        min=0.0,
        max=0.01,
        precision=5,
        update=_on_depth_bias_changed
    )
    
    # Brush settings (Phase 4)
    bpy.types.Scene.npr_brush_size = bpy.props.FloatProperty(
        name="Brush Size",
        description="Size multiplier for brush strokes",
        default=1.0,
        min=0.01,
        max=10.0
    )
    
    bpy.types.Scene.npr_brush_opacity = bpy.props.FloatProperty(
        name="Brush Opacity",
        description="Opacity of brush strokes",
        default=0.8,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.npr_brush_spacing = bpy.props.FloatProperty(
        name="Brush Spacing",
        description="Spacing between stamps relative to brush size",
        default=0.3,
        min=0.05,
        max=2.0
    )
    
    bpy.types.Scene.npr_brush_color = bpy.props.FloatVectorProperty(
        name="Brush Color",
        description="Color of brush strokes",
        subtype='COLOR',
        default=(0.8, 0.2, 0.2),
        min=0.0,
        max=1.0,
        size=3
    )
    
    bpy.types.Scene.npr_brush_pattern = bpy.props.EnumProperty(
        name="Brush Pattern",
        description="Pattern of gaussians in each brush stamp",
        items=[
            ('CIRCULAR', "Circular", "Gaussians arranged in a circle"),
            ('LINE', "Line", "Gaussians arranged in a line"),
            ('GRID', "Grid", "Gaussians arranged in a grid"),
        ],
        default='CIRCULAR'
    )
    
    bpy.types.Scene.npr_brush_num_gaussians = bpy.props.IntProperty(
        name="Gaussians per Stamp",
        description="Number of gaussians in each brush stamp",
        default=20,
        min=1,
        max=500
    )
    
    # Deformation settings (Phase 4)
    bpy.types.Scene.npr_enable_deformation = bpy.props.BoolProperty(
        name="Enable Deformation",
        description="Apply spline-based deformation to strokes",
        default=False
    )


def _unregister_scene_props():
    """Unregister scene-level properties."""
    props_to_remove = [
        'npr_use_depth_test',
        'npr_depth_bias',
        'npr_brush_size',
        'npr_brush_opacity',
        'npr_brush_spacing',
        'npr_brush_color',
        'npr_brush_pattern',
        'npr_brush_num_gaussians',
        'npr_enable_deformation',
    ]
    
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)


def _on_depth_test_changed(self, context):
    """Callback when depth test setting changes."""
    from .viewport_renderer import GaussianViewportRenderer
    renderer = GaussianViewportRenderer.get_instance()
    renderer.use_depth_test = context.scene.npr_use_depth_test
    renderer.request_redraw()


def _on_depth_bias_changed(self, context):
    """Callback when depth bias setting changes."""
    from .viewport_renderer import GaussianViewportRenderer
    renderer = GaussianViewportRenderer.get_instance()
    renderer.depth_bias = context.scene.npr_depth_bias
    renderer.request_redraw()


# Panel classes
panel_classes = [
    NPR_PT_ViewportPanel,
    NPR_PT_PaintingPanel,
    NPR_PT_DependenciesPanel,
]


def register_panels():
    """Register UI panels."""
    _register_scene_props()
    
    for cls in panel_classes:
        bpy.utils.register_class(cls)


def unregister_panels():
    """Unregister UI panels."""
    for cls in reversed(panel_classes):
        bpy.utils.unregister_class(cls)
    
    _unregister_scene_props()
