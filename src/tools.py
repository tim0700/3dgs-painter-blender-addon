# tools.py
# WorkSpaceTool definitions for 3DGS Painter addon

import bpy
from bpy.types import WorkSpaceTool


class GaussianPaintTool(WorkSpaceTool):
    """Gaussian Splat painting tool for 3D Viewport"""
    
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    
    bl_idname = "threegds.gaussian_paint_tool"
    bl_label = "Gaussian Paint"
    bl_description = (
        "Paint Gaussian splats in the viewport.\n"
        "Left-click and drag to paint strokes.\n"
        "Adjust brush settings in the N-panel"
    )
    
    # Use a brush icon
    bl_icon = "brush.sculpt.paint"
    
    # Cursor style when tool is active
    bl_cursor = 'PAINT_BRUSH'
    
    # No gizmo needed
    bl_widget = None
    
    # Keymap: bind mouse events to the stroke operator
    bl_keymap = (
        # Primary stroke - left mouse drag
        ("threegds.gaussian_paint_stroke",
         {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": [("mode", 'ADD')]}),
        
        # Future: Erase with Ctrl+LMB
        # ("threegds.gaussian_paint_stroke",
        #  {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
        #  {"properties": [("mode", 'REMOVE')]}),
    )
    
    def draw_settings(context, layout, tool):
        """Draw tool settings in the Tool Header."""
        scene = context.scene
        
        # Quick access to key brush settings
        row = layout.row(align=True)
        row.prop(scene, "npr_brush_size", text="Size")
        row.prop(scene, "npr_brush_opacity", text="Opacity")
        
        row = layout.row(align=True)
        row.prop(scene, "npr_brush_color", text="")
        row.prop(scene, "npr_brush_pattern", text="")


def register_tools():
    """Register painting tools."""
    bpy.utils.register_tool(
        GaussianPaintTool,
        after={"builtin.cursor"},  # Position after 3D cursor tool
        separator=True,  # Add visual separator before this tool
        group=False
    )


def unregister_tools():
    """Unregister painting tools."""
    bpy.utils.unregister_tool(GaussianPaintTool)
