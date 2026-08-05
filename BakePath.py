import bpy, math, time
import numpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.props import *
from .common import *
from . import ListItem, node_connections, node_arrangements

PATH_SHRINKWRAP_NAME = 'YP Path Shrinkwrap'

def get_path_curve_object(layer):
    curve_obj = None
    if is_bl_newer_than(2, 79) and getattr(layer, 'path_curve_object', None):
        curve_obj = layer.path_curve_object
    if not curve_obj and layer.path_curve_object_name:
        curve_obj = bpy.data.objects.get(layer.path_curve_object_name)
    if curve_obj and curve_obj.type != 'CURVE':
        return None
    return curve_obj

def set_path_curve_object(layer, curve_obj):
    if is_bl_newer_than(2, 79):
        layer.path_curve_object = curve_obj
    layer.path_curve_object_name = curve_obj.name if curve_obj else ''

def find_path_bake_layer_for_curve(curve_obj):
    '''
    Find the UcuPaint layer linked to this path/shape curve.
    Returns (yp, layer_index) or (None, -1).
    '''
    if not curve_obj or curve_obj.type != 'CURVE':
        return None, -1
    parent = curve_obj.parent
    if not parent or parent.type != 'MESH':
        return None, -1

    for slot in parent.material_slots:
        mat = slot.material
        if not mat or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'GROUP' or not node.node_tree:
                continue
            yp = getattr(node.node_tree, 'yp', None)
            if not yp or not getattr(yp, 'is_ypaint_node', False):
                continue
            for i, layer in enumerate(yp.layers):
                if not getattr(layer, 'enable_path_bake', False):
                    continue
                linked = get_path_curve_object(layer)
                if linked == curve_obj:
                    return yp, i
                # Name fallback if the pointer was cleared / stale
                if getattr(layer, 'path_curve_object_name', '') == curve_obj.name:
                    return yp, i
    return None, -1

def get_mesh_for_path_curve(curve_obj):
    '''
    If curve_obj is a path/shape bake curve linked to a UcuPaint layer on its
    parent mesh, return that mesh. Otherwise None.
    '''
    yp, _idx = find_path_bake_layer_for_curve(curve_obj)
    if yp is None:
        return None
    return curve_obj.parent if curve_obj else None

def select_path_bake_layer_for_curve(curve_obj):
    '''Make the UcuPaint layer linked to this curve the active layer.'''
    yp, idx = find_path_bake_layer_for_curve(curve_obj)
    if yp is None or idx < 0 or idx >= len(yp.layers):
        return False

    from . import ListItem

    # Classic list binds to active_layer_index
    if yp.active_layer_index != idx:
        yp.active_layer_index = idx

    # Dynamic list binds to active_item_index — set it directly (do not use
    # refresh_list_items(repoint_active=True), which preserves the previous item)
    item_idx = ListItem.get_layer_item_index(yp.layers[idx])
    if item_idx < 0:
        ListItem.refresh_list_items(yp, repoint_active=False)
        item_idx = ListItem.get_layer_item_index(yp.layers[idx])
    if item_idx >= 0 and yp.active_item_index != item_idx:
        # Avoid recursive layer index updates when item already maps to this layer
        yp.halt_update = True
        try:
            yp.active_item_index = item_idx
        finally:
            yp.halt_update = False

    # Force UI redraw
    wm = bpy.context.window_manager
    if hasattr(wm, 'ypui'):
        wm.ypui.need_update = True
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()

    return True

_last_path_curve_layer_sync = None
_pending_path_curve_layer_sync = None
_pending_layer_to_curve_tree = None
_pending_layer_to_curve_index = None
_last_layer_to_curve_key = None
_path_sel_sync_lock = False

def clear_path_selection_sync_state():
    '''Drop pending layer↔curve sync (call after deleting curves / layers).'''
    global _pending_path_curve_layer_sync, _pending_layer_to_curve_tree
    global _pending_layer_to_curve_index, _last_layer_to_curve_key, _last_path_curve_layer_sync
    _pending_path_curve_layer_sync = None
    _pending_layer_to_curve_tree = None
    _pending_layer_to_curve_index = None
    _last_layer_to_curve_key = None
    _last_path_curve_layer_sync = None

def _deferred_path_curve_layer_sync():
    '''Timer callback — RNA writes are unsafe directly from depsgraph/msgbus.'''
    global _pending_path_curve_layer_sync, _last_path_curve_layer_sync, _path_sel_sync_lock
    curve_obj = _pending_path_curve_layer_sync
    _pending_path_curve_layer_sync = None
    if not curve_obj or _path_sel_sync_lock:
        return None
    # A layer-list change is in flight — don't snap the layer back to an old curve
    if _pending_layer_to_curve_tree is not None:
        return None
    # Object may have been deleted between schedule and run
    try:
        _ = curve_obj.name
    except ReferenceError:
        _last_path_curve_layer_sync = None
        return None
    # Stale: user already moved selection (e.g. layer list picked another layer)
    try:
        if bpy.context.object != curve_obj:
            return None
    except Exception:
        return None
    _path_sel_sync_lock = True
    try:
        if select_path_bake_layer_for_curve(curve_obj):
            _last_path_curve_layer_sync = curve_obj.as_pointer() if hasattr(curve_obj, 'as_pointer') else curve_obj.name
    finally:
        _path_sel_sync_lock = False
    return None

def sync_active_path_curve_layer_selection():
    '''
    If the active object is a linked path/shape curve, activate its layer.
    Safe to call from msgbus / depsgraph handlers (defers the actual write).
    '''
    global _last_path_curve_layer_sync, _pending_path_curve_layer_sync
    if _path_sel_sync_lock:
        return
    # Don't fight an in-progress layer→curve sync (layer list click wins)
    if _pending_layer_to_curve_tree is not None:
        return
    try:
        obj = bpy.context.object
    except Exception:
        return
    if not obj or obj.type != 'CURVE':
        _last_path_curve_layer_sync = None
        return

    # Only for curves actually linked to a path/shape layer
    yp, idx = find_path_bake_layer_for_curve(obj)
    if yp is None or idx < 0:
        _last_path_curve_layer_sync = None
        return

    key = obj.as_pointer() if hasattr(obj, 'as_pointer') else obj.name
    if key == _last_path_curve_layer_sync and yp.active_layer_index == idx:
        return

    _pending_path_curve_layer_sync = obj
    if not bpy.app.timers.is_registered(_deferred_path_curve_layer_sync):
        bpy.app.timers.register(_deferred_path_curve_layer_sync, first_interval=0.0)

def _select_only_object(obj):
    if not obj:
        return
    try:
        selected = list(bpy.context.selected_objects)
    except Exception:
        selected = []
    for o in selected:
        if o != obj:
            set_object_select(o, False)
    set_object_select(obj, True)
    set_active_object(obj)

def _deferred_layer_to_curve_sync():
    '''When a path/shape layer becomes active, select its curve (and vice versa leave).'''
    global _pending_layer_to_curve_tree, _pending_layer_to_curve_index
    global _last_path_curve_layer_sync, _path_sel_sync_lock
    tree = _pending_layer_to_curve_tree
    want_idx = _pending_layer_to_curve_index
    _pending_layer_to_curve_tree = None
    _pending_layer_to_curve_index = None
    if not tree or _path_sel_sync_lock:
        return None
    yp = getattr(tree, 'yp', None)
    if not yp or getattr(yp, 'halt_update', False):
        return None
    if len(yp.layers) == 0 or yp.active_layer_index < 0 or yp.active_layer_index >= len(yp.layers):
        return None
    # Another layer click already superseded this deferred job
    if want_idx is not None and int(yp.active_layer_index) != int(want_idx):
        return None

    layer = yp.layers[yp.active_layer_index]
    curve_obj = None
    if getattr(layer, 'enable_path_bake', False):
        curve_obj = get_path_curve_object(layer)

    try:
        active = bpy.context.object
    except Exception:
        active = None

    mesh = None
    if curve_obj and curve_obj.parent and curve_obj.parent.type == 'MESH':
        mesh = curve_obj.parent
    elif active:
        if active.type == 'MESH':
            mesh = active
        elif active.type == 'CURVE' and active.parent and active.parent.type == 'MESH':
            mesh = active.parent

    _path_sel_sync_lock = True
    try:
        if curve_obj:
            if active != curve_obj:
                _select_only_object(curve_obj)
            _last_path_curve_layer_sync = (
                curve_obj.as_pointer() if hasattr(curve_obj, 'as_pointer') else curve_obj.name
            )
        else:
            # Left a path/shape layer: if a linked path curve is active, return to the mesh
            if active and active.type == 'CURVE':
                linked_yp, _idx = find_path_bake_layer_for_curve(active)
                if linked_yp is not None and mesh:
                    _select_only_object(mesh)
            _last_path_curve_layer_sync = None
    finally:
        _path_sel_sync_lock = False
    return None

def schedule_sync_active_layer_to_path_curve(yp):
    '''Defer layer→curve object selection (safe from active_layer_index update).'''
    global _pending_layer_to_curve_tree, _pending_layer_to_curve_index
    global _last_layer_to_curve_key, _pending_path_curve_layer_sync
    if _path_sel_sync_lock or not yp:
        return
    if getattr(yp, 'halt_update', False):
        return
    tree = getattr(yp, 'id_data', None)
    if not tree:
        return
    # Skip no-op refreshes (yp.active_layer_index = yp.active_layer_index)
    # so paint-slot updates don't keep yanking selection onto the curve.
    tree_key = tree.as_pointer() if hasattr(tree, 'as_pointer') else id(tree)
    key = (tree_key, int(yp.active_layer_index))
    if key == _last_layer_to_curve_key:
        return
    _last_layer_to_curve_key = key
    # Layer list click wins: cancel any stale curve→layer job still queued from
    # the previously active path curve (that was the "snap back" bug).
    _pending_path_curve_layer_sync = None
    _pending_layer_to_curve_tree = tree
    _pending_layer_to_curve_index = int(yp.active_layer_index)
    if not bpy.app.timers.is_registered(_deferred_layer_to_curve_sync):
        bpy.app.timers.register(_deferred_layer_to_curve_sync, first_interval=0.0)

def resolve_path_bake_mesh(obj=None):
    '''Return a mesh suitable for path/shape baking from the active or given object.'''
    if obj is None:
        obj = bpy.context.object
    if not obj:
        return None
    if obj.type == 'MESH':
        return obj
    if obj.type == 'CURVE':
        return get_mesh_for_path_curve(obj)
    return None

def remove_path_curve_object(layer):
    '''Unlink and delete the path/shape curve owned by this layer, if unused elsewhere.'''
    curve_obj = get_path_curve_object(layer)
    set_path_curve_object(layer, None)
    if not curve_obj:
        return

    # Keep the curve if another path/shape layer still references it
    parent = curve_obj.parent if curve_obj.parent and curve_obj.parent.type == 'MESH' else None
    if parent:
        for slot in parent.material_slots:
            mat = slot.material
            if not mat or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type != 'GROUP' or not node.node_tree:
                    continue
                yp = getattr(node.node_tree, 'yp', None)
                if not yp or not getattr(yp, 'is_ypaint_node', False):
                    continue
                for other in yp.layers:
                    if get_path_curve_object(other) == curve_obj:
                        return

    # Layer↔curve sync often makes the curve the active object — switch to the
    # parent mesh before deleting so context.object is not a removed StructRNA.
    try:
        active = bpy.context.object
    except Exception:
        active = None
    if parent and active == curve_obj:
        global _last_path_curve_layer_sync, _path_sel_sync_lock
        _path_sel_sync_lock = True
        try:
            set_object_select(curve_obj, False)
            set_object_select(parent, True)
            set_active_object(parent)
            clear_path_selection_sync_state()
        finally:
            _path_sel_sync_lock = False
    else:
        clear_path_selection_sync_state()

    curve_data = curve_obj.data
    remove_datablock(bpy.data.objects, curve_obj)
    if curve_data and curve_data.users == 0:
        remove_datablock(bpy.data.curves, curve_data)

def duplicate_path_curve_for_layer(layer, duplicated_curves=None):
    '''
    Duplicate the path/shape curve linked to a layer and reassign it.
    Used when duplicating/pasting layers so each layer gets its own curve.
    '''
    if duplicated_curves is None:
        duplicated_curves = {}

    if not getattr(layer, 'enable_path_bake', False):
        return None

    original = get_path_curve_object(layer)
    if not original or original.type != 'CURVE':
        return None

    if original in duplicated_curves:
        new_curve = duplicated_curves[original]
    else:
        nname = get_unique_name(original.name, bpy.data.objects)
        custom_collection = None
        if is_bl_newer_than(2, 80) and len(original.users_collection) > 0:
            custom_collection = original.users_collection[0]
        elif original.parent and is_bl_newer_than(2, 80) and len(original.parent.users_collection) > 0:
            custom_collection = original.parent.users_collection[0]

        new_curve = original.copy()
        new_curve.name = nname
        # Make curve data single-user so editing the copy doesn't affect the original
        if original.data:
            new_curve.data = original.data.copy()
            new_curve.data.name = get_unique_name(original.data.name, bpy.data.curves)

        link_object(bpy.context.scene, new_curve, custom_collection)
        duplicated_curves[original] = new_curve

        # Keep shrinkwrap targeting the same mesh (parent / existing target)
        if getattr(layer, 'path_enable_shrinkwrap', True):
            target = new_curve.parent
            if target and target.type == 'MESH':
                _apply_layer_shrinkwrap(layer, new_curve, target)
            else:
                mod = get_path_shrinkwrap_modifier(new_curve)
                if mod and mod.target:
                    _apply_layer_shrinkwrap(layer, new_curve, mod.target)

    set_path_curve_object(layer, new_curve)
    return new_curve

def get_path_shrinkwrap_modifier(curve_obj):
    if not curve_obj:
        return None
    for mod in curve_obj.modifiers:
        if mod.type == 'SHRINKWRAP' and mod.name == PATH_SHRINKWRAP_NAME:
            return mod
    # Fallback: any shrinkwrap (user may have renamed)
    for mod in curve_obj.modifiers:
        if mod.type == 'SHRINKWRAP':
            return mod
    return None

def configure_path_shrinkwrap(mod, target_obj, method='NEAREST_SURFACEPOINT',
                              project_axes=(False, False, False)):
    '''Apply path shrinkwrap settings (Nearest, or Project along chosen axes).'''
    mod.target = target_obj
    if method == 'PROJECT':
        mod.wrap_method = 'PROJECT'
        if hasattr(mod, 'use_negative_direction'):
            mod.use_negative_direction = True
        if hasattr(mod, 'use_positive_direction'):
            mod.use_positive_direction = True
        ax, ay, az = project_axes
        if hasattr(mod, 'use_project_x'):
            mod.use_project_x = bool(ax)
        if hasattr(mod, 'use_project_y'):
            mod.use_project_y = bool(ay)
        if hasattr(mod, 'use_project_z'):
            mod.use_project_z = bool(az)
    else:
        mod.wrap_method = 'NEAREST_SURFACEPOINT'
    if hasattr(mod, 'wrap_mode'):
        mod.wrap_mode = 'ON_SURFACE'
    mod.offset = 0.0
    mod.show_viewport = True
    mod.show_render = True
    if hasattr(mod, 'show_in_editmode'):
        mod.show_in_editmode = True
    if hasattr(mod, 'show_on_cage'):
        mod.show_on_cage = True

def ensure_path_shrinkwrap(curve_obj, target_obj=None, method='NEAREST_SURFACEPOINT',
                           project_axes=(False, False, False)):
    '''Add or update Shrinkwrap on the path curve, targeting the parent mesh.'''
    if not curve_obj or curve_obj.type != 'CURVE':
        return None
    if target_obj is None:
        target_obj = curve_obj.parent
    if not target_obj or target_obj.type != 'MESH':
        return None

    mod = get_path_shrinkwrap_modifier(curve_obj)
    if not mod:
        mod = curve_obj.modifiers.new(PATH_SHRINKWRAP_NAME, 'SHRINKWRAP')
    elif mod.name != PATH_SHRINKWRAP_NAME:
        mod.name = PATH_SHRINKWRAP_NAME
    configure_path_shrinkwrap(mod, target_obj, method=method, project_axes=project_axes)
    return mod

def remove_path_shrinkwrap(curve_obj):
    if not curve_obj:
        return
    # Remove all YP path shrinkwraps (and legacy unnamed ones we created)
    to_remove = [m for m in curve_obj.modifiers if m.type == 'SHRINKWRAP' and m.name == PATH_SHRINKWRAP_NAME]
    for mod in to_remove:
        curve_obj.modifiers.remove(mod)

def _layer_shrinkwrap_method(layer):
    return getattr(layer, 'path_shrinkwrap_method', 'NEAREST_SURFACEPOINT') or 'NEAREST_SURFACEPOINT'

def _layer_project_axes(layer):
    '''(x, y, z) project-axis flags from a path/shape layer.'''
    return (
        bool(getattr(layer, 'path_project_x', False)),
        bool(getattr(layer, 'path_project_y', False)),
        bool(getattr(layer, 'path_project_z', False)),
    )

def _apply_layer_shrinkwrap(layer, curve_obj, target_obj):
    '''Push the layer's shrinkwrap method + axes onto the curve modifier.'''
    if not curve_obj or not target_obj:
        return None
    return ensure_path_shrinkwrap(
        curve_obj, target_obj,
        method=_layer_shrinkwrap_method(layer),
        project_axes=_layer_project_axes(layer),
    )

def update_path_enable_shrinkwrap(self, context):
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    if self.path_enable_shrinkwrap:
        target = curve_obj.parent if curve_obj.parent else context.object
        if target and target.type == 'MESH':
            _apply_layer_shrinkwrap(self, curve_obj, target)
    else:
        remove_path_shrinkwrap(curve_obj)

def update_path_shrinkwrap_method(self, context):
    '''Re-apply shrinkwrap when Nearest / Project mode changes.'''
    if not getattr(self, 'path_enable_shrinkwrap', False):
        return
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    target = curve_obj.parent if curve_obj.parent else getattr(context, 'object', None)
    if target and target.type == 'MESH':
        _apply_layer_shrinkwrap(self, curve_obj, target)

def update_path_project_axis(self, context):
    '''Re-apply Project axes on the curve shrinkwrap when X/Y/Z toggles change.'''
    if not getattr(self, 'path_enable_shrinkwrap', False):
        return
    if _layer_shrinkwrap_method(self) != 'PROJECT':
        return
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    target = curve_obj.parent if curve_obj.parent else getattr(context, 'object', None)
    if target and target.type == 'MESH':
        _apply_layer_shrinkwrap(self, curve_obj, target)

def create_path_curve(target_obj, name='Path', use_shrinkwrap=True, cyclic=False,
                      shrinkwrap_method='NEAREST_SURFACEPOINT',
                      project_axes=(False, False, False)):
    scene = bpy.context.scene
    curve_name = get_unique_name(name, bpy.data.objects)
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 64

    spline = curve_data.splines.new('BEZIER')

    # Place default curve near the object / cursor
    if is_bl_newer_than(2, 80):
        cursor_loc = scene.cursor.location.copy()
    else:
        cursor_loc = scene.cursor_location.copy()

    size = max(target_obj.dimensions) * 0.25 if target_obj.dimensions.length > 0 else 0.25
    size = max(size, 0.1)

    # Build a surface BVH for snapping control points
    bvh = None
    bm = None
    try:
        import bmesh as _bmesh
        bm = _bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = target_obj.evaluated_get(depsgraph)
        try:
            bm.from_object(eval_obj, depsgraph)
        except TypeError:
            bm.from_mesh(target_obj.data)
        bm.transform(target_obj.matrix_world)
        bvh = BVHTree.FromBMesh(bm)
    except Exception:
        bvh = None

    def _snap(p):
        if not bvh:
            return p
        hit = bvh.find_nearest(p)
        if hit and hit[0] is not None:
            return hit[0].copy()
        return p

    if cyclic:
        # Closed shape: 4-point cyclic bezier (rounded square / ellipse)
        count = 4
        spline.bezier_points.add(count - 1)
        radius = size * 0.5
        for i, bp in enumerate(spline.bezier_points):
            angle = (i / float(count)) * math.pi * 2.0
            co = cursor_loc + Vector((math.cos(angle) * radius, math.sin(angle) * radius, 0.0))
            bp.co = _snap(co)
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
        spline.use_cyclic_u = True
    else:
        # Open ribbon path: 2-point bezier
        spline.bezier_points.add(1)
        p0 = _snap(cursor_loc + Vector((-size * 0.5, 0.0, 0.0)))
        p1 = _snap(cursor_loc + Vector((size * 0.5, 0.0, 0.0)))
        bp0 = spline.bezier_points[0]
        bp1 = spline.bezier_points[1]
        for bp, co in ((bp0, p0), (bp1, p1)):
            bp.co = co
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
        spline.use_cyclic_u = False

    if bm is not None:
        try:
            bm.free()
        except Exception:
            pass

    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    custom_collection = None
    if is_bl_newer_than(2, 80) and len(target_obj.users_collection) > 0:
        custom_collection = target_obj.users_collection[0]
    link_object(scene, curve_obj, custom_collection)

    # Parent to mesh so it moves with the object, but keep world positions
    curve_obj.parent = target_obj
    curve_obj.matrix_parent_inverse = target_obj.matrix_world.inverted()

    if use_shrinkwrap:
        ensure_path_shrinkwrap(
            curve_obj, target_obj, method=shrinkwrap_method, project_axes=project_axes
        )

    return curve_obj

def ensure_curve_cyclic(curve_obj, cyclic=True):
    if not curve_obj or curve_obj.type != 'CURVE':
        return
    for spline in curve_obj.data.splines:
        spline.use_cyclic_u = cyclic

def update_path_mode(self, context):
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    if self.path_mode == 'SHAPE':
        ensure_curve_cyclic(curve_obj, True)

def _curve_has_enabled_shrinkwrap(curve_obj):
    mod = get_path_shrinkwrap_modifier(curve_obj)
    return bool(mod and mod.show_viewport and mod.target)

def _curve_shrinkwrap_method(curve_obj, fallback=None):
    '''Return active shrinkwrap wrap_method, or fallback / None if disabled.'''
    if not _curve_has_enabled_shrinkwrap(curve_obj):
        return fallback
    mod = get_path_shrinkwrap_modifier(curve_obj)
    if not mod:
        return fallback
    return getattr(mod, 'wrap_method', 'NEAREST_SURFACEPOINT') or 'NEAREST_SURFACEPOINT'

def _curve_to_polyline(curve_obj, resolution=64, shrinkwrap_method=None):
    '''Return world-space polyline points sampled from a CURVE object.'''
    method = shrinkwrap_method if shrinkwrap_method is not None else _curve_shrinkwrap_method(curve_obj)

    # Project (or no shrinkwrap): keep authored silhouette — do not nearest-snap
    if method == 'PROJECT' or method is None or not _curve_has_enabled_shrinkwrap(curve_obj):
        points = _sample_bezier_math(curve_obj, resolution)
        if len(points) >= 2:
            return points
        return _sample_evaluated_curve(curve_obj, resolution)

    # Nearest: sample evaluated / snap to closest surface
    if _curve_has_enabled_shrinkwrap(curve_obj):
        points = _sample_evaluated_curve(curve_obj, resolution)
        if len(points) >= 2:
            return points

    points = _sample_bezier_math(curve_obj, resolution)
    if len(points) >= 2:
        return points
    return _sample_evaluated_curve(curve_obj, resolution)

def _sample_evaluated_curve(curve_obj, resolution):
    '''Sample curve after modifiers (e.g. Shrinkwrap) via evaluated mesh conversion.'''
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = curve_obj.evaluated_get(depsgraph)

    mesh = None
    try:
        mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=False, depsgraph=depsgraph)
    except TypeError:
        try:
            mesh = bpy.data.meshes.new_from_object(eval_obj)
        except Exception:
            mesh = None
    except Exception:
        mesh = None

    if mesh and len(mesh.vertices) >= 2:
        mw = curve_obj.matrix_world
        raw = [mw @ v.co for v in mesh.vertices]
        remove_datablock(bpy.data.meshes, mesh)
        if len(raw) >= 2:
            return _resample_polyline(raw, resolution)
    elif mesh:
        remove_datablock(bpy.data.meshes, mesh)

    # Curves often don't convert to a useful mesh without bevel.
    # Emulate Shrinkwrap for bake sampling.
    mod = get_path_shrinkwrap_modifier(curve_obj)
    target = mod.target if mod else None
    points = _sample_bezier_math(curve_obj, resolution)
    if not target or target.type != 'MESH' or len(points) < 2:
        return points

    wrap_method = getattr(mod, 'wrap_method', 'NEAREST_SURFACEPOINT')
    # Project: keep authored silhouette. Nearest: snap to closest surface point.
    if wrap_method == 'PROJECT':
        return points

    try:
        import bmesh as _bmesh
        bm = _bmesh.new()
        eval_target = target.evaluated_get(depsgraph)
        try:
            bm.from_object(eval_target, depsgraph)
        except TypeError:
            bm.from_mesh(target.data)
        bm.transform(target.matrix_world)
        bvh = BVHTree.FromBMesh(bm)
        offset = float(mod.offset) if mod else 0.0
        projected = []
        for p in points:
            hit = bvh.find_nearest(p)
            if hit and hit[0] is not None:
                loc, normal, _, _ = hit
                if offset and normal is not None:
                    n = Vector(normal)
                    if n.length > 1e-8:
                        n.normalize()
                        loc = loc + n * offset
                projected.append(Vector(loc))
            else:
                projected.append(p)
        bm.free()
        return projected
    except Exception:
        return points

def _curve_is_cyclic(curve_obj):
    if not curve_obj or curve_obj.type != 'CURVE' or not curve_obj.data:
        return False
    return any(getattr(s, 'use_cyclic_u', False) for s in curve_obj.data.splines)

def _sample_bezier_math(curve_obj, resolution):
    '''Fallback bezier sampling without mesh conversion (all splines concatenated).'''
    loops = _sample_bezier_math_loops(curve_obj, resolution)
    if not loops:
        return []
    if len(loops) == 1:
        return loops[0][0]
    # Ribbon/legacy callers expect one polyline — join with gaps (not ideal for shapes)
    out = []
    for loop, _cyclic in loops:
        if out and (out[-1] - loop[0]).length > 1e-6:
            out.append(loop[0].copy())
        out.extend(loop)
    return out

def _sample_bezier_math_loops(curve_obj, resolution):
    '''
    Sample each spline as its own world-space polyline.
    Returns list of (points, cyclic) so multi-spline ribbons never bridge.
    '''
    mw = curve_obj.matrix_world
    loops = []
    curve = curve_obj.data
    res_each = max(16, int(resolution))

    for spline in curve.splines:
        points = []
        cyclic = bool(getattr(spline, 'use_cyclic_u', False))

        if spline.type != 'BEZIER':
            pts = [mw @ Vector(p.co[:3]) for p in spline.points]
            if len(pts) < (3 if cyclic else 2):
                continue
            if cyclic and (pts[0] - pts[-1]).length > 1e-6:
                pts = list(pts) + [pts[0].copy()]
            points = _resample_polyline(pts, res_each)
        else:
            bps = spline.bezier_points
            if len(bps) < 2:
                continue
            segs = len(bps) if cyclic else (len(bps) - 1)
            if segs < 1:
                continue
            samples_per_seg = max(2, res_each // max(segs, 1))
            for i in range(segs):
                p0 = bps[i]
                p1 = bps[(i + 1) % len(bps)]
                knot0 = mw @ p0.co
                handle0 = mw @ p0.handle_right
                handle1 = mw @ p1.handle_left
                knot1 = mw @ p1.co
                for j in range(samples_per_seg):
                    if i > 0 and j == 0:
                        continue
                    t = j / float(samples_per_seg - 1) if samples_per_seg > 1 else 0.0
                    points.append(_bezier_point(knot0, handle0, handle1, knot1, t))
            if len(points) < 2:
                continue
            if cyclic and (points[0] - points[-1]).length > 1e-6:
                points.append(points[0].copy())
            points = _resample_polyline(points, res_each)

        if cyclic and len(points) >= 2 and (points[0] - points[-1]).length < 1e-4:
            points = points[:-1]
        if len(points) >= 2:
            loops.append((points, cyclic))
    return loops

def _snap_polyline_nearest_to_target(points, target_obj, offset=0.0):
    '''Snap polyline points to nearest surface on target mesh (world space).'''
    if not points or not target_obj or target_obj.type != 'MESH':
        return points
    try:
        import bmesh as _bmesh
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm = _bmesh.new()
        eval_target = target_obj.evaluated_get(depsgraph)
        try:
            bm.from_object(eval_target, depsgraph)
        except TypeError:
            bm.from_mesh(target_obj.data)
        bm.transform(target_obj.matrix_world)
        bvh = BVHTree.FromBMesh(bm)
        projected = []
        for p in points:
            hit = bvh.find_nearest(p)
            if hit and hit[0] is not None:
                loc, normal, _, _ = hit
                if offset and normal is not None:
                    n = Vector(normal)
                    if n.length > 1e-8:
                        n.normalize()
                        loc = loc + n * offset
                projected.append(Vector(loc))
            else:
                projected.append(p)
        bm.free()
        return projected
    except Exception:
        return points

def _curve_to_polylines(curve_obj, resolution=64, shrinkwrap_method=None):
    '''
    Return a list of (points, cyclic) world-space polylines, one per spline.
    Nearest: snap each loop to the surface. Project: keep authored silhouette.
    '''
    loops = _sample_bezier_math_loops(curve_obj, resolution)
    if not loops:
        return []
    method = shrinkwrap_method if shrinkwrap_method is not None else _curve_shrinkwrap_method(curve_obj)
    if method == 'PROJECT' or method is None or not _curve_has_enabled_shrinkwrap(curve_obj):
        return loops
    mod = get_path_shrinkwrap_modifier(curve_obj)
    if not mod or not mod.target:
        return loops
    offset = float(mod.offset) if mod else 0.0
    return [
        (_snap_polyline_nearest_to_target(loop, mod.target, offset=offset), cyclic)
        for loop, cyclic in loops
    ]

def _bezier_point(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u * u * u) * p0 + 3 * (u * u) * t * p1 + 3 * u * (t * t) * p2 + (t * t * t) * p3

def _resample_polyline(points, count):
    if len(points) < 2 or count < 2:
        return points[:]

    lengths = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        total += (points[i] - points[i - 1]).length
        lengths.append(total)

    if total <= 1e-8:
        return [points[0]] * count

    result = []
    for i in range(count):
        target = (i / float(count - 1)) * total
        # Find segment
        j = 1
        while j < len(lengths) and lengths[j] < target:
            j += 1
        j = min(j, len(points) - 1)
        seg_len = lengths[j] - lengths[j - 1]
        if seg_len < 1e-12:
            result.append(points[j].copy())
        else:
            t = (target - lengths[j - 1]) / seg_len
            result.append(points[j - 1].lerp(points[j], t))
    return result

def _build_mesh_bvh_and_uvs(obj, uv_name):
    '''Build world-space BVH and per-loop UV data for UV interpolation.'''
    mesh = obj.data
    uv_layer = mesh.uv_layers.get(uv_name) if hasattr(mesh, 'uv_layers') else None
    if not uv_layer and hasattr(mesh, 'uv_layers') and len(mesh.uv_layers) > 0:
        uv_layer = mesh.uv_layers.active
    if not uv_layer:
        return None, None, None, None

    mw = obj.matrix_world
    # Normal matrix for transforming normals
    try:
        normal_mw = mw.to_3x3().inverted_safe().transposed()
    except Exception:
        normal_mw = mw.to_3x3()

    # Triangulate via bmesh for stable face indices
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.transform(mw)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    uv_lay = bm.loops.layers.uv.get(uv_layer.name)
    if not uv_lay:
        bm.free()
        return None, None, None, None

    bvh = BVHTree.FromBMesh(bm)

    # Store triangle verts + uvs for barycentric UV lookup
    tris = []  # list of (v0, v1, v2, uv0, uv1, uv2, normal)
    for face in bm.faces:
        if len(face.loops) != 3:
            continue
        verts = [loop.vert.co.copy() for loop in face.loops]
        uvs = [loop[uv_lay].uv.copy() for loop in face.loops]
        tris.append((verts[0], verts[1], verts[2], uvs[0], uvs[1], uvs[2], face.normal.copy()))

    # Keep bm for now? Free after building arrays — BVH owns its data
    bm.free()

    return bvh, tris, mw, normal_mw

def _nearest_uv(bvh, tris, point, max_dist=None):
    hit = bvh.find_nearest(point)
    if hit is None or hit[0] is None:
        return None
    loc, normal, index, dist = hit
    if max_dist is not None and dist > max_dist:
        return None
    if index < 0 or index >= len(tris):
        return None

    v0, v1, v2, uv0, uv1, uv2, face_n = tris[index]
    # Barycentric UV
    try:
        # Use 2D barycentric in the triangle plane via areas
        uv = _barycentric_uv(loc, v0, v1, v2, uv0, uv1, uv2)
    except Exception:
        uv = (uv0 + uv1 + uv2) / 3.0

    n = face_n if face_n.length > 1e-8 else Vector(normal)
    if n.length > 1e-8:
        n.normalize()
    else:
        n = Vector((0, 0, 1))

    return loc, n, Vector((uv.x, uv.y)), dist

def _barycentric_uv(p, a, b, c, uva, uvb, uvc):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return (uva + uvb + uvc) / 3.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return uva * u + uvb * v + uvc * w

# ---------------------------------------------------------------------------
# Surface-space bake core
#
# Stamping curve samples into UV can never be exact: the result depends on the
# sample spacing and every curve point has to guess a UV. These helpers invert
# the mapping — each texel that belongs to the mesh is turned back into its 3D
# surface position and tested against the real curve geometry, so a bake is the
# true footprint of the curve on the surface.
# ---------------------------------------------------------------------------

_MAX_SURFACE_TEXELS = 8000000
_OCCLUSION_TEST_LIMIT = 60000

def _falloff_profile(dist, half_width, power=1.0):
    '''Cross-section opacity: 1 at the center, 0 at the edge.'''
    t = numpy.clip(dist / max(float(half_width), 1e-12), 0.0, 1.0)
    a = 1.0 - t * t * (3.0 - 2.0 * t)
    if power != 1.0:
        a = numpy.power(numpy.maximum(a, 0.0), max(float(power), 1e-4))
    return a.astype(numpy.float32)

def _smooth_rows(arr, radius, cyclic):
    '''Box blur along the first axis so a per-sample frame varies smoothly.'''
    if radius < 1 or arr.shape[0] < 3:
        return arr
    pad = int(min(radius, arr.shape[0] - 1))
    padded = numpy.pad(arr, ((pad, pad), (0, 0)), mode='wrap' if cyclic else 'edge')
    kernel = numpy.full(pad * 2 + 1, 1.0 / float(pad * 2 + 1), dtype=numpy.float32)
    out = numpy.empty_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = numpy.convolve(padded[:, col], kernel, mode='valid')
    return out

def _frame_from_dir_and_normal(dirs, nrms):
    '''
    Orthonormal (axis, side) for every direction: axis is the normal made
    perpendicular to the direction, side is across both.
    Shared along the curve so both faces of a UV seam use the same width axis.
    '''
    axis = nrms - numpy.einsum('ij,ij->i', nrms, dirs)[:, None] * dirs
    lens = numpy.linalg.norm(axis, axis=1)
    bad = lens < 1e-5
    if numpy.any(bad):
        helper = numpy.zeros((int(bad.sum()), 3), dtype=numpy.float32)
        d = dirs[bad]
        helper[numpy.arange(helper.shape[0]), numpy.argmin(numpy.abs(d), axis=1)] = 1.0
        alt = numpy.cross(d, helper)
        alt_lens = numpy.linalg.norm(alt, axis=1)
        alt_lens[alt_lens < 1e-12] = 1.0
        axis[bad] = alt / alt_lens[:, None]
        lens[bad] = 1.0
    axis = (axis / lens[:, None]).astype(numpy.float32)
    side = numpy.cross(dirs, axis).astype(numpy.float32)
    side_lens = numpy.linalg.norm(side, axis=1)
    side_lens[side_lens < 1e-12] = 1.0
    return axis, (side / side_lens[:, None]).astype(numpy.float32)

def _quantize_co(v, scale=1e5):
    return (int(round(float(v[0]) * scale)),
            int(round(float(v[1]) * scale)),
            int(round(float(v[2]) * scale)))

def _build_uv_seam_pairs(verts, uvs, tri_indices=None):
    '''
    3D edges that have two different UV images (UV seams).
    Returns list of (a3, b3, uv_a0, uv_b0, uv_a1, uv_b1) as float32 arrays.
    '''
    if tri_indices is None:
        tri_indices = numpy.arange(verts.shape[0], dtype=numpy.int64)
    else:
        tri_indices = numpy.asarray(tri_indices, dtype=numpy.int64)
    edges = {}
    for ti in tri_indices.tolist():
        for e in range(3):
            i0, i1 = e, (e + 1) % 3
            a = verts[ti, i0]
            b = verts[ti, i1]
            ka, kb = _quantize_co(a), _quantize_co(b)
            if ka <= kb:
                key = (ka, kb)
                ua, ub = uvs[ti, i0], uvs[ti, i1]
                a3, b3 = a, b
            else:
                key = (kb, ka)
                ua, ub = uvs[ti, i1], uvs[ti, i0]
                a3, b3 = b, a
            bucket = edges.get(key)
            entry = (ua.copy(), ub.copy(), a3.copy(), b3.copy())
            if bucket is None:
                edges[key] = [entry]
            else:
                bucket.append(entry)

    pairs = []
    for entries in edges.values():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            ua0, ub0, a3, b3 = entries[i]
            for j in range(i + 1, len(entries)):
                ua1, ub1, _, _ = entries[j]
                if (abs(float(ua0[0] - ua1[0])) + abs(float(ua0[1] - ua1[1])) > 1e-4
                        or abs(float(ub0[0] - ub1[0])) + abs(float(ub0[1] - ub1[1])) > 1e-4):
                    pairs.append((
                        a3.astype(numpy.float32), b3.astype(numpy.float32),
                        ua0.astype(numpy.float32), ub0.astype(numpy.float32),
                        ua1.astype(numpy.float32), ub1.astype(numpy.float32),
                    ))
    return pairs

def _splat_coverage_uv_segment(coverage, u0, v0, u1, v1, alpha, radius_px=2.0):
    '''Max-blend a short UV segment into a coverage buffer (island-border seal).'''
    img_h, img_w = coverage.shape
    du = (u1 - u0) * img_w
    dv = (v1 - v0) * img_h
    steps = max(1, int(math.ceil(math.hypot(du, dv) * 2.0)))
    r = max(1, int(math.ceil(radius_px)))
    for s in range(steps + 1):
        t = s / float(steps)
        px = (u0 * (1.0 - t) + u1 * t) * img_w - 0.5
        py = (v0 * (1.0 - t) + v1 * t) * img_h - 0.5
        cx, cy = int(round(px)), int(round(py))
        for y in range(max(0, cy - r), min(img_h, cy + r + 1)):
            for x in range(max(0, cx - r), min(img_w, cx + r + 1)):
                if (x - px) * (x - px) + (y - py) * (y - py) <= radius_px * radius_px:
                    if alpha > coverage[y, x]:
                        coverage[y, x] = alpha

def _seal_uv_seams_path(coverage, seam_pairs, segments, half_w, hover_ceiling, img_w, img_h):
    '''
    Paint both UV images of every seam edge that lies under the ribbon.
    Guarantees the atlas border on each island is filled so the 3D view is seamless.
    '''
    if not seam_pairs or segments is None:
        return
    start, delta, delta2, seg_len, arc_start, _total = segments
    end = start + delta
    box_min = numpy.minimum(start, end) - (half_w + hover_ceiling)
    box_max = numpy.maximum(start, end) + (half_w + hover_ceiling)
    radius_px = 1.0
    for a3, b3, ua0, ub0, ua1, ub1 in seam_pairs:
        mid = 0.5 * (a3 + b3)
        # Cheap reject: no segment bbox reaches this edge
        if not numpy.any(numpy.all((box_max >= mid) & (box_min <= mid), axis=1)):
            continue
        edge = b3 - a3
        edge_len = float(numpy.linalg.norm(edge))
        n_samp = max(2, int(math.ceil(edge_len / max(half_w * 0.35, 1e-5))) + 1)
        for s in range(n_samp):
            t = s / float(n_samp - 1)
            p = a3 * (1.0 - t) + b3 * t
            rel = p[None, :] - start
            tt = numpy.einsum('ij,ij->i', rel, delta) / delta2
            numpy.clip(tt, 0.0, 1.0, out=tt)
            proj = start + tt[:, None] * delta
            diff = p[None, :] - proj
            d2 = numpy.einsum('ij,ij->i', diff, diff)
            best = int(numpy.argmin(d2))
            dist = math.sqrt(float(d2[best]))
            if dist > half_w * 1.05:
                continue
            alpha = float(max(0.0, 1.0 - dist / max(half_w, 1e-9)))
            if alpha < 1e-3:
                continue
            u0 = float(ua0[0] * (1.0 - t) + ub0[0] * t)
            v0 = float(ua0[1] * (1.0 - t) + ub0[1] * t)
            u1 = float(ua1[0] * (1.0 - t) + ub1[0] * t)
            v1 = float(ua1[1] * (1.0 - t) + ub1[1] * t)
            # Point stamps only — stroking the full UV edge paints along mesh
            # edges and shows up as jagged bleed where the ribbon crosses them.
            _splat_coverage_uv_segment(coverage, u0, v0, u0, v0, alpha, radius_px)
            _splat_coverage_uv_segment(coverage, u1, v1, u1, v1, alpha, radius_px)

def _seal_uv_seams_shape(coverage, seam_pairs, mask_sample, origin, u_axis, v_axis, n_axis, depth_limit, img_w, img_h):
    '''Seal both UV images of seam edges that fall inside the shape mask.'''
    if not seam_pairs:
        return
    radius_px = 1.0
    for a3, b3, ua0, ub0, ua1, ub1 in seam_pairs:
        mid = 0.5 * (a3 + b3)
        rel = mid - origin
        pu = float(numpy.dot(rel, u_axis))
        pv = float(numpy.dot(rel, v_axis))
        pd = float(numpy.dot(rel, n_axis))
        if abs(pd) > depth_limit:
            continue
        alpha = float(mask_sample(pu, pv))
        if alpha < 1e-3:
            continue
        # Point stamps at the edge midpoint (both UV images) — avoid full-edge
        # strokes that thicken along mesh edges in the 3D view.
        u0 = 0.5 * (float(ua0[0]) + float(ub0[0]))
        v0 = 0.5 * (float(ua0[1]) + float(ub0[1]))
        u1 = 0.5 * (float(ua1[0]) + float(ub1[0]))
        v1 = 0.5 * (float(ua1[1]) + float(ub1[1]))
        _splat_coverage_uv_segment(coverage, u0, v0, u0, v0, alpha, radius_px)
        _splat_coverage_uv_segment(coverage, u1, v1, u1, v1, alpha, radius_px)

def _tris_to_arrays(tris):
    '''Triangle soup as arrays: verts (T,3,3), uvs (T,3,2), unit normals (T,3).'''
    verts = numpy.array(
        [[[t[0].x, t[0].y, t[0].z],
          [t[1].x, t[1].y, t[1].z],
          [t[2].x, t[2].y, t[2].z]] for t in tris],
        dtype=numpy.float32
    )
    uvs = numpy.array(
        [[[t[3].x, t[3].y], [t[4].x, t[4].y], [t[5].x, t[5].y]] for t in tris],
        dtype=numpy.float32
    )
    nrms = numpy.array([[t[6].x, t[6].y, t[6].z] for t in tris], dtype=numpy.float32)
    lens = numpy.linalg.norm(nrms, axis=1)
    lens[lens < 1e-12] = 1.0
    nrms /= lens[:, None]
    return verts, uvs, nrms

def _tris_near_points(bvh, tri_count, points, radius):
    '''Triangle indices within radius of any point (None if BVH lacks range query).'''
    query = getattr(bvh, 'find_nearest_range', None)
    if not query:
        return None
    found = set()
    for p in points:
        try:
            hits = query(p, radius)
        except Exception:
            return None
        if not hits:
            continue
        for hit in hits:
            index = hit[2] if hit else None
            if index is not None and 0 <= index < tri_count:
                found.add(index)
    if not found:
        return numpy.zeros(0, dtype=numpy.int64)
    return numpy.fromiter(sorted(found), dtype=numpy.int64, count=len(found))

def _gather_surface_texels(tri_indices, verts, uvs, nrms, img_w, img_h, dilate_px=1.0):
    '''
    Rasterize triangles in UV space and map every covered texel back onto the
    surface. Returns (ys, xs, points, normals, world_per_px) or None.

    dilate_px grows each UV triangle so island borders still receive the texels
    that sit on the shared 3D edge. Keep this modest — large dilate clamps
    barycentrics onto mesh edges and creates jagged ribbon bleed.
    '''
    tri_idx = numpy.asarray(tri_indices)
    if tri_idx.size == 0:
        return None

    # Everything that does not depend on the texel grid is done in one batch
    tri_v = verts[tri_idx].astype(numpy.float64)
    tri_uv = uvs[tri_idx].astype(numpy.float64)
    px = tri_uv[:, :, 0] * img_w
    py = tri_uv[:, :, 1] * img_h
    ax, bx, cx = px[:, 0], px[:, 1], px[:, 2]
    ay, by, cy = py[:, 0], py[:, 1], py[:, 2]

    area2 = numpy.abs((by - cy) * (ax - cx) + (cx - bx) * (ay - cy))
    safe_area2 = numpy.maximum(area2, 1e-12)
    # Only triangles that are mapped inside this UV tile can be baked
    on_sheet = (
        (px.max(axis=1) >= -dilate_px) & (px.min(axis=1) <= img_w + dilate_px)
        & (py.max(axis=1) >= -dilate_px) & (py.min(axis=1) <= img_h + dilate_px)
    )
    inv_den = numpy.where(
        on_sheet & (area2 > 1e-12),
        1.0 / numpy.where(area2 > 1e-12, (by - cy) * (ax - cx) + (cx - bx) * (ay - cy), 1.0),
        0.0
    )
    # Grow each triangle by dilate_px so UV island borders leave no gap
    tol0 = dilate_px * numpy.hypot(bx - cx, by - cy) / safe_area2
    tol1 = dilate_px * numpy.hypot(cx - ax, cy - ay) / safe_area2
    tol2 = dilate_px * numpy.hypot(ax - bx, ay - by) / safe_area2

    edge_a = tri_v[:, 1] - tri_v[:, 0]
    edge_b = tri_v[:, 2] - tri_v[:, 0]
    crs_x = edge_a[:, 1] * edge_b[:, 2] - edge_a[:, 2] * edge_b[:, 1]
    crs_y = edge_a[:, 2] * edge_b[:, 0] - edge_a[:, 0] * edge_b[:, 2]
    crs_z = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    area_3d = 0.5 * numpy.sqrt(crs_x * crs_x + crs_y * crs_y + crs_z * crs_z)
    scale = numpy.sqrt(numpy.maximum(area_3d, 1e-20) / (0.5 * safe_area2))

    bx0 = numpy.clip(numpy.floor(px.min(axis=1) - dilate_px - 1.0), 0, img_w - 1)
    bx1 = numpy.clip(numpy.ceil(px.max(axis=1) + dilate_px + 1.0), 0, img_w - 1)
    by0 = numpy.clip(numpy.floor(py.min(axis=1) - dilate_px - 1.0), 0, img_h - 1)
    by1 = numpy.clip(numpy.ceil(py.max(axis=1) + dilate_px + 1.0), 0, img_h - 1)

    l_ax, l_bx, l_cx = ax.tolist(), bx.tolist(), cx.tolist()
    l_ay, l_by, l_cy = ay.tolist(), by.tolist(), cy.tolist()
    l_inv = inv_den.tolist()
    l_t0, l_t1, l_t2 = tol0.tolist(), tol1.tolist(), tol2.tolist()
    l_scale = scale.tolist()
    l_x0, l_x1 = bx0.astype(numpy.int64).tolist(), bx1.astype(numpy.int64).tolist()
    l_y0, l_y1 = by0.astype(numpy.int64).tolist(), by1.astype(numpy.int64).tolist()

    ys_all = []
    xs_all = []
    pts_all = []
    nrm_all = []
    scale_all = []
    total = 0

    for i in range(tri_idx.shape[0]):
        inv = l_inv[i]
        if inv == 0.0:
            continue
        x0, x1 = l_x0[i], l_x1[i]
        y0, y1 = l_y0[i], l_y1[i]
        if x1 < x0 or y1 < y0:
            continue

        p_ax, p_bx, p_cx = l_ax[i], l_bx[i], l_cx[i]
        p_ay, p_by, p_cy = l_ay[i], l_by[i], l_cy[i]

        # Barycentric coordinates of every texel center in the bounding box
        gx = numpy.arange(x0, x1 + 1, dtype=numpy.float64) + (0.5 - p_cx)
        gy = numpy.arange(y0, y1 + 1, dtype=numpy.float64) + (0.5 - p_cy)
        w0 = ((p_by - p_cy) * gx[None, :] + (p_cx - p_bx) * gy[:, None]) * inv
        w1 = ((p_cy - p_ay) * gx[None, :] + (p_ax - p_cx) * gy[:, None]) * inv
        w2 = 1.0 - w0 - w1

        covered = (w0 >= -l_t0[i]) & (w1 >= -l_t1[i]) & (w2 >= -l_t2[i])
        rows, cols = numpy.nonzero(covered)
        if rows.size == 0:
            continue

        # Do not clamp — clamping snaps dilated texels onto triangle edges and
        # creates jagged ribbon bleed where the path crosses mesh edges.
        b0 = w0[rows, cols]
        b1 = w1[rows, cols]
        b2 = w2[rows, cols]
        wsum = b0 + b1 + b2
        wsum[numpy.abs(wsum) < 1e-12] = 1.0
        b0 /= wsum
        b1 /= wsum
        b2 /= wsum

        tri = tri_v[i]
        pts = (b0[:, None] * tri[0][None, :]
               + b1[:, None] * tri[1][None, :]
               + b2[:, None] * tri[2][None, :])
        count = pts.shape[0]

        ys_all.append((rows + y0).astype(numpy.int32))
        xs_all.append((cols + x0).astype(numpy.int32))
        pts_all.append(pts.astype(numpy.float32))
        nrm_all.append(numpy.broadcast_to(nrms[tri_idx[i]], (count, 3)))
        scale_all.append(numpy.full(count, l_scale[i], dtype=numpy.float32))

        total += count
        if total >= _MAX_SURFACE_TEXELS:
            break

    if not ys_all:
        return None
    return (
        numpy.concatenate(ys_all),
        numpy.concatenate(xs_all),
        numpy.concatenate(pts_all),
        numpy.concatenate(nrm_all),
        numpy.concatenate(scale_all),
    )

def _uv_tri_texel_area(uvs, tri_idx, img_w, img_h):
    '''Doubled texel-space area of each triangle.'''
    tuv = uvs[tri_idx].astype(numpy.float64)
    px = tuv[:, :, 0] * img_w
    py = tuv[:, :, 1] * img_h
    return numpy.abs(
        (py[:, 1] - py[:, 2]) * (px[:, 0] - px[:, 2])
        + (px[:, 2] - px[:, 1]) * (py[:, 0] - py[:, 2])
    )

def _median_texel_size(verts, uvs, tri_idx, img_w, img_h):
    '''Median world size of one texel over the given triangles.'''
    if tri_idx.size == 0:
        return 0.0
    area2 = _uv_tri_texel_area(uvs, tri_idx, img_w, img_h)
    tv = verts[tri_idx].astype(numpy.float64)
    edge_a = tv[:, 1] - tv[:, 0]
    edge_b = tv[:, 2] - tv[:, 0]
    crs_x = edge_a[:, 1] * edge_b[:, 2] - edge_a[:, 2] * edge_b[:, 1]
    crs_y = edge_a[:, 2] * edge_b[:, 0] - edge_a[:, 0] * edge_b[:, 2]
    crs_z = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    area_3d2 = numpy.sqrt(crs_x * crs_x + crs_y * crs_y + crs_z * crs_z)
    valid = area2 > 1e-9
    if not numpy.any(valid):
        return 0.0
    return float(numpy.median(numpy.sqrt(area_3d2[valid] / area2[valid])))

def _batch_tri_indices(uvs, tri_idx, img_w, img_h, max_texels=2000000):
    '''Split candidate triangles into batches with a bounded texel count.'''
    if tri_idx.size == 0:
        return []
    cost = _uv_tri_texel_area(uvs, tri_idx, img_w, img_h) * 0.5 + 8.0
    groups = numpy.floor(numpy.cumsum(cost) / float(max_texels)).astype(numpy.int64)
    edges = numpy.nonzero(numpy.diff(groups))[0] + 1
    return numpy.split(tri_idx, edges)

def _polyline_to_segments(points, cyclic):
    '''(start, delta, delta^2, length, arc start, total length) segment arrays.'''
    coords = numpy.array([[p.x, p.y, p.z] for p in points], dtype=numpy.float32)
    if cyclic and coords.shape[0] >= 3:
        coords = numpy.vstack([coords, coords[:1]])
    if coords.shape[0] < 2:
        return None
    start = coords[:-1]
    delta = coords[1:] - start
    delta2 = numpy.maximum(numpy.einsum('ij,ij->i', delta, delta), 1e-20).astype(numpy.float32)
    length = numpy.sqrt(delta2).astype(numpy.float32)
    arc_start = numpy.concatenate([
        numpy.zeros(1, dtype=numpy.float32), numpy.cumsum(length)[:-1]
    ]).astype(numpy.float32)
    return start, delta, delta2, length, arc_start, float(length.sum())

def _sample_surface_gap_normals(bvh, tris, polyline, far_away, max_samples=256):
    '''
    Surface normal + gap under each curve sample.
    Dense polylines are thinned first so BVH queries stay cheap; results are
    resampled back to the full sample count.
    '''
    count = len(polyline)
    if count == 0:
        return (
            numpy.zeros((0, 3), dtype=numpy.float32),
            numpy.zeros(0, dtype=numpy.float32),
        )
    # Query a capped subset, then interpolate along the polyline index
    query = _decimate_points(polyline, max_samples)
    q_n = len(query)
    q_nrm = numpy.empty((q_n, 3), dtype=numpy.float32)
    q_gap = numpy.empty(q_n, dtype=numpy.float32)
    for i, p in enumerate(query):
        hit = _nearest_uv(bvh, tris, p)
        if hit:
            n = hit[1]
            q_nrm[i] = (n.x, n.y, n.z)
            q_gap[i] = float(hit[3])
        else:
            q_nrm[i] = (0.0, 0.0, 1.0)
            q_gap[i] = far_away
    if q_n == count:
        return q_nrm, q_gap
    # Map full indices onto the decimated query
    src = numpy.linspace(0.0, q_n - 1, count, dtype=numpy.float32)
    i0 = numpy.floor(src).astype(numpy.int32)
    i1 = numpy.minimum(i0 + 1, q_n - 1)
    t = (src - i0.astype(numpy.float32))[:, None]
    nrm = q_nrm[i0] * (1.0 - t) + q_nrm[i1] * t
    gap = q_gap[i0] * (1.0 - src + i0.astype(numpy.float32)) + q_gap[i1] * (src - i0.astype(numpy.float32))
    return nrm.astype(numpy.float32), gap.astype(numpy.float32)

def _build_one_path_spline_bundle(polyline, cyclic, bvh, tris, far_away):
    '''Frame + segment arrays for a single spline.'''
    nrm_pts, gap_pts = _sample_surface_gap_normals(bvh, tris, polyline, far_away)
    segments = _polyline_to_segments(polyline, cyclic)
    if segments is None:
        return None
    start, delta, delta2, seg_len, arc_start, total_len = segments
    seg_count = delta.shape[0]
    nrm_pts = _smooth_rows(nrm_pts, 2, cyclic)
    if cyclic and nrm_pts.shape[0] >= 3:
        seg_nrm = nrm_pts + numpy.roll(nrm_pts, -1, axis=0)
        seg_gap = numpy.maximum(gap_pts, numpy.roll(gap_pts, -1))
    else:
        seg_nrm = nrm_pts[:-1] + nrm_pts[1:]
        seg_gap = numpy.maximum(gap_pts[:-1], gap_pts[1:])
    nrm_lens = numpy.linalg.norm(seg_nrm, axis=1)
    nrm_lens[nrm_lens < 1e-12] = 1.0
    seg_nrm = (seg_nrm / nrm_lens[:, None]).astype(numpy.float32)
    seg_gap = seg_gap[:seg_count].astype(numpy.float32)
    seg_dir = (delta / seg_len[:, None]).astype(numpy.float32)
    seg_axis, seg_side = _frame_from_dir_and_normal(seg_dir, seg_nrm)
    open_start = numpy.zeros(seg_count, dtype=bool)
    open_end = numpy.zeros(seg_count, dtype=bool)
    if not cyclic and seg_count > 0:
        open_start[0] = True
        open_end[-1] = True
    return {
        'segments': segments,
        'seg_nrm': seg_nrm,
        'seg_gap': seg_gap,
        'seg_dir': seg_dir,
        'seg_axis': seg_axis,
        'seg_side': seg_side,
        'path_len': max(total_len, 1e-9),
        'open_start': open_start,
        'open_end': open_end,
        'cyclic': bool(cyclic),
    }

def _build_path_spline_bundles(prepared, bvh, tris, far_away):
    '''
    One bundle per spline. Overlapping ribbons must be evaluated separately so a
    texel claimed by the nearer curve (but outside its width) can still be
    painted by another curve that actually covers it.
    '''
    bundles = []
    for polyline, cyclic in prepared:
        bundle = _build_one_path_spline_bundle(polyline, cyclic, bvh, tris, far_away)
        if bundle is not None:
            bundles.append(bundle)
    return bundles

def _merge_path_spline_bundles(prepared, bvh, tris, far_away):
    '''Legacy helper: pack every spline into one discontinuous segment soup.'''
    bundles = _build_path_spline_bundles(prepared, bvh, tris, far_away)
    if not bundles:
        return None
    return {
        'segments': (
            numpy.concatenate([b['segments'][0] for b in bundles]),
            numpy.concatenate([b['segments'][1] for b in bundles]),
            numpy.concatenate([b['segments'][2] for b in bundles]),
            numpy.concatenate([b['segments'][3] for b in bundles]),
            numpy.concatenate([b['segments'][4] for b in bundles]),
            float(sum(b['segments'][3].sum() for b in bundles)),
        ),
        'seg_nrm': numpy.concatenate([b['seg_nrm'] for b in bundles]),
        'seg_gap': numpy.concatenate([b['seg_gap'] for b in bundles]),
        'seg_dir': numpy.concatenate([b['seg_dir'] for b in bundles]),
        'seg_axis': numpy.concatenate([b['seg_axis'] for b in bundles]),
        'seg_side': numpy.concatenate([b['seg_side'] for b in bundles]),
        'path_len': numpy.concatenate([
            numpy.full(b['segments'][0].shape[0], b['path_len'], dtype=numpy.float32)
            for b in bundles
        ]),
        'open_start': numpy.concatenate([b['open_start'] for b in bundles]),
        'open_end': numpy.concatenate([b['open_end'] for b in bundles]),
        'spline_count': len(bundles),
    }

def _closest_on_polyline(points, segments, search_radius, chunk_elems=1500000):
    '''
    Closest point on the polyline for every point.
    Returns (distance, arc length, segment t, segment index, closest point).
    Distance stays inf where no segment is within search_radius.

    Strategy: hash segment starts into a coarse grid, pick the nearest sample
    per point, then refine against a few neighbouring segments only.
    '''
    start, delta, delta2, seg_len, arc_start, _total = segments
    count = points.shape[0]
    seg_count = start.shape[0]
    dist = numpy.full(count, numpy.inf, dtype=numpy.float32)
    arc = numpy.zeros(count, dtype=numpy.float32)
    seg_t = numpy.zeros(count, dtype=numpy.float32)
    seg_i = numpy.zeros(count, dtype=numpy.int32)
    closest = numpy.zeros((count, 3), dtype=numpy.float32)
    if count == 0 or seg_count == 0:
        return dist, arc, seg_t, seg_i, closest

    radius = max(float(search_radius), 1e-9)
    cell = radius

    # Cheap reject: ignore texels outside the dilated curve bounds
    end_pts = start + delta
    world_lo = numpy.minimum(start, end_pts).min(axis=0) - radius
    world_hi = numpy.maximum(start, end_pts).max(axis=0) + radius
    in_bounds = numpy.all((points >= world_lo) & (points <= world_hi), axis=1)
    active = numpy.nonzero(in_bounds)[0]
    if active.size == 0:
        return dist, arc, seg_t, seg_i, closest

    # Stamp each segment start into its own cell. Queries search the 3×3×3
    # neighbourhood so a point still sees every sample within ~radius.
    grid = {}
    skeys = numpy.floor(start / cell).astype(numpy.int64)
    for si in range(seg_count):
        key = (int(skeys[si, 0]), int(skeys[si, 1]), int(skeys[si, 2]))
        bucket = grid.get(key)
        if bucket is None:
            grid[key] = [si]
        else:
            bucket.append(si)
    if not grid:
        return dist, arc, seg_t, seg_i, closest
    for key, vals in list(grid.items()):
        grid[key] = numpy.asarray(vals, dtype=numpy.int32)

    sub = points[active]
    keys = numpy.floor(sub / cell).astype(numpy.int64)
    order = numpy.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    if active.size > 1:
        edges = numpy.nonzero(numpy.any(numpy.diff(sorted_keys, axis=0) != 0, axis=1))[0] + 1
        buckets = numpy.split(order, edges)
    else:
        buckets = [order]

    refine = 2
    r2 = radius * radius
    refine_offs = numpy.arange(-refine, refine + 1, dtype=numpy.int32)

    for bucket in buckets:
        local = bucket  # indices into sub/keys
        p0 = int(local[0])
        cx, cy, cz = int(keys[p0, 0]), int(keys[p0, 1]), int(keys[p0, 2])
        sample_lists = []
        for ix in range(cx - 1, cx + 2):
            for iy in range(cy - 1, cy + 2):
                for iz in range(cz - 1, cz + 2):
                    hit = grid.get((ix, iy, iz))
                    if hit is not None:
                        sample_lists.append(hit)
        if not sample_lists:
            continue
        samples = numpy.unique(numpy.concatenate(sample_lists)) if len(sample_lists) > 1 else sample_lists[0]
        if samples.size == 0:
            continue

        sample_pts = start[samples]
        step = max(256, int(chunk_elems // max(int(samples.size), 1)))
        for base in range(0, local.shape[0], step):
            loc = local[base:base + step]
            where = active[loc]
            chunk = sub[loc]
            diff_s = chunk[:, None, :] - sample_pts[None, :, :]
            d2_s = numpy.einsum('msk,msk->ms', diff_s, diff_s)
            best_s = numpy.argmin(d2_s, axis=1)
            seed = samples[best_s]
            rough = d2_s[numpy.arange(chunk.shape[0]), best_s]
            cand = rough <= (r2 * 4.0)
            if not numpy.any(cand):
                continue

            where_c = where[cand]
            seed_c = seed[cand]
            chunk_c = chunk[cand]

            # Per-point window of neighbouring segments — fully vectorized refine
            near = numpy.clip(seed_c[:, None] + refine_offs[None, :], 0, seg_count - 1)
            seg_a = start[near]
            seg_d = delta[near]
            seg_2 = delta2[near]
            rel = chunk_c[:, None, :] - seg_a
            t = numpy.einsum('msk,msk->ms', rel, seg_d) / seg_2
            numpy.clip(t, 0.0, 1.0, out=t)
            proj = seg_a + t[:, :, None] * seg_d
            diff = chunk_c[:, None, :] - proj
            d2 = numpy.einsum('msk,msk->ms', diff, diff)

            best = numpy.argmin(d2, axis=1)
            rows = numpy.arange(chunk_c.shape[0])
            picked = near[rows, best]
            best_t = t[rows, best]
            best_d2 = d2[rows, best]
            ok = best_d2 <= r2
            if not numpy.any(ok):
                continue
            idx_ok = where_c[ok]
            best_d = numpy.sqrt(best_d2[ok])
            dist[idx_ok] = best_d
            seg_t[idx_ok] = best_t[ok]
            seg_i[idx_ok] = picked[ok]
            arc[idx_ok] = arc_start[picked[ok]] + best_t[ok] * seg_len[picked[ok]]
            closest[idx_ok] = proj[rows[ok], best[ok]]

    return dist, arc, seg_t, seg_i, closest

def _reject_occluded(bvh, origins, targets, grazing=0.15):
    '''
    True where the mesh blocks the straight line from origin to target.
    Surfaces the line only skims are ignored: those are neighbouring faces of a
    mesh edge or UV seam the ray runs along, not something standing in the way.
    '''
    count = targets.shape[0]
    blocked = numpy.zeros(count, dtype=bool)
    if count == 0 or count > _OCCLUSION_TEST_LIMIT:
        return blocked
    org = origins.astype(numpy.float64)
    tgt = targets.astype(numpy.float64)
    for i in range(count):
        origin = Vector((org[i, 0], org[i, 1], org[i, 2]))
        target = Vector((tgt[i, 0], tgt[i, 1], tgt[i, 2]))
        ray = target - origin
        span = ray.length
        if span < 1e-6:
            continue
        ray /= span
        try:
            hit = bvh.ray_cast(origin + ray * (span * 0.03), ray, span * 0.92)
        except Exception:
            hit = None
        if not hit or hit[0] is None:
            continue
        normal = hit[1]
        if normal is not None and abs(normal.dot(ray)) < grazing:
            continue
        blocked[i] = True
    return blocked

class _BakeAccumulator:
    '''Collects texel coverage from every loop, then resolves it per texel.'''

    def __init__(self, img_w, img_h, use_texture):
        self.img_w = img_w
        self.img_h = img_h
        self.use_texture = use_texture
        self.count = 0
        self._ys = []
        self._xs = []
        self._alpha = []
        self._rgb = []

    def add(self, ys, xs, alpha, rgb=None):
        keep = alpha > 1e-4
        kept = int(numpy.count_nonzero(keep))
        if kept == 0:
            return 0
        self._ys.append(ys[keep])
        self._xs.append(xs[keep])
        self._alpha.append(alpha[keep].astype(numpy.float32))
        if self.use_texture:
            if rgb is None:
                self._rgb.append(numpy.ones((kept, 3), dtype=numpy.float32))
            else:
                self._rgb.append(rgb[keep].astype(numpy.float32))
        self.count += kept
        return kept

    def resolve(self):
        '''(coverage, rgb) buffers — the strongest sample wins each texel.'''
        coverage = numpy.zeros((self.img_h, self.img_w), dtype=numpy.float32)
        if not self._ys:
            return coverage, None
        ys = numpy.concatenate(self._ys)
        xs = numpy.concatenate(self._xs)
        alpha = numpy.concatenate(self._alpha)
        numpy.maximum.at(coverage, (ys, xs), alpha)
        rgb_buf = None
        if self.use_texture and self._rgb:
            rgb_buf = numpy.zeros((self.img_h, self.img_w, 3), dtype=numpy.float32)
            order = numpy.argsort(alpha, kind='stable')
            rgb_buf[ys[order], xs[order]] = numpy.concatenate(self._rgb)[order]
        return coverage, rgb_buf

def _read_image_pixels(image, img_w, img_h):
    if is_bl_newer_than(2, 83):
        pxs = numpy.empty(shape=img_w * img_h * 4, dtype=numpy.float32)
        image.pixels.foreach_get(pxs)
    else:
        pxs = numpy.array(image.pixels[:], dtype=numpy.float32)
    pxs.shape = (img_h, img_w, 4)
    return pxs

def _write_image_pixels(image, pxs):
    flat = pxs.ravel()
    if is_bl_newer_than(2, 83):
        image.pixels.foreach_set(flat)
    else:
        image.pixels = flat.tolist()
    image.update()

def _load_texture_pixels(path_texture):
    '''(pixels (h,w,4), width, height) for an optional stamp texture.'''
    if not path_texture or path_texture.size[0] < 1 or path_texture.size[1] < 1:
        return None, 0, 0
    tw, th = path_texture.size[0], path_texture.size[1]
    if is_bl_newer_than(2, 83):
        tex_pxs = numpy.empty(shape=tw * th * 4, dtype=numpy.float32)
        path_texture.pixels.foreach_get(tex_pxs)
    else:
        tex_pxs = numpy.array(path_texture.pixels[:], dtype=numpy.float32)
    tex_pxs.shape = (th, tw, 4)
    return tex_pxs, tw, th

def _composite_bake(pxs, coverage, rgb_buf, color, clear):
    '''Blend a coverage buffer into the destination pixels.'''
    cr, cg, cb, ca = [float(c) for c in color]
    src_a = numpy.clip(coverage * ca, 0.0, 1.0).astype(numpy.float32)
    touched = src_a > 1e-5
    tint = numpy.array((cr, cg, cb), dtype=numpy.float32)
    if rgb_buf is not None:
        src_rgb = rgb_buf * tint[None, None, :]
    else:
        src_rgb = numpy.broadcast_to(tint, (pxs.shape[0], pxs.shape[1], 3))

    if clear:
        for ch in range(3):
            pxs[:, :, ch] = numpy.where(touched, src_rgb[:, :, ch], 0.0)
        pxs[:, :, 3] = numpy.where(touched, src_a, 0.0)
        return int(numpy.count_nonzero(touched))

    dst_a = pxs[:, :, 3].copy()
    out_a = src_a + dst_a * (1.0 - src_a)
    safe = numpy.maximum(out_a, 1e-8)
    for ch in range(3):
        pxs[:, :, ch] = numpy.where(
            touched,
            (src_rgb[:, :, ch] * src_a + pxs[:, :, ch] * dst_a * (1.0 - src_a)) / safe,
            pxs[:, :, ch],
        )
    pxs[:, :, 3] = numpy.where(touched, numpy.minimum(out_a, 1.0), dst_a)
    return int(numpy.count_nonzero(touched))

def _sample_path_texture_batch(tex_pxs, tw, th, u_len, v_across, tile_u=1.0, rotation_deg=0.0):
    '''Vectorized bilinear texture sample. u_len/v_across are 1D float arrays.'''
    uu = numpy.mod(u_len * float(tile_u), 1.0)
    uu = numpy.where(uu < 0.0, uu + 1.0, uu)
    vv = v_across * 0.5 + 0.5

    if rotation_deg:
        rad = math.radians(rotation_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x = uu - 0.5
        y = vv - 0.5
        uu = x * cos_a - y * sin_a + 0.5
        vv = x * sin_a + y * cos_a + 0.5
        uu = numpy.mod(uu, 1.0)
        vv = numpy.mod(vv, 1.0)
        uu = numpy.where(uu < 0.0, uu + 1.0, uu)
        vv = numpy.where(vv < 0.0, vv + 1.0, vv)
    else:
        vv = numpy.clip(vv, 0.0, 1.0)

    x = uu * float(tw - 1)
    y = vv * float(th - 1)
    x0 = numpy.floor(x).astype(numpy.int32)
    y0 = numpy.floor(y).astype(numpy.int32)
    x1 = numpy.minimum(x0 + 1, tw - 1)
    y1 = numpy.minimum(y0 + 1, th - 1)
    fx = (x - x0).astype(numpy.float32)[:, None]
    fy = (y - y0).astype(numpy.float32)[:, None]
    c00 = tex_pxs[y0, x0]
    c10 = tex_pxs[y0, x1]
    c01 = tex_pxs[y1, x0]
    c11 = tex_pxs[y1, x1]
    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy

def _fit_polyline_plane(polyline):
    '''Return (center, axis_u, axis_v, normal) for the best-fit plane of the polyline.'''
    n = len(polyline)
    center = Vector((0.0, 0.0, 0.0))
    for p in polyline:
        center += p
    center /= float(max(n, 1))

    coords = numpy.array([[p.x, p.y, p.z] for p in polyline], dtype=numpy.float64)
    centered = coords - numpy.array((center.x, center.y, center.z), dtype=numpy.float64)
    cov = centered.T @ centered / float(max(n, 1))
    try:
        _eigvals, eigvecs = numpy.linalg.eigh(cov)
        normal = Vector(eigvecs[:, 0].tolist())
    except Exception:
        normal = Vector((0.0, 0.0, 1.0))
    if normal.length < 1e-8:
        normal = Vector((0.0, 0.0, 1.0))
    else:
        normal.normalize()

    return _plane_frame_from_center_normal(center, normal)

def _plane_frame_from_center_normal(center, normal):
    '''Orthonormal (center, axis_u, axis_v, normal) for a projection plane.'''
    n = Vector(normal)
    if n.length < 1e-8:
        n = Vector((0.0, 0.0, 1.0))
    else:
        n.normalize()
    axis_u = n.cross(Vector((0.0, 0.0, 1.0)))
    if axis_u.length < 1e-6:
        axis_u = n.cross(Vector((0.0, 1.0, 0.0)))
    if axis_u.length < 1e-8:
        axis_u = Vector((1.0, 0.0, 0.0))
    else:
        axis_u.normalize()
    axis_v = n.cross(axis_u)
    if axis_v.length < 1e-8:
        axis_v = Vector((0.0, 1.0, 0.0))
    else:
        axis_v.normalize()
    return center, axis_u, axis_v, n

def _world_project_normal(curve_obj, project_axes, fallback_obj=None):
    '''
    World-space bake/shrinkwrap normal from enabled X/Y/Z flags.
    Axes are taken in the curve object's local space (same as Shrinkwrap Project).
    Returns None when no axis is enabled (caller should auto-fit).
    '''
    ax, ay, az = project_axes
    local = Vector((1.0 if ax else 0.0, 1.0 if ay else 0.0, 1.0 if az else 0.0))
    if local.length < 1e-8:
        return None
    space = curve_obj if (curve_obj and curve_obj.type == 'CURVE') else fallback_obj
    if space is not None:
        normal = space.matrix_world.to_3x3() @ local
    else:
        normal = local
    if normal.length < 1e-8:
        return None
    normal.normalize()
    return normal

def _find_viewport_region_3d(context=None):
    '''Return (space, region_3d) for the active / first 3D Viewport, or (None, None).'''
    ctx = context if context is not None else bpy.context
    rv3d = getattr(ctx, 'region_data', None)
    if rv3d is not None:
        space = getattr(ctx, 'space_data', None)
        return space, rv3d

    screen = getattr(ctx, 'screen', None)
    if screen is None:
        window = getattr(ctx, 'window', None)
        screen = getattr(window, 'screen', None) if window else None
    if screen is None:
        wm = getattr(ctx, 'window_manager', None)
        if wm and getattr(wm, 'windows', None):
            for window in wm.windows:
                if window.screen:
                    screen = window.screen
                    break
    if screen is None:
        return None, None

    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        space = area.spaces.active if hasattr(area.spaces, 'active') else None
        if space is None and len(area.spaces) > 0:
            space = area.spaces[0]
        rv3d = getattr(space, 'region_3d', None) if space else None
        if rv3d is not None:
            return space, rv3d
    return None, None

def _get_viewport_view_state(context=None):
    '''
    Capture the active 3D Viewport look: direction, rotation, orbit center, distance.
    Returns a dict or None.
    '''
    _space, rv3d = _find_viewport_region_3d(context)
    if rv3d is None:
        return None
    try:
        rotation = rv3d.view_rotation.copy()
    except Exception:
        return None
    direction = rotation @ Vector((0.0, 0.0, -1.0))
    if direction.length < 1e-8:
        return None
    direction.normalize()
    try:
        location = rv3d.view_location.copy()
    except Exception:
        location = Vector((0.0, 0.0, 0.0))
    try:
        distance = float(rv3d.view_distance)
    except Exception:
        distance = 10.0
    return {
        'direction': direction,
        'rotation': rotation,
        'location': location,
        'distance': distance,
    }

def _get_viewport_view_direction(context=None):
    '''
    World-space direction the active 3D Viewport is looking (into the screen).
    Used when baking with Project Axis → View.
    '''
    state = _get_viewport_view_state(context)
    if state is None:
        return None
    return state['direction']

def layer_has_saved_path_view(layer):
    return bool(getattr(layer, 'path_view_saved', False))

def get_layer_saved_view_direction(layer):
    '''Return saved bake view direction, or None.'''
    if not layer_has_saved_path_view(layer):
        return None
    d = Vector(getattr(layer, 'path_view_direction', (0.0, 0.0, -1.0)))
    if d.length < 1e-8:
        return None
    d.normalize()
    return d

def save_layer_path_view(layer, context=None, state=None):
    '''Store the current (or provided) viewport state on the layer for later rebakes.'''
    if state is None:
        state = _get_viewport_view_state(context)
    if state is None:
        return False
    direction = state['direction']
    rotation = state['rotation']
    location = state['location']
    distance = state['distance']
    layer.path_view_direction = (float(direction.x), float(direction.y), float(direction.z))
    # Quaternion: Blender stores (w, x, y, z)
    layer.path_view_rotation = (
        float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
    )
    layer.path_view_location = (float(location.x), float(location.y), float(location.z))
    layer.path_view_distance = max(float(distance), 1e-4)
    layer.path_view_saved = True
    return True

def clear_layer_path_view(layer):
    layer.path_view_saved = False

def apply_layer_saved_view_to_viewport(layer, context=None):
    '''Restore the 3D Viewport to the layer's saved bake view. Returns True on success.'''
    if not layer_has_saved_path_view(layer):
        return False
    _space, rv3d = _find_viewport_region_3d(context)
    if rv3d is None:
        return False
    from mathutils import Quaternion
    rot = getattr(layer, 'path_view_rotation', (1.0, 0.0, 0.0, 0.0))
    loc = getattr(layer, 'path_view_location', (0.0, 0.0, 0.0))
    dist = float(getattr(layer, 'path_view_distance', 10.0))
    try:
        rv3d.view_rotation = Quaternion((rot[0], rot[1], rot[2], rot[3]))
        rv3d.view_location = Vector((loc[0], loc[1], loc[2]))
        rv3d.view_distance = max(dist, 1e-4)
        if hasattr(rv3d, 'update'):
            rv3d.update()
    except Exception:
        return False
    return True

def resolve_path_bake_view_direction(layer, context=None, capture_view=False):
    '''
    Direction for Project Axis → View.
    - capture_view=True (single-layer Bake): use live viewport and save it.
    - capture_view=False (Bake All / rebake): prefer saved view, else live viewport (and save).
    Returns (direction Vector or None, error_message or None).
    '''
    if not getattr(layer, 'path_project_view', False):
        return None, None

    if not capture_view:
        saved = get_layer_saved_view_direction(layer)
        if saved is not None:
            return saved, None

    state = _get_viewport_view_state(context)
    if state is None:
        if layer_has_saved_path_view(layer):
            # Viewport missing but we still have a saved direction
            saved = get_layer_saved_view_direction(layer)
            if saved is not None:
                return saved, None
        return None, "No 3D Viewport found — open a 3D View or turn off Project Axis 'View'"

    save_layer_path_view(layer, state=state)
    return state['direction'], None

def _project_polyline_along_direction(points, bvh, direction, max_dist):
    '''
    Project each point onto the mesh along +/- direction (Shrinkwrap Project style).
    Picks the hit closest to the original point. Misses keep the original.
    '''
    if not points or bvh is None:
        return list(points) if points else []
    d = Vector(direction)
    if d.length < 1e-8:
        return [Vector(p) for p in points]
    d.normalize()
    reach = max(float(max_dist), 1e-4)
    out = []
    for p in points:
        p = Vector(p)
        best = None
        best_gap = None
        for sign in (1.0, -1.0):
            ray = d * sign
            # Start behind the point so both sides of a thin shell can hit
            origin = p - ray * reach
            try:
                hit = bvh.ray_cast(origin, ray, reach * 2.0)
            except Exception:
                hit = None
            if not hit or hit[0] is None:
                continue
            loc = Vector(hit[0])
            gap = (loc - p).length
            if best is None or gap < best_gap:
                best = loc
                best_gap = gap
        out.append(best if best is not None else p)
    return out

def _decimate_points(points, max_points):
    '''Uniformly thin a point list — keeps the shape, caps the field cost.'''
    count = len(points)
    if count <= max_points:
        return list(points)
    step = count / float(max_points)
    return [points[min(count - 1, int(i * step))] for i in range(max_points)]

def _decimate_polyline_for_width(points, half_w, max_points=384):
    '''
    Thin a curve so segment spacing tracks ribbon width.
    Dense samples make closest-point O(points×segments) explode; the ribbon
    only needs a few segments across its width.
    '''
    count = len(points)
    if count <= 2:
        return list(points)
    total = 0.0
    for i in range(1, count):
        total += (points[i] - points[i - 1]).length
    spacing = max(float(half_w) * 0.4, total / float(max(max_points - 1, 1)), 1e-6)
    target = int(math.ceil(total / spacing)) + 1
    target = max(2, min(int(max_points), target))
    return _decimate_points(points, target)

def _rasterize_polygon_grid(poly_x, poly_y, grid_w, grid_h):
    '''Even-odd fill of a closed polygon given in grid coordinates.'''
    mask = numpy.zeros((grid_h, grid_w), dtype=numpy.float32)
    if poly_x.shape[0] < 3 or grid_w < 1 or grid_h < 1:
        return mask

    x0 = poly_x.astype(numpy.float64)
    y0 = poly_y.astype(numpy.float64)
    x1 = numpy.roll(x0, -1)
    y1 = numpy.roll(y0, -1)
    rows = numpy.arange(grid_h, dtype=numpy.float64) + 0.5

    spans = (y0[None, :] > rows[:, None]) != (y1[None, :] > rows[:, None])
    row_idx, edge_idx = numpy.nonzero(spans)
    if row_idx.size == 0:
        return mask

    dy = (y1 - y0)[edge_idx]
    dy = numpy.where(numpy.abs(dy) < 1e-12, 1e-12, dy)
    cross_x = x0[edge_idx] + (rows[row_idx] - y0[edge_idx]) / dy * (x1 - x0)[edge_idx]
    # Every crossing toggles the cells whose center sits to its right
    col = numpy.ceil(cross_x - 0.5).astype(numpy.int64)
    numpy.clip(col, 0, grid_w, out=col)
    toggles = numpy.zeros((grid_h, grid_w + 1), dtype=numpy.int32)
    numpy.add.at(toggles, (row_idx, col), 1)
    parity = numpy.cumsum(toggles, axis=1) & 1
    mask[:] = parity[:, :grid_w].astype(numpy.float32)
    return mask

def _sample_grid_bilinear(grid, gx, gy):
    '''Bilinear lookup in a (h, w) grid at continuous grid coordinates.'''
    grid_h, grid_w = grid.shape
    x = numpy.clip(gx - 0.5, 0.0, grid_w - 1.0)
    y = numpy.clip(gy - 0.5, 0.0, grid_h - 1.0)
    x0 = numpy.floor(x).astype(numpy.int32)
    y0 = numpy.floor(y).astype(numpy.int32)
    x1 = numpy.minimum(x0 + 1, grid_w - 1)
    y1 = numpy.minimum(y0 + 1, grid_h - 1)
    fx = (x - x0).astype(numpy.float32)
    fy = (y - y0).astype(numpy.float32)
    top = grid[y0, x0] * (1.0 - fx) + grid[y0, x1] * fx
    bottom = grid[y1, x0] * (1.0 - fx) + grid[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy

def _box_blur_mask(mask, radius):
    '''Separable box blur for soft shape edges. radius in pixels.'''
    r = int(max(0, round(radius)))
    if r <= 0:
        return mask
    h, w = mask.shape
    inv = 1.0 / float(2 * r + 1)

    # Horizontal
    padded = numpy.pad(mask, ((0, 0), (r, r)), mode='edge')
    cs = numpy.cumsum(padded, axis=1, dtype=numpy.float64)
    cs = numpy.concatenate([numpy.zeros((h, 1), dtype=numpy.float64), cs], axis=1)
    horiz = ((cs[:, 2 * r + 1:2 * r + 1 + w] - cs[:, :w]) * inv).astype(numpy.float32)

    # Vertical
    padded = numpy.pad(horiz, ((r, r), (0, 0)), mode='edge')
    cs = numpy.cumsum(padded, axis=0, dtype=numpy.float64)
    cs = numpy.concatenate([numpy.zeros((1, w), dtype=numpy.float64), cs], axis=0)
    vert = ((cs[2 * r + 1:2 * r + 1 + h, :] - cs[:h, :]) * inv).astype(numpy.float32)
    return numpy.clip(vert, 0.0, 1.0)

def bake_shape_to_image(obj, image, curve_obj, uv_name, resolution,
                        color=(1, 1, 1, 1), falloff=1.0, clear=True,
                        path_texture=None, tile_u=1.0, rotation_deg=0.0,
                        feather_px=2.0, project_distance=None,
                        shrinkwrap_method=None, project_axes=(False, False, False),
                        project_normal=None):
    '''
    Fill a closed (cyclic) curve onto the UV image as a solid vector shape.

    The outline is projected onto a plane and every surface texel under that
    plane is tested against the real outline. Enable path_project_x/y/z or
    path_project_view to force that plane's normal; otherwise the plane is fitted.
    Returns (ok, message).

    shrinkwrap_method: 'NEAREST_SURFACEPOINT' | 'PROJECT' | None (read from modifier)
    '''
    if not obj or obj.type != 'MESH':
        return False, 'Active object must be a mesh'
    if not curve_obj or curve_obj.type != 'CURVE':
        return False, 'Shape object must be a Curve'
    if not image:
        return False, 'No target image'

    # Ensure cyclic for shape baking
    ensure_curve_cyclic(curve_obj, True)

    bvh, tris, _, _ = _build_mesh_bvh_and_uvs(obj, uv_name)
    if not bvh or not tris:
        return False, "UV map '%s' not found on mesh" % uv_name

    img_w, img_h = image.size[0], image.size[1]
    if img_w < 1 or img_h < 1:
        return False, 'Invalid image size'

    shape_res = min(max(int(resolution), 32), 1024)
    method = shrinkwrap_method if shrinkwrap_method is not None else _curve_shrinkwrap_method(
        curve_obj, fallback='NEAREST_SURFACEPOINT'
    )
    if isinstance(method, str):
        method = method.upper()
    use_project = (method == 'PROJECT')

    obj_dim = max(obj.dimensions) if obj.dimensions.length > 0 else 1.0
    if project_distance is None:
        project_distance = max(obj_dim * 0.5, 0.1)

    # Project keeps the authored outline, Nearest uses the surface-snapped one
    if use_project:
        loop_entries = _sample_bezier_math_loops(curve_obj, shape_res)
    else:
        loop_entries = _curve_to_polylines(
            curve_obj, resolution=shape_res, shrinkwrap_method='NEAREST_SURFACEPOINT'
        )
        if not loop_entries:
            loop_entries = _sample_bezier_math_loops(curve_obj, shape_res)
    loops = [loop for loop, _cyclic in loop_entries if len(loop) >= 3]
    if not loops:
        return False, 'Closed shape needs at least 3 points'

    forced_normal = None
    proj_tag = ''
    if project_normal is not None:
        forced_normal = Vector(project_normal)
        if forced_normal.length > 1e-8:
            forced_normal.normalize()
            proj_tag = ' view'
        else:
            forced_normal = None
    if forced_normal is None:
        forced_normal = _world_project_normal(curve_obj, project_axes, fallback_obj=obj)
        if forced_normal is not None:
            proj_tag = ' axis'

    verts, tri_uvs, tri_nrms = _tris_to_arrays(tris)
    tex_pxs, tw, th = _load_texture_pixels(path_texture)
    use_tex = tex_pxs is not None

    accum = _BakeAccumulator(img_w, img_h, use_tex)
    feather = max(0.0, float(feather_px))
    loops_baked = 0
    seal_jobs = []

    for loop in loops:
        outline = list(loop)
        if len(outline) >= 4 and (outline[0] - outline[-1]).length < 1e-6:
            outline = outline[:-1]
        outline = _decimate_points(outline, 512)
        if len(outline) < 3:
            continue

        center = Vector((0.0, 0.0, 0.0))
        for p in outline:
            center += p
        center /= float(len(outline))
        if forced_normal is not None:
            center, axis_u, axis_v, plane_n = _plane_frame_from_center_normal(
                center, forced_normal
            )
        else:
            center, axis_u, axis_v, plane_n = _fit_polyline_plane(outline)

        # Face the outline plane the same way as the surface it sits on
        surface_n = Vector((0.0, 0.0, 0.0))
        stride = max(1, len(outline) // 24)
        for p in outline[::stride]:
            hit = _nearest_uv(bvh, tris, p, max_dist=project_distance)
            if hit:
                surface_n += hit[1]
        if surface_n.length > 1e-8 and surface_n.dot(plane_n) < 0.0:
            plane_n = -plane_n
            axis_v = -axis_v

        origin_np = numpy.array((center.x, center.y, center.z), dtype=numpy.float32)
        u_np = numpy.array((axis_u.x, axis_u.y, axis_u.z), dtype=numpy.float32)
        v_np = numpy.array((axis_v.x, axis_v.y, axis_v.z), dtype=numpy.float32)
        n_np = numpy.array((plane_n.x, plane_n.y, plane_n.z), dtype=numpy.float32)

        loop_rel = numpy.array(
            [[p.x, p.y, p.z] for p in outline], dtype=numpy.float32
        ) - origin_np
        loop_u = loop_rel @ u_np
        loop_v = loop_rel @ v_np
        loop_depth = loop_rel @ n_np

        u_min, u_max = float(loop_u.min()), float(loop_u.max())
        v_min, v_max = float(loop_v.min()), float(loop_v.max())
        span_u = max(u_max - u_min, 1e-6)
        span_v = max(v_max - v_min, 1e-6)
        extent = max(span_u, span_v)
        wobble = float(numpy.abs(loop_depth).max())
        # How far the surface may curve away from the outline plane
        depth_limit = max(wobble * 3.0 + extent * 0.35, obj_dim * 0.01, 1e-5)

        # Candidate triangles: inside the outline prism and facing the plane
        tri_rel = verts - origin_np
        tri_u = tri_rel @ u_np
        tri_v = tri_rel @ v_np
        tri_d = tri_rel @ n_np
        pad = extent * 0.02 + 1e-6
        candidates = (
            (tri_u.max(axis=1) >= u_min - pad) & (tri_u.min(axis=1) <= u_max + pad)
            & (tri_v.max(axis=1) >= v_min - pad) & (tri_v.min(axis=1) <= v_max + pad)
            & (tri_d.max(axis=1) >= -depth_limit) & (tri_d.min(axis=1) <= depth_limit)
            & ((tri_nrms @ n_np) > -0.35)
        )
        cand_idx = numpy.nonzero(candidates)[0]
        if cand_idx.size == 0:
            continue

        # Outline mask in plane space, finer than a texel so edges stay crisp
        texel = _median_texel_size(verts, tri_uvs, cand_idx, img_w, img_h)
        if texel <= 0.0:
            continue
        cell = max(texel * 0.45, extent * 1e-4)
        feather_world = feather * texel
        margin = feather_world * 2.5 + cell * 4.0
        grid_w = int(math.ceil((span_u + margin * 2.0) / cell)) + 1
        grid_h = int(math.ceil((span_v + margin * 2.0) / cell)) + 1
        max_cells = 8000000
        if grid_w * grid_h > max_cells:
            cell *= math.sqrt(float(grid_w) * float(grid_h) / max_cells)
            grid_w = int(math.ceil((span_u + margin * 2.0) / cell)) + 1
            grid_h = int(math.ceil((span_v + margin * 2.0) / cell)) + 1
        grid_u0 = u_min - margin
        grid_v0 = v_min - margin

        mask = _rasterize_polygon_grid(
            (loop_u - grid_u0) / cell, (loop_v - grid_v0) / cell, grid_w, grid_h
        )
        if feather_world > cell:
            mask = _box_blur_mask(mask, feather_world / cell)
        if falloff <= 0.0:
            mask = (mask >= 0.5).astype(numpy.float32)
        elif falloff != 1.0:
            mask = numpy.power(numpy.clip(mask, 0.0, 1.0), float(falloff)).astype(numpy.float32)

        visible_depth = max(wobble * 1.5, extent * 0.05, texel * 2.0)
        loop_pixels = 0

        for batch in _batch_tri_indices(tri_uvs, cand_idx, img_w, img_h):
            gathered = _gather_surface_texels(batch, verts, tri_uvs, tri_nrms, img_w, img_h)
            if gathered is None:
                continue
            ys, xs, pts, nrm, _scale = gathered

            rel = pts - origin_np
            p_u = rel @ u_np
            p_v = rel @ v_np
            p_d = rel @ n_np

            keep = (numpy.abs(p_d) <= depth_limit) & ((nrm @ n_np) > -0.35)
            if not numpy.any(keep):
                continue
            ys = ys[keep]
            xs = xs[keep]
            pts = pts[keep]
            p_u = p_u[keep]
            p_v = p_v[keep]
            p_d = p_d[keep]

            gx = (p_u - grid_u0) / cell
            gy = (p_v - grid_v0) / cell
            alpha = _sample_grid_bilinear(mask, gx, gy)
            alpha = numpy.where(
                (gx >= 0.0) & (gx <= grid_w) & (gy >= 0.0) & (gy <= grid_h), alpha, 0.0
            ).astype(numpy.float32)

            # Surface that sits well off the plane has to be visible from it
            far_idx = numpy.nonzero((alpha > 1e-4) & (numpy.abs(p_d) > visible_depth))[0]
            if far_idx.size:
                plane_pts = pts[far_idx] - p_d[far_idx][:, None] * n_np[None, :]
                blocked = _reject_occluded(bvh, plane_pts, pts[far_idx])
                alpha[far_idx[blocked]] = 0.0

            rgb = None
            if use_tex:
                sampled = _sample_path_texture_batch(
                    tex_pxs, tw, th,
                    (p_u - u_min) / span_u, (p_v - v_min) / span_v * 2.0 - 1.0,
                    tile_u=tile_u, rotation_deg=rotation_deg
                )
                alpha = alpha * sampled[:, 3]
                rgb = sampled[:, :3]

            loop_pixels += accum.add(ys, xs, alpha, rgb)

        if loop_pixels:
            loops_baked += 1
            seal_jobs.append((
                cand_idx, origin_np, u_np, v_np, n_np, depth_limit,
                grid_u0, grid_v0, cell, mask
            ))

    if accum.count == 0:
        return False, 'Shape did not cover any pixels (check curve position / UV map)'

    coverage, rgb_buf = accum.resolve()
    for cand_idx, origin_np, u_np, v_np, n_np, depth_limit, grid_u0, grid_v0, cell, mask in seal_jobs:
        seam_pairs = _build_uv_seam_pairs(verts, tri_uvs, cand_idx)

        def _mask_at(pu, pv, _mask=mask, _u0=grid_u0, _v0=grid_v0, _cell=cell):
            gx = numpy.array([(pu - _u0) / _cell], dtype=numpy.float64)
            gy = numpy.array([(pv - _v0) / _cell], dtype=numpy.float64)
            return float(_sample_grid_bilinear(_mask, gx, gy)[0])

        _seal_uv_seams_shape(
            coverage, seam_pairs, _mask_at, origin_np, u_np, v_np, n_np,
            depth_limit, img_w, img_h
        )
    pxs = _read_image_pixels(image, img_w, img_h)
    filled = _composite_bake(pxs, coverage, rgb_buf, color, clear)
    _write_image_pixels(image, pxs)

    msg = 'Filled shape (%d loop%s, %d / %d pixels) [%s%s]' % (
        loops_baked, 's' if loops_baked != 1 else '', filled, img_w * img_h,
        'Project' if use_project else 'Nearest', proj_tag
    )
    if loops_baked < len(loops):
        msg += ' — %d loop(s) found no surface' % (len(loops) - loops_baked)
    return True, msg

def bake_path_to_image(obj, image, curve_obj, uv_name, width, resolution, width_samples,
                       color=(1, 1, 1, 1), falloff=1.0, clear=True,
                       path_texture=None, tile_u=1.0, rotation_deg=0.0,
                       project_distance=None, fill_gaps=True, mode='RIBBON',
                       feather_px=2.0, shrinkwrap_method=None,
                       project_axes=(False, False, False), project_normal=None):
    '''
    Bake a ribbon (or a filled closed shape) from a real 3D curve onto a UV image.

    The ribbon is the surface footprint of the curve: every texel is measured
    against the curve itself, so opacity is even and the edges land exactly at
    half the path width. width_samples / fill_gaps are kept for older callers
    and no longer change the result.
    Returns (ok, message).
    '''
    if mode == 'SHAPE':
        return bake_shape_to_image(
            obj, image, curve_obj, uv_name, resolution,
            color=color, falloff=falloff, clear=clear,
            path_texture=path_texture, tile_u=tile_u, rotation_deg=rotation_deg,
            feather_px=feather_px, project_distance=project_distance,
            shrinkwrap_method=shrinkwrap_method, project_axes=project_axes,
            project_normal=project_normal
        )

    if not obj or obj.type != 'MESH':
        return False, 'Active object must be a mesh'
    if not curve_obj or curve_obj.type != 'CURVE':
        return False, 'Path object must be a Curve'
    if not image:
        return False, 'No target image'

    bvh, tris, _, _ = _build_mesh_bvh_and_uvs(obj, uv_name)
    if not bvh or not tris:
        return False, "UV map '%s' not found on mesh" % uv_name

    img_w, img_h = image.size[0], image.size[1]
    if img_w < 1 or img_h < 1:
        return False, 'Invalid image size'

    method = shrinkwrap_method if shrinkwrap_method is not None else _curve_shrinkwrap_method(
        curve_obj, fallback='NEAREST_SURFACEPOINT'
    )
    obj_dim = max(obj.dimensions) if obj.dimensions.length > 0 else 1.0
    half_w = max(float(width), 1e-6) * 0.5
    # How far the curve may hover over the surface and still paint it
    if project_distance is None:
        project_distance = max(half_w * 8.0, obj_dim * 0.08)
    hover_ceiling = max(float(project_distance), half_w * 2.0)

    view_dir = None
    if project_normal is not None:
        view_dir = Vector(project_normal)
        if view_dir.length > 1e-8:
            view_dir.normalize()
        else:
            view_dir = None

    # One polyline per spline — never bridge gaps between distinct curves
    if view_dir is not None:
        # View bake: keep authored silhouette, then raycast onto the mesh
        loop_entries = _sample_bezier_math_loops(curve_obj, max(int(resolution), 32))
        if not loop_entries:
            loop_entries = _curve_to_polylines(
                curve_obj, resolution=max(int(resolution), 32), shrinkwrap_method='PROJECT'
            )
    else:
        loop_entries = _curve_to_polylines(
            curve_obj, resolution=max(int(resolution), 32), shrinkwrap_method=method
        )
    raw_loops = []
    for polyline, cyclic in loop_entries:
        if view_dir is not None:
            polyline = _project_polyline_along_direction(
                polyline, bvh, view_dir, max(hover_ceiling * 4.0, obj_dim)
            )
        if cyclic and len(polyline) >= 3 and (polyline[0] - polyline[-1]).length < 1e-4:
            polyline = polyline[:-1]
        if len(polyline) >= 2:
            raw_loops.append((polyline, cyclic))
    if not raw_loops:
        return False, 'Curve has too few points to bake'

    # Thin dense samples — closest-point cost scales with segment count
    prepared = [
        (_decimate_polyline_for_width(polyline, half_w), cyclic)
        for polyline, cyclic in raw_loops
    ]
    prepared = [(pl, cy) for pl, cy in prepared if len(pl) >= 2]
    if not prepared:
        return False, 'Curve has too few points to bake'

    verts, tri_uvs, tri_nrms = _tris_to_arrays(tris)
    tex_pxs, tw, th = _load_texture_pixels(path_texture)
    use_tex = tex_pxs is not None

    # Closest-point radius must cover curve hover above the surface.
    depth_reach = hover_ceiling + half_w
    search_r = math.sqrt(depth_reach * depth_reach + (half_w * 1.1) ** 2)
    query_pts = []
    for polyline, _cyclic in prepared:
        query_pts.extend(_decimate_points(polyline, 512))
    if len(query_pts) > 1:
        spacing = max(
            (query_pts[i + 1] - query_pts[i]).length for i in range(len(query_pts) - 1)
        )
    else:
        spacing = 0.0
    # Triangle gather only needs lateral ribbon reach + the real curve/surface gap.
    # Using hover_ceiling here pulled in nearly the whole mesh and made closest-point O(n²).
    gap_samples = []
    stride = max(1, len(query_pts) // 48)
    for p in query_pts[::stride]:
        try:
            hit = bvh.find_nearest(p)
        except Exception:
            hit = None
        if hit and hit[0] is not None:
            gap_samples.append(float(hit[3]))
    median_gap = float(numpy.median(gap_samples)) if gap_samples else hover_ceiling
    tri_r = half_w * 3.0 + median_gap * 2.0 + spacing * 0.5
    cand_idx = _tris_near_points(bvh, len(tris), query_pts, tri_r)
    if cand_idx is None:
        curve_np = numpy.array([[p.x, p.y, p.z] for p in query_pts], dtype=numpy.float32)
        lo = curve_np.min(axis=0) - tri_r
        hi = curve_np.max(axis=0) + tri_r
        tri_lo = verts.min(axis=1)
        tri_hi = verts.max(axis=1)
        cand_idx = numpy.nonzero(numpy.all((tri_hi >= lo) & (tri_lo <= hi), axis=1))[0]
    if cand_idx.size == 0:
        return False, 'No mesh surface near the curve (move the curve closer)'

    seam_pairs = _build_uv_seam_pairs(verts, tri_uvs, cand_idx)
    accum = _BakeAccumulator(img_w, img_h, use_tex)
    far_away = obj_dim * 1000.0 + 1.0

    bundles = _build_path_spline_bundles(prepared, bvh, tris, far_away)
    if not bundles:
        return False, 'Curve has too few points to bake'
    splines_baked = len(bundles)

    # Gather surface texels once; evaluate each spline separately so crossings
    # max-blend instead of the nearer curve punching a hole in the other.
    gathered_batches = []
    for batch in _batch_tri_indices(tri_uvs, cand_idx, img_w, img_h):
        gathered = _gather_surface_texels(batch, verts, tri_uvs, tri_nrms, img_w, img_h)
        if gathered is not None:
            gathered_batches.append(gathered)

    for bundle in bundles:
        segments = bundle['segments']
        seg_nrm = bundle['seg_nrm']
        seg_gap = bundle['seg_gap']
        seg_dir = bundle['seg_dir']
        seg_axis = bundle['seg_axis']
        seg_side = bundle['seg_side']
        path_len = bundle['path_len']
        open_start = bundle['open_start']
        open_end = bundle['open_end']

        for ys, xs, pts, nrm, world_per_px in gathered_batches:
            dist, arc, seg_t, seg_i, closest = _closest_on_polyline(pts, segments, search_r)
            offset = pts - closest
            near = numpy.isfinite(dist)
            tangent = seg_dir[seg_i]
            axis = seg_axis[seg_i]
            side = seg_side[seg_i]
            hover = seg_gap[seg_i]
            slack = numpy.minimum(world_per_px, half_w * 0.5)

            # Width lives only on the shared curve side axis — identical on every face
            # of a UV seam / mesh edge, so the ribbon cannot jump.
            lat = numpy.einsum('ij,ij->i', offset, side)
            depth = numpy.einsum('ij,ij->i', offset, axis)
            d_lat = numpy.where(near, numpy.abs(lat), numpy.float32(numpy.inf))

            # Depth is a reach gate only. Excess depth beyond the local hover fades.
            in_reach = near & (numpy.abs(depth) <= hover_ceiling + slack)
            excess = numpy.maximum(numpy.abs(depth) - (hover + half_w + slack), 0.0)
            d_reach = numpy.sqrt(d_lat * d_lat + excess * excess)

            # Crease wrap: faces that turn away still get a cylinder around the curve
            away = numpy.einsum('ij,ij->i', nrm, seg_nrm[seg_i]) <= -0.15
            along = numpy.einsum('ij,ij->i', offset, tangent)
            perp = offset - along[:, None] * tangent
            d_cyl = numpy.sqrt(numpy.einsum('ij,ij->i', perp, perp))
            wrap = near & away & (d_cyl <= half_w + slack) & (numpy.abs(depth) <= hover_ceiling + half_w)

            d_eff = numpy.where(
                in_reach, d_reach,
                numpy.where(wrap, d_cyl, numpy.float32(numpy.inf))
            )

            keep = d_eff <= half_w + slack
            keep &= ~(
                (open_start[seg_i] & (seg_t <= 1e-6))
                | (open_end[seg_i] & (seg_t >= 1.0 - 1e-6))
            )
            if not numpy.any(keep):
                continue

            ys_k = ys[keep]
            xs_k = xs[keep]
            pts_k = pts[keep]
            d_eff = d_eff[keep]
            lat = lat[keep]
            arc = arc[keep]
            closest_k = closest[keep]
            world_per_px_k = world_per_px[keep]
            away_k = away[keep]

            suspect = numpy.nonzero(away_k)[0]
            if suspect.size:
                if suspect.size > _OCCLUSION_TEST_LIMIT:
                    blocked = numpy.ones(suspect.size, dtype=bool)
                else:
                    blocked = _reject_occluded(bvh, closest_k[suspect], pts_k[suspect])
                if numpy.any(blocked):
                    visible = numpy.ones(ys_k.shape[0], dtype=bool)
                    visible[suspect[blocked]] = False
                    ys_k = ys_k[visible]
                    xs_k = xs_k[visible]
                    d_eff = d_eff[visible]
                    lat = lat[visible]
                    arc = arc[visible]
                    world_per_px_k = world_per_px_k[visible]
            if ys_k.shape[0] == 0:
                continue

            edge = numpy.clip(
                (half_w - d_eff) / numpy.clip(world_per_px_k * 0.7, 1e-9, half_w * 0.5) + 0.5,
                0.0, 1.0
            ).astype(numpy.float32)
            if falloff <= 0.0:
                alpha = edge
            else:
                alpha = _falloff_profile(d_eff, half_w, falloff) * edge

            rgb = None
            if use_tex:
                across = lat / half_w
                sampled = _sample_path_texture_batch(
                    tex_pxs, tw, th, arc / path_len,
                    numpy.clip(across, -1.0, 1.0),
                    tile_u=tile_u, rotation_deg=rotation_deg
                )
                alpha = alpha * sampled[:, 3]
                rgb = sampled[:, :3]

            accum.add(ys_k, xs_k, alpha.astype(numpy.float32), rgb)

    if accum.count == 0:
        return False, 'No path pixels found on the surface (move the curve closer to the mesh)'

    coverage, rgb_buf = accum.resolve()
    for bundle in bundles:
        _seal_uv_seams_path(
            coverage, seam_pairs, bundle['segments'], half_w, hover_ceiling, img_w, img_h
        )
    pxs = _read_image_pixels(image, img_w, img_h)
    filled = _composite_bake(pxs, coverage, rgb_buf, color, clear)
    _write_image_pixels(image, pxs)

    method_tag = 'View' if view_dir is not None else (
        'Project' if method == 'PROJECT' else 'Nearest'
    )
    return True, 'Baked path ribbon (%d pixels, %d spline(s)) [%s]' % (
        filled, splines_baked, method_tag
    )


def _poll_curve_object(self, obj):
    return obj and obj.type == 'CURVE'

class BaseBakePath():
    '''Mixin for path-bake settings on layers.'''

    enable_path_bake : BoolProperty(
        name = 'Enable Path Bake',
        description = 'This layer can bake a ribbon or closed shape from a real 3D Bezier curve',
        default = False
    )

    path_mode : EnumProperty(
        name = 'Path Mode',
        description = 'Ribbon stamps along an open curve; Shape fills a closed cyclic curve like a vector silhouette',
        items = (
            ('RIBBON', 'Ribbon', 'Bake a ribbon along an open path'),
            ('SHAPE', 'Shape', 'Bake a filled closed shape (cyclic curve)'),
        ),
        default = 'RIBBON',
        update = update_path_mode
    )

    path_enable_shrinkwrap : BoolProperty(
        name = 'Shrinkwrap to Mesh',
        description = 'Add a Shrinkwrap modifier on the path curve targeting the parent mesh',
        default = True,
        update = update_path_enable_shrinkwrap
    )

    path_shrinkwrap_method : EnumProperty(
        name = 'Shrinkwrap Method',
        description = 'How the path curve sticks to the mesh',
        items = (
            ('NEAREST_SURFACEPOINT', 'Nearest', 'Nearest Surface Point (snap to closest mesh location)'),
            ('PROJECT', 'Project', 'Shrinkwrap Project along +/- of the chosen axes'),
        ),
        default = 'NEAREST_SURFACEPOINT',
        update = update_path_shrinkwrap_method
    )

    path_project_x : BoolProperty(
        name = 'X',
        description = "Project along the curve object's X axis (Shrinkwrap Project + shape bake plane)",
        default = False,
        update = update_path_project_axis
    )
    path_project_y : BoolProperty(
        name = 'Y',
        description = "Project along the curve object's Y axis (Shrinkwrap Project + shape bake plane)",
        default = False,
        update = update_path_project_axis
    )
    path_project_z : BoolProperty(
        name = 'Z',
        description = "Project along the curve object's Z axis (Shrinkwrap Project + shape bake plane)",
        default = False,
        update = update_path_project_axis
    )
    path_project_view : BoolProperty(
        name = 'View',
        description = 'Bake along the 3D Viewport looking direction (ribbons project onto the mesh; shapes use it as the plane normal). The view is saved on bake and reused for Bake All / rebake',
        default = False
    )

    path_view_saved : BoolProperty(
        name = 'Has Saved Bake View',
        description = 'Whether this layer stored a viewport for View projection rebakes',
        default = False
    )

    path_view_direction : FloatVectorProperty(
        name = 'Saved View Direction',
        description = 'World-space look direction used for View projection baking',
        size = 3,
        default = (0.0, 0.0, -1.0)
    )

    path_view_rotation : FloatVectorProperty(
        name = 'Saved View Rotation',
        description = 'Viewport rotation (quaternion wxyz) for restoring the bake view',
        size = 4,
        default = (1.0, 0.0, 0.0, 0.0)
    )

    path_view_location : FloatVectorProperty(
        name = 'Saved View Location',
        description = 'Viewport orbit center for restoring the bake view',
        size = 3,
        default = (0.0, 0.0, 0.0)
    )

    path_view_distance : FloatProperty(
        name = 'Saved View Distance',
        description = 'Viewport distance for restoring the bake view',
        default = 10.0,
        min = 0.0001
    )

    if is_bl_newer_than(2, 79):
        path_curve_object : PointerProperty(
            name = 'Path Curve',
            description = 'Bezier curve object used for path/ribbon baking',
            type = bpy.types.Object,
            poll = _poll_curve_object
        )

    path_curve_object_name : StringProperty(
        name = 'Path Curve Name',
        default = ''
    )

    path_width : FloatProperty(
        name = 'Path Width',
        description = 'Ribbon width in world units',
        default = 0.05, min = 0.0001, max = 100.0, precision = 4, step = 1
    )

    path_resolution : IntProperty(
        name = 'Path Resolution',
        description = 'How finely the curve itself is sampled — higher follows tight curvature more closely',
        default = 512, min = 8, max = 4096
    )

    # Kept so older scenes keep loading; the bake no longer samples across width
    path_width_samples : IntProperty(
        name = 'Width Samples',
        description = 'Unused — ribbon width is now measured exactly',
        default = 48, min = 2, max = 256
    )

    path_color : FloatVectorProperty(
        name = 'Path Color',
        description = 'Color stamped along the path',
        size = 4, subtype = 'COLOR',
        default = (1.0, 1.0, 1.0, 1.0),
        min = 0.0, max = 1.0
    )

    path_falloff : FloatProperty(
        name = 'Edge Falloff',
        description = 'Softness of the ribbon edge or shape fringe (0 = hard, higher = softer)',
        default = 1.0, min = 0.0, max = 8.0
    )

    path_shape_feather : FloatProperty(
        name = 'Shape Feather',
        description = 'Soft edge width for filled shapes, in pixels',
        default = 2.0, min = 0.0, max = 64.0
    )

    path_clear_before_bake : BoolProperty(
        name = 'Clear Before Bake',
        description = 'Clear the layer image before stamping the path',
        default = True
    )

    path_tile_u : FloatProperty(
        name = 'Tile Along Path',
        description = 'How many times the path texture repeats along the curve',
        default = 1.0, min = 0.01, max = 64.0
    )

    path_texture_rotation : FloatProperty(
        name = 'Texture Rotation',
        description = 'Rotate the tiled path texture in degrees',
        default = 0.0, min = -360.0, max = 360.0, step = 100, precision = 1
    )

    if is_bl_newer_than(2, 79):
        path_texture : PointerProperty(
            name = 'Path Texture',
            description = 'Optional texture stamped along the path (U = length, V = width). Leave empty for solid color',
            type = bpy.types.Image
        )

class YNewPathLayer(bpy.types.Operator):
    '''Create an IMAGE layer with a real 3D Bezier curve for path/ribbon or closed shape baking'''
    bl_idname = 'wm.y_new_path_layer'
    bl_label = 'New Path Layer'
    bl_options = {'REGISTER', 'UNDO'}

    name : StringProperty(name='Name', default='Path')

    path_mode : EnumProperty(
        name = 'Mode',
        items = (
            ('RIBBON', 'Ribbon', 'Open path ribbon'),
            ('SHAPE', 'Shape', 'Closed filled shape'),
        ),
        default = 'RIBBON'
    )

    width : IntProperty(name='Width', default=1024, min=1, max=16384)
    height : IntProperty(name='Height', default=1024, min=1, max=16384)

    uv_map : StringProperty(name='UV Map', default='')

    path_width : FloatProperty(
        name = 'Path Width',
        default = 0.05, min = 0.0001, max = 100.0, precision = 4
    )

    hdr : BoolProperty(name='32-bit Float', default=False)

    @classmethod
    def poll(cls, context):
        return get_active_ypaint_node() and resolve_path_bake_mesh(context.object) is not None

    def invoke(self, context, event):
        obj = resolve_path_bake_mesh(context.object)
        node = get_active_ypaint_node()
        yp = node.node_tree.yp
        ypup = get_user_preferences()

        default_name = 'Shape' if self.path_mode == 'SHAPE' else 'Path'
        self.name = get_unique_name(default_name, bpy.data.images)
        self.width = ypup.default_new_image_size
        self.height = ypup.default_new_image_size

        # Default UV
        self.uv_map = ''
        if obj.type == 'MESH' and obj.data.uv_layers.active:
            self.uv_map = obj.data.uv_layers.active.name
        elif len(yp.uvs) > 0:
            self.uv_map = yp.uvs[0].name

        # Guess a reasonable default width from object size
        dim = max(obj.dimensions) if obj.dimensions.length > 0 else 1.0
        self.path_width = max(dim * 0.02, 0.01)

        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        obj = resolve_path_bake_mesh(context.object)

        layout.prop(self, 'name')
        layout.prop(self, 'path_mode', expand=True)
        row = layout.row(align=True)
        row.prop(self, 'width')
        row.prop(self, 'height')
        layout.prop(self, 'hdr')
        if obj and obj.type == 'MESH':
            layout.prop_search(self, 'uv_map', obj.data, 'uv_layers', text='UV Map', icon='GROUP_UVS')
        if self.path_mode == 'RIBBON':
            layout.prop(self, 'path_width')

    def execute(self, context):
        T = time.time()
        obj = resolve_path_bake_mesh(context.object)
        mat = get_active_material()
        node = get_active_ypaint_node()
        yp = node.node_tree.yp
        wm = context.window_manager
        ypui = wm.ypui

        if not obj:
            self.report({'ERROR'}, 'Active object must be a mesh (or its path/shape curve)')
            return {'CANCELLED'}

        if not self.uv_map:
            self.report({'ERROR'}, 'UV Map is required')
            return {'CANCELLED'}

        if bpy.data.images.get(self.name):
            self.report({'ERROR'}, "Image named '%s' is already available!" % self.name)
            return {'CANCELLED'}

        colorspace = get_linear_color_name() if self.hdr else get_srgb_name()
        color = (0, 0, 0, 0)

        img = bpy.data.images.new(
            name=self.name, width=self.width, height=self.height,
            alpha=True, float_buffer=self.hdr
        )
        img.generated_type = 'BLANK'
        img.generated_color = color
        if hasattr(img, 'use_alpha'):
            img.use_alpha = True
        if img.is_float and is_bl_newer_than(2, 80):
            img.alpha_mode = 'PREMUL'
        img.colorspace_settings.name = colorspace

        update_image_editor_image(context, img)

        is_shape = self.path_mode == 'SHAPE'
        curve_label = self.name + (' Shape' if is_shape else ' Curve')
        curve_obj = create_path_curve(
            obj, name=curve_label, use_shrinkwrap=True, cyclic=is_shape,
            shrinkwrap_method='NEAREST_SURFACEPOINT'
        )

        from . import Layer

        # Keep the mesh active while building the layer (curve has no material)
        set_active_object(obj)

        yp.halt_update = True
        layer = Layer.add_new_layer(
            node.node_tree, self.name, 'IMAGE',
            0, 'MIX', 'MIX', 'BUMP_MAP', 'UV',
            uv_name=self.uv_map, image=img
        )

        layer.enable_path_bake = True
        layer.path_mode = self.path_mode
        layer.path_enable_shrinkwrap = True
        layer.path_shrinkwrap_method = 'NEAREST_SURFACEPOINT'
        set_path_curve_object(layer, curve_obj)
        layer.path_width = self.path_width
        _apply_layer_shrinkwrap(layer, curve_obj, obj)
        if is_shape:
            ensure_curve_cyclic(curve_obj, True)

        yp.halt_update = False

        node_connections.reconnect_yp_nodes(node.node_tree)
        node_arrangements.rearrange_yp_nodes(node.node_tree)

        # Keep the mesh active so Ucupaint UI stays on the material setup.
        set_object_select(curve_obj, True)
        set_active_object(obj)

        ypui.layer_ui.expand_content = True
        ypui.layer_ui.expand_source = True
        ypui.need_update = True
        ListItem.refresh_list_items(yp)

        # Initial bake so something is visible immediately
        ok, msg = bake_path_to_image(
            obj, img, curve_obj, self.uv_map,
            width=layer.path_width,
            resolution=layer.path_resolution,
            width_samples=layer.path_width_samples,
            color=tuple(layer.path_color),
            falloff=layer.path_falloff,
            clear=True,
            mode=layer.path_mode,
            feather_px=layer.path_shape_feather,
            shrinkwrap_method=_layer_shrinkwrap_method(layer),
            project_axes=_layer_project_axes(layer),
        )
        kind = 'Shape' if is_shape else 'Path'
        if not ok:
            self.report({'WARNING'}, '%s layer created, but initial bake skipped: %s' % (kind, msg))
        else:
            self.report({'INFO'}, '%s layer created. Edit the curve, then Bake Path.' % kind)

        print('INFO: %s layer' % kind, layer.name, 'created in', '{:0.2f}'.format((time.time() - T) * 1000), 'ms!')
        wm.yptimer.time = str(time.time())
        return {'FINISHED'}

class YBakePathToLayer(bpy.types.Operator):
    '''Bake the linked 3D Bezier path as a ribbon into this IMAGE layer'''
    bl_idname = 'wm.y_bake_path_to_layer'
    bl_label = 'Bake Path'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        if not layer or layer.type != 'IMAGE':
            return False
        return get_path_curve_object(layer) is not None

    def execute(self, context):
        obj = resolve_path_bake_mesh(context.object)
        if not obj:
            self.report({'ERROR'}, 'Active object must be a mesh (or its path/shape curve)')
            return {'CANCELLED'}

        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer or layer.type != 'IMAGE':
            self.report({'ERROR'}, 'Active layer must be an IMAGE layer')
            return {'CANCELLED'}

        T = time.time()
        ok, msg = bake_layer_path(obj, layer, context=context, capture_view=True)

        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        for area in context.screen.areas:
            area.tag_redraw()

        self.report({'INFO'}, msg + ' ({:0.0f} ms)'.format((time.time() - T) * 1000))
        return {'FINISHED'}

def bake_layer_path(obj, layer, context=None, capture_view=False):
    '''
    Bake one path/shape layer. Returns (ok, message).

    capture_view: when Project Axis → View is on, True reads the live viewport
    and stores it on the layer (single Bake Path/Shape). False prefers the
    saved view so Bake All / rebake stay consistent after you move the camera.
    '''
    curve_obj = get_path_curve_object(layer)
    if not curve_obj:
        return False, "Layer '%s' has no path curve" % layer.name

    layer.path_curve_object_name = curve_obj.name

    source = get_layer_source(layer)
    image = source.image if source else None
    if not image:
        return False, "Layer '%s' has no image" % layer.name

    path_tex = None
    if is_bl_newer_than(2, 79):
        path_tex = getattr(layer, 'path_texture', None)

    method = _layer_shrinkwrap_method(layer)
    axes = _layer_project_axes(layer)
    project_normal, view_err = resolve_path_bake_view_direction(
        layer, context=context, capture_view=capture_view
    )
    if view_err:
        return False, view_err

    if getattr(layer, 'path_enable_shrinkwrap', True):
        _apply_layer_shrinkwrap(layer, curve_obj, obj)

    ok, msg = bake_path_to_image(
        obj, image, curve_obj, layer.uv_name,
        width=layer.path_width,
        resolution=layer.path_resolution,
        width_samples=layer.path_width_samples,
        color=tuple(layer.path_color),
        falloff=layer.path_falloff,
        clear=layer.path_clear_before_bake,
        path_texture=path_tex,
        tile_u=layer.path_tile_u,
        rotation_deg=layer.path_texture_rotation,
        mode=layer.path_mode,
        feather_px=layer.path_shape_feather,
        shrinkwrap_method=method,
        project_axes=axes,
        project_normal=project_normal,
    )
    if ok:
        image.update()
        if project_normal is not None:
            if capture_view:
                msg += ' [view captured]'
            elif layer_has_saved_path_view(layer):
                msg += ' [saved view]'
    return ok, msg

def iter_bakeable_path_layers(yp):
    '''Yield IMAGE layers that have path bake enabled and a linked curve.'''
    for layer in yp.layers:
        if layer.type != 'IMAGE':
            continue
        if not getattr(layer, 'enable_path_bake', False):
            continue
        if get_path_curve_object(layer) is None:
            continue
        yield layer

class YBakeAllPaths(bpy.types.Operator):
    '''Bake all ribbon and closed-shape path layers'''
    bl_idname = 'wm.y_bake_all_paths'
    bl_label = 'Bake All Paths'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if resolve_path_bake_mesh(context.object) is None:
            return False
        node = get_active_ypaint_node()
        if not node:
            return False
        return any(True for _ in iter_bakeable_path_layers(node.node_tree.yp))

    def execute(self, context):
        obj = resolve_path_bake_mesh(context.object)
        node = get_active_ypaint_node()
        if not obj:
            self.report({'ERROR'}, 'Active object must be a mesh (or its path/shape curve)')
            return {'CANCELLED'}
        if not node:
            self.report({'ERROR'}, 'No Ucupaint node found')
            return {'CANCELLED'}

        layers = list(iter_bakeable_path_layers(node.node_tree.yp))
        if not layers:
            self.report({'ERROR'}, 'No path/shape layers with curves to bake')
            return {'CANCELLED'}

        T = time.time()
        ok_count = 0
        fail_msgs = []
        for layer in layers:
            ok, msg = bake_layer_path(obj, layer, context=context)
            if ok:
                ok_count += 1
                print('INFO: Baked path layer', layer.name + ':', msg)
            else:
                fail_msgs.append('%s: %s' % (layer.name, msg))
                print('WARNING: Failed path bake for', layer.name + ':', msg)

        for area in context.screen.areas:
            area.tag_redraw()

        elapsed = (time.time() - T) * 1000
        if ok_count == 0:
            self.report({'ERROR'}, 'Bake All failed. ' + '; '.join(fail_msgs[:3]))
            return {'CANCELLED'}

        summary = 'Baked %d / %d path layers (%.0f ms)' % (ok_count, len(layers), elapsed)
        if fail_msgs:
            self.report({'WARNING'}, summary + ' - some failed: ' + '; '.join(fail_msgs[:2]))
        else:
            self.report({'INFO'}, summary)
        return {'FINISHED'}

class YResizePathBakeImage(bpy.types.Operator):
    '''Change the resolution of this path/shape layer's bake image'''
    bl_idname = 'wm.y_resize_path_bake_image'
    bl_label = 'Resize Bake Image'
    bl_options = {'REGISTER', 'UNDO'}

    width : IntProperty(name='Width', default=1024, min=1, max=16384)
    height : IntProperty(name='Height', default=1024, min=1, max=16384)
    rebake : BoolProperty(
        name = 'Rebake After Resize',
        description = 'Bake the path/shape again after changing image size',
        default = True
    )

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        if not layer or layer.type != 'IMAGE' or not getattr(layer, 'enable_path_bake', False):
            return False
        source = get_layer_source(layer)
        return bool(source and source.image)

    def invoke(self, context, event):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        source = get_layer_source(layer) if layer else None
        image = source.image if source else None
        if image and image.size[0] > 0 and image.size[1] > 0:
            self.width = image.size[0]
            self.height = image.size[1]
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.prop(self, 'width')
        row.prop(self, 'height')
        layout.prop(self, 'rebake')

    def execute(self, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer:
            self.report({'ERROR'}, 'No active layer')
            return {'CANCELLED'}

        source = get_layer_source(layer)
        image = source.image if source else None
        if not image:
            self.report({'ERROR'}, 'Layer has no image')
            return {'CANCELLED'}

        if image.size[0] == self.width and image.size[1] == self.height:
            self.report({'INFO'}, 'Image already has this size')
            return {'CANCELLED'}

        # Prefer Blender's image.scale when available
        try:
            if hasattr(image, 'scale'):
                image.scale(self.width, self.height)
            else:
                # Fallback via UV editor resize (same approach as wm.y_resize_image)
                area = context.area
                ori_ui_type = getattr(area, 'ui_type', None) if area else None
                if area and hasattr(area, 'ui_type'):
                    area.ui_type = 'UV'
                    if hasattr(context, 'space_data') and context.space_data:
                        context.space_data.image = image
                    bpy.ops.image.resize(size=(self.width, self.height))
                    if ori_ui_type is not None:
                        area.ui_type = ori_ui_type
                else:
                    self.report({'ERROR'}, 'Could not resize image in this context')
                    return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, 'Resize failed: %s' % str(e))
            return {'CANCELLED'}

        if image.source == 'GENERATED':
            if hasattr(image, 'generated_width'):
                image.generated_width = self.width
            if hasattr(image, 'generated_height'):
                image.generated_height = self.height

        image.update()

        msg = 'Resized bake image to %d x %d' % (self.width, self.height)
        if self.rebake:
            obj = resolve_path_bake_mesh(context.object)
            if not obj:
                self.report({'WARNING'}, msg + ' (rebake skipped: no mesh)')
                return {'FINISHED'}
            ok, bake_msg = bake_layer_path(obj, layer, context=context)
            if ok:
                self.report({'INFO'}, msg + ' and rebaked')
            else:
                self.report({'WARNING'}, msg + ' but rebake failed: %s' % bake_msg)
        else:
            self.report({'INFO'}, msg + ' — bake again to refresh content')

        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}

class YSelectPathCurve(bpy.types.Operator):
    '''Select the path curve object linked to this layer'''
    bl_idname = 'wm.y_select_path_curve'
    bl_label = 'Select Path Curve'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return layer and get_path_curve_object(layer)

    def execute(self, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        curve_obj = get_path_curve_object(layer) if layer else None
        if not curve_obj:
            self.report({'ERROR'}, 'Path curve not found')
            return {'CANCELLED'}

        for o in context.selected_objects:
            set_object_select(o, False)
        set_object_select(curve_obj, True)
        set_active_object(curve_obj)
        # Operator context can write RNA directly
        global _last_path_curve_layer_sync
        _last_path_curve_layer_sync = None
        select_path_bake_layer_for_curve(curve_obj)

        self.report({'INFO'}, "Path curve selected. UcuPaint stays on the parent mesh.")
        return {'FINISHED'}

class YCreatePathCurveForLayer(bpy.types.Operator):
    '''Create a new Bezier curve and link it to this IMAGE layer for path baking'''
    bl_idname = 'wm.y_create_path_curve_for_layer'
    bl_label = 'Create Path Curve'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = resolve_path_bake_mesh(context.object)
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return obj is not None and layer and layer.type == 'IMAGE'

    def execute(self, context):
        obj = resolve_path_bake_mesh(context.object)
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer or not obj:
            return {'CANCELLED'}

        is_shape = layer.path_mode == 'SHAPE'
        curve_obj = create_path_curve(
            obj,
            name=layer.name + (' Shape' if is_shape else ' Curve'),
            use_shrinkwrap=layer.path_enable_shrinkwrap,
            cyclic=is_shape,
            shrinkwrap_method=_layer_shrinkwrap_method(layer),
            project_axes=_layer_project_axes(layer),
        )
        layer.enable_path_bake = True
        set_path_curve_object(layer, curve_obj)
        if layer.path_enable_shrinkwrap:
            _apply_layer_shrinkwrap(layer, curve_obj, obj)
        if is_shape:
            ensure_curve_cyclic(curve_obj, True)

        set_object_select(curve_obj, True)
        set_active_object(obj)

        self.report({'INFO'}, "Created %s curve '%s'" % ('shape' if is_shape else 'path', curve_obj.name))
        return {'FINISHED'}

def iter_path_bake_layers_with_image(yp):
    '''Yield path/shape bake layers that already have a bake image.'''
    for layer in yp.layers:
        if layer.type != 'IMAGE':
            continue
        if not getattr(layer, 'enable_path_bake', False):
            continue
        source = get_layer_source(layer)
        if source and source.image:
            yield layer

def apply_path_bake_as_mask(path_layer, target_layer, socket_input_name='Alpha',
                            blend_type='MULTIPLY', disable_source_layer=True,
                            interpolation='Linear'):
    '''
    Share a path/shape bake image as a UV IMAGE mask on another layer.
    Returns (mask, error_message).
    '''
    from . import Mask

    if not path_layer or not target_layer:
        return None, 'Missing path or target layer'
    if path_layer == target_layer:
        return None, 'Pick a different layer to mask — cannot use a bake on itself'

    source = get_layer_source(path_layer)
    image = source.image if source else None
    if not image:
        return None, "Path/shape layer '%s' has no image — bake it first" % path_layer.name

    for mask in target_layer.masks:
        if mask.type != 'IMAGE':
            continue
        mask_source = get_mask_source(mask)
        if mask_source and mask_source.image == image:
            return mask, "Layer '%s' already uses '%s' as a mask" % (
                target_layer.name, path_layer.name
            )

    if hasattr(image, 'colorspace_settings'):
        noncolor = get_noncolor_name()
        if image.colorspace_settings.name != noncolor and not image.is_dirty:
            image.colorspace_settings.name = noncolor

    uv_name = path_layer.uv_name or target_layer.uv_name or ''
    mask = Mask.add_new_mask(
        target_layer, path_layer.name, 'IMAGE', 'UV', uv_name,
        image=image, blend_type=blend_type, socket_input_name=socket_input_name,
        interpolation=interpolation
    )

    target_layer.enable_masks = True
    if disable_source_layer:
        path_layer.enable = False

    node_connections.reconnect_layer_nodes(target_layer)
    node_arrangements.rearrange_layer_nodes(target_layer)
    node_connections.reconnect_yp_nodes(target_layer.id_data)
    node_arrangements.rearrange_yp_nodes(target_layer.id_data)

    return mask, None

class YUsePathBakeAsMask(bpy.types.Operator):
    '''Add a path or shape bake as a UV mask on another layer (ideal for masking decals)'''
    bl_idname = 'wm.y_use_path_bake_as_mask'
    bl_label = 'Use Path/Shape as Mask'
    bl_description = 'Use a path or shape bake image as a UV mask on a layer. Rebaking the path updates the mask automatically'
    bl_options = {'REGISTER', 'UNDO'}

    mode : EnumProperty(
        name = 'Mode',
        items = (
            ('PICK_TARGET', 'Pick Target Layer', 'Use the current path/shape bake as a mask on another layer'),
            ('PICK_PATH', 'Pick Path/Shape', 'Pick a path/shape bake to mask the current layer'),
        ),
        default = 'PICK_TARGET'
    )

    path_layer_name : StringProperty(name='Path / Shape', default='')
    target_layer_name : StringProperty(name='Target Layer', default='')
    path_coll : CollectionProperty(type=bpy.types.PropertyGroup)
    target_coll : CollectionProperty(type=bpy.types.PropertyGroup)

    socket_input_name : EnumProperty(
        name = 'Source Input',
        description = 'Path/shape bakes store coverage in Alpha; Color works if the bake is white',
        items = (
            ('Alpha', 'Alpha', 'Use image alpha (recommended for path/shape coverage)'),
            ('Color', 'Color', 'Use image color'),
        ),
        default = 'Alpha'
    )

    blend_type : EnumProperty(
        name = 'Blend',
        description = 'How the mask combines with the layer',
        items = mask_blend_type_items,
        default = 3 if is_bl_newer_than(2, 90) else None,
    )

    disable_source_layer : BoolProperty(
        name = 'Disable Path/Shape Layer',
        description = 'Turn off the path/shape layer so it only contributes as a mask (rebakes still update the shared image)',
        default = True
    )

    @classmethod
    def poll(cls, context):
        return get_active_ypaint_node() is not None

    def _resolve_context_layer(self, context):
        layer = getattr(context, 'layer', None)
        if layer:
            return layer
        node = get_active_ypaint_node()
        if not node:
            return None
        return get_active_layer(node.node_tree.yp)

    def invoke(self, context, event):
        node = get_active_ypaint_node()
        if not node:
            self.report({'ERROR'}, 'No UcuPaint node')
            return {'CANCELLED'}
        yp = node.node_tree.yp
        context_layer = self._resolve_context_layer(context)

        self.path_coll.clear()
        path_layers = list(iter_path_bake_layers_with_image(yp))
        for layer in path_layers:
            self.path_coll.add().name = layer.name

        self.target_coll.clear()
        # Prefer Decal layers at the top of the picker
        others = [l for l in yp.layers if context_layer is None or l != context_layer]
        decals = [l for l in others if getattr(l, 'texcoord_type', '') == 'Decal']
        rest = [l for l in others if l not in decals]
        for layer in decals + rest:
            label = layer.name
            if getattr(layer, 'texcoord_type', '') == 'Decal':
                label = layer.name  # name is enough; sorted first
            self.target_coll.add().name = label

        if self.mode == 'PICK_TARGET':
            if context_layer and getattr(context_layer, 'enable_path_bake', False):
                self.path_layer_name = context_layer.name
            elif self.path_coll:
                self.path_layer_name = self.path_coll[0].name
            else:
                self.report({'ERROR'}, 'No path/shape bake with an image found')
                return {'CANCELLED'}

            # Default target: active layer if different, else first Decal, else first other
            active = get_active_layer(yp)
            if active and active.name != self.path_layer_name and active.name in [c.name for c in self.target_coll]:
                self.target_layer_name = active.name
            elif decals:
                self.target_layer_name = decals[0].name
            elif self.target_coll:
                self.target_layer_name = self.target_coll[0].name
            else:
                self.report({'ERROR'}, 'No other layer to mask')
                return {'CANCELLED'}

        else:  # PICK_PATH — mask the current layer
            if not context_layer:
                self.report({'ERROR'}, 'No active layer')
                return {'CANCELLED'}
            self.target_layer_name = context_layer.name
            # Exclude self from path list if it is a path layer
            self.path_coll.clear()
            for layer in path_layers:
                if layer == context_layer:
                    continue
                self.path_coll.add().name = layer.name
            if not self.path_coll:
                self.report({'ERROR'}, 'No other path/shape bake with an image found')
                return {'CANCELLED'}
            self.path_layer_name = self.path_coll[0].name

        target = yp.layers.get(self.target_layer_name)
        if target and len(target.masks) == 0:
            self.blend_type = 'MULTIPLY'

        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        if self.mode == 'PICK_TARGET':
            col.label(text='Path / Shape:  %s' % self.path_layer_name)
            col.separator()
            col.label(text='Mask this layer:')
            col.prop_search(self, 'target_layer_name', self, 'target_coll', text='', icon='NODE')
        else:
            col.label(text='Target layer:  %s' % self.target_layer_name)
            col.separator()
            col.label(text='Use this path/shape:')
            col.prop_search(self, 'path_layer_name', self, 'path_coll', text='', icon='CURVE_DATA' if is_bl_newer_than(2, 80) else 'CURVE_BEZCURVE')

        col.separator()
        row = col.row(align=True)
        row.label(text='Input')
        row.prop(self, 'socket_input_name', expand=True)
        col.prop(self, 'blend_type', text='Blend')
        col.prop(self, 'disable_source_layer')

    def execute(self, context):
        node = get_active_ypaint_node()
        if not node:
            self.report({'ERROR'}, 'No UcuPaint node')
            return {'CANCELLED'}
        yp = node.node_tree.yp

        path_layer = yp.layers.get(self.path_layer_name)
        target_layer = yp.layers.get(self.target_layer_name)
        if not path_layer:
            self.report({'ERROR'}, "Path/shape layer '%s' not found" % self.path_layer_name)
            return {'CANCELLED'}
        if not target_layer:
            self.report({'ERROR'}, "Target layer '%s' not found" % self.target_layer_name)
            return {'CANCELLED'}

        mask, err = apply_path_bake_as_mask(
            path_layer, target_layer,
            socket_input_name=self.socket_input_name,
            blend_type=self.blend_type,
            disable_source_layer=self.disable_source_layer
        )
        if err and mask is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        if err and mask is not None:
            self.report({'WARNING'}, err)
            return {'CANCELLED'}

        from .subtree import check_yp_linear_nodes
        check_yp_linear_nodes(yp)

        ypui = context.window_manager.ypui
        ypui.layer_ui.expand_masks = True
        ypui.need_update = True

        # Jump UI to the masked (decal) layer
        idx = get_layer_index(target_layer)
        if idx >= 0:
            yp.active_layer_index = idx

        tag = 'Decal' if getattr(target_layer, 'texcoord_type', '') == 'Decal' else 'layer'
        self.report(
            {'INFO'},
            "Using '%s' as mask on %s '%s'" % (path_layer.name, tag, target_layer.name)
        )
        return {'FINISHED'}

class YCapturePathBakeView(bpy.types.Operator):
    '''Store the current 3D Viewport as this layer's bake view (used by Bake All / rebake)'''
    bl_idname = 'wm.y_capture_path_bake_view'
    bl_label = 'Capture Bake View'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return layer is not None and getattr(layer, 'enable_path_bake', False)

    def execute(self, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer:
            self.report({'ERROR'}, 'No path/shape layer')
            return {'CANCELLED'}

        if not save_layer_path_view(layer, context=context):
            self.report({'ERROR'}, 'No 3D Viewport found to capture')
            return {'CANCELLED'}

        # Enable View so the saved direction is actually used
        if not layer.path_project_view:
            layer.path_project_view = True

        self.report({'INFO'}, "Saved bake view for '%s'" % layer.name)
        return {'FINISHED'}

class YClearPathBakeView(bpy.types.Operator):
    '''Forget the saved bake view for this layer'''
    bl_idname = 'wm.y_clear_path_bake_view'
    bl_label = 'Clear Bake View'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return layer is not None and layer_has_saved_path_view(layer)

    def execute(self, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer:
            return {'CANCELLED'}
        clear_layer_path_view(layer)
        self.report({'INFO'}, "Cleared saved bake view for '%s'" % layer.name)
        return {'FINISHED'}

class YGotoPathBakeView(bpy.types.Operator):
    '''Move the 3D Viewport to this layer's saved bake view'''
    bl_idname = 'wm.y_goto_path_bake_view'
    bl_label = 'Go to Bake View'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return layer is not None and layer_has_saved_path_view(layer)

    def execute(self, context):
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer:
            return {'CANCELLED'}
        if not apply_layer_saved_view_to_viewport(layer, context=context):
            self.report({'ERROR'}, 'Could not restore bake view (no 3D Viewport?)')
            return {'CANCELLED'}
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        self.report({'INFO'}, "Viewport set to saved bake view for '%s'" % layer.name)
        return {'FINISHED'}

def draw_path_bake_ui(layout, context, layer):
    '''Draw Path Bake controls for an IMAGE layer.'''
    if layer.type != 'IMAGE':
        return

    box = layout.box()
    col = box.column(align=True)

    is_shape = layer.path_mode == 'SHAPE'
    header = 'Shape Bake' if is_shape else 'Path Bake'
    icon = 'MESH_CIRCLE' if is_shape and is_bl_newer_than(2, 80) else ('CURVE_DATA' if is_bl_newer_than(2, 80) else 'CURVE_BEZCURVE')

    row = col.row(align=True)
    row.label(text=header, icon=icon)
    row.prop(layer, 'enable_path_bake', text='')

    if not layer.enable_path_bake:
        return

    col.separator()
    col.prop(layer, 'path_mode', expand=True)

    curve_obj = get_path_curve_object(layer)

    if is_bl_newer_than(2, 79):
        col.prop(layer, 'path_curve_object', text='Curve')
    else:
        col.prop(layer, 'path_curve_object_name', text='Curve')

    brow = col.row(align=True)
    brow.context_pointer_set('layer', layer)
    if curve_obj:
        brow.operator('wm.y_select_path_curve', text='Select Curve', icon='CURVE_DATA' if is_bl_newer_than(2, 80) else 'CURVE_BEZCURVE')
    else:
        brow.operator('wm.y_create_path_curve_for_layer', text='Create Curve', icon='ADD' if is_bl_newer_than(2, 80) else 'ZOOMIN')

    sw = col.row(align=True)
    sw.prop(layer, 'path_enable_shrinkwrap', text='Shrinkwrap')
    method_row = sw.row(align=True)
    method_row.enabled = layer.path_enable_shrinkwrap
    method_row.prop(layer, 'path_shrinkwrap_method', expand=True)

    axis_row = col.row(align=True)
    axis_row.label(text='Project Axis')
    # X/Y/Z: Project shrinkwrap + shape plane. View: bake-time for ribbon and shape.
    axes_on = (
        (layer.path_enable_shrinkwrap and layer.path_shrinkwrap_method == 'PROJECT')
        or layer.path_mode == 'SHAPE'
    )
    axes = axis_row.row(align=True)
    axes.enabled = axes_on
    axes.prop(layer, 'path_project_x', text='X', toggle=True)
    axes.prop(layer, 'path_project_y', text='Y', toggle=True)
    axes.prop(layer, 'path_project_z', text='Z', toggle=True)
    axis_row.prop(layer, 'path_project_view', text='View', toggle=True)

    if layer.path_project_view:
        view_row = col.row(align=True)
        view_row.context_pointer_set('layer', layer)
        if layer_has_saved_path_view(layer):
            view_row.label(text='Bake view saved', icon='SOLO_ON')
            view_row.operator('wm.y_goto_path_bake_view', text='', icon='VIEWZOOM' if is_bl_newer_than(2, 80) else 'ZOOM_SELECTED')
            view_row.operator('wm.y_capture_path_bake_view', text='', icon='FILE_REFRESH')
            view_row.operator('wm.y_clear_path_bake_view', text='', icon='X')
        else:
            view_row.label(text='No saved view yet')
            view_row.operator('wm.y_capture_path_bake_view', text='Capture View', icon='CAMERA_DATA')

    col.separator()
    # Bake image resolution (Layer Source Info is read-only for FILE images)
    source = get_layer_source(layer)
    bake_img = source.image if source else None
    size_row = col.row(align=True)
    if bake_img and bake_img.size[0] > 0:
        size_row.label(text='Image: %d x %d' % (bake_img.size[0], bake_img.size[1]))
    else:
        size_row.label(text='Image: -')
    size_row.context_pointer_set('layer', layer)
    size_row.operator('wm.y_resize_path_bake_image', text='Resize', icon='FULLSCREEN_ENTER')

    col.prop(layer, 'path_resolution')
    if is_shape:
        col.prop(layer, 'path_shape_feather')
        col.prop(layer, 'path_falloff', text='Feather Falloff')
    else:
        col.prop(layer, 'path_width')
        col.prop(layer, 'path_falloff')
    col.prop(layer, 'path_color')
    if is_bl_newer_than(2, 79):
        col.prop(layer, 'path_texture', text='Texture')
        if getattr(layer, 'path_texture', None):
            col.prop(layer, 'path_tile_u')
            col.prop(layer, 'path_texture_rotation', text='Texture Rotate')
    col.prop(layer, 'path_clear_before_bake')

    col.separator()
    brow = col.row(align=True)
    brow.context_pointer_set('layer', layer)
    brow.scale_y = 1.3
    brow.enabled = curve_obj is not None
    bake_label = 'Bake Shape' if is_shape else 'Bake Path'
    brow.operator('wm.y_bake_path_to_layer', text=bake_label, icon_value=lib_get_bake_icon())

    node = get_active_ypaint_node()
    can_bake_all = False
    if node:
        can_bake_all = any(True for _ in iter_bakeable_path_layers(node.node_tree.yp))
    brow = col.row(align=True)
    brow.scale_y = 1.1
    brow.enabled = can_bake_all
    brow.operator('wm.y_bake_all_paths', text='Bake All Paths', icon_value=lib_get_bake_icon())

    col.separator()
    mask_row = col.row(align=True)
    mask_row.context_pointer_set('layer', layer)
    mask_row.scale_y = 1.15
    has_img = bake_img is not None
    mask_row.enabled = has_img
    mask_icon = 'MOD_MASK' if is_bl_newer_than(2, 80) else 'IMAGE_ALPHA'
    op = mask_row.operator(
        'wm.y_use_path_bake_as_mask',
        text='Use as Mask on Layer…',
        icon=mask_icon
    )
    op.mode = 'PICK_TARGET'

def lib_get_bake_icon():
    from . import lib
    return lib.get_icon('bake')

def register():
    bpy.utils.register_class(YNewPathLayer)
    bpy.utils.register_class(YBakePathToLayer)
    bpy.utils.register_class(YBakeAllPaths)
    bpy.utils.register_class(YResizePathBakeImage)
    bpy.utils.register_class(YSelectPathCurve)
    bpy.utils.register_class(YCreatePathCurveForLayer)
    bpy.utils.register_class(YUsePathBakeAsMask)
    bpy.utils.register_class(YCapturePathBakeView)
    bpy.utils.register_class(YClearPathBakeView)
    bpy.utils.register_class(YGotoPathBakeView)

def unregister():
    bpy.utils.unregister_class(YNewPathLayer)
    bpy.utils.unregister_class(YBakePathToLayer)
    bpy.utils.unregister_class(YBakeAllPaths)
    bpy.utils.unregister_class(YResizePathBakeImage)
    bpy.utils.unregister_class(YSelectPathCurve)
    bpy.utils.unregister_class(YCreatePathCurveForLayer)
    bpy.utils.unregister_class(YUsePathBakeAsMask)
    bpy.utils.unregister_class(YCapturePathBakeView)
    bpy.utils.unregister_class(YClearPathBakeView)
    bpy.utils.unregister_class(YGotoPathBakeView)
