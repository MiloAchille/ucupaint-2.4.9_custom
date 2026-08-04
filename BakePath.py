import bpy, math, time
import numpy
from mathutils import Vector, Matrix
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
            method = _layer_shrinkwrap_method(layer)
            target = new_curve.parent
            if target and target.type == 'MESH':
                ensure_path_shrinkwrap(new_curve, target, method=method)
            else:
                mod = get_path_shrinkwrap_modifier(new_curve)
                if mod and mod.target:
                    ensure_path_shrinkwrap(new_curve, mod.target, method=method)

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

def configure_path_shrinkwrap(mod, target_obj, method='NEAREST_SURFACEPOINT'):
    '''Apply path shrinkwrap settings (Nearest Surface Point or Project +/-).'''
    mod.target = target_obj
    if method == 'PROJECT':
        mod.wrap_method = 'PROJECT'
        if hasattr(mod, 'use_negative_direction'):
            mod.use_negative_direction = True
        if hasattr(mod, 'use_positive_direction'):
            mod.use_positive_direction = True
        # No project axes — match Blender Project setup with only +/- directions
        if hasattr(mod, 'use_project_x'):
            mod.use_project_x = False
        if hasattr(mod, 'use_project_y'):
            mod.use_project_y = False
        if hasattr(mod, 'use_project_z'):
            mod.use_project_z = False
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

def ensure_path_shrinkwrap(curve_obj, target_obj=None, method='NEAREST_SURFACEPOINT'):
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
    configure_path_shrinkwrap(mod, target_obj, method=method)
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

def update_path_enable_shrinkwrap(self, context):
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    if self.path_enable_shrinkwrap:
        target = curve_obj.parent if curve_obj.parent else context.object
        if target and target.type == 'MESH':
            ensure_path_shrinkwrap(curve_obj, target, method=_layer_shrinkwrap_method(self))
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
        ensure_path_shrinkwrap(curve_obj, target, method=_layer_shrinkwrap_method(self))

def create_path_curve(target_obj, name='Path', use_shrinkwrap=True, cyclic=False,
                      shrinkwrap_method='NEAREST_SURFACEPOINT'):
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
        ensure_path_shrinkwrap(curve_obj, target_obj, method=shrinkwrap_method)

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
        return loops[0]
    # Ribbon/legacy callers expect one polyline — join with gaps (not ideal for shapes)
    out = []
    for loop in loops:
        if out and (out[-1] - loop[0]).length > 1e-6:
            out.append(loop[0].copy())
        out.extend(loop)
    return out

def _sample_bezier_math_loops(curve_obj, resolution):
    '''
    Sample each spline as its own world-space polyline.
    Multi-cyclic shapes stay separate so fill does not bridge between loops.
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
        if len(points) >= 3:
            loops.append(points)
        elif len(points) >= 2:
            loops.append(points)
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
    Return a list of world-space polylines, one per spline.
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
        _snap_polyline_nearest_to_target(loop, mod.target, offset=offset)
        for loop in loops
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

def _uv_from_tri_hit(tris, loc, normal, index, dist):
    if index < 0 or index >= len(tris):
        return None
    v0, v1, v2, uv0, uv1, uv2, face_n = tris[index]
    try:
        uv = _barycentric_uv(loc, v0, v1, v2, uv0, uv1, uv2)
    except Exception:
        uv = (uv0 + uv1 + uv2) / 3.0
    n = face_n if face_n.length > 1e-8 else Vector(normal)
    if n.length > 1e-8:
        n.normalize()
    else:
        n = Vector((0, 0, 1))
    return loc, n, Vector((uv.x, uv.y)), dist

def _raycast_uv_along_dir(bvh, tris, point, direction, max_dist):
    '''
    Raycast ± along direction (shape plane normal) and return the closest hit UV.
    Matches Shrinkwrap Project with +/- and no axis.
    '''
    if direction.length < 1e-8:
        return None
    d = direction.normalized()
    max_dist = float(max_dist) if max_dist is not None else 1.0
    best = None
    for sign in (1.0, -1.0):
        # Start slightly behind the point so coplanar/on-surface points still hit
        origin = point - d * sign * min(max_dist * 0.001, 1e-4)
        try:
            hit = bvh.ray_cast(origin, d * sign, max_dist)
        except Exception:
            hit = None
        if not hit or hit[0] is None:
            continue
        loc, normal, index, dist = hit
        # Distance from original point
        dist_from_p = (Vector(loc) - point).length
        if best is None or dist_from_p < best[3]:
            mapped = _uv_from_tri_hit(tris, Vector(loc), normal, index, dist_from_p)
            if mapped:
                best = mapped
    return best

def _project_shape_polyline_project_to_uv(polyline, bvh, tris, project_distance):
    '''
    Project-mode UV mapping: raycast each authored point along the shape plane
    normal (±). Misses (overhang past the mesh) are filled with a plane->UV
    affine trained only on successful ray hits.

    Avoids the old full-affine path that warped multi-island / non-linear UVs
    into fan-shaped garbage.
    '''
    if len(polyline) < 3:
        return [], 0

    center, axis_u, axis_v, plane_n = _fit_polyline_plane(polyline)

    def to_plane(p):
        d = p - center
        return (d.dot(axis_u), d.dot(axis_v))

    uvs = [None] * len(polyline)
    fit_pts = []
    fit_uvs = []
    seam_jumps = 0
    prev_uv = None
    for i, p in enumerate(polyline):
        hit = _raycast_uv_along_dir(bvh, tris, p, plane_n, project_distance)
        if not hit:
            continue
        _loc, _n, uv, _dist = hit
        uvs[i] = uv.copy()
        fit_pts.append(p)
        fit_uvs.append(uv)
        if prev_uv is not None and (uv - prev_uv).length > 0.25:
            seam_jumps += 1
        prev_uv = uv

    misses = [i for i, uv in enumerate(uvs) if uv is None]
    xu = xv = None
    if misses and len(fit_pts) >= 3:
        A = numpy.array([[to_plane(p)[0], to_plane(p)[1], 1.0] for p in fit_pts], dtype=numpy.float64)
        bu = numpy.array([uv.x for uv in fit_uvs], dtype=numpy.float64)
        bv = numpy.array([uv.y for uv in fit_uvs], dtype=numpy.float64)
        try:
            xu, _, _, _ = numpy.linalg.lstsq(A, bu, rcond=None)
            xv, _, _, _ = numpy.linalg.lstsq(A, bv, rcond=None)
        except Exception:
            xu = xv = None

    if xu is not None and xv is not None:
        for i in misses:
            s, t = to_plane(polyline[i])
            uvs[i] = Vector((
                float(xu[0] * s + xu[1] * t + xu[2]),
                float(xv[0] * s + xv[1] * t + xv[2]),
            ))

    # Any remaining gaps: nearest as last resort (keeps polygon closed)
    out = []
    for i, p in enumerate(polyline):
        if uvs[i] is not None:
            out.append(uvs[i])
            continue
        hit = _nearest_uv(bvh, tris, p, max_dist=project_distance)
        if hit:
            out.append(hit[2].copy())
    return out, seam_jumps

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

def _soft_falloff(t, power=1.0):
    '''t in 0..1 (0=center, 1=edge). Returns opacity.'''
    t = max(0.0, min(1.0, t))
    # Hard edge when falloff is 0
    if power <= 0.0:
        return 1.0 if t < 1.0 - 1e-6 else 0.0
    # Smoothstep-like edge
    a = 1.0 - t * t * (3.0 - 2.0 * t)
    if power != 1.0:
        a = pow(max(a, 0.0), power)
    return a

def _blend_pixel(pxs, px, py, rgba, strength):
    if strength <= 1e-6:
        return
    cr, cg, cb, ca = rgba
    a = ca * strength
    if a <= 1e-6:
        return
    dst = pxs[py, px]
    out_a = a + dst[3] * (1.0 - a)
    if out_a > 1e-6:
        pxs[py, px, 0] = (cr * a + dst[0] * dst[3] * (1.0 - a)) / out_a
        pxs[py, px, 1] = (cg * a + dst[1] * dst[3] * (1.0 - a)) / out_a
        pxs[py, px, 2] = (cb * a + dst[2] * dst[3] * (1.0 - a)) / out_a
        pxs[py, px, 3] = min(1.0, out_a)
    else:
        pxs[py, px, 3] = 0.0

def _splat_rgba(pxs, width, height, u, v, rgba, strength, radius_px=1.0):
    '''Stamp into float32 array shaped (height, width, 4), with optional brush radius in pixels.'''
    if strength <= 1e-6:
        return
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return

    x = u * (width - 1)
    y = v * (height - 1)
    radius = max(0.75, float(radius_px))
    r_ceil = int(math.ceil(radius)) + 1
    x0 = max(0, int(math.floor(x)) - r_ceil)
    x1 = min(width - 1, int(math.ceil(x)) + r_ceil)
    y0 = max(0, int(math.floor(y)) - r_ceil)
    y1 = min(height - 1, int(math.ceil(y)) + r_ceil)
    r2 = radius * radius

    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 > r2:
                continue
            # Soft disk falloff inside brush
            w = 1.0 - math.sqrt(d2) / radius
            w = w * w
            _blend_pixel(pxs, px, py, rgba, strength * w)

def _splat_coverage_max(cov, width, height, u, v, strength, radius_px=1.25):
    '''Max-blend soft coverage stamp — no alpha-over banding between overlaps.'''
    if strength <= 1e-6:
        return
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return
    x = u * (width - 1)
    y = v * (height - 1)
    radius = max(0.75, float(radius_px))
    r_ceil = int(math.ceil(radius)) + 1
    x0 = max(0, int(math.floor(x)) - r_ceil)
    x1 = min(width - 1, int(math.ceil(x)) + r_ceil)
    y0 = max(0, int(math.floor(y)) - r_ceil)
    y1 = min(height - 1, int(math.ceil(y)) + r_ceil)
    inv_r = 1.0 / radius
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            dx = px - x
            dy = py - y
            d = math.sqrt(dx * dx + dy * dy) * inv_r
            if d > 1.0:
                continue
            w = 1.0 - d
            w = w * w
            val = strength * w
            if val > cov[py, px]:
                cov[py, px] = val

def _barycentric_2d(px, py, ax, ay, bx, by, cx, cy):
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-12:
        return None
    inv = 1.0 / den
    v = (v2x * v1y - v1x * v2y) * inv
    w = (v0x * v2y - v2x * v0y) * inv
    u = 1.0 - v - w
    return u, v, w

def _fill_uv_tri_coverage(cov, img_w, img_h, uv_a, s_a, uv_b, s_b, uv_c, s_c):
    '''Rasterize a UV triangle into a coverage buffer with interpolated strength.'''
    ax, ay = uv_a.x * (img_w - 1), uv_a.y * (img_h - 1)
    bx, by = uv_b.x * (img_w - 1), uv_b.y * (img_h - 1)
    cx, cy = uv_c.x * (img_w - 1), uv_c.y * (img_h - 1)
    min_x = max(0, int(math.floor(min(ax, bx, cx))))
    max_x = min(img_w - 1, int(math.ceil(max(ax, bx, cx))))
    min_y = max(0, int(math.floor(min(ay, by, cy))))
    max_y = min(img_h - 1, int(math.ceil(max(ay, by, cy))))
    if min_x > max_x or min_y > max_y:
        return 0
    filled = 0
    for py in range(min_y, max_y + 1):
        # Sample at pixel centers
        fy = py + 0.5
        for px in range(min_x, max_x + 1):
            fx = px + 0.5
            bary = _barycentric_2d(fx, fy, ax, ay, bx, by, cx, cy)
            if not bary:
                continue
            u, v, w = bary
            if u < -1e-4 or v < -1e-4 or w < -1e-4:
                continue
            s = s_a * u + s_b * v + s_c * w
            if s <= 1e-6:
                continue
            if s > cov[py, px]:
                cov[py, px] = s
                filled += 1
    return filled

def _fill_uv_quad_coverage(cov, img_w, img_h, uv00, s00, uv10, s10, uv11, s11, uv01, s01):
    '''Fill UV quad (00-10-11-01) as two triangles into coverage.'''
    n = _fill_uv_tri_coverage(cov, img_w, img_h, uv00, s00, uv10, s10, uv11, s11)
    n += _fill_uv_tri_coverage(cov, img_w, img_h, uv00, s00, uv11, s11, uv01, s01)
    return n

def _parallel_transport_side(prev_side, prev_tangent, tangent):
    '''Carry ribbon side along the path so mesh-normal flips don't invert the frame.'''
    if prev_side is None or prev_tangent is None or tangent is None:
        return None
    if prev_tangent.length < 1e-8 or tangent.length < 1e-8:
        return None
    t0 = prev_tangent.normalized()
    t1 = tangent.normalized()
    side = prev_side.copy()
    axis = t0.cross(t1)
    if axis.length > 1e-8:
        try:
            angle = t0.angle(t1)
        except Exception:
            angle = 0.0
        if abs(angle) > 1e-8:
            side = Matrix.Rotation(angle, 3, axis.normalized()) @ side
    elif t0.dot(t1) < 0.0:
        side = -side
    side = side - t1 * side.dot(t1)
    if side.length < 1e-8:
        return None
    return side.normalized()

def _sample_path_texture(tex_pxs, tw, th, u_len, v_across, tile_u=1.0, rotation_deg=0.0):
    '''Sample path stamp texture. u_len 0..1 along path, v_across -1..1 across width.'''
    uu = (u_len * tile_u) % 1.0
    if uu < 0.0:
        uu += 1.0
    vv = v_across * 0.5 + 0.5

    if rotation_deg:
        rad = math.radians(rotation_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x = uu - 0.5
        y = vv - 0.5
        uu = x * cos_a - y * sin_a + 0.5
        vv = x * sin_a + y * cos_a + 0.5
        # Wrap so rotated tiles still repeat cleanly
        uu = uu % 1.0
        if uu < 0.0:
            uu += 1.0
        vv = vv % 1.0
        if vv < 0.0:
            vv += 1.0
    else:
        vv = max(0.0, min(1.0, vv))

    x = uu * (tw - 1)
    y = vv * (th - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, tw - 1)
    y1 = min(y0 + 1, th - 1)
    fx = x - x0
    fy = y - y0
    c00 = tex_pxs[y0, x0]
    c10 = tex_pxs[y0, x1]
    c01 = tex_pxs[y1, x0]
    c11 = tex_pxs[y1, x1]
    c0 = c00 * (1 - fx) + c10 * fx
    c1 = c01 * (1 - fx) + c11 * fx
    return c0 * (1 - fy) + c1 * fy

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

def _uv_pixel_dist(uv_a, uv_b, img_w, img_h):
    dx = (uv_a.x - uv_b.x) * (img_w - 1)
    dy = (uv_a.y - uv_b.y) * (img_h - 1)
    return math.sqrt(dx * dx + dy * dy)

def _project_polyline_to_uv(polyline, bvh, tris, project_distance):
    uvs = []
    seam_jumps = 0
    prev = None
    for p in polyline:
        hit = _nearest_uv(bvh, tris, p, max_dist=project_distance)
        if not hit:
            continue
        _, _, uv, _ = hit
        if prev is not None:
            # Rough seam detection in UV units
            if (uv - prev).length > 0.25:
                seam_jumps += 1
        uvs.append(uv.copy())
        prev = uv
    return uvs, seam_jumps

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

    axis_u = normal.cross(Vector((0.0, 0.0, 1.0)))
    if axis_u.length < 1e-6:
        axis_u = normal.cross(Vector((0.0, 1.0, 0.0)))
    if axis_u.length < 1e-8:
        axis_u = Vector((1.0, 0.0, 0.0))
    else:
        axis_u.normalize()
    axis_v = normal.cross(axis_u)
    if axis_v.length < 1e-8:
        axis_v = Vector((0.0, 1.0, 0.0))
    else:
        axis_v.normalize()
    return center, axis_u, axis_v, normal

def _project_shape_polyline_to_uv(polyline, bvh, tris, project_distance,
                                  preserve_silhouette=False):
    '''
    Map a closed world-space shape into UV continuously (Nearest / hybrid).

    On-mesh points keep nearest UVs; clear overhangs use plane->UV affine.
    For Project mode prefer _project_shape_polyline_project_to_uv (raycast).
    '''
    if len(polyline) < 3:
        return [], 0

    center, axis_u, axis_v, _normal = _fit_polyline_plane(polyline)

    def to_plane(p):
        d = p - center
        return (d.dot(axis_u), d.dot(axis_v))

    # Per-point hits (None if no hit within project_distance)
    point_hits = []  # list of (uv, dist) or None
    hit_dists = []
    hits = []  # (index, p, uv, dist)
    prev_uv = None
    seam_jumps = 0
    for i, p in enumerate(polyline):
        hit = _nearest_uv(bvh, tris, p, max_dist=project_distance)
        if not hit:
            point_hits.append(None)
            continue
        _loc, _n, uv, dist = hit
        point_hits.append((uv.copy(), dist))
        hit_dists.append(dist)
        hits.append((i, p, uv, dist))
        if prev_uv is not None and (uv - prev_uv).length > 0.25:
            seam_jumps += 1
        prev_uv = uv

    if len(hits) < 3:
        return _project_polyline_to_uv(polyline, bvh, tris, project_distance)

    sorted_d = sorted(hit_dists)
    median = sorted_d[len(sorted_d) // 2]
    on_thresh = max(median * 6.0, sorted_d[0] * 12.0, 1e-5)
    if project_distance is not None:
        on_thresh = min(on_thresh, max(project_distance * 0.35, median * 3.0 + 1e-6))

    overhang_indices = set()
    for i, ph in enumerate(point_hits):
        if ph is None or ph[1] > on_thresh:
            overhang_indices.add(i)

    # Hybrid only: no clear overhang → pure nearest
    if not preserve_silhouette and len(overhang_indices) < 3:
        uvs = []
        for ph in point_hits:
            if ph is not None:
                uvs.append(ph[0].copy())
        if len(uvs) < 3:
            return _project_polyline_to_uv(polyline, bvh, tris, project_distance)
        return uvs, seam_jumps

    # Train affine from on-surface samples (closest hits for Project / overhang)
    samples_st = []
    samples_uv = []
    for _i, p, uv, dist in hits:
        if not preserve_silhouette and dist > on_thresh:
            continue
        # For Project silhouette: prefer closer hits as trainers, still allow all
        samples_st.append(to_plane(p))
        samples_uv.append((uv.x, uv.y))

    if preserve_silhouette:
        # Prefer the closest third of hits so edge-collapsed far points don't bias the fit
        closest = sorted(hits, key=lambda h: h[3])[:max(3, max(3, len(hits) // 2))]
        samples_st = []
        samples_uv = []
        for _i, p, uv, _dist in closest:
            samples_st.append(to_plane(p))
            samples_uv.append((uv.x, uv.y))
    elif len(samples_st) < 3:
        closest = sorted(hits, key=lambda h: h[3])[:max(3, len(hits) // 3)]
        samples_st = []
        samples_uv = []
        for _i, p, uv, _dist in closest:
            samples_st.append(to_plane(p))
            samples_uv.append((uv.x, uv.y))

    xu = xv = None
    affine_ok = False
    if len(samples_st) >= 3:
        A = numpy.array([[s, t, 1.0] for s, t in samples_st], dtype=numpy.float64)
        bu = numpy.array([uv[0] for uv in samples_uv], dtype=numpy.float64)
        bv = numpy.array([uv[1] for uv in samples_uv], dtype=numpy.float64)
        try:
            xu, _, _, _ = numpy.linalg.lstsq(A, bu, rcond=None)
            xv, _, _, _ = numpy.linalg.lstsq(A, bv, rcond=None)
            pred_u = A @ xu
            pred_v = A @ xv
            err = float(numpy.sqrt(numpy.mean((pred_u - bu) ** 2 + (pred_v - bv) ** 2)))
            if preserve_silhouette:
                # Project must keep silhouette even across mild unwrap distortion
                affine_ok = err <= 0.20
            else:
                affine_ok = err <= 0.08 and seam_jumps < 3
        except Exception:
            affine_ok = False

    if not affine_ok:
        if preserve_silhouette:
            # Last resort for Project: still try affine from all hits (ignore residual)
            try:
                A = numpy.array([[to_plane(p)[0], to_plane(p)[1], 1.0] for _i, p, _uv, _d in hits], dtype=numpy.float64)
                bu = numpy.array([uv.x for _i, _p, uv, _d in hits], dtype=numpy.float64)
                bv = numpy.array([uv.y for _i, _p, uv, _d in hits], dtype=numpy.float64)
                xu, _, _, _ = numpy.linalg.lstsq(A, bu, rcond=None)
                xv, _, _, _ = numpy.linalg.lstsq(A, bv, rcond=None)
                affine_ok = True
            except Exception:
                affine_ok = False
        if not affine_ok:
            uvs = []
            for ph in point_hits:
                if ph is not None:
                    uvs.append(ph[0].copy())
            if len(uvs) < 3:
                return _project_polyline_to_uv(polyline, bvh, tris, project_distance)
            return uvs, seam_jumps

    def affine_uv(p):
        s, t = to_plane(p)
        return Vector((
            float(xu[0] * s + xu[1] * t + xu[2]),
            float(xv[0] * s + xv[1] * t + xv[2]),
        ))

    # Project: every outline point through affine (preserves authored silhouette)
    if preserve_silhouette:
        return [affine_uv(p) for p in polyline], seam_jumps

    # Hybrid: nearest on-mesh, affine only for clear overhang / missed points
    uvs = []
    for i, p in enumerate(polyline):
        if i in overhang_indices:
            uvs.append(affine_uv(p))
        else:
            ph = point_hits[i]
            uvs.append(ph[0].copy() if ph is not None else affine_uv(p))
    return uvs, seam_jumps

def _decimate_poly(poly, max_points=96):
    '''Keep polygon under max_points by uniform stride (closed loop).'''
    n = len(poly)
    if n <= max_points:
        return poly
    step = n / float(max_points)
    out = [poly[int(i * step) % n] for i in range(max_points)]
    return out

def _densify_poly_edges(poly, max_edge_px=2.0):
    '''Subdivide long polygon edges so scanline fill follows the outline tightly.'''
    n = len(poly)
    if n < 3:
        return poly
    out = []
    max_edge = max(0.5, float(max_edge_px))
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        out.append((x0, y0))
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= max_edge:
            continue
        steps = int(math.ceil(dist / max_edge))
        for s in range(1, steps):
            t = s / float(steps)
            out.append((x0 + dx * t, y0 + dy * t))
    return out

def _inflate_poly(poly, amount_px):
    '''Push polygon vertices outward along averaged edge normals (pixel space).'''
    n = len(poly)
    if n < 3 or abs(amount_px) < 1e-8:
        return poly

    # Signed area to know winding. Image Y grows downward, so the geometric
    # left-of-edge normal for a CCW poly points inward — flip for outward.
    area = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    sign = -1.0 if area >= 0.0 else 1.0

    out = []
    for i in range(n):
        x_prev, y_prev = poly[(i - 1) % n]
        x, y = poly[i]
        x_next, y_next = poly[(i + 1) % n]

        e0x, e0y = x - x_prev, y - y_prev
        e1x, e1y = x_next - x, y_next - y
        l0 = math.sqrt(e0x * e0x + e0y * e0y) or 1.0
        l1 = math.sqrt(e1x * e1x + e1y * e1y) or 1.0
        # Inward normals of incoming/outgoing edges, then flip to outward
        n0x, n0y = -sign * e0y / l0, sign * e0x / l0
        n1x, n1y = -sign * e1y / l1, sign * e1x / l1
        nx, ny = n0x + n1x, n0y + n1y
        nl = math.sqrt(nx * nx + ny * ny)
        if nl < 1e-8:
            nx, ny = n0x, n0y
            nl = math.sqrt(nx * nx + ny * ny) or 1.0
        nx /= nl
        ny /= nl
        out.append((x + nx * amount_px, y + ny * amount_px))
    return out

def _scanline_fill_mask(poly, img_w, img_h, min_x, max_x, min_y, max_y):
    '''Fast even-odd scanline fill into a float mask (img_h, img_w).

    Uses pixel-center coverage on both axes (x+0.5, y+0.5).
    '''
    mask = numpy.zeros((img_h, img_w), dtype=numpy.float32)
    n = len(poly)
    if n < 3:
        return mask

    for y in range(min_y, max_y + 1):
        y_c = y + 0.5
        hits = []
        j = n - 1
        for i in range(n):
            x0, y0 = poly[j]
            x1, y1 = poly[i]
            if (y0 > y_c) != (y1 > y_c):
                denom = y1 - y0
                if abs(denom) < 1e-30:
                    denom = 1e-30
                hits.append(x0 + (y_c - y0) / denom * (x1 - x0))
            j = i
        if len(hits) < 2:
            continue
        hits.sort()
        for k in range(0, len(hits) - 1, 2):
            x_a = hits[k]
            x_b = hits[k + 1]
            if x_b < x_a:
                x_a, x_b = x_b, x_a
            # Pixel x is covered when its center x+0.5 lies strictly inside (x_a, x_b)
            x0 = max(min_x, int(math.ceil(x_a - 0.5)))
            x1 = min(max_x, int(math.floor(x_b - 0.5)))
            if x1 >= x0:
                mask[y, x0:x1 + 1] = 1.0
    return mask

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
                        shrinkwrap_method=None):
    '''
    Project a closed (cyclic) curve to UV and fill it as a solid vector shape.
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

    if project_distance is None:
        dim = max(obj.dimensions) if obj.dimensions.length > 0 else 1.0
        project_distance = max(dim * 0.5, 0.1)

    def _close_loop(pts):
        if len(pts) < 3:
            return pts
        if (pts[0] - pts[-1]).length > 1e-6:
            return list(pts) + [pts[0].copy()]
        return pts

    def _count_overhang(pts):
        if len(pts) < 3:
            return 0
        dists = []
        misses = 0
        for p in pts:
            hit = _nearest_uv(bvh, tris, p, max_dist=project_distance)
            if not hit:
                misses += 1
                continue
            dists.append(hit[3])
        if len(dists) < 3:
            return misses
        sd = sorted(dists)
        median = sd[len(sd) // 2]
        thresh = max(median * 6.0, sd[0] * 12.0, 1e-5)
        if project_distance is not None:
            thresh = min(thresh, max(project_distance * 0.35, median * 3.0 + 1e-6))
        n_far = sum(1 for d in dists if d > thresh)
        return n_far + misses

    # Authored loops always; Nearest also builds surface-snapped loops
    authored_loops = [_close_loop(L) for L in _sample_bezier_math_loops(curve_obj, shape_res) if len(L) >= 3]
    if use_project:
        # Project: bake the authored silhouette (matches Project viewport intent)
        source_loops = authored_loops
    else:
        source_loops = [
            _close_loop(L) for L in _curve_to_polylines(
                curve_obj, resolution=shape_res, shrinkwrap_method='NEAREST_SURFACEPOINT'
            ) if len(L) >= 3
        ]
        if not source_loops:
            source_loops = authored_loops

    if not source_loops:
        return False, 'Closed shape needs at least 3 projected points'

    feather = max(0.0, float(feather_px))
    max_poly = min(max(shape_res, 64), 1024)
    mask = numpy.zeros((img_h, img_w), dtype=numpy.float32)
    seam_jumps = 0
    all_uvs = []
    all_xs = []
    all_ys = []
    loops_baked = 0

    for li, polyline in enumerate(source_loops):
        authored = authored_loops[li] if li < len(authored_loops) else polyline
        if use_project:
            # Raycast ± along shape plane normal (true Project); affine only for misses
            uvs, jumps = _project_shape_polyline_project_to_uv(
                authored, bvh, tris, project_distance
            )
        else:
            use_overhang = len(authored) >= 3 and _count_overhang(authored) >= 3
            if use_overhang:
                uvs, jumps = _project_shape_polyline_to_uv(
                    authored, bvh, tris, project_distance, preserve_silhouette=False
                )
            else:
                uvs, jumps = _project_polyline_to_uv(polyline, bvh, tris, project_distance)
        seam_jumps += jumps
        if len(uvs) < 3:
            continue
        if (uvs[0] - uvs[-1]).length < 1e-6:
            uvs = uvs[:-1]
        if len(uvs) < 3:
            continue

        poly = [(uv.x * img_w, uv.y * img_h) for uv in uvs]
        poly = _decimate_poly(poly, max_points=max_poly)
        poly = _densify_poly_edges(poly, max_edge_px=1.5)
        if feather <= 0.5:
            poly = _inflate_poly(poly, 0.6)

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        pad = int(math.ceil(max(feather, 0.6))) + 2
        min_x = max(0, int(math.floor(min(xs) - pad)))
        max_x = min(img_w - 1, int(math.ceil(max(xs) + pad)))
        min_y = max(0, int(math.floor(min(ys) - pad)))
        max_y = min(img_h - 1, int(math.ceil(max(ys) + pad)))

        loop_mask = _scanline_fill_mask(poly, img_w, img_h, min_x, max_x, min_y, max_y)
        if feather > 0.5:
            crop = loop_mask[min_y:max_y + 1, min_x:max_x + 1]
            crop = _box_blur_mask(crop, feather)
            if falloff <= 0.0:
                crop = (crop >= 0.5).astype(numpy.float32)
            elif falloff != 1.0:
                crop = numpy.power(numpy.clip(crop, 0.0, 1.0), float(falloff)).astype(numpy.float32)
            loop_mask[min_y:max_y + 1, min_x:max_x + 1] = crop

        numpy.maximum(mask, loop_mask, out=mask)
        all_uvs.extend(uvs)
        all_xs.extend(xs)
        all_ys.extend(ys)
        loops_baked += 1

    if loops_baked == 0 or not all_xs:
        return False, 'Could not project the shape onto the mesh UVs'

    uvs = all_uvs
    xs = all_xs
    ys = all_ys

    if is_bl_newer_than(2, 83):
        pxs = numpy.empty(shape=img_w * img_h * 4, dtype=numpy.float32)
        image.pixels.foreach_get(pxs)
    else:
        pxs = numpy.array(image.pixels[:], dtype=numpy.float32)
    pxs.shape = (img_h, img_w, 4)

    if clear:
        pxs[:] = 0.0

    cr, cg, cb, ca = [float(c) for c in color]
    alpha = mask * ca
    filled = int(numpy.count_nonzero(alpha > 1e-4))
    if filled == 0:
        return False, 'Shape did not cover any pixels (check UV projection / curve position)'

    tex_pxs = None
    tw = th = 0
    if path_texture and path_texture.size[0] > 0 and path_texture.size[1] > 0:
        tw, th = path_texture.size[0], path_texture.size[1]
        if is_bl_newer_than(2, 83):
            tex_pxs = numpy.empty(shape=tw * th * 4, dtype=numpy.float32)
            path_texture.pixels.foreach_get(tex_pxs)
        else:
            tex_pxs = numpy.array(path_texture.pixels[:], dtype=numpy.float32)
        tex_pxs.shape = (th, tw, 4)

    if tex_pxs is None:
        # Solid fill — vectorized over (only where shape contributes)
        m = alpha > 1e-4
        if numpy.any(m):
            src_a = alpha[m]
            dst_a = pxs[:, :, 3][m]
            out_a = src_a + dst_a * (1.0 - src_a)
            safe = numpy.maximum(out_a, 1e-8)
            for ch, cval in enumerate((cr, cg, cb)):
                dst_c = pxs[:, :, ch][m]
                pxs[:, :, ch][m] = (cval * src_a + dst_c * dst_a * (1.0 - src_a)) / safe
            pxs[:, :, 3][m] = numpy.minimum(out_a, 1.0)
    else:
        # Textured fill — vectorized bilinear sample + blend (was a slow Python loop)
        bbox_w = max(max(xs) - min(xs), 1e-6)
        bbox_h = max(max(ys) - min(ys), 1e-6)
        m = alpha > 1e-4
        ys_idx, xs_idx = numpy.nonzero(m)
        if len(xs_idx):
            u_local = (xs_idx.astype(numpy.float32) + 0.5 - min(xs)) / bbox_w
            v_local = (ys_idx.astype(numpy.float32) + 0.5 - min(ys)) / bbox_h
            sampled = _sample_path_texture_batch(
                tex_pxs, tw, th, u_local, v_local * 2.0 - 1.0,
                tile_u=tile_u, rotation_deg=rotation_deg
            )
            # Matches previous per-pixel path: strength = mask, src_a = tex_a * alpha
            src_a = sampled[:, 3] * alpha[ys_idx, xs_idx]
            dst_a = pxs[ys_idx, xs_idx, 3]
            out_a = src_a + dst_a * (1.0 - src_a)
            safe = numpy.maximum(out_a, 1e-8)
            for ch, cval in enumerate((cr, cg, cb)):
                src_c = sampled[:, ch] * cval
                dst_c = pxs[ys_idx, xs_idx, ch]
                pxs[ys_idx, xs_idx, ch] = (src_c * src_a + dst_c * dst_a * (1.0 - src_a)) / safe
            pxs[ys_idx, xs_idx, 3] = numpy.minimum(out_a, 1.0)

    flat = pxs.ravel()
    if is_bl_newer_than(2, 83):
        image.pixels.foreach_set(flat)
    else:
        image.pixels = flat.tolist()
    image.update()

    total = img_w * img_h
    msg = 'Filled shape (%d loop%s, %d / %d pixels)' % (
        loops_baked, 's' if loops_baked != 1 else '', filled, total
    )
    msg += ' [%s]' % ('Project' if use_project else 'Nearest')
    uv_xs = [uv.x for uv in uvs]
    uv_ys = [uv.y for uv in uvs]
    if min(uv_xs) < -0.01 or max(uv_xs) > 1.01 or min(uv_ys) < -0.01 or max(uv_ys) > 1.01:
        msg += ' — overhangs UV sheet'
    if seam_jumps > 0:
        msg += ' — warning: %d UV seam jumps (shape may look wrong across islands)' % seam_jumps
    return True, msg

def bake_path_to_image(obj, image, curve_obj, uv_name, width, resolution, width_samples,
                       color=(1, 1, 1, 1), falloff=1.0, clear=True,
                       path_texture=None, tile_u=1.0, rotation_deg=0.0,
                       project_distance=None, fill_gaps=True, mode='RIBBON',
                       feather_px=2.0, shrinkwrap_method=None):
    '''
    Stamp a ribbon or filled closed shape from a real 3D curve onto a UV image.
    Returns (ok, message).
    '''
    if mode == 'SHAPE':
        return bake_shape_to_image(
            obj, image, curve_obj, uv_name, resolution,
            color=color, falloff=falloff, clear=clear,
            path_texture=path_texture, tile_u=tile_u, rotation_deg=rotation_deg,
            feather_px=feather_px, project_distance=project_distance,
            shrinkwrap_method=shrinkwrap_method
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
    polyline = _curve_to_polyline(
        curve_obj, resolution=max(resolution, 8), shrinkwrap_method=method
    )
    if len(polyline) < 2:
        return False, 'Curve has too few points to bake'
    width_samples = max(width_samples, 2)
    cyclic = _curve_is_cyclic(curve_obj)
    # Evaluated/shrinkwrap sampling can leave a duplicate close point
    if cyclic and len(polyline) >= 3 and (polyline[0] - polyline[-1]).length < 1e-4:
        polyline = polyline[:-1]
    if len(polyline) < 2:
        return False, 'Curve has too few points to bake'

    # Max search distance: generous default based on object size + path width
    if project_distance is None:
        dim = max(obj.dimensions) if obj.dimensions.length > 0 else 1.0
        project_distance = max(dim * 0.5, width * 4.0, 0.1)

    # Read destination pixels
    if is_bl_newer_than(2, 83):
        pxs = numpy.empty(shape=img_w * img_h * 4, dtype=numpy.float32)
        image.pixels.foreach_get(pxs)
    else:
        pxs = numpy.array(image.pixels[:], dtype=numpy.float32)
    pxs.shape = (img_h, img_w, 4)

    if clear:
        pxs[:] = 0.0

    # Optional path texture
    tex_pxs = None
    tw = th = 0
    if path_texture and path_texture.size[0] > 0 and path_texture.size[1] > 0:
        tw, th = path_texture.size[0], path_texture.size[1]
        if is_bl_newer_than(2, 83):
            tex_pxs = numpy.empty(shape=tw * th * 4, dtype=numpy.float32)
            path_texture.pixels.foreach_get(tex_pxs)
        else:
            tex_pxs = numpy.array(path_texture.pixels[:], dtype=numpy.float32)
        tex_pxs.shape = (th, tw, 4)

    half_w = width * 0.5
    n_len = len(polyline)
    # UV seam jump threshold in pixels — don't bridge / fill across UV islands
    seam_px = max(img_w, img_h) * 0.08

    # Collect per-width-lane samples: list of (uv, u_len, v_across, strength) or None
    lanes = [[] for _ in range(width_samples)]
    prev_side = None
    prev_tangent = None

    for i in range(n_len):
        p = polyline[i]
        if cyclic and n_len >= 3:
            prev_p = polyline[(i - 1) % n_len]
            next_p = polyline[(i + 1) % n_len]
            tangent = next_p - prev_p
        elif i == 0:
            tangent = (polyline[1] - polyline[0])
        elif i == n_len - 1:
            tangent = (polyline[-1] - polyline[-2])
        else:
            tangent = (polyline[i + 1] - polyline[i - 1])
        if tangent.length < 1e-10:
            for lane in lanes:
                lane.append(None)
            continue
        tangent.normalize()

        center = _nearest_uv(bvh, tris, p, max_dist=project_distance)
        if not center:
            for lane in lanes:
                lane.append(None)
            continue
        c_loc, c_normal, _, _ = center

        raw_side = c_normal.cross(tangent)
        if raw_side.length < 1e-8:
            raw_side = tangent.cross(Vector((0, 0, 1)))
            if raw_side.length < 1e-8:
                raw_side = tangent.cross(Vector((0, 1, 0)))
        if raw_side.length < 1e-8:
            for lane in lanes:
                lane.append(None)
            continue
        raw_side.normalize()
        side = _parallel_transport_side(prev_side, prev_tangent, tangent)
        if side is None:
            side = raw_side
        prev_side = side
        prev_tangent = tangent.copy()

        u_len = i / float(max(n_len - 1, 1))
        # Lift along normal so creases on low-poly meshes still find a surface hit
        lift = max(half_w * 0.35, project_distance * 0.01)

        for j in range(width_samples):
            if width_samples == 1:
                v_across = 0.0
            else:
                v_across = (j / float(width_samples - 1)) * 2.0 - 1.0

            offset = side * (v_across * half_w)
            sample_pos = c_loc + offset + c_normal * lift
            hit = _nearest_uv(bvh, tris, sample_pos, max_dist=project_distance)
            if not hit:
                hit = _nearest_uv(bvh, tris, c_loc + offset, max_dist=project_distance)
            if not hit:
                lanes[j].append(None)
                continue

            _, _, uv, _ = hit
            strength = _soft_falloff(abs(v_across), power=falloff)
            lanes[j].append((uv.copy(), u_len, v_across, strength))

    # Continuous ribbon raster: coverage-max (no alpha-over sectioning) + UV quads
    cov = numpy.zeros((img_h, img_w), dtype=numpy.float32)
    # Optional texture accumulation (RGB weighted by coverage contribution)
    use_tex = tex_pxs is not None
    if use_tex:
        acc_rgb = numpy.zeros((img_h, img_w, 3), dtype=numpy.float32)
        acc_w = numpy.zeros((img_h, img_w), dtype=numpy.float32)

    def _rgba_at(u_len, v_across):
        if use_tex:
            rgba = _sample_path_texture(
                tex_pxs, tw, th, u_len, v_across,
                tile_u=tile_u, rotation_deg=rotation_deg
            )
            return (
                float(rgba[0]) * color[0],
                float(rgba[1]) * color[1],
                float(rgba[2]) * color[2],
                float(rgba[3]) * color[3],
            )
        return color

    def _stamp_cov(uv, strength, radius_px=1.35):
        _splat_coverage_max(cov, img_w, img_h, uv.x, uv.y, strength, radius_px=radius_px)

    def _stamp_tex(uv, u_len, v_across, strength, radius_px=1.35):
        '''Max coverage + weighted RGB for textured ribbons.'''
        if strength <= 1e-6:
            return
        if uv.x < 0.0 or uv.x > 1.0 or uv.y < 0.0 or uv.y > 1.0:
            return
        rgba = _rgba_at(u_len, v_across)
        x = uv.x * (img_w - 1)
        y = uv.y * (img_h - 1)
        radius = max(0.75, float(radius_px))
        r_ceil = int(math.ceil(radius)) + 1
        x0 = max(0, int(math.floor(x)) - r_ceil)
        x1 = min(img_w - 1, int(math.ceil(x)) + r_ceil)
        y0 = max(0, int(math.floor(y)) - r_ceil)
        y1 = min(img_h - 1, int(math.ceil(y)) + r_ceil)
        inv_r = 1.0 / radius
        cr, cg, cb, ca = rgba
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                dx = px - x
                dy = py - y
                d = math.sqrt(dx * dx + dy * dy) * inv_r
                if d > 1.0:
                    continue
                w = (1.0 - d)
                w = w * w * strength * ca
                if w <= 1e-6:
                    continue
                if w > cov[py, px]:
                    cov[py, px] = w
                acc_rgb[py, px, 0] += cr * w
                acc_rgb[py, px, 1] += cg * w
                acc_rgb[py, px, 2] += cb * w
                acc_w[py, px] += w

    filled = 0
    # 1) Fill continuous UV quads between consecutive cross-sections
    n_frames = n_len
    for i in range(n_frames):
        i1 = i + 1
        if i1 >= n_frames:
            if cyclic and n_frames >= 3:
                i1 = 0
            else:
                break
        for j in range(width_samples - 1):
            a = lanes[j][i]
            b = lanes[j + 1][i]
            c = lanes[j + 1][i1]
            d = lanes[j][i1]
            if a is None or b is None or c is None or d is None:
                continue
            uv_a, _, _, s_a = a
            uv_b, _, _, s_b = b
            uv_c, _, _, s_c = c
            uv_d, _, _, s_d = d
            # Skip UV island hops
            if (
                _uv_pixel_dist(uv_a, uv_b, img_w, img_h) > seam_px
                or _uv_pixel_dist(uv_b, uv_c, img_w, img_h) > seam_px
                or _uv_pixel_dist(uv_c, uv_d, img_w, img_h) > seam_px
                or _uv_pixel_dist(uv_d, uv_a, img_w, img_h) > seam_px
                or _uv_pixel_dist(uv_a, uv_d, img_w, img_h) > seam_px
                or _uv_pixel_dist(uv_b, uv_c, img_w, img_h) > seam_px
            ):
                continue
            filled += _fill_uv_quad_coverage(
                cov, img_w, img_h,
                uv_a, s_a, uv_b, s_b, uv_c, s_c, uv_d, s_d,
            )
            if use_tex:
                # Seed texture weights at corners (quad fill is coverage-only)
                _stamp_tex(uv_a, a[1], a[2], s_a, radius_px=1.0)
                _stamp_tex(uv_b, b[1], b[2], s_b, radius_px=1.0)
                _stamp_tex(uv_c, c[1], c[2], s_c, radius_px=1.0)
                _stamp_tex(uv_d, d[1], d[2], s_d, radius_px=1.0)

    # 2) Dense coverage stamps along each lane (antialias / fill tiny gaps)
    for lane in lanes:
        for i, sample in enumerate(lane):
            if sample is None:
                continue
            uv, u_len, v_across, strength = sample
            if use_tex:
                _stamp_tex(uv, u_len, v_across, strength, radius_px=1.25)
            else:
                _stamp_cov(uv, strength, radius_px=1.25)
            filled += 1
            if not fill_gaps:
                continue
            # Bridge to next valid sample
            nxt = None
            if i + 1 < len(lane):
                nxt = lane[i + 1]
            elif cyclic and len(lane) >= 3:
                nxt = lane[0]
            if nxt is None:
                continue
            uv2, u_len2, v_across2, strength2 = nxt
            dist_px = _uv_pixel_dist(uv, uv2, img_w, img_h)
            if dist_px < 0.35 or dist_px > seam_px:
                continue
            steps = max(1, int(math.ceil(dist_px / 0.4)))
            for s in range(1, steps):
                t = s / float(steps)
                uv_i = uv.lerp(uv2, t)
                u_i = u_len * (1.0 - t) + u_len2 * t
                v_i = v_across * (1.0 - t) + v_across2 * t
                str_i = strength * (1.0 - t) + strength2 * t
                if use_tex:
                    _stamp_tex(uv_i, u_i, v_i, str_i, radius_px=1.1)
                else:
                    _stamp_cov(uv_i, str_i, radius_px=1.1)
                filled += 1

    if filled == 0 or float(numpy.max(cov)) <= 1e-6:
        return False, 'No path samples projected onto the mesh (move the curve closer to the surface)'

    # Composite coverage into image — uniform opacity, no stamp banding
    cr, cg, cb, ca = [float(c) for c in color]
    src_a = numpy.clip(cov * ca, 0.0, 1.0).astype(numpy.float32)
    m = src_a > 1e-6

    if use_tex:
        safe = numpy.maximum(acc_w, 1e-8)[:, :, None]
        rgb = (acc_rgb / safe).astype(numpy.float32)
        has = acc_w > 1e-6
        out_r = numpy.where(has, rgb[:, :, 0], cr).astype(numpy.float32)
        out_g = numpy.where(has, rgb[:, :, 1], cg).astype(numpy.float32)
        out_b = numpy.where(has, rgb[:, :, 2], cb).astype(numpy.float32)
    else:
        out_r = numpy.full((img_h, img_w), cr, dtype=numpy.float32)
        out_g = numpy.full((img_h, img_w), cg, dtype=numpy.float32)
        out_b = numpy.full((img_h, img_w), cb, dtype=numpy.float32)

    if clear:
        pxs[:, :, 0] = numpy.where(m, out_r, 0.0)
        pxs[:, :, 1] = numpy.where(m, out_g, 0.0)
        pxs[:, :, 2] = numpy.where(m, out_b, 0.0)
        pxs[:, :, 3] = numpy.where(m, src_a, 0.0)
    else:
        dst_a = pxs[:, :, 3]
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_a = numpy.maximum(out_a, 1e-8)
        for ch, src in enumerate((out_r, out_g, out_b)):
            pxs[:, :, ch] = numpy.where(
                m,
                (src * src_a + pxs[:, :, ch] * dst_a * (1.0 - src_a)) / safe_a,
                pxs[:, :, ch],
            )
        pxs[:, :, 3] = numpy.where(m, numpy.minimum(out_a, 1.0), dst_a)

    flat = pxs.ravel()
    if is_bl_newer_than(2, 83):
        image.pixels.foreach_set(flat)
    else:
        image.pixels = flat.tolist()
    image.update()

    return True, 'Baked path ribbon (%d samples, coverage fill)' % filled

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
            ('PROJECT', 'Project', 'Shrinkwrap Project (+/- directions, no axis)'),
        ),
        default = 'NEAREST_SURFACEPOINT',
        update = update_path_shrinkwrap_method
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
        description = 'Number of samples along the curve length',
        default = 512, min = 8, max = 4096
    )

    path_width_samples : IntProperty(
        name = 'Width Samples',
        description = 'Number of samples across the ribbon width',
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
        ensure_path_shrinkwrap(curve_obj, obj, method=layer.path_shrinkwrap_method)
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
            shrinkwrap_method=_layer_shrinkwrap_method(layer)
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
        ok, msg = bake_layer_path(obj, layer)

        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        for area in context.screen.areas:
            area.tag_redraw()

        self.report({'INFO'}, msg + ' ({:0.0f} ms)'.format((time.time() - T) * 1000))
        return {'FINISHED'}

def bake_layer_path(obj, layer):
    '''Bake one path/shape layer. Returns (ok, message).'''
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
    if getattr(layer, 'path_enable_shrinkwrap', True):
        ensure_path_shrinkwrap(curve_obj, obj, method=method)

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
    )
    if ok:
        image.update()
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
            ok, msg = bake_layer_path(obj, layer)
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
            ok, bake_msg = bake_layer_path(obj, layer)
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
            shrinkwrap_method=_layer_shrinkwrap_method(layer)
        )
        layer.enable_path_bake = True
        set_path_curve_object(layer, curve_obj)
        if layer.path_enable_shrinkwrap:
            ensure_path_shrinkwrap(curve_obj, obj, method=_layer_shrinkwrap_method(layer))
        if is_shape:
            ensure_curve_cyclic(curve_obj, True)

        set_object_select(curve_obj, True)
        set_active_object(obj)

        self.report({'INFO'}, "Created %s curve '%s'" % ('shape' if is_shape else 'path', curve_obj.name))
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
        col.prop(layer, 'path_width_samples')
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

def unregister():
    bpy.utils.unregister_class(YNewPathLayer)
    bpy.utils.unregister_class(YBakePathToLayer)
    bpy.utils.unregister_class(YBakeAllPaths)
    bpy.utils.unregister_class(YResizePathBakeImage)
    bpy.utils.unregister_class(YSelectPathCurve)
    bpy.utils.unregister_class(YCreatePathCurveForLayer)
