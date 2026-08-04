import bpy, re
from . import lib
from bpy.props import *
from bpy.app.handlers import persistent
from mathutils import Matrix
from .common import *

def get_decal_object(entity):
    m1 = re.match(r'^yp\.layers\[(\d+)\]$', entity.path_from_id())
    m2 = re.match(r'^yp\.layers\[(\d+)\]\.masks\[(\d+)\]$', entity.path_from_id())

    if m1: tree = get_tree(entity)
    elif m2: tree = get_mask_tree(entity)
    else: return None

    decal_obj = None
    texcoord = tree.nodes.get(entity.texcoord)
    if texcoord and hasattr(texcoord, 'object'): decal_obj = texcoord.object

    return decal_obj

def get_decal_shrinkwrap_constraint(decal_obj):
    cs = [c for c in decal_obj.constraints if c.type == 'SHRINKWRAP']
    if len(cs) > 0: return cs[0]
    return None

def any_decal_inside_layer(layer):
    if layer.texcoord_type == 'Decal':
        return True

    for mask in layer.masks:
        if mask.texcoord_type == 'Decal':
            return True

    return False

def remove_decal_object(tree, entity):
    if not tree: return
    # NOTE: This will remove the texcoord object even if the entity is not using decal
    #if entity.texcoord_type == 'Decal':
    texcoord = tree.nodes.get(entity.texcoord)
    if texcoord and hasattr(texcoord, 'object') and texcoord.object:
        decal_obj = texcoord.object
        texcoord.object = None
        remove_entity_decal_mirrors(entity, tree)
        # Scene collection keeps a user; original threshold was <= 2
        if decal_obj.type == 'EMPTY' and decal_obj.users <= 2:
            remove_datablock(bpy.data.objects, decal_obj)
    else:
        remove_entity_decal_mirrors(entity, tree)

# ---------------------------------------------------------------------------
# Object-space decal mirrors (across the parent / main mesh local axes)
# ---------------------------------------------------------------------------

_DECAL_MIRROR_SCALES = {
    'x': (-1.0, 1.0, 1.0),
    'y': (1.0, -1.0, 1.0),
    'z': (1.0, 1.0, -1.0),
    'xy': (-1.0, -1.0, 1.0),
    'xz': (-1.0, 1.0, -1.0),
    'yz': (1.0, -1.0, -1.0),
    'xyz': (-1.0, -1.0, -1.0),
}

def _decal_mirror_slots(entity):
    mx = bool(getattr(entity, 'decal_mirror_x', False))
    my = bool(getattr(entity, 'decal_mirror_y', False))
    mz = bool(getattr(entity, 'decal_mirror_z', False))
    slots = []
    if mx: slots.append('x')
    if my: slots.append('y')
    if mz: slots.append('z')
    if mx and my: slots.append('xy')
    if mx and mz: slots.append('xz')
    if my and mz: slots.append('yz')
    if mx and my and mz: slots.append('xyz')
    return slots

def _mirror_prop(slot, kind):
    return 'decal_mirror_%s_%s' % (kind, slot)

def _mirror_decal_process_scale(primary_proc, mirror_proc, slot):
    '''Copy aspect Scale from primary (do not negate — UV flip handles mirroring).'''
    if not primary_proc or not mirror_proc:
        return
    if 'Scale' not in primary_proc.inputs or 'Scale' not in mirror_proc.inputs:
        return
    src = primary_proc.inputs['Scale'].default_value
    mirror_proc.inputs['Scale'].default_value = (
        float(src[0]),
        float(src[1]),
        float(src[2]) if len(src) > 2 else 1.0,
    )

def _mirror_slot_needs_uv_flip(slot):
    '''
    mirrored_world_matrix rebuilds a right-handed frame from reflected +Y/+Z.
    For an odd number of parent-axis mirrors that undoes texture handedness
    (extra flip on empty local X / U). Compensate with u' = 1 - u.
    '''
    scale = _DECAL_MIRROR_SCALES.get(slot, (1.0, 1.0, 1.0))
    return (scale[0] * scale[1] * scale[2]) < 0.0

def _scale_matrix(scale_xyz):
    return Matrix.Diagonal((scale_xyz[0], scale_xyz[1], scale_xyz[2], 1.0))

def get_decal_mirror_parent(empty):
    '''Main object used as the symmetry reference (usually the mesh the empty is parented to).'''
    if empty and empty.parent:
        return empty.parent
    return None

def mirrored_world_matrix(primary, slot, parent=None):
    '''
    World matrix of primary mirrored across the parent mesh's local axes for `slot`.

    Rebuilds a right-handed basis from the mirrored projection (+Z) and up (+Y)
    axes so the decal stays surface-aligned and upright (not rolled upside-down).
    Falls back to world axes if the empty has no parent.
    '''
    from mathutils import Vector

    scale = _DECAL_MIRROR_SCALES.get(slot)
    if not scale or not primary:
        return primary.matrix_world.copy() if primary else Matrix.Identity(4)

    S3 = Matrix.Diagonal((scale[0], scale[1], scale[2]))

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        Mw = primary.evaluated_get(depsgraph).matrix_world.copy()
    except Exception:
        Mw = primary.matrix_world.copy()

    parent = parent or get_decal_mirror_parent(primary)
    if parent:
        try:
            Mp = parent.matrix_world.copy()
            Ml = Mp.inverted() @ Mw
        except Exception:
            Mp = Matrix.Identity(4)
            Ml = Mw.copy()
    else:
        Mp = Matrix.Identity(4)
        Ml = Mw.copy()

    loc = Ml.to_translation()
    rot = Ml.to_3x3()
    scl = Ml.to_scale()

    # Reflect position across enabled parent-local planes
    loc_m = S3 @ loc

    # Reflect the empty's local axes in parent space, then rebuild a proper
    # right-handed frame (S@R@S / negative scale was rolling decals upside-down).
    z_m = S3 @ (rot @ Vector((0.0, 0.0, 1.0)))
    y_m = S3 @ (rot @ Vector((0.0, 1.0, 0.0)))
    if z_m.length < 1e-8:
        z_m = Vector((0.0, 0.0, 1.0))
    else:
        z_m.normalize()
    # Keep up as orthogonal to projection as possible
    y_m = y_m - z_m * y_m.dot(z_m)
    if y_m.length < 1e-8:
        # Degenerate: pick any up
        helper = Vector((0.0, 1.0, 0.0)) if abs(z_m.y) < 0.9 else Vector((1.0, 0.0, 0.0))
        y_m = helper - z_m * helper.dot(z_m)
    y_m.normalize()
    x_m = y_m.cross(z_m)
    if x_m.length < 1e-8:
        x_m = Vector((1.0, 0.0, 0.0))
    else:
        x_m.normalize()
    # Re-orthogonalize up for numerical safety
    y_m = z_m.cross(x_m)
    y_m.normalize()

    rot_m = Matrix((x_m, y_m, z_m)).transposed().to_4x4()
    loc_m_mat = Matrix.Translation(loc_m)
    scl_m = Matrix.Diagonal((abs(scl.x), abs(scl.y), abs(scl.z), 1.0))
    Ml_m = loc_m_mat @ rot_m @ scl_m
    return Mp @ Ml_m

def _create_mirror_empty(primary, slot):
    scene = bpy.context.scene
    parent = get_decal_mirror_parent(primary)
    name = get_unique_name('%s.mirror.%s' % (primary.name, slot.upper()), bpy.data.objects)
    empty = bpy.data.objects.new(name, None)
    if is_bl_newer_than(2, 80):
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = getattr(primary, 'empty_display_size', 1.0) * 0.85
    else:
        empty.empty_draw_type = 'SINGLE_ARROW'

    custom_collection = None
    if parent and is_bl_newer_than(2, 80) and len(parent.users_collection) > 0:
        custom_collection = parent.users_collection[0]
    elif is_bl_newer_than(2, 80) and len(primary.users_collection) > 0:
        custom_collection = primary.users_collection[0]
    link_object(scene, empty, custom_collection)

    if parent:
        empty.parent = parent
        empty.matrix_parent_inverse = primary.matrix_parent_inverse.copy()

    empty.yp_decal.is_mirror = True
    empty.yp_decal.mirror_source_name = primary.name
    empty.yp_decal.mirror_slot = slot
    # Keep mirrors out of the way while still evaluating
    if hasattr(empty, 'hide_select'):
        empty.hide_select = True

    empty.matrix_world = mirrored_world_matrix(primary, slot, parent)
    return empty

def sync_decal_mirror_empty(mirror_obj, primary=None):
    if not mirror_obj or not getattr(mirror_obj, 'yp_decal', None):
        return
    if not mirror_obj.yp_decal.is_mirror:
        return
    slot = mirror_obj.yp_decal.mirror_slot or 'x'
    if primary is None:
        primary = bpy.data.objects.get(mirror_obj.yp_decal.mirror_source_name)
    if not primary:
        return
    target = mirrored_world_matrix(primary, slot)
    # Avoid feedback loops when nothing changed
    diff = 0.0
    for i in range(4):
        for j in range(4):
            d = target[i][j] - mirror_obj.matrix_world[i][j]
            diff += d * d
    if diff > 1e-12:
        mirror_obj.matrix_world = target

def sync_mirrors_for_primary(primary):
    if not primary:
        return
    for obj in bpy.data.objects:
        yd = getattr(obj, 'yp_decal', None)
        if not yd or not yd.is_mirror:
            continue
        if yd.mirror_source_name == primary.name:
            sync_decal_mirror_empty(obj, primary)

def remove_entity_decal_mirrors(entity, tree=None):
    '''Remove mirror empties + their nodes for an entity.'''
    if tree is None:
        m1 = re.match(r'^yp\.layers\[(\d+)\]$', entity.path_from_id())
        m2 = re.match(r'^yp\.layers\[(\d+)\]\.masks\[(\d+)\]$', entity.path_from_id())
        if m1:
            tree = get_tree(entity)
        elif m2:
            tree = get_mask_tree(entity)

    for slot in _DECAL_MIRROR_SCALES:
        obj_prop = _mirror_prop(slot, 'obj')
        name = getattr(entity, obj_prop, '')
        if name:
            obj = bpy.data.objects.get(name)
            if obj:
                # Clear texcoord reference first
                if tree:
                    tex = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'texcoord'), ''))
                    if tex and getattr(tex, 'object', None) == obj:
                        tex.object = None
                if obj.users <= 2:
                    remove_datablock(bpy.data.objects, obj)
            setattr(entity, obj_prop, '')
        if tree:
            for kind in ('texcoord', 'process', 'gt', 'mix', 'max', 'flipmul', 'flipadd'):
                remove_node(tree, entity, _mirror_prop(slot, kind))

def ensure_entity_decal_mirrors(entity, tree, primary):
    '''Create/update mirror empties and nodes for enabled axes; remove unused ones.'''
    if not primary:
        remove_entity_decal_mirrors(entity, tree)
        return

    active = set(_decal_mirror_slots(entity))
    primary_proc = tree.nodes.get(entity.decal_process)

    for slot, scale in _DECAL_MIRROR_SCALES.items():
        obj_prop = _mirror_prop(slot, 'obj')
        tex_prop = _mirror_prop(slot, 'texcoord')
        proc_prop = _mirror_prop(slot, 'process')
        gt_prop = _mirror_prop(slot, 'gt')
        mix_prop = _mirror_prop(slot, 'mix')
        max_prop = _mirror_prop(slot, 'max')

        if slot not in active:
            # Tear down
            name = getattr(entity, obj_prop, '')
            if name:
                obj = bpy.data.objects.get(name)
                tex = tree.nodes.get(getattr(entity, tex_prop, ''))
                if tex and getattr(tex, 'object', None) == obj:
                    tex.object = None
                if obj and obj.users <= 2:
                    remove_datablock(bpy.data.objects, obj)
                setattr(entity, obj_prop, '')
            for kind in ('texcoord', 'process', 'gt', 'mix', 'max', 'flipmul', 'flipadd'):
                remove_node(tree, entity, _mirror_prop(slot, kind))
            continue

        # Empty
        mirror = bpy.data.objects.get(getattr(entity, obj_prop, ''))
        if not mirror:
            mirror = _create_mirror_empty(primary, slot)
            setattr(entity, obj_prop, mirror.name)
        else:
            mirror.yp_decal.is_mirror = True
            mirror.yp_decal.mirror_source_name = primary.name
            mirror.yp_decal.mirror_slot = slot
            sync_decal_mirror_empty(mirror, primary)

        # TexCoord bound to mirror empty
        tex = check_new_node(tree, entity, tex_prop, 'ShaderNodeTexCoord', 'TexCoord Mirror ' + slot.upper())
        tex.object = mirror

        # Decal Process (same aspect as primary; texture flip is post-UV below)
        proc = check_new_node(tree, entity, proc_prop, 'ShaderNodeGroup', 'Decal Process ' + slot.upper())
        if not proc.node_tree:
            proc.node_tree = get_node_tree_lib(lib.DECAL_PROCESS)
        _mirror_decal_process_scale(primary_proc, proc, slot)

        check_new_node(tree, entity, gt_prop, 'ShaderNodeMath', 'Decal Mirror GT ' + slot.upper())
        mix_data = 'VECTOR' if is_bl_newer_than(3, 4) else 'RGBA'
        check_new_mix_node(tree, entity, mix_prop, 'Decal Mirror Mix ' + slot.upper(), data_type=mix_data)
        check_new_node(tree, entity, max_prop, 'ShaderNodeMath', 'Decal Mirror Max ' + slot.upper())

        # Post-UV flip nodes (u' = 1-u) for odd mirror parity — see wire_decal_projection
        flip_mul = check_new_node(
            tree, entity, _mirror_prop(slot, 'flipmul'),
            'ShaderNodeVectorMath', 'Decal Mirror UV Flip Mul ' + slot.upper()
        )
        if flip_mul.operation != 'MULTIPLY':
            flip_mul.operation = 'MULTIPLY'
        flip_add = check_new_node(
            tree, entity, _mirror_prop(slot, 'flipadd'),
            'ShaderNodeVectorMath', 'Decal Mirror UV Flip Add ' + slot.upper()
        )
        if flip_add.operation != 'ADD':
            flip_add.operation = 'ADD'

def update_decal_mirror(self, context):
    yp = self.id_data.yp
    if getattr(yp, 'halt_update', False):
        return

    m1 = re.match(r'^yp\.layers\[(\d+)\]$', self.path_from_id())
    m2 = re.match(r'^yp\.layers\[(\d+)\]\.masks\[(\d+)\]$', self.path_from_id())
    if not m1 and not m2:
        return

    layer = self if m1 else yp.layers[int(m2.group(1))]
    entity = self
    tree = get_tree(layer)
    check_entity_decal_nodes(entity, tree)

    from .node_connections import reconnect_layer_nodes
    from .node_arrangements import rearrange_layer_nodes
    reconnect_layer_nodes(layer)
    rearrange_layer_nodes(layer)

def wire_decal_projection(tree, entity, object_vector, decal_process, distance_socket=None):
    '''
    Wire primary Decal Process, then any object-mirrored copies.
    Returns (uv_output, alpha_output). UV follows the strongest clip alpha so one image sample covers all copies.
    '''
    from .node_connections import create_link

    create_link(tree, object_vector, decal_process.inputs[0])
    if distance_socket is not None and len(decal_process.inputs) > 1:
        create_link(tree, distance_socket, decal_process.inputs[1])

    cur_uv = decal_process.outputs[0]
    cur_a = decal_process.outputs[1]

    for slot in _decal_mirror_slots(entity):
        tex = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'texcoord'), ''))
        proc = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'process'), ''))
        gt = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'gt'), ''))
        mix = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'mix'), ''))
        mx = tree.nodes.get(getattr(entity, _mirror_prop(slot, 'max'), ''))
        if not tex or not proc or not gt or not mix or not mx:
            continue

        # Keep aspect in sync
        _mirror_decal_process_scale(decal_process, proc, slot)

        create_link(tree, tex.outputs['Object'], proc.inputs[0])
        if distance_socket is not None and len(proc.inputs) > 1:
            create_link(tree, distance_socket, proc.inputs[1])

        # True mirror image: flip U after Decal Process when RH rebuild undid handedness
        mirror_uv = proc.outputs[0]
        if _mirror_slot_needs_uv_flip(slot):
            flip_mul = check_new_node(
                tree, entity, _mirror_prop(slot, 'flipmul'),
                'ShaderNodeVectorMath', 'Decal Mirror UV Flip Mul ' + slot.upper()
            )
            flip_add = check_new_node(
                tree, entity, _mirror_prop(slot, 'flipadd'),
                'ShaderNodeVectorMath', 'Decal Mirror UV Flip Add ' + slot.upper()
            )
            if flip_mul.operation != 'MULTIPLY':
                flip_mul.operation = 'MULTIPLY'
            if flip_add.operation != 'ADD':
                flip_add.operation = 'ADD'
            # u' = 1 - u  (v unchanged)
            flip_mul.inputs[1].default_value = (-1.0, 1.0, 1.0)
            flip_add.inputs[1].default_value = (1.0, 0.0, 0.0)
            create_link(tree, proc.outputs[0], flip_mul.inputs[0])
            create_link(tree, flip_mul.outputs[0], flip_add.inputs[0])
            mirror_uv = flip_add.outputs[0]

        if gt.operation != 'GREATER_THAN':
            gt.operation = 'GREATER_THAN'
        create_link(tree, proc.outputs[1], gt.inputs[0])
        create_link(tree, cur_a, gt.inputs[1])

        a_idx, b_idx, out_idx = get_mix_color_indices(mix)
        create_link(tree, cur_uv, mix.inputs[a_idx])
        create_link(tree, mirror_uv, mix.inputs[b_idx])
        create_link(tree, gt.outputs[0], mix.inputs[0])

        if mx.operation != 'MAXIMUM':
            mx.operation = 'MAXIMUM'
        create_link(tree, cur_a, mx.inputs[0])
        create_link(tree, proc.outputs[1], mx.inputs[1])

        cur_uv = mix.outputs[out_idx]
        cur_a = mx.outputs[0]

    return cur_uv, cur_a

def update_enable_decal_object_constraint(self, context):
    obj = context.object
    decal_obj = self.id_data
    decal_const = get_decal_shrinkwrap_constraint(decal_obj)

    if self.enable_shrinkwrap:
        if not decal_const and obj:
            c = decal_obj.constraints.new('SHRINKWRAP')
            c.target = obj
            if is_bl_newer_than(2, 80):
                c.use_track_normal = True
                c.track_axis = 'TRACK_Z'
    else:
        if decal_const:
            decal_obj.constraints.remove(decal_const)

def create_decal_empty():
    obj = bpy.context.object
    scene = bpy.context.scene
    empty_name = get_unique_name('Decal', bpy.data.objects)
    empty = bpy.data.objects.new(empty_name, None)
    if is_bl_newer_than(2, 80):
        empty.empty_display_type = 'SINGLE_ARROW'
    else: empty.empty_draw_type = 'SINGLE_ARROW'
    custom_collection = obj.users_collection[0] if is_bl_newer_than(2, 80) and len(obj.users_collection) > 0 else None
    link_object(scene, empty, custom_collection)
    if is_bl_newer_than(2, 80):
        empty.location = scene.cursor.location.copy()
        empty.rotation_euler = scene.cursor.rotation_euler.copy()
    else: 
        empty.location = scene.cursor_location.copy()

    # Parent empty to active object
    empty.parent = obj
    empty.matrix_parent_inverse = obj.matrix_world.inverted()

    return empty

def check_entity_decal_nodes(entity, tree=None):
    yp = entity.id_data.yp
    m1 = re.match(r'^yp\.layers\[(\d+)\]$', entity.path_from_id())
    m2 = re.match(r'^yp\.layers\[(\d+)\]\.masks\[(\d+)\]$', entity.path_from_id())

    if m1: 
        entity_enabled = get_layer_enabled(entity)
        source = get_layer_source(entity)
        if not tree: tree = get_tree(entity)
        layer = entity
        mask = None
    elif m2: 
        entity_enabled = get_mask_enabled(entity)
        source = get_mask_source(entity)
        layer = yp.layers[int(m2.group(1))]
        if not tree: tree = get_tree(entity)
        mask = entity
    else: return

    # Get height channel
    height_ch = get_height_channel(layer)

    # Create texcoord node if decal is used
    texcoord = tree.nodes.get(entity.texcoord)
    if entity_enabled and entity.texcoord_type == 'Decal' and is_mapping_possible(entity.type):

        # Set image extension type to clip
        image = None
        if entity.type == 'IMAGE' and source:
            image = source.image

        # Create new empty object if there's no texcoord yet
        if not texcoord:
            empty = create_decal_empty()
            texcoord = new_node(tree, entity, 'texcoord', 'ShaderNodeTexCoord', 'TexCoord')
            texcoord.object = empty

        decal_process = tree.nodes.get(entity.decal_process)
        if not decal_process:
            decal_process = new_node(tree, entity, 'decal_process', 'ShaderNodeGroup', 'Decal Process')
            decal_process.node_tree = get_node_tree_lib(lib.DECAL_PROCESS)

            # Set image extension only after decal process node is initialized
            if image and source:
                entity.original_image_extension = source.extension
                source.extension = 'CLIP'

        # Set decal aspect ratio
        if image and image.size[0] > 0 and image.size[1] > 0:
            if image.size[0] > image.size[1]:
                decal_process.inputs['Scale'].default_value = (image.size[1] / image.size[0], 1.0, 1.0)
            else: decal_process.inputs['Scale'].default_value = (1.0, image.size[0] / image.size[1], 1.0)

        # Object-space mirrors across the parent mesh local axes
        primary_empty = texcoord.object if texcoord else None
        ensure_entity_decal_mirrors(entity, tree, primary_empty)

        # Create decal alpha nodes
        if mask:

            # Check if height channel is enabled
            height_root_ch = get_root_height_channel(yp)
            height_ch_enabled = get_channel_enabled(height_ch) if height_ch else False

            decal_alpha = check_new_node(tree, mask, 'decal_alpha', 'ShaderNodeMath', 'Decal Alpha')
            if decal_alpha.operation != 'MULTIPLY':
                decal_alpha.operation = 'MULTIPLY'

            if height_ch and height_ch_enabled and height_root_ch.enable_smooth_bump:
                for letter in nsew_letters:
                    decal_alpha = check_new_node(tree, mask, 'decal_alpha_' + letter, 'ShaderNodeMath', 'Decal Alpha ' + letter.upper())
                    if decal_alpha.operation != 'MULTIPLY':
                        decal_alpha.operation = 'MULTIPLY'
            else:
                for letter in nsew_letters:
                    remove_node(tree, mask, 'decal_alpha_' + letter)

        else:

            for i, ch in enumerate(layer.channels):
                root_ch = yp.channels[i]
                ch_enabled = get_channel_enabled(ch)
                if ch_enabled:
                    decal_alpha = check_new_node(tree, ch, 'decal_alpha', 'ShaderNodeMath', 'Decal Alpha')
                    if decal_alpha.operation != 'MULTIPLY':
                        decal_alpha.operation = 'MULTIPLY'
                else:
                    remove_node(tree, ch, 'decal_alpha')

                if root_ch.type == 'NORMAL':
                    if ch_enabled and root_ch.enable_smooth_bump:
                        for letter in nsew_letters:
                            decal_alpha = check_new_node(tree, ch, 'decal_alpha_' + letter, 'ShaderNodeMath', 'Decal Alpha ' + letter.upper())
                            if decal_alpha.operation != 'MULTIPLY':
                                decal_alpha.operation = 'MULTIPLY'
                    else:
                        for letter in nsew_letters:
                            remove_node(tree, ch, 'decal_alpha_' + letter)
    else:

        if not texcoord or not hasattr(texcoord, 'object') or not texcoord.object: 
            remove_node(tree, entity, 'texcoord')
        remove_node(tree, entity, 'decal_process')
        remove_entity_decal_mirrors(entity, tree)

        if mask: 
            remove_node(tree, mask, 'decal_alpha')

            if height_ch:
                for letter in nsew_letters:
                    remove_node(tree, mask, 'decal_alpha_' + letter)
        else:
            for i, ch in enumerate(layer.channels):
                root_ch = yp.channels[i]
                remove_node(tree, ch, 'decal_alpha')

                if root_ch.type == 'NORMAL':
                    for letter in nsew_letters:
                        remove_node(tree, ch, 'decal_alpha_' + letter)

        # Recover image extension type
        if entity.type == 'IMAGE' and entity.original_texcoord == 'Decal' and entity.original_image_extension != '':
            source = get_mask_source(mask) if mask else get_layer_source(layer)
            if source:
                source.extension = entity.original_image_extension
                entity.original_image_extension = ''

    # Save original texcoord type
    if entity.original_texcoord != entity.texcoord_type:
        entity.original_texcoord = entity.texcoord_type

class YSelectDecalObject(bpy.types.Operator):
    bl_idname = "wm.y_select_decal_object"
    bl_label = "Select Decal Object"
    bl_description = "Select Decal Object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        group_node = get_active_ypaint_node()
        return group_node and hasattr(context, 'entity')

    def execute(self, context):
        scene = context.scene

        decal_obj = get_decal_object(context.entity)
        if decal_obj:
            try: bpy.ops.object.mode_set(mode='OBJECT')
            except: pass
            bpy.ops.object.select_all(action='DESELECT')
            if decal_obj.name not in get_scene_objects():
                parent = decal_obj.parent
                custom_collection = parent.users_collection[0] if is_bl_newer_than(2, 80) and parent and len(parent.users_collection) > 0 else None
                link_object(scene, decal_obj, custom_collection)
            set_active_object(decal_obj)
            set_object_select(decal_obj, True)
        else: return {'CANCELLED'}

        return {'FINISHED'}

class YSetDecalObjectPositionToCursor(bpy.types.Operator):
    bl_idname = "wm.y_set_decal_object_position_to_sursor"
    bl_label = "Set Decal Position to Cursor"
    bl_description = "Set the position of the decal object to the 3D cursor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        group_node = get_active_ypaint_node()
        return group_node and hasattr(context, 'entity')

    def execute(self, context):
        scene = bpy.context.scene
        entity = context.entity

        m1 = re.match(r'^yp\.layers\[(\d+)\]$', entity.path_from_id())
        m2 = re.match(r'^yp\.layers\[(\d+)\]\.masks\[(\d+)\]$', entity.path_from_id())

        if m1: tree = get_tree(entity)
        elif m2: tree = get_mask_tree(entity)
        else: return {'CANCELLED'}

        texcoord = tree.nodes.get(entity.texcoord)

        if texcoord and hasattr(texcoord, 'object') and texcoord.object:
            # Move decal object to 3D cursor
            if is_bl_newer_than(2, 80):
                texcoord.object.location = scene.cursor.location.copy()
                texcoord.object.rotation_euler = scene.cursor.rotation_euler.copy()
            else: 
                texcoord.object.location = scene.cursor_location.copy()

        else: return {'CANCELLED'}

        return {'FINISHED'}

class BaseDecal():

    decal_distance_value : FloatProperty(
        name = 'Decal Distance',
        description = 'Distance between surface and the decal object',
        min=0.0, max=100.0, default=0.5, precision=3
    )

    original_texcoord : EnumProperty(
        name = 'Original Texture Coordinate Type',
        items = mask_texcoord_type_items,
        default = 'UV'
    )

    original_image_extension : StringProperty(
        name = 'Original Image Extension Type',
        default = ''
    )

    decal_mirror_x : BoolProperty(
        name = 'X',
        description = "Also project a mirrored copy across the main object's local X axis",
        default = False,
        update = update_decal_mirror
    )
    decal_mirror_y : BoolProperty(
        name = 'Y',
        description = "Also project a mirrored copy across the main object's local Y axis",
        default = False,
        update = update_decal_mirror
    )
    decal_mirror_z : BoolProperty(
        name = 'Z',
        description = "Also project a mirrored copy across the main object's local Z axis",
        default = False,
        update = update_decal_mirror
    )

    # Mirror empties + nodes (slots: x y z xy xz yz xyz)
    decal_mirror_obj_x : StringProperty(default='')
    decal_mirror_texcoord_x : StringProperty(default='')
    decal_mirror_process_x : StringProperty(default='')
    decal_mirror_gt_x : StringProperty(default='')
    decal_mirror_mix_x : StringProperty(default='')
    decal_mirror_max_x : StringProperty(default='')
    decal_mirror_flipmul_x : StringProperty(default='')
    decal_mirror_flipadd_x : StringProperty(default='')

    decal_mirror_obj_y : StringProperty(default='')
    decal_mirror_texcoord_y : StringProperty(default='')
    decal_mirror_process_y : StringProperty(default='')
    decal_mirror_gt_y : StringProperty(default='')
    decal_mirror_mix_y : StringProperty(default='')
    decal_mirror_max_y : StringProperty(default='')
    decal_mirror_flipmul_y : StringProperty(default='')
    decal_mirror_flipadd_y : StringProperty(default='')

    decal_mirror_obj_z : StringProperty(default='')
    decal_mirror_texcoord_z : StringProperty(default='')
    decal_mirror_process_z : StringProperty(default='')
    decal_mirror_gt_z : StringProperty(default='')
    decal_mirror_mix_z : StringProperty(default='')
    decal_mirror_max_z : StringProperty(default='')
    decal_mirror_flipmul_z : StringProperty(default='')
    decal_mirror_flipadd_z : StringProperty(default='')

    decal_mirror_obj_xy : StringProperty(default='')
    decal_mirror_texcoord_xy : StringProperty(default='')
    decal_mirror_process_xy : StringProperty(default='')
    decal_mirror_gt_xy : StringProperty(default='')
    decal_mirror_mix_xy : StringProperty(default='')
    decal_mirror_max_xy : StringProperty(default='')
    decal_mirror_flipmul_xy : StringProperty(default='')
    decal_mirror_flipadd_xy : StringProperty(default='')

    decal_mirror_obj_xz : StringProperty(default='')
    decal_mirror_texcoord_xz : StringProperty(default='')
    decal_mirror_process_xz : StringProperty(default='')
    decal_mirror_gt_xz : StringProperty(default='')
    decal_mirror_mix_xz : StringProperty(default='')
    decal_mirror_max_xz : StringProperty(default='')
    decal_mirror_flipmul_xz : StringProperty(default='')
    decal_mirror_flipadd_xz : StringProperty(default='')

    decal_mirror_obj_yz : StringProperty(default='')
    decal_mirror_texcoord_yz : StringProperty(default='')
    decal_mirror_process_yz : StringProperty(default='')
    decal_mirror_gt_yz : StringProperty(default='')
    decal_mirror_mix_yz : StringProperty(default='')
    decal_mirror_max_yz : StringProperty(default='')
    decal_mirror_flipmul_yz : StringProperty(default='')
    decal_mirror_flipadd_yz : StringProperty(default='')

    decal_mirror_obj_xyz : StringProperty(default='')
    decal_mirror_texcoord_xyz : StringProperty(default='')
    decal_mirror_process_xyz : StringProperty(default='')
    decal_mirror_gt_xyz : StringProperty(default='')
    decal_mirror_mix_xyz : StringProperty(default='')
    decal_mirror_max_xyz : StringProperty(default='')
    decal_mirror_flipmul_xyz : StringProperty(default='')
    decal_mirror_flipadd_xyz : StringProperty(default='')

class YPaintDecalObjectProps(bpy.types.PropertyGroup):
    enable_shrinkwrap : BoolProperty(
        name = 'Enable Decal Shrinkwrap Constraint',
        description = 'Enable shrinkwrap constraint, so decal object always follow the target object',
        default = False,
        update = update_enable_decal_object_constraint
    )

    last_operator : StringProperty(default='')
    last_operator_pointer : StringProperty(default='')

    is_mirror : BoolProperty(
        name = 'Is Mirror Empty',
        description = 'Internal: this empty is a mirrored copy of another decal object',
        default = False
    )
    mirror_source_name : StringProperty(
        name = 'Mirror Source',
        description = 'Name of the primary decal empty this mirrors',
        default = ''
    )
    mirror_slot : StringProperty(
        name = 'Mirror Slot',
        description = 'Which axis combination this mirror represents (x/y/z/xy/...)',
        default = ''
    )

def apply_decal_constraint_transforms(op):
    for obj in bpy.context.selected_objects:
        if not obj.yp_decal.enable_shrinkwrap: continue
        if obj.yp_decal.is_mirror: continue

        # Get constraint
        c = get_decal_shrinkwrap_constraint(obj)
        if not c or c.mute: continue

        if obj.yp_decal.last_operator != op.bl_idname or obj.yp_decal.last_operator_pointer != str(op.as_pointer()):
            obj.yp_decal.last_operator = op.bl_idname
            obj.yp_decal.last_operator_pointer = str(op.as_pointer())

            # Apply the constraint after transforming
            mat = obj.matrix_world.copy()
            try: 
                c.mute = True
                obj.matrix_world = mat
                c.mute = False
            except Exception as e: print('EXCEPTIION:', e)

        sync_mirrors_for_primary(obj)

@persistent
def ypaint_decal_constraint_update(scene):
    # NOTE: Only apply constraint transformations when the active object enable the decal contstraint flag
    # This is to improve performance since there's no need to check every selected objects
    obj = bpy.context.object if hasattr(bpy.context, 'object') else None
    if obj and obj.yp_decal.enable_shrinkwrap and bpy.context.active_operator:
        op = bpy.context.active_operator
        # NOTE: Using depsgraph updates is slightly faster than using `startswith`, but only works on Blender 2.80+
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for update in depsgraph.updates:
            if update.is_updated_transform:
                apply_decal_constraint_transforms(op)
                break

    # Keep object-space mirror empties aligned with their primary decal
    _sync_decal_mirrors_from_depsgraph()

@persistent
def ypaint_decal_constraint_update_legacy(scene):
    # NOTE: Only apply constraint transformations when the active object enable the decal contstraint flag
    # This is to improve performance since there's no need to check every selected objects
    obj = bpy.context.object
    if obj and obj.yp_decal.enable_shrinkwrap and bpy.context.active_operator:
        op = bpy.context.active_operator
        if op.bl_idname.startswith('TRANSFORM_OT'):
            apply_decal_constraint_transforms(op)
    _sync_decal_mirrors_from_depsgraph()

_mirror_sync_depth = 0

def _sync_decal_mirrors_from_depsgraph():
    '''Push primary decal transforms into mirror empties when the primary (or its parent) moves.'''
    global _mirror_sync_depth
    if _mirror_sync_depth > 0:
        return
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        return

    touched = set()
    for update in depsgraph.updates:
        if not getattr(update, 'is_updated_transform', False):
            continue
        id_data = update.id
        obj = getattr(id_data, 'original', id_data)
        if not isinstance(obj, bpy.types.Object):
            continue
        yd = getattr(obj, 'yp_decal', None)
        if yd and yd.is_mirror:
            continue
        touched.add(obj)

    if not touched:
        return

    primaries = set()
    for obj in bpy.data.objects:
        yd = getattr(obj, 'yp_decal', None)
        if not yd or not yd.is_mirror:
            continue
        src = bpy.data.objects.get(yd.mirror_source_name)
        if not src:
            continue
        if src in touched or (src.parent and src.parent in touched):
            primaries.add(src)

    if not primaries:
        return

    _mirror_sync_depth += 1
    try:
        for primary in primaries:
            sync_mirrors_for_primary(primary)
    finally:
        _mirror_sync_depth -= 1

def register():
    bpy.utils.register_class(YSelectDecalObject)
    bpy.utils.register_class(YSetDecalObjectPositionToCursor)
    bpy.utils.register_class(YPaintDecalObjectProps)

    # YPaint Props
    bpy.types.Object.yp_decal = PointerProperty(type=YPaintDecalObjectProps)

    # Handlers
    if is_bl_newer_than(2, 80):
        bpy.app.handlers.depsgraph_update_post.append(ypaint_decal_constraint_update)
    else: bpy.app.handlers.scene_update_pre.append(ypaint_decal_constraint_update_legacy)

def unregister():
    bpy.utils.unregister_class(YSelectDecalObject)
    bpy.utils.unregister_class(YSetDecalObjectPositionToCursor)
    bpy.utils.unregister_class(YPaintDecalObjectProps)

    # Handlers
    if is_bl_newer_than(2, 80):
        bpy.app.handlers.depsgraph_update_post.remove(ypaint_decal_constraint_update)
    else: bpy.app.handlers.scene_update_pre.remove(ypaint_decal_constraint_update_legacy)
