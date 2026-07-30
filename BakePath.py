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

def configure_path_shrinkwrap(mod, target_obj):
    '''Apply the standard path shrinkwrap settings.'''
    mod.target = target_obj
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

def ensure_path_shrinkwrap(curve_obj, target_obj=None):
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
    configure_path_shrinkwrap(mod, target_obj)
    return mod

def remove_path_shrinkwrap(curve_obj):
    if not curve_obj:
        return
    # Remove all YP path shrinkwraps (and legacy unnamed ones we created)
    to_remove = [m for m in curve_obj.modifiers if m.type == 'SHRINKWRAP' and m.name == PATH_SHRINKWRAP_NAME]
    for mod in to_remove:
        curve_obj.modifiers.remove(mod)

def update_path_enable_shrinkwrap(self, context):
    curve_obj = get_path_curve_object(self)
    if not curve_obj:
        return
    if self.path_enable_shrinkwrap:
        target = curve_obj.parent if curve_obj.parent else context.object
        if target and target.type == 'MESH':
            ensure_path_shrinkwrap(curve_obj, target)
    else:
        remove_path_shrinkwrap(curve_obj)

def create_path_curve(target_obj, name='Path', use_shrinkwrap=True):
    scene = bpy.context.scene
    curve_name = get_unique_name(name, bpy.data.objects)
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(1)  # 2 points total

    # Place a short default curve near the object / cursor
    if is_bl_newer_than(2, 80):
        cursor_loc = scene.cursor.location.copy()
    else:
        cursor_loc = scene.cursor_location.copy()

    size = max(target_obj.dimensions) * 0.25 if target_obj.dimensions.length > 0 else 0.25
    size = max(size, 0.1)

    p0 = cursor_loc + Vector((-size * 0.5, 0.0, 0.0))
    p1 = cursor_loc + Vector((size * 0.5, 0.0, 0.0))

    # Snap default endpoints onto the mesh surface when possible
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
        for i, p in enumerate((p0, p1)):
            hit = bvh.find_nearest(p)
            if hit and hit[0] is not None:
                if i == 0:
                    p0 = hit[0].copy()
                else:
                    p1 = hit[0].copy()
        bm.free()
    except Exception:
        pass

    bp0 = spline.bezier_points[0]
    bp1 = spline.bezier_points[1]
    for bp, co in ((bp0, p0), (bp1, p1)):
        bp.co = co
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    custom_collection = None
    if is_bl_newer_than(2, 80) and len(target_obj.users_collection) > 0:
        custom_collection = target_obj.users_collection[0]
    link_object(scene, curve_obj, custom_collection)

    # Parent to mesh so it moves with the object, but keep world positions
    curve_obj.parent = target_obj
    curve_obj.matrix_parent_inverse = target_obj.matrix_world.inverted()

    if use_shrinkwrap:
        ensure_path_shrinkwrap(curve_obj, target_obj)

    return curve_obj

def _curve_has_enabled_shrinkwrap(curve_obj):
    mod = get_path_shrinkwrap_modifier(curve_obj)
    return bool(mod and mod.show_viewport and mod.target)

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
    # Emulate Nearest Surface Point shrinkwrap for bake sampling.
    mod = get_path_shrinkwrap_modifier(curve_obj)
    target = mod.target if mod else None
    points = _sample_bezier_math(curve_obj, resolution)
    if not target or target.type != 'MESH' or len(points) < 2:
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

def _curve_to_polyline(curve_obj, resolution=64):
    '''Return world-space polyline points sampled from a CURVE object.'''
    # When Shrinkwrap is on, sample evaluated / projected geometry so baking follows the surface
    if _curve_has_enabled_shrinkwrap(curve_obj):
        points = _sample_evaluated_curve(curve_obj, resolution)
        if len(points) >= 2:
            return points

    # Prefer mathematical bezier sampling (reliable even without bevel/extrude)
    points = _sample_bezier_math(curve_obj, resolution)
    if len(points) >= 2:
        return points

    # Fallback: convert evaluated curve to mesh (works for poly / modifiers)
    return _sample_evaluated_curve(curve_obj, resolution)

def _sample_bezier_math(curve_obj, resolution):
    '''Fallback bezier sampling without mesh conversion.'''
    mw = curve_obj.matrix_world
    points = []
    curve = curve_obj.data

    for spline in curve.splines:
        if spline.type != 'BEZIER':
            # Poly / NURBS: use control points as coarse polyline
            pts = [mw @ Vector(p.co[:3]) for p in spline.points]
            points.extend(pts)
            continue

        bps = spline.bezier_points
        if len(bps) < 2:
            continue

        segs = len(bps) - 1
        if spline.use_cyclic_u:
            segs = len(bps)

        samples_per_seg = max(2, resolution // max(segs, 1))

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
        return []
    return _resample_polyline(points, resolution)

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

def _soft_falloff(t, power=1.0):
    '''t in 0..1 (0=center, 1=edge). Returns opacity.'''
    t = max(0.0, min(1.0, t))
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

def _uv_pixel_dist(uv_a, uv_b, img_w, img_h):
    dx = (uv_a.x - uv_b.x) * (img_w - 1)
    dy = (uv_a.y - uv_b.y) * (img_h - 1)
    return math.sqrt(dx * dx + dy * dy)

def bake_path_to_image(obj, image, curve_obj, uv_name, width, resolution, width_samples,
                       color=(1, 1, 1, 1), falloff=1.0, clear=True,
                       path_texture=None, tile_u=1.0, rotation_deg=0.0,
                       project_distance=None, fill_gaps=True):
    '''
    Stamp a ribbon from a real 3D curve onto a UV image.
    Returns (ok, message).
    '''
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

    polyline = _curve_to_polyline(curve_obj, resolution=max(resolution, 8))
    if len(polyline) < 2:
        return False, 'Curve has too few points to bake'
    width_samples = max(width_samples, 2)

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
    # UV seam jump threshold in pixels — don't bridge across UV islands
    seam_px = max(img_w, img_h) * 0.08

    # Collect per-width-lane samples: list of (uv, u_len, v_across, strength) or None
    lanes = [[] for _ in range(width_samples)]
    stamps = 0

    for i in range(n_len):
        p = polyline[i]
        if i == 0:
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

        side = c_normal.cross(tangent)
        if side.length < 1e-8:
            side = tangent.cross(Vector((0, 0, 1)))
            if side.length < 1e-8:
                side = tangent.cross(Vector((0, 1, 0)))
        side.normalize()

        u_len = i / float(n_len - 1)
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
                # Fallback without lift
                hit = _nearest_uv(bvh, tris, c_loc + offset, max_dist=project_distance)
            if not hit:
                lanes[j].append(None)
                continue

            _, _, uv, _ = hit
            strength = _soft_falloff(abs(v_across), power=falloff)
            lanes[j].append((uv.copy(), u_len, v_across, strength))

    def _rgba_at(u_len, v_across):
        if tex_pxs is not None:
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

    # Stamp each lane, bridging gaps in UV space between consecutive hits
    for lane in lanes:
        for i, sample in enumerate(lane):
            if sample is None:
                continue
            uv, u_len, v_across, strength = sample
            rgba = _rgba_at(u_len, v_across)
            _splat_rgba(pxs, img_w, img_h, uv.x, uv.y, rgba, strength, radius_px=1.25)
            stamps += 1

            if not fill_gaps or i + 1 >= len(lane):
                continue
            nxt = lane[i + 1]
            if nxt is None:
                continue

            uv2, u_len2, v_across2, strength2 = nxt
            dist_px = _uv_pixel_dist(uv, uv2, img_w, img_h)
            if dist_px < 0.5 or dist_px > seam_px:
                # Too close (already covered) or UV seam jump — don't interpolate
                continue

            steps = int(math.ceil(dist_px))
            # Brush radius scales with spacing so the ribbon stays continuous
            radius = max(1.0, dist_px / max(steps, 1) * 0.9)
            for s in range(1, steps):
                t = s / float(steps)
                uv_i = uv.lerp(uv2, t)
                u_i = u_len * (1.0 - t) + u_len2 * t
                v_i = v_across * (1.0 - t) + v_across2 * t
                str_i = strength * (1.0 - t) + strength2 * t
                rgba_i = _rgba_at(u_i, v_i)
                _splat_rgba(pxs, img_w, img_h, uv_i.x, uv_i.y, rgba_i, str_i, radius_px=radius)
                stamps += 1

    if stamps == 0:
        return False, 'No path samples projected onto the mesh (move the curve closer to the surface)'

    flat = pxs.ravel()
    if is_bl_newer_than(2, 83):
        image.pixels.foreach_set(flat)
    else:
        image.pixels = flat.tolist()
    image.update()

    return True, 'Baked %d path samples (res %d x %d)' % (stamps, n_len, width_samples)

def _poll_curve_object(self, obj):
    return obj and obj.type == 'CURVE'

class BaseBakePath():
    '''Mixin for path-bake settings on layers.'''

    enable_path_bake : BoolProperty(
        name = 'Enable Path Bake',
        description = 'This layer can bake a ribbon from a real 3D Bezier curve',
        default = False
    )

    path_enable_shrinkwrap : BoolProperty(
        name = 'Shrinkwrap to Mesh',
        description = 'Add a Shrinkwrap modifier on the path curve targeting the parent mesh (Nearest Surface Point)',
        default = True,
        update = update_path_enable_shrinkwrap
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
        description = 'Softness of the ribbon edge (higher = softer)',
        default = 1.0, min = 0.1, max = 8.0
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
    '''Create an IMAGE layer with a real 3D Bezier curve for path/ribbon baking'''
    bl_idname = 'wm.y_new_path_layer'
    bl_label = 'New Path Layer'
    bl_options = {'REGISTER', 'UNDO'}

    name : StringProperty(name='Name', default='Path')

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
        obj = context.object
        return get_active_ypaint_node() and obj and obj.type == 'MESH'

    def invoke(self, context, event):
        obj = context.object
        node = get_active_ypaint_node()
        yp = node.node_tree.yp
        ypup = get_user_preferences()

        self.name = get_unique_name('Path', bpy.data.images)
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
        obj = context.object

        layout.prop(self, 'name')
        row = layout.row(align=True)
        row.prop(self, 'width')
        row.prop(self, 'height')
        layout.prop(self, 'hdr')
        if obj.type == 'MESH':
            layout.prop_search(self, 'uv_map', obj.data, 'uv_layers', text='UV Map', icon='GROUP_UVS')
        layout.prop(self, 'path_width')

    def execute(self, context):
        T = time.time()
        obj = context.object
        mat = get_active_material()
        node = get_active_ypaint_node()
        yp = node.node_tree.yp
        wm = context.window_manager
        ypui = wm.ypui

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

        # Create real bezier curve in the scene
        curve_obj = create_path_curve(obj, name=self.name + ' Curve', use_shrinkwrap=True)

        from . import Layer

        yp.halt_update = True
        layer = Layer.add_new_layer(
            node.node_tree, self.name, 'IMAGE',
            0, 'MIX', 'MIX', 'BUMP_MAP', 'UV',
            uv_name=self.uv_map, image=img
        )

        layer.enable_path_bake = True
        layer.path_enable_shrinkwrap = True
        set_path_curve_object(layer, curve_obj)
        layer.path_width = self.path_width
        # Ensure shrinkwrap is present (prop default is True; set after curve link)
        ensure_path_shrinkwrap(curve_obj, obj)

        yp.halt_update = False

        node_connections.reconnect_yp_nodes(node.node_tree)
        node_arrangements.rearrange_yp_nodes(node.node_tree)

        # Keep the mesh active so Ucupaint UI stays on the material setup.
        # Still select the curve so it's easy to find in the viewport/outliner.
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
            clear=True
        )
        if not ok:
            self.report({'WARNING'}, 'Path layer created, but initial bake skipped: ' + msg)
        else:
            self.report({'INFO'}, 'Path layer created. Edit the curve, then Bake Path.')

        print('INFO: Path layer', layer.name, 'created in', '{:0.2f}'.format((time.time() - T) * 1000), 'ms!')
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
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, 'Active object must be a mesh')
            return {'CANCELLED'}

        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer or layer.type != 'IMAGE':
            self.report({'ERROR'}, 'Active layer must be an IMAGE layer')
            return {'CANCELLED'}

        curve_obj = get_path_curve_object(layer)
        if not curve_obj:
            self.report({'ERROR'}, 'No path curve linked to this layer')
            return {'CANCELLED'}

        # Keep name in sync
        layer.path_curve_object_name = curve_obj.name

        source = get_layer_source(layer)
        image = source.image if source else None
        if not image:
            self.report({'ERROR'}, 'Layer has no image')
            return {'CANCELLED'}

        path_tex = None
        if is_bl_newer_than(2, 79):
            path_tex = getattr(layer, 'path_texture', None)

        T = time.time()
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
            rotation_deg=layer.path_texture_rotation
        )

        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        # Refresh viewport / paint canvas
        image.update()
        for area in context.screen.areas:
            area.tag_redraw()

        self.report({'INFO'}, msg + ' ({:0.0f} ms)'.format((time.time() - T) * 1000))
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

        # Hint: reselect the mesh to return to the Ucupaint panel
        self.report({'INFO'}, "Path curve selected. Reselect the mesh to show Ucupaint again.")
        return {'FINISHED'}

class YCreatePathCurveForLayer(bpy.types.Operator):
    '''Create a new Bezier curve and link it to this IMAGE layer for path baking'''
    bl_idname = 'wm.y_create_path_curve_for_layer'
    bl_label = 'Create Path Curve'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            if node:
                layer = get_active_layer(node.node_tree.yp)
        return obj and obj.type == 'MESH' and layer and layer.type == 'IMAGE'

    def execute(self, context):
        obj = context.object
        layer = getattr(context, 'layer', None)
        if not layer:
            node = get_active_ypaint_node()
            layer = get_active_layer(node.node_tree.yp) if node else None
        if not layer:
            return {'CANCELLED'}

        curve_obj = create_path_curve(obj, name=layer.name + ' Curve', use_shrinkwrap=layer.path_enable_shrinkwrap)
        layer.enable_path_bake = True
        set_path_curve_object(layer, curve_obj)
        if layer.path_enable_shrinkwrap:
            ensure_path_shrinkwrap(curve_obj, obj)

        set_object_select(curve_obj, True)
        set_active_object(obj)

        self.report({'INFO'}, "Created path curve '%s'" % curve_obj.name)
        return {'FINISHED'}

def draw_path_bake_ui(layout, context, layer):
    '''Draw Path Bake controls for an IMAGE layer.'''
    if layer.type != 'IMAGE':
        return

    box = layout.box()
    col = box.column(align=True)

    row = col.row(align=True)
    row.label(text='Path Bake', icon='CURVE_DATA' if is_bl_newer_than(2, 80) else 'CURVE_BEZCURVE')
    row.prop(layer, 'enable_path_bake', text='')

    if not layer.enable_path_bake:
        return

    col.separator()

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

    col.prop(layer, 'path_enable_shrinkwrap')

    col.separator()
    col.prop(layer, 'path_width')
    col.prop(layer, 'path_resolution')
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
    brow.operator('wm.y_bake_path_to_layer', text='Bake Path', icon_value=lib_get_bake_icon())

def lib_get_bake_icon():
    from . import lib
    return lib.get_icon('bake')

def register():
    bpy.utils.register_class(YNewPathLayer)
    bpy.utils.register_class(YBakePathToLayer)
    bpy.utils.register_class(YSelectPathCurve)
    bpy.utils.register_class(YCreatePathCurveForLayer)

def unregister():
    bpy.utils.unregister_class(YNewPathLayer)
    bpy.utils.unregister_class(YBakePathToLayer)
    bpy.utils.unregister_class(YSelectPathCurve)
    bpy.utils.unregister_class(YCreatePathCurveForLayer)
