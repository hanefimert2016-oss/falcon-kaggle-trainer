import bpy
import math
import os
import random
from mathutils import Vector

OUT = os.environ.get("IRONMAN_OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
random.seed(50)

# ---------- scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 144
scene.render.fps = 24
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world.color = (0.003, 0.004, 0.008)

# Color management
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass

# Cycles GPU, falling back cleanly if the runtime exposes CUDA differently.
scene.render.engine = "BLENDER_EEVEE_NEXT"
gpu_backend = "EEVEE_NEXT"
try:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 3
    prefs = bpy.context.preferences.addons["cycles"].preferences
    activated = False
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            enabled = 0
            for dev in prefs.devices:
                if dev.type in {"OPTIX", "CUDA"}:
                    dev.use = True
                    enabled += 1
            if enabled:
                gpu_backend = "CYCLES_" + backend
                activated = True
                break
        except Exception as e:
            print("GPU backend", backend, "not available:", e)
    if not activated:
        print("Cycles GPU device enumeration did not expose CUDA/OptiX; using engine fallback")
except Exception as e:
    print("Cycles setup failed, using Eevee Next:", e)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    gpu_backend = "EEVEE_NEXT"

print("RENDER_BACKEND", gpu_backend, flush=True)

# ---------- helpers ----------
def material(name, rgba, metallic=0.0, rough=0.35, emission=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    if emission is not None:
        # Blender 4.x uses Emission Color; 3.x uses Emission.
        inp = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        if inp:
            inp.default_value = emission
        sinp = bsdf.inputs.get("Emission Strength")
        if sinp:
            sinp.default_value = strength
    return m

RED = material("Nano Red", (0.33, 0.008, 0.012, 1), 0.92, 0.16)
RED2 = material("Nano Red Dark", (0.085, 0.004, 0.006, 1), 0.9, 0.2)
GOLD = material("Titanium Gold", (0.62, 0.22, 0.035, 1), 0.92, 0.14)
BLACK = material("Undersuit", (0.012, 0.016, 0.022, 1), 0.55, 0.28)
GUN = material("Gunmetal", (0.045, 0.055, 0.07, 1), 0.9, 0.18)
CYAN = material("Arc Energy", (0.01, 0.3, 0.42, 1), 0.25, 0.12, (0.02, 0.8, 1.0, 1), 14.0)
WHITE = material("Eye Energy", (0.75, 0.95, 1.0, 1), 0.2, 0.1, (0.3, 0.9, 1.0, 1), 20.0)
FLOOR = material("Floor", (0.012, 0.014, 0.02, 1), 0.65, 0.28)

character = []

def mark(obj):
    obj["export_character"] = True
    character.append(obj)
    return obj

def smooth(obj):
    if obj.type == "MESH":
        for p in obj.data.polygons:
            p.use_smooth = True

def bevel(obj, width=0.035, segments=3):
    mod = obj.modifiers.new("MicroBevel", "BEVEL")
    mod.width = width
    mod.segments = segments

def add_uv(name, loc, scale, mat, segments=32):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=16, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    smooth(o)
    return mark(o)

def add_cube(name, loc, scale, mat, rot=(0,0,0), bev=0.04):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    bevel(o, bev, 3)
    return mark(o)

def add_cyl(name, loc, radius, depth, mat, rot=(0,0,0), vertices=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    bevel(o, min(0.025, radius*0.18), 2)
    smooth(o)
    return mark(o)

def cyl_between(name, a, b, radius, mat):
    a = Vector(a); b = Vector(b)
    d = b - a
    mid = (a+b)*0.5
    o = add_cyl(name, mid, radius, d.length, mat)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector((0,0,1)).rotation_difference(d.normalized())
    return o

def animate_assembly(o, start_frame, duration=18, source=(0,-0.82,2.43), overshoot=1.08):
    final_loc = o.location.copy()
    final_scale = o.scale.copy()
    final_rot_mode = o.rotation_mode
    if o.rotation_mode == "QUATERNION":
        final_rot = o.rotation_quaternion.copy()
    else:
        final_rot = o.rotation_euler.copy()

    jitter = Vector((random.uniform(-0.18,0.18), random.uniform(-0.15,0.08), random.uniform(-0.12,0.12)))
    o.location = Vector(source) + jitter
    o.scale = final_scale * 0.025
    if o.rotation_mode == "QUATERNION":
        o.rotation_quaternion = final_rot @ Vector((0,0,1)).rotation_difference(Vector((random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1))).normalized())
    else:
        o.rotation_euler.rotate_axis("Z", random.uniform(-1.2,1.2))
    o.keyframe_insert("location", frame=start_frame)
    o.keyframe_insert("scale", frame=start_frame)
    if o.rotation_mode == "QUATERNION":
        o.keyframe_insert("rotation_quaternion", frame=start_frame)
    else:
        o.keyframe_insert("rotation_euler", frame=start_frame)

    mid = start_frame + int(duration*0.58)
    arc = (final_loc + Vector(source))*0.5 + Vector((random.uniform(-0.25,0.25), -0.45, 0.22))
    o.location = arc
    o.scale = final_scale * 0.48
    o.keyframe_insert("location", frame=mid)
    o.keyframe_insert("scale", frame=mid)

    end = start_frame + duration
    o.location = final_loc
    o.scale = final_scale * overshoot
    if o.rotation_mode == "QUATERNION":
        o.rotation_quaternion = final_rot
        o.keyframe_insert("rotation_quaternion", frame=end)
    else:
        o.rotation_euler = final_rot
        o.keyframe_insert("rotation_euler", frame=end)
    o.keyframe_insert("location", frame=end)
    o.keyframe_insert("scale", frame=end)

    o.scale = final_scale
    o.keyframe_insert("scale", frame=end+4)

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

# ---------- undersuit ----------
add_uv("Undersuit_Torso", (0,0,2.30), (0.67,0.36,0.82), BLACK, 40)
add_uv("Undersuit_Pelvis", (0,0,1.55), (0.55,0.33,0.43), BLACK, 32)
add_uv("Undersuit_Head", (0,0,3.32), (0.34,0.31,0.44), BLACK, 32)
for side in (-1,1):
    cyl_between(f"Undersuit_UpperArm_{side}", (0.62*side,0,2.62), (0.83*side,0,2.05), 0.19, BLACK)
    cyl_between(f"Undersuit_Forearm_{side}", (0.83*side,0,2.05), (0.88*side,-0.02,1.47), 0.17, BLACK)
    cyl_between(f"Undersuit_Thigh_{side}", (0.32*side,0,1.45), (0.36*side,0,0.82), 0.24, BLACK)
    cyl_between(f"Undersuit_Shin_{side}", (0.36*side,0,0.82), (0.31*side,-0.01,0.20), 0.19, BLACK)

# ---------- armor final pieces ----------
armor = []
def armor_obj(o, phase):
    o["nano_phase"] = phase
    armor.append(o)
    return o

# Chest layered plates
armor_obj(add_uv("Chest_Carapace", (0,-0.055,2.46), (0.70,0.31,0.73), RED, 40), 0)
armor_obj(add_cube("Chest_Gold_V", (0,-0.36,2.49), (0.24,0.055,0.36), GOLD, rot=(0.0,0.0,0.0), bev=0.06), 1)
for i,(z,w) in enumerate([(2.18,0.48),(2.04,0.44),(1.91,0.40)]):
    armor_obj(add_cube(f"Ab_Plate_{i}", (0,-0.34,z), (w,0.07,0.085), RED if i!=1 else GOLD, bev=0.035), 2+i)
armor_obj(add_cube("Pelvis_Belt", (0,-0.27,1.62), (0.48,0.09,0.10), GOLD, bev=0.04), 5)
armor_obj(add_cube("Pelvis_Core", (0,-0.20,1.48), (0.47,0.16,0.19), RED, bev=0.06), 5)

# Arc reactor
bpy.ops.mesh.primitive_torus_add(major_radius=0.17, minor_radius=0.032, major_segments=48, minor_segments=12, location=(0,-0.405,2.43), rotation=(math.radians(90),0,0))
reactor_ring = mark(bpy.context.object); reactor_ring.name="Arc_Reactor_Ring"; reactor_ring.data.materials.append(GUN)
add_cyl("Arc_Reactor_Core", (0,-0.43,2.43), 0.135, 0.045, CYAN, rot=(math.radians(90),0,0), vertices=48)
for i in range(8):
    a=2*math.pi*i/8
    armor_obj(add_cube(f"Reactor_Petal_{i}", (0.23*math.cos(a),-0.39,2.43+0.23*math.sin(a)), (0.055,0.035,0.09), GOLD, rot=(0,0,-a), bev=0.02), 0)

# Shoulders and arms
for side in (-1,1):
    s="L" if side<0 else "R"
    armor_obj(add_uv(f"Shoulder_{s}", (0.70*side,-0.01,2.64), (0.31,0.28,0.30), RED, 32), 5)
    armor_obj(add_cube(f"ShoulderCap_{s}", (0.72*side,-0.22,2.68), (0.25,0.09,0.18), GOLD, rot=(0,0,math.radians(10)*side), bev=0.05), 6)
    armor_obj(cyl_between(f"BicepArmor_{s}", (0.72*side,-0.02,2.48),(0.84*side,-0.03,2.12),0.225,RED), 7)
    armor_obj(add_cube(f"BicepGold_{s}", (0.82*side,-0.23,2.28), (0.12,0.055,0.19), GOLD, rot=(0,0,math.radians(6)*side), bev=0.035), 8)
    armor_obj(cyl_between(f"ForearmArmor_{s}", (0.86*side,-0.02,1.99),(0.89*side,-0.05,1.54),0.205,RED), 10)
    armor_obj(add_cube(f"ForearmGold_{s}", (0.89*side,-0.25,1.76), (0.13,0.05,0.20), GOLD, bev=0.035), 11)
    gaunt = armor_obj(add_uv(f"Gauntlet_{s}", (0.90*side,-0.04,1.43), (0.21,0.24,0.20), RED, 32), 12)
    armor_obj(add_cyl(f"PalmRepulsor_{s}", (0.90*side,-0.285,1.43), 0.075, 0.03, CYAN, rot=(math.radians(90),0,0), vertices=32), 13)

# Legs
for side in (-1,1):
    s="L" if side<0 else "R"
    armor_obj(cyl_between(f"ThighArmor_{s}", (0.31*side,-0.03,1.43),(0.36*side,-0.03,0.93),0.285,RED), 13)
    armor_obj(add_cube(f"ThighGold_{s}", (0.34*side,-0.29,1.18), (0.12,0.045,0.20), GOLD, bev=0.035), 14)
    armor_obj(add_uv(f"Knee_{s}", (0.36*side,-0.13,0.79), (0.23,0.20,0.19), GOLD, 28), 15)
    armor_obj(cyl_between(f"ShinArmor_{s}", (0.35*side,-0.03,0.71),(0.31*side,-0.05,0.27),0.225,RED), 16)
    armor_obj(add_cube(f"ShinGold_{s}", (0.33*side,-0.25,0.48), (0.11,0.05,0.20), GOLD, bev=0.03), 17)
    armor_obj(add_cube(f"Boot_{s}", (0.31*side,-0.11,0.11), (0.24,0.34,0.12), RED, bev=0.055), 18)
    armor_obj(add_cyl(f"FootThruster_{s}", (0.31*side,0.19,0.08), 0.07, 0.025, CYAN, vertices=32), 18)

# Neck and helmet
armor_obj(add_cyl("Neck_Ring", (0,0,2.91), 0.28, 0.18, GOLD, vertices=40), 16)
helmet = armor_obj(add_uv("Helmet_Shell", (0,0,3.33), (0.39,0.34,0.46), RED, 48), 20)
face = armor_obj(add_uv("Faceplate", (0,-0.275,3.34), (0.30,0.12,0.36), GOLD, 40), 22)
armor_obj(add_cube("Jaw", (0,-0.29,3.12), (0.29,0.08,0.10), RED, bev=0.04), 22)
for side in (-1,1):
    armor_obj(add_cube(f"Eye_{side}", (0.115*side,-0.405,3.43), (0.085,0.018,0.025), WHITE, rot=(0,0,math.radians(7)*side), bev=0.012), 24)
    armor_obj(add_cube(f"Temple_{side}", (0.31*side,-0.16,3.33), (0.06,0.10,0.20), RED2, bev=0.03), 21)

# Surface panel details
for side in (-1,1):
    for i,z in enumerate((2.58,2.40,2.22)):
        armor_obj(add_cube(f"ChestPanel_{side}_{i}", (0.36*side,-0.36,z), (0.16,0.035,0.09), RED2 if i%2 else GOLD, rot=(0,0,math.radians(7)*side), bev=0.02), 2+i)

# ---------- nanotech assembly ----------
for o in armor:
    phase = int(o.get("nano_phase", 0))
    start = 10 + phase*3 + random.randint(0,3)
    animate_assembly(o, start, duration=16 + random.randint(0,6))

# Nano-particle swarm, sharing one mesh to keep the file compact.
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1, location=(0,-0.82,2.43))
nano_seed = bpy.context.object
nano_seed.name = "NanoSeed"
nano_seed.data.materials.append(RED)
mark(nano_seed)
nano_mesh = nano_seed.data
bpy.data.objects.remove(nano_seed, do_unlink=True)
character.remove(nano_seed)

nano_final_regions = []
for i in range(120):
    # Final points hug the full body silhouette.
    t = random.random()
    if t < 0.32:
        x=random.uniform(-0.58,0.58); z=random.uniform(1.85,2.82); y=random.uniform(-0.38,-0.20)
    elif t < 0.52:
        side=random.choice((-1,1)); x=random.uniform(0.67,0.93)*side; z=random.uniform(1.42,2.68); y=random.uniform(-0.25,-0.08)
    elif t < 0.82:
        side=random.choice((-1,1)); x=random.uniform(0.22,0.43)*side; z=random.uniform(0.20,1.50); y=random.uniform(-0.24,-0.06)
    else:
        x=random.uniform(-0.29,0.29); z=random.uniform(3.02,3.60); y=random.uniform(-0.39,-0.18)
    nano_final_regions.append((x,y,z))

for i,final in enumerate(nano_final_regions):
    o = bpy.data.objects.new(f"NanoTile_{i:03d}", nano_mesh)
    bpy.context.collection.objects.link(o)
    mark(o)
    o.data = nano_mesh
    o.location = final
    o.scale = (0.020,0.009,0.030)
    if i % 13 == 0:
        # Linked mesh material is shared; leave red for consistency.
        pass
    dist = (Vector(final)-Vector((0,-0.82,2.43))).length
    start = 6 + int(dist*12) + random.randint(0,8)
    animate_assembly(o, start, duration=13+random.randint(0,8), overshoot=1.35)

# ---------- floor ----------
bpy.ops.mesh.primitive_plane_add(size=30, location=(0,0,-0.03))
floor=bpy.context.object
floor.data.materials.append(FLOOR)
bev = floor.modifiers.new("Floor bevel","BEVEL"); bev.width=0.02

# subtle glowing ring on floor
bpy.ops.mesh.primitive_torus_add(major_radius=2.15, minor_radius=0.012, major_segments=96, minor_segments=8, location=(0,0,0.012))
ring=bpy.context.object
ring.data.materials.append(CYAN)

# ---------- lights ----------
def area(name, loc, energy, size, color):
    data=bpy.data.lights.new(name, "AREA")
    data.energy=energy; data.shape="DISK"; data.size=size; data.color=color
    o=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(o); o.location=loc
    look_at(o,(0,0,2.1)); return o

area("Key", (-4,-4,6.5), 1350, 5.0, (1.0,0.72,0.58))
area("Fill", (4,-3,4.5), 950, 4.0, (0.35,0.58,1.0))
area("Rim", (0,3.5,5.8), 1550, 3.5, (0.72,0.10,0.06))
area("Top", (0,0,7.5), 1100, 3.0, (0.8,0.9,1.0))

# ---------- camera ----------
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0,0,1.95))
target=bpy.context.object
bpy.ops.object.camera_add(location=(4.9,-8.8,3.15))
cam=bpy.context.object
cam.data.lens=64
track=cam.constraints.new(type="TRACK_TO")
track.target=target; track.track_axis="TRACK_NEGATIVE_Z"; track.up_axis="UP_Y"
scene.camera=cam
cam.location=(4.9,-8.8,3.15); cam.keyframe_insert("location",frame=1)
cam.location=(3.4,-7.8,2.85); cam.keyframe_insert("location",frame=82)
cam.location=(2.6,-7.2,2.7); cam.keyframe_insert("location",frame=144)

# ---------- export/save ----------
# Set interpolation to Bezier with restrained handles for a premium mechanical feel.
for o in character+[cam]:
    if o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"

blend_path=os.path.join(OUT,"ironman_nanotech_scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

# Export only the character assets, not studio objects.
bpy.ops.object.select_all(action="DESELECT")
for o in character:
    if o and o.name in bpy.context.view_layer.objects:
        o.select_set(True)
        try:
            bpy.context.view_layer.objects.active=o
        except Exception:
            pass
scene.frame_set(144)
glb_path=os.path.join(OUT,"ironman_nanotech_animated.glb")
bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format="GLB",
    use_selection=True,
    export_animations=True,
    export_apply=True,
)

# Final poster.
scene.render.filepath=os.path.join(OUT,"ironman_nanotech_poster.png")
scene.render.image_settings.file_format="PNG"
scene.render.resolution_x=1920
scene.render.resolution_y=1080
bpy.ops.render.render(write_still=True)

# Full 6 second nanotech transformation preview.
scene.frame_set(1)
scene.render.image_settings.file_format="FFMPEG"
scene.render.ffmpeg.format="MPEG4"
scene.render.ffmpeg.codec="H264"
scene.render.ffmpeg.constant_rate_factor="MEDIUM"
scene.render.filepath=os.path.join(OUT,"ironman_nanotech_transform.mp4")
bpy.ops.render.render(animation=True)

# Human-readable report.
report = {
    "render_backend": gpu_backend,
    "frames": [scene.frame_start, scene.frame_end],
    "fps": scene.render.fps,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "armor_piece_count": len(armor),
    "nano_tile_count": len(nano_final_regions),
    "outputs": [
        "ironman_nanotech_scene.blend",
        "ironman_nanotech_animated.glb",
        "ironman_nanotech_poster.png",
        "ironman_nanotech_transform.mp4",
    ],
}
import json
with open(os.path.join(OUT,"ironman_nanotech_manifest.json"),"w",encoding="utf-8") as f:
    json.dump(report,f,indent=2)

for fn in report["outputs"]:
    p=os.path.join(OUT,fn)
    if not os.path.exists(p) or os.path.getsize(p)<1024:
        raise RuntimeError("Missing or suspiciously small output: "+p)
print("IRONMAN_NANOTECH_DONE", json.dumps(report), flush=True)
