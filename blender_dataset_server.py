import bpy
import cv2
import random
import os
from math import radians
#import cv2
import numpy as np
import json

# Set the directory where the images and data will be saved
output_dir = "new_dataset_random/"

# Path to your .blend file
blend_file_path = "./ArucoBlender.blend"

# Open the .blend file
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# Set the render engine to 'CYCLES'
bpy.context.scene.render.engine = 'CYCLES'

# Set device type to 'CUDA' (for NVIDIA) or 'OPTIX' (for NVIDIA RTX)
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

# Set rendering device to GPU
bpy.context.scene.cycles.device = 'GPU'

# Get available devices
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.get_devices()


# Print available devices to check which GPUs are available
print("Available devices:")
for i, device in enumerate(prefs.devices):
    print(f"{i}: {device.name} (Type: {device.type}, Use: {device.use})")

# Disable all devices
for device in prefs.devices:
    device.use = False

# Enable only a specific GPU (e.g., the second GPU in the list, index 1)
prefs.devices[0].use = True  # Adjust the index based on your device list
prefs.devices[1].use = True
# Alternatively, if you know the name of the GPU, you can use:
# for device in prefs.devices:
#     if 'GPU_NAME' in device.name:
#         device.use = True
print(prefs.devices[0].name)
print(prefs.devices[1].name)
# Check enabled devices
print("Enabled devices:", [d.name for d in prefs.devices if d.use])

# Set the Noise Threshold
bpy.context.scene.cycles.adaptive_threshold = 0.0100

# Set Max Samples
bpy.context.scene.cycles.samples = 4096

# Set Min Samples
bpy.context.scene.cycles.adaptive_min_samples = 0

# Set Time Limit
bpy.context.scene.cycles.time_limit = 0

# Enable Denoising
bpy.context.scene.cycles.use_denoising = True

# Path to the image file (adjust this path to where your texture is located)
image_path = "./aruco_Example.png"

# Get the object (replace 'ArucoObjectName' with the actual object name in Blender)
obj = bpy.data.objects['NewAruco']

# Ensure the object has an active material; if not, create one
if not obj.data.materials:
    mat = bpy.data.materials.new(name="ArucoMaterial")
    obj.data.materials.append(mat)
else:
    mat = obj.data.materials[0]  # Use the first material

# Enable 'use_nodes' so we can apply the texture
mat.use_nodes = True
nodes = mat.node_tree.nodes

# Remove the default node (if exists) and add a new Principled BSDF shader
for node in nodes:
    nodes.remove(node)  # Clean up all existing nodes if needed

# Add a new Principled BSDF node
principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

# Add an Image Texture node
tex_image = nodes.new(type='ShaderNodeTexImage')

# Load the image into the Image Texture node
img = bpy.data.images.load(image_path)
tex_image.image = img

# Connect the Image Texture node to the Base Color input of the Principled BSDF node
mat.node_tree.links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])

# Add an Output node and connect the Principled BSDF to the material output
material_output = nodes.new(type='ShaderNodeOutputMaterial')
mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

# Optionally, adjust other settings like the Alpha (transparency), subsurface, etc.:
# Example: Set the alpha to 1 (fully opaque) for the material.
principled_bsdf.inputs['Alpha'].default_value = 1.0

# Update the object so it uses the material
obj.active_material = mat

# List of objects to hide from rendering (replace these names with your actual object names)
objects_to_hide = ['Aruco','white','meshparentnode.65300','meshparentnode.67608', 'PIANTANA_1','PIANTANA_2','PIN','PRIMA_RULLIERA','RULLO_TRONCO-CONICI','SALTO','SECONDA_RULLIERA', 'vergella', 'IRB6640_235_255__04 Zona operativa']
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
# Iterate over each object and set hide_render to True
for obj_name in objects_to_hide:
    obj = bpy.data.objects.get(obj_name)
    if obj:
        obj.hide_render = True  # This will prevent the object from being rendered

# Ensure there is adequate lighting
if "Light" not in bpy.data.objects:
    # Create a new light if one does not exist
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 10))
    light = bpy.context.object
    light.data.energy = 1000  # Adjust light intensity
#winSize = (3,3)
#zeroZone = (-1, -1)
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.0001)
# Intrinsic parameters from datasheet
W=14.2*0.001 #mm (blender's camera -> width of a pixel)
w=4112
H=7.5 *0.001 #mm (blender's camera -> height of a pixel)
h=2176
f=6.5 * 0.001 #m
du=W/w
dv=H/h
cx=(w-1)/2
cy=(h-1)/2
K_true = np.array(
    [[f/(du), 0.,          cx],
     [0.,         f/(dv),  cy],
     [0.,         0.,      1]])   


seed_value = 42
random.seed(seed_value)

trueRotation=np.eye(3,3)
trueRotation[1,1]=-1
trueRotation[2,2]=-1
trueTranslation=np.array([8.75677, -3.96, 8.8125])
RT=np.vstack((np.hstack((trueRotation,trueTranslation.reshape((3,1)))),np.array([0,0,0,1])))
# Set render output to RGB
bpy.context.scene.render.image_settings.file_format = 'PNG'  # Can be set to 'JPEG', 'TIFF', etc.
bpy.context.scene.render.image_settings.color_mode = 'RGB'  # Set to RGB
if bpy.context.scene.use_nodes:
    for node in bpy.context.scene.node_tree.nodes:
        if node.type == 'RGBTOBW':
            bpy.context.scene.node_tree.nodes.remove(node)  # Remove any RGB to BW nodes
# Set viewport shading to Material Preview
#bpy.context.space_data.shading.type = 'MATERIAL'  # 'SOLID', 'MATERIAL', 'RENDERED'


aruco = bpy.data.objects["NewAruco"]
def randomize_texture(obj):
    # Ensure the object has a material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="RandomMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Remove existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add new Principled BSDF shader
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Set base color to white
    principled_bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # White color
    

# Add noise texture for dust with random scale
    noise_texture = nodes.new(type='ShaderNodeTexNoise')
    noise_texture.inputs['Scale'].default_value = random.uniform(30.0, 100.0)  # Randomize scale for different dust size
    
    # Add a ColorRamp to control dust intensity and distribution
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = random.uniform(0.3, 0.6)  # Randomize the start position of the dust
    color_ramp.color_ramp.elements[0].color = (random.uniform(0, 0.2), random.uniform(0, 0.2), random.uniform(0, 0.2), 1)  # Random dark dust color
    color_ramp.color_ramp.elements[1].position = random.uniform(0.7, 0.9)  # Randomize where dust transitions back to white
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # White background color
    
    # Add Mapping and Texture Coordinate nodes
    mapping_node = nodes.new(type="ShaderNodeMapping")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    
    # Randomize rotation in the Mapping node for different dust locations
    mapping_node.inputs['Rotation'].default_value[0] = radians(random.uniform(0.0, 360.0))
    mapping_node.inputs['Rotation'].default_value[1] = radians(random.uniform(0.0, 360.0))
    mapping_node.inputs['Rotation'].default_value[2] = radians(random.uniform(0.0, 360.0))
    
    # Add a MixRGB node to mix the noise texture with the base color
    mix_rgb = nodes.new(type="ShaderNodeMixRGB")
    mix_rgb.blend_type = 'MIX'
    mix_rgb.inputs['Fac'].default_value = random.uniform(0.4, 0.9)  # Randomize dust visibility (0 = no dust, 1 = full dust)
    
    # Add another randomization for variation of the dust effect over the surface
    random_factor_texture = nodes.new(type='ShaderNodeTexNoise')
    random_factor_texture.inputs['Scale'].default_value = random.uniform(5.0, 15.0)  # Random factor to vary the dust intensity across the surface
    
    # Add a ColorRamp to control how random factor affects the mix
    random_factor_ramp = nodes.new(type="ShaderNodeValToRGB")
    random_factor_ramp.color_ramp.elements[0].position = random.uniform(0.3, 0.5)  # Random distribution for dust intensity
    random_factor_ramp.color_ramp.elements[1].position = random.uniform(0.6, 0.9)
    
    # Link the nodes
    links.new(texture_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], noise_texture.inputs["Vector"])
    links.new(noise_texture.outputs["Fac"], color_ramp.inputs["Fac"])
    
    # Connect ColorRamp to MixRGB for dust effect
    links.new(color_ramp.outputs["Color"], mix_rgb.inputs['Color1'])  # Dust color (based on noise)
    links.new(principled_bsdf.outputs['BSDF'], mix_rgb.inputs['Color2'])  # Base white color
    
    # Randomize dust intensity via random_factor_texture
    links.new(random_factor_texture.outputs["Fac"], random_factor_ramp.inputs["Fac"])
    links.new(random_factor_ramp.outputs["Color"], mix_rgb.inputs["Fac"])  # Use this to control the Mix factor dynamically
    
    # Add Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Link the mix_rgb shader to the Material Output node
    links.new(mix_rgb.outputs['Color'], material_output.inputs['Surface'])

def get_aruco_corners(obj):
    # Get the vertices of the mesh
    mesh = obj.data
    corners_3d_local = [v.co for v in mesh.vertices]
    
    aruco_face_indices = [0, 1, 2, 3]  # Assuming 4 ArUco corners
    
    aruco_face_corners_local = [corners_3d_local[i] for i in aruco_face_indices]
    
    # Get the object's world matrix
    world_matrix = obj.matrix_world
    
    # Convert local coordinates to world coordinates
    aruco_face_corners_world = [world_matrix @ corner.to_4d() for corner in aruco_face_corners_local]
    
    # Extract x, y, z coordinates from the world coordinates
    aruco_face_corners_world = [corner.xyz for corner in aruco_face_corners_world]
    
    # Reorder corners to match the desired order: [0, 1, 3, 2]
    ordered_corners_3d_world = [aruco_face_corners_world[i] for i in [2,3,1,0]]
    
    return ordered_corners_3d_world

def project_to_2d(k_true, RT, corners_3d):
    # Convert 3D corners to homogeneous coordinates (4x1 vectors)
    corners_3d_homogeneous = np.array([[corner.x, corner.y, corner.z, 1] for corner in corners_3d]).T  # Shape (4, N)
    
    # Project 3D points into 2D
    projected_points = np.dot(k_true, np.dot(RT, corners_3d_homogeneous)[:3,:])  # Shape (3, N)
    
    projected_points /= projected_points[2, :]  # Normalize by the third row
    
    # Extract x, y pixel coordinates (ignoring the homogeneous coordinate)
    corners_2d = projected_points[:2, :].T  # Shape (N, 2)
    
    return corners_2d.tolist()

# Get the object to be moved
obj = bpy.data.objects["Robot_tool"]
white_obj = bpy.data.objects.get('NewWhite')
# Define the ranges for the movements along X, Y, Z

y_range = (-4.45, -3.2)
z_range = (6.6, 7.7)

# Define the number of random positions
num_positions =300
output_json = []
start=0
# Iterate through the positions
for i in range(start,num_positions):
    # Randomize the location
#    apply_random_texture(white_obj)
    randomize_texture(white_obj)  
    print(i)
    obj.location.y = random.uniform(*y_range)
    obj.location.z = random.uniform(*z_range)
    if i < 750:
        x_range = (-9.7, -8.8)

        obj.location.x = random.uniform(*x_range)

    # Randomize the rotation
        obj.rotation_euler.x = random.uniform(radians(25), radians(270))
        obj.rotation_euler.y = random.uniform(radians(60), radians(110))
        obj.rotation_euler.z = random.uniform(radians(70), radians(150))
    else:
        x_range = (-8.8,-7.4)
        obj.location.x = random.uniform(*x_range)

    # Randomize the rotation
        obj.rotation_euler.x = random.uniform(radians(25), radians(270))
        obj.rotation_euler.y = random.uniform(radians(110), radians(180))
        obj.rotation_euler.z = random.uniform(radians(150), radians(180))

    bpy.context.scene.view_settings.view_transform = 'Filmic'  
# Update the scene
    bpy.context.view_layer.update()

    # Define the file paths
    img_file = os.path.join(output_dir, f"render_{i:05d}.png")

    # Set Color Management settings
    bpy.context.scene.view_settings.view_transform = 'Standard'  # Change as needed
    bpy.context.scene.view_settings.look = 'None'
 
# Render the image
    bpy.context.scene.render.filepath = img_file
    bpy.ops.render.render(write_still=True)

    # Save the object data (matrix world and vertices)
    aruco_corners_3d = get_aruco_corners(aruco)
    true_corners_2d = project_to_2d(K_true, RT, aruco_corners_3d)

    # Store data in a dictionary
    data = {
        "image_file": img_file,
        "true": true_corners_2d
    }
        
    # Append the data for this position to the list
    output_json.append(data)

# Write the collected data to JSON after all iterations
with open("new_dataset1.json", "w") as outfile:
    json.dump(output_json, outfile, indent=4)
