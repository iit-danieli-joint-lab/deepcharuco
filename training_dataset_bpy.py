import bpy
import random
import os
from math import radians
import cv2
import numpy as np
import json

# Set the directory where the images and data will be saved
output_dir = "training_dataset/"

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


winSize = (3,3)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.0001)
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
     [0.,         0.,          1.]]
)
seed_value = 42
random.seed(seed_value)

RT = np.array([
    [ 0.99999998, -0.00001684 , 0.00018797 , 8.75476102],
    [-0.00001684 ,-1.          , 0.00001036 ,-3.96008424],
    [ 0.00018797, -0.00001037, -0.99999998,  8.81389337],
    [ 0.      ,    0.     ,     0.       ,   1.        ]
])

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()
#detector = cv2.aruco.ArucoDetector(dictionary, parameters)

aruco = bpy.data.objects["Aruco"]

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
    ordered_corners_3d_world = [aruco_face_corners_world[i] for i in [0, 1, 3, 2]]
    
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

# Define the ranges for the movements along X, Y, Z

y_range = (-4.45, -3.2)
z_range = (6.6, 7.7)

# Define the number of random positions
num_positions = 10
output_json = []

# Iterate through the positions
for i in range(num_positions):
    # Randomize the location
    print(i)
    obj.location.y = random.uniform(*y_range)
    obj.location.z = random.uniform(*z_range)
    if i < 5:
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
    # Update the scene
    bpy.context.view_layer.update()

    # Define the file paths
    img_file = os.path.join(output_dir, f"render_{i:03d}.png")

    # Render the image
    bpy.context.scene.render.filepath = img_file
    bpy.ops.render.render(write_still=True)

    # Save the object data (matrix world and vertices)
    aruco_corners_3d = get_aruco_corners(aruco)
    true_corners_2d = project_to_2d(K_true, RT, aruco_corners_3d)

    detected_corners = None
    if os.path.exists(img_file):    
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)


        if len(corners) > 0:
            # Only consider the first detected marker for simplicity
            corners = cv2.cornerSubPix(image, corners[0][0], winSize, zeroZone, criteria)
            detected_corners = corners.tolist()

    # Store data in a dictionary
    data = {
        "image_file": img_file,
        "true": true_corners_2d
    }
    
    if detected_corners is not None:
        data["detected"] = detected_corners
    
    # Append the data for this position to the list
    output_json.append(data)

# Write the collected data to JSON after all iterations
with open("training_dataset.json", "w") as outfile:
    json.dump(output_json, outfile, indent=4)
