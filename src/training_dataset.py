import bpy
import random
import os
from math import radians
import cv2
import numpy as np
import json

# Set the directory where the images and data will be saved
output_dir = "/home/alessia/Desktop/training_dataset/"

winSize = (3,3)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.0001)
#intrinsics parameters from datasheet
W=14.2*0.001 #mm (blender's camera -> width of a pixel)
w=4112
H=7.5 *0.001 #mm (blender's camera -> height of a pixel)
h=2176
f=6.5 * 0.001 #m
#du=0.00000345 
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

RT=np.array([[ 0.99999998, -0.00001684 , 0.00018797 , 8.75476102],
 [-0.00001684 ,-1.          ,0.00001036 ,-3.96008424],
 [ 0.00018797, -0.00001037, -0.99999998,  8.81389337],
 [ 0.      ,    0.     ,     0.       ,   1.        ]])

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

aruco = bpy.data.objects["Aruco"]


def get_aruco_corners(obj):
    # Get the vertices of the mesh
    mesh = obj.data
    corners_3d_local = [v.co for v in mesh.vertices]
    
    
    aruco_face_indices = [0, 1, 2, 3]  
    
    
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
    
    projected_points /= projected_points[2, :]  # Divide by the third row to normalize
    
    # Extract x, y pixel coordinates (ignoring the homogeneous coordinate)
    corners_2d = projected_points[:2, :].T  # Shape (N, 2)
    
    return corners_2d.tolist()

# Get the object to be moved
obj = bpy.data.objects["Robot_tool"]

# Define the ranges for the movements along X, Y, Z
x_range = (-9.7, -8.1)
y_range = (-3.35, -3.5)
z_range = (7.4, 7.7)

# Define the number of random positions
num_positions = 3
# Write to a JSON file
with open("/home/alessia/Desktop/training_dataset.json", "w") as outfile:
    # Iterate through the positions
    for i in range(num_positions):
        # Randomize the location
        obj.location.x = random.uniform(*x_range)
        obj.location.y = random.uniform(*y_range)
        obj.location.z = random.uniform(*z_range)

        # Randomize the rotation
        obj.rotation_euler.x = random.uniform(radians(45),radians(65))
        obj.rotation_euler.y = random.uniform(radians(59),radians(65))
        obj.rotation_euler.z = random.uniform(radians(125), radians(145))

        # Update the scene
        bpy.context.view_layer.update()

        # Define the file paths
        img_file = os.path.join(output_dir, f"render_{i:03d}.png")

        # Render the image
        #bpy.context.scene.render.filepath = img_file
        #bpy.ops.render.render(write_still=True)
        # Save the object data (matrix world and vertices)
        aruco_corners_3d = get_aruco_corners(aruco)

        true_corners_2d = project_to_2d(K_true, RT, aruco_corners_3d)
        #corners_data = {"true": aruco_corners_2d}
        #json.dump(corners_data, outfile)
        

    # Check if the image file exists after rendering
        if os.path.exists(img_file):    
            image = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
            corners, ids, rejected = detector.detectMarkers(image)

            # If a marker was detected
            if len(corners) > 0:
                # Only consider the first detected marker for simplicity
                corners=cv2.cornerSubPix(image,corners[0][0],winSize,zeroZone,criteria)
                detected_corners = corners.tolist()
                    
            data = {
                "image_file": img_file,  # Store the image file path
                "true": true_corners_2d       # Store the 'true' corner positions
            }
            
            # If detected corners are available, add them to the dictionary
            if detected_corners is not None:
                data["detected"] = detected_corners

            # Save to JSON file
            json.dump(data, outfile)
