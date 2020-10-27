#Dependencies
import glob
import os
import sys
import time
import numpy as np
import carla
from IPython.display import display, clear_output
import logging
import random
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Creating a client
client = carla.Client("127.0.0.1", 2000)
#client = carla.Client("52.232.96.45", 2000)
client.set_timeout(10.0)
client.reload_world()
for mapName in client.get_available_maps():
    print(mapName)

ego_vehicle = None
ego_cam = None
world = client.get_world()
# world.set_weather(carla.WeatherParameters())
today = datetime.now()
if today.hour < 10:
    h = "0"+ str(today.hour)
else:
    h = str(today.hour)
if today.minute < 10:
    m = "0"+str(today.minute)
else:
    m = str(today.minute)
#directory = "data/" + today.strftime('%Y%m%d_')+ h + m + "_npy"
directory = "/home/juliendo/TESTDATA/" + today.strftime('%Y%m%d_')+ h + m + "_npy"
WIDTH = 200
HEIGHT = 88
print(directory)

try:
    os.makedirs(directory)
except:
    print("Directory already exists")
try:
    inputs_file = open(directory + "/inputs.npy","ba+") 
    outputs_file = open(directory + "/outputs.npy","ba+")     
except:
    print("Files could not be opened")
    
#Spawn vehicle
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name','ego')
ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color',ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
    print('\nVehicle spawned')
else: 
    logging.warning('Could not found any spawn points')
     
#Adding a RGB camera sensor
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute("image_size_x",str(WIDTH))
cam_bp.set_attribute("image_size_y",str(HEIGHT))
cam_bp.set_attribute("fov",str(105))
cam_location = carla.Location(2,0,1)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location,cam_rotation)
ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

#Function to convert image to a numpy array
def process_image(image):
    #Get raw image in 8bit format
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #Reshape image to RGBA
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))
    #Taking only RGB
    processed_image = raw_image[:, :, :3]/255
    return processed_image

#Save required data
def save_image(carla_image):
    image = process_image(carla_image)
    ego_control = ego_vehicle.get_control()
    data = [ego_control.steer, ego_control.throttle, ego_control.brake]
    np.save(inputs_file, image)
    np.save(outputs_file, data)
     
#enable auto pilot
ego_vehicle.set_autopilot(True)

#Attach event listeners
ego_cam.listen(save_image)

try:
    i = 0
    while i < 25000:
        world_snapshot = world.wait_for_tick()
        clear_output(wait=True)
        display(f"{str(i)} frames saved")
        i += 1
except:
    print('\nSimulation error.')
        
if ego_vehicle is not None:
    if ego_cam is not None:
        ego_cam.stop()
        ego_cam.destroy()
    ego_vehicle.destroy()
inputs_file.close()
outputs_file.close()
print("Data retrieval finished")
print(directory)