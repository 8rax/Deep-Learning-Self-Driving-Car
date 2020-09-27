# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import random
import time
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

#Parameters for sensor image that we will retrive
IM_WIDTH=640
IM_HEIGHT=480

#Function to flatten the image retrieved by the sensor
def process_img(image):
    i = np.array(image.raw_data)
    #RGBA = 4 (get the alpha)
    #reshape the image to get a picture
    i2= i.reshape((IM_HEIGHT,IM_WIDTH, 4))
    #take everything but only the RGB not the Alpha
    i3=i2[:,:,:3]
    #Show the image
    cv2.imshow("",i3)
    cv2.waitKey(1)
    #Normalize the data
    return i3/255.0

#create empty list of actors
actor_list=[]

try: 
    #The client is the module the user runs to ask for information or changes in the simulation. A client runs with an IP and a specific port. It communicates with the server via terminal.
    client=carla.Client("localhost",2000)
    #connect to carla environment
    client.set_timeout(2.0)
    #The world is an object representing the simulation. It acts as an abstract layer containing the main methods to spawn actors, change the weather, 
    #get the current state of the world, etc. There is only one world per simulation. It will be destroyed and substituted for a new one when the map is changed.
    world=client.get_world()
    #Returns the blueprints(attributes)
    #An actor is anything that plays a role in the simulation (Vehicles, Walkers, Sensors,The spectator)

    #Blueprints are already-made actor layouts necessary to spawn an actor. Basically, models with animations and a set of attributes. 
    #Some of these attributes can be customized by the user, others don't.

    blueprint_library=world.get_blueprint_library()

    #We search for the Tesla Model 3 blueprint and take the index 0
    bp=blueprint_library.filter("model3")[0]
    print(bp)

    #We select a random spawn point from the map
    spawn_point=random.choice(world.get_map().get_spawn_points())

    #We spawn the Model S3 vehicle at the selected random spawn point
    vehicle=world.spawn_actor(bp, spawn_point)

    #We make the vehicle go staight
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    #Finally, let's not forget to add this vehicle to our list of actors that we need to track and clean up:
    actor_list.append(vehicle)

    #Setup blueprint of sensor camera and set attribute concerning image size and field of view
    cam_bp=blueprint_library.find("sensor.camera.rgb")
    # change the dimensions of the image
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov","110")

    #Now we will have to spawn the sensor on the hood of the car, those locations below are adapted for the TESLA model 3 but have to be changed for other types of cars
    spawn_point=carla.Transform(carla.Location(x=2.5, z=0.7))

    #Spawn the blueprint of the sensor, at the selected spawn point ON the attached vehicle
    sensor=world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    #Append to actor list
    actor_list.append(sensor)

    #Get gata from the sensor that we created
    sensor.listen(lambda data: process_img(data))

    #We let it run for 15 seconds
    time.sleep(15)

finally: 
    #We kill all the actors
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
