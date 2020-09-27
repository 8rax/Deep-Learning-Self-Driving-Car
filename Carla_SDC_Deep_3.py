
#https://pythonprogramming.net/reinforcement-learning-agent-self-driving-autonomous-cars-carla-python/
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
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model


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

#Flag Whether we want to show the camera
SHOW_PREVIEW = False
#We know those ones
IM_WIDTH = 640
IM_HEIGHT = 480
#Length of every run
SECONDS_PER_EPISODE=10
REPLAY_MEMORY_SIZE = 5000 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16 # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.8 #how much gpu you wanna use
MIN_REWARD = -200
EPISODES = 100
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10


#very step an agent takes comes not only with a plausible prediction (depending on exploration/epsilon), but also with a fitment! This means that we're training and predicting at 
#the same time, but it's pretty essential that our agent gets the most FPS (frames per second) possible. We'd also like our trainer to go as quick as possible. To achieve this, we 
#can use either multiprocessing, or threading. Threading allows us to keep things still fairly simple. Later, I will likely at least open-source some multiprocessing code for this task at some point, but, for now, threading it is.


class CarEnv: 
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None 

    def __init__(self):
        self.client = carla.Client("localhost",2000)
        self.client = set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb.set_attribute("fov",f"110")
        
        transform=carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor= self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: process_img(data))

        #Just to make it start recording, apparently passing an empty command makes it react
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        time.sleep(4)

        colsensor=self.blueprint_library.find("sensor.other.collision")
        self.colsensor=self.world.spawn_actor(colsensor,transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_date(event))

        while self.front_camera is None: 
            time.sleep(0.01)

        #Everything is set, we can start the episode

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)

    #Function to flatten the image retrieved by the sensor
    def process_img(image):
        i = np.array(image.raw_data)
        #RGBA = 4 (get the alpha)
        #reshape the image to get a picture
        i2= i.reshape((self.im_height,self.im_width, 4))
        #take everything but only the RGB not the Alpha
        i3=i2[:,:,:3]
        if self.SHOW_CAM:
            #Show the image
            cv2.imshow("",i3)
            cv2.waitKey(1)
        #Normalize the data
        self.front_camera=i3

    #Go left, straight, right
    def step(self,action):
        if action==0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action==1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action ==2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v=self.vehicle.get_velocity()
        #Convert velocity into km/h

        kmh=int(3.6*math.sqrt(v.x**2+v.y**2+v.z**2))

        #if there has been a collision
        if len(self.collision_hist) !=0:
            done=True
            #Penalty
            reward=-200
        elif kmh < 50:
            done = False
            #Small penalty if we go > 50
            reward = -1
        else:
            done=False
            reward=1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done=True

        return self.front_camera, reward, done, None


class DQNAgent:

    def __init__(self):
        self.model = self.create_model()

        #This is the same concept as from the reinforcement learning tutorials, where we have a main network, which is constantly evolving, and then the 
        #target network, which we update every n things, where n is whatever you want and things is something like steps or episodes.
        #Target model this is what we .predict against every step
        #Here, you can see there are apparently two models: self.model and self.target_model. What's going on here? So every step we take, we want to update Q values, but we also are
        #trying to predict from our model. Especially initially, our model is starting off as random, and it's being updated every single step, per every single episode. 
        #What ensues here are massive fluctuations that are super confusing to our model. This is why we almost always train neural networks with batches (that and the time-savings).
        #One way this is solved is through a concept of memory replay, whereby we actually have two models.

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights()) #one model is trained, the other is kept same for x episodes to predict against and then update every x episodes


        #Along these lines, we have a variable here called replay_memory. Replay memory is yet another way that we attempt to keep some sanity in a model that is 
        #getting trained every single step of an episode. We still have the issue of training/fitting a model on one sample of data. This is still a problem with neural networks. 
        #Thus, we're instead going to maintain a sort of "memory" for our agent. In our case, we'll remember 1000 previous actions, and then we will fit our model on a random selection 
        #of these previous 1000 actions. This helps to "smooth out" some of the crazy fluctuations that we'd otherwise be seeing. Like our target_model, we'll get a better idea of what's 
        #going on here when we actually get to the part of the code that deals with this I think.


        #As we train, we train from randomly selected data from our replay memory:
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #Memory of previous actions

        #Reporting of the metrics
        #For the same reasons as before (the RL tutorial), we will be modifying TensorBoard: -> we do not want it to create a log at each episode
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # will track when it's time to update the target model
        self.graph = tf.get_default_graph()

        self.terminate = False # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False # waiting for TF to get rolling

        #We're going to use elf.training_initialized to track when TensorFlow is ready to get going. The first predictions/fitments when a model begins take extra long, so we're going to just pass some 
        #nonsense information initially to prime our model to actually get moving.

    def create_model(self):
        base_model= Xception(weights=None, include_top=False, input_shape(IM_HEIGHT, IM_WIDTH,3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        #3 actions, 3 predictions left, right, straight
        predictions = Dense(3, activation="linera")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

        #Here, we're just going to make use of the premade Xception model, but you could make some other model, or import a different one. 
        #Note that we're adding GlobalAveragePooling to our ouput layer, as well as obviously adding the 3 neuron output that is each possible action for the agent to take. 

    def update_replay_memory(self, transition):
        # All the information to train the models and upate the qs
        # transition = (current_state, action, reward, new_state, done)
        #We need a quick method in our DQNAgent for updating replay memory:
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):
        #We need enough memory to actually train and not just do random stuff over and over again with high epsilon
        #To begin, we only want to train if we have a bare minimum of samples in replay memory:
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #If we don't have enough samples, we'll just return and be done. If we do, then we will begin our training. First, we need to grab a random minibatch:
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        #Once we have our minibatch, we want to grab our current and future q values
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255 

        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255 

        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        y = []

        X = []
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):


            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q

            else:
                new_q = reward 

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q


            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        log_this_step = False

        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # Fit on all samples as one batch, log only on terminal state

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

            #Notice that we're setting the tensorboard callback, only if log_this_step is true. If it's false, then we'll still fit, we just wont log to TensorBoard.

        #Next, we want to continue tracking for logging:
        if log_this_step:
            self.target_update_counter +=1

        #Finally, we'll check to see if it's time to update our target_model:
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.tarfet_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #First, we need a method to get q values (basically to make a prediction)
        return self.model.predict(np.array(state).reshape(-1 * state.shape)/255)[0]

    def train_in_loop(self):
        #Finally, we just need to actually do training:
        X = np.random.uniform(size = (1, IM_HEIGHT, IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size=(1,3)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X,y,verbose=False,batch_size=1)

        self.training_initialized = True
        #To start, we use some random data like above to initialize, then we begin our infinite loop:
        while True:
            if self.terminate:
                return

            self.train()
            time.sleep(0.01)
