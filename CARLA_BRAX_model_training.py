import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
#import SGD from keras.optimizers
from keras.optimizers import SGD
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.compat.v1.keras.backend import set_session

#GPU LIMIT

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 4 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    tf.config.experimental.set_virtual_device_configuration(
        gpus[1],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#Strategy
STRATEGY = tf.distribute.experimental.MultiWorkerMirroredStrategy()

#Previous model to continue training
#load_previous = None
load_previous = f"/home/juliendo/MODELS/First_model.20201011_1739_npy.testing.h5"

#Variables
MEMORY_FRACTION = 0.6
DATASET = "20201012_1653_npy_used"
DIRECTORY = f"/home/juliendo/TESTDATA/{DATASET}"
WIDTH = 200
HEIGHT = 88
EPOCHS=10
MODEL_NAME = "Xception"
TRAINING_BATCH_SIZE = 16


#Optimizers
OPT = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#OPT = keras.optimizers.Adam(learning_rate=0.001)

#We open the training data
INPUTS_FILE = open(DIRECTORY + "/inputs.npy","br") 
OUTPUTS_FILE = open(DIRECTORY + "/outputs.npy","br")  

#We get the data in
inputs = []
outputs = []

while True:
    try:
        input = np.load(INPUTS_FILE)
        inputs.append(input)
    except: 
        break
while True:
    try:
        output = np.load(OUTPUTS_FILE)
        outputs.append(output)
    except: 
        break

with STRATEGY.scope():
    input_np = np.array(inputs)
    output_np = np.array(outputs)

print(input_np.shape)    
    
#we close everything
inputs = None
outputs = None

INPUTS_FILE.close()
OUTPUTS_FILE.close()

#We take out the first 400 frames to avoid having the car idle
input_np = input_np[400:,:,:]
output_np = output_np[400:,:]

#Let's print some metrics
print("Input Shape")
print(input_np.shape)
print("-------------------------------------------------------------------------------------")
    
print("Output Shape")
print(output_np.shape)
print("-------------------------------------------------------------------------------------")

print("Input min axis 0")
print(input_np.min(axis=0))
print("-------------------------------------------------------------------------------------")

print("Input max axis 0")
print(input_np.max(axis=0))
print("-------------------------------------------------------------------------------------")

print("First line of input")
print(input_np[0])
print("-------------------------------------------------------------------------------------")

print("Output Shape")
print(output_np.shape)
print("-------------------------------------------------------------------------------------")

print("First line of output")
print(output_np[0])
print("-------------------------------------------------------------------------------------")

with STRATEGY.scope():
    x_train, x_test, y_train, y_test = train_test_split(input_np, output_np)
    
    
with STRATEGY.scope():
    if load_previous is not None:
        model = models.load_model(load_previous)
        print(f"loaded model:{load_previous}")
    else:
        
        #Spintronics model
        #model = models.Sequential()
        #model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, 3)))
        #model.add(layers.MaxPooling2D((2, 2)))
        #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        #model.add(layers.MaxPooling2D((2, 2)))
        #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        #model.add(layers.Flatten())
        #model.add(layers.Dense(128, activation='sigmoid'))
        #model.add(layers.Dense(3))
        
        
        #JDO Model
        base_model= Xception(weights=None, include_top=False, input_shape=(HEIGHT, WIDTH,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #3 actions, 3 predictions left, right, straight
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs = base_model.input, outputs = predictions)      
        
model.summary()

with STRATEGY.scope():
    if load_previous is None:
        #model.compile(optimizer=OPT, loss="categorical_crossentropy", metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    my_callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        #tf.keras.callbacks.ModelCheckpoint(filepath='models/model.' + DATASET + '.{epoch:02d}.{val_accuracy:.2f}-{val_loss:.2f}.h5', save_weights_only=False),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    
#my_callbacks = []

with STRATEGY.scope():
    #history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=my_callbacks)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=my_callbacks)
    
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_loss, test_acc)
#model_file = f"models/workshop_model.{DATASET}.{test_acc:.2f}-{test_loss:.2f}.h5"
model_file = f"/home/juliendo/MODELS/First_model.{DATASET}.testing.h5"
model.save(model_file)
print(model_file)
