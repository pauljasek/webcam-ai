# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:26:29 2016

@author: Paul Jasek
"""

import tensorflow as tf, numpy as np, cv2, random, functools
from time import sleep

SHOW_IMAGE = True
SHOW_FILTERS = True
KEY_INPUT = False

STATE_SIZE = 256
OBSERVATION_SIZE = STATE_SIZE
TRAIN_TIME = 1000 # 1000

INPUT_WIDTH = 80 * 1 #80
INPUT_HEIGHT = 60 * 1 #60

images = []
generated_images = []
MAX_IMAGES_LENGTH = 3000
MAX_GENERATED_IMAGES_LENGTH = 100

RECENT_IN_BATCH = 1
OLD_IN_BATCH = 9
BATCH_SIZE = RECENT_IN_BATCH + OLD_IN_BATCH
TIME_SIZE = 15

LOOK_AHEAD = TIME_SIZE - 1
SIMILARITY_CHECK_STEPS = 50

MAIN_ACTIVATION = tf.nn.relu

def linear(x):
    return x

def get_reshape(shape):
    return list(map(lambda x: -1 if x == None else x, shape))

def base_shape(shape):
    return list(map(lambda x: 1 if x == -1 or x == None else x, shape))

class Conv():
    def __init__(self, activation, i, size=(5,5), num_channels=3, num_filters=32, strides=1):
        self.conv_filter = tf.Variable(tf.truncated_normal([size[0], size[1], num_channels, num_filters], stddev=0.1))
        self.conv_bias = tf.Variable(tf.truncated_normal([num_filters], stddev=0.1))
        
        self.activation = activation
        self.strides = strides
        self.num_channels = num_channels
        self.num_filters = num_filters
        
        self.out = tf.nn.conv2d(i, self.conv_filter, strides=[1,strides,strides,1], padding='SAME')
        self.out = activation(tf.nn.bias_add(self.out, self.conv_bias))
    
    def apply(self, i):
        out = tf.nn.conv2d(i, self.conv_filter, strides=[1,self.strides,self.strides,1], padding='SAME')
        return self.activation(tf.nn.bias_add(out, self.conv_bias))

    def get_shape(self):
        return self.out.get_shape()
    
    def get_reshape(self):
        return list(map(lambda x: -1 if x == None else x, self.out.get_shape().as_list()))
    
    def get_base_shape(self):
        return list(map(lambda x: 1 if x == None else x, self.out.get_shape().as_list()))

 
class Deconv():
    def __init__(self, activation, i, size, output_shape, strides):
        self.input_shape = i.get_shape().as_list()
        self.deconv_filter = tf.Variable(tf.truncated_normal([size[0], size[1], output_shape[3], self.input_shape[3]], stddev=0.1))    
        self.deconv_bias = tf.Variable(tf.truncated_normal(base_shape(output_shape), stddev=0.1))
        
        self.activation = activation
        self.strides = strides
        self.output_shape = output_shape
        
        self.batch_size = tf.shape(i)[0]
        self.deconv_shape = tf.stack([self.batch_size, output_shape[1], output_shape[2], output_shape[3]])
        
        self.out = self.apply(i)
    
    def apply(self, i):
        self.batch_size = tf.shape(i)[0]
        self.deconv_shape = tf.stack([self.batch_size, self.output_shape[1], self.output_shape[2], self.output_shape[3]])
        return self.activation(tf.nn.conv2d_transpose(i, self.deconv_filter, self.deconv_shape, strides=[1,self.strides,self.strides,1], padding='SAME') + self.deconv_bias)

    def get_shape(self):
        return self.out.get_shape()

class Fc():
    def __init__(self, activation, i, num_units):
        self.weights_shape = [base_shape(i.get_shape().as_list())[1], num_units]
        self.weights = tf.Variable(tf.truncated_normal(self.weights_shape, stddev=0.1))
        self.biases = tf.Variable(tf.truncated_normal([num_units], stddev=0.1))
        
        self.activation = activation
        
        self.out = activation(tf.matmul(i, self.weights) + self.biases)
    
    def apply(self,i):
        return self.activation(tf.matmul(i, self.weights) + self.biases)
    
    def get_shape(self):
        return self.out.get_shape()

input_images = tf.placeholder(tf.float32, [None,None,INPUT_HEIGHT,INPUT_WIDTH,3])

input_shape = tf.shape(input_images)
batch_size = input_shape[0]
time_size = input_shape[1]

reshaped_input_images = tf.reshape(input_images, [-1,INPUT_HEIGHT,INPUT_WIDTH,3])
conv1 = Conv(MAIN_ACTIVATION, reshaped_input_images, size=(7,7), num_channels=3, num_filters=16, strides=4)
conv2 = Conv(MAIN_ACTIVATION, conv1.out, size=(5,5), num_channels=16, num_filters=24, strides=2)
conv3 = Conv(MAIN_ACTIVATION, conv2.out, size=(3,3), num_channels=24, num_filters=32, strides=2)
conv4 = Conv(MAIN_ACTIVATION, conv3.out, size=(3,3), num_channels=32, num_filters=48, strides=1)

print(input_images.get_shape())
print(conv1.get_shape())
print(conv2.get_shape())
print(conv3.get_shape())
print(conv4.get_shape())
final_size = functools.reduce(lambda x,y: x*y, filter(lambda x: x != None, conv4.get_shape().as_list()))
print(final_size)

#observation = Fc(tf.nn.relu, tf.reshape(conv3.out, [1,-1]),STATE_SIZE)
#previous_state = tf.placeholder(tf.float32, [1, STATE_SIZE])

#concatenated = tf.concat(1, [observation.out, previous_state])
#fc1 = Fc(tf.nn.relu, concatenated, int(STATE_SIZE * 3/2))
#gates_branch = Fc(tf.nn.relu, fc1.out, STATE_SIZE)
#forget_gates = Fc(tf.nn.sigmoid, gates_branch.out, STATE_SIZE)
#input_gates = Fc(tf.nn.sigmoid, gates_branch.out, STATE_SIZE)
#inputs = Fc(tf.nn.tanh, fc1.out, STATE_SIZE)

#state = tf.mul(previous_state, forget_gates.out) + tf.mul(input_gates.out, inputs.out)

#fc2 = Fc(tf.nn.relu, state, STATE_SIZE * 2)
#fc3 = Fc(tf.nn.relu, fc2.out, STATE_SIZE * 2)

#next_state = Fc(linear, fc3.out, STATE_SIZE)

#start = Fc(tf.nn.relu, state, 3328)

observation = Fc(MAIN_ACTIVATION, tf.reshape(conv4.out, [-1, final_size]),STATE_SIZE)
observations = tf.reshape(observation.out, [batch_size, time_size, STATE_SIZE])

gru = tf.contrib.rnn.GRUCell(STATE_SIZE)

state = tf.placeholder(tf.float32, [None, STATE_SIZE])

#output, output_state = gru(observations, state)
output, output_state = tf.nn.dynamic_rnn(gru, observations, time_major=False, initial_state=state)

print(output.get_shape())

output_reshaped = tf.reshape(output, [-1, STATE_SIZE])


start = Fc(MAIN_ACTIVATION, output_reshaped, final_size)

#reshaped_start = tf.reshape(start.out, conv4.get_shape().as_list())
reshaped_start = tf.reshape(start.out, conv4.get_reshape())

deconv1 = Deconv(MAIN_ACTIVATION, reshaped_start, (5,5), conv3.get_shape().as_list(), 1)
deconv2 = Deconv(MAIN_ACTIVATION, deconv1.out, (5,5), conv2.get_shape().as_list(), 2)
deconv3 = Deconv(MAIN_ACTIVATION, deconv2.out, (7,7), conv1.get_shape().as_list(), 2)
autoencoder_output = Deconv(tf.nn.sigmoid, deconv3.out, (9,9), reshaped_input_images.get_shape().as_list(), 4)

truth_images = tf.placeholder(tf.float32, [None,None,INPUT_HEIGHT,INPUT_WIDTH,3])
truth_images_reshaped = tf.reshape(truth_images, [batch_size*time_size,INPUT_HEIGHT,INPUT_WIDTH,3])

autoencoder_loss = 0.5 * tf.reduce_sum(tf.square(truth_images_reshaped - autoencoder_output.out))
train_autoencoder = tf.train.AdamOptimizer(learning_rate=0.001).minimize(autoencoder_loss)

fc1 = Fc(MAIN_ACTIVATION, output_reshaped, STATE_SIZE * 2)
fc2 = Fc(MAIN_ACTIVATION, fc1.out, STATE_SIZE * 2)

predicted_state = Fc(tf.tanh, fc2.out, STATE_SIZE)

first_states = tf.reshape(tf.slice(output, [0,0,0], [-1, 1, STATE_SIZE]), [-1,STATE_SIZE])

look_ahead_predictions = [predicted_state.apply(fc2.apply(fc1.apply(first_states)))]
for i in range(LOOK_AHEAD-1):
    look_ahead_predictions.append(predicted_state.apply(fc2.apply(fc1.apply(look_ahead_predictions[i]))))
look_ahead_predictions = tf.stack(look_ahead_predictions, axis=1)

print(look_ahead_predictions.get_shape())

truth_predictions = tf.stop_gradient(tf.slice(output, [0,1,0], [-1, LOOK_AHEAD, STATE_SIZE]))

similarity_checks = [predicted_state.apply(fc2.apply(fc1.apply(first_states)))]
for i in range(SIMILARITY_CHECK_STEPS-1):
    similarity_checks.append(predicted_state.apply(fc2.apply(fc1.apply(similarity_checks[i]))))
similarity_checks = tf.stack(similarity_checks, axis=1)

#prediction_difference = tf.reduce_sum(tf.square(tf.slice(similarity_checks,[0,1,0],[-1,SIMILARITY_CHECK_STEPS-1,STATE_SIZE]) - tf.stop_gradient(tf.slice(similarity_checks, [0,0,0], [-1,SIMILARITY_CHECK_STEPS-1,STATE_SIZE]))))
_, deviation_in_predictions = tf.nn.moments(tf.reshape(similarity_checks, [-1,STATE_SIZE]), axes=[1])

next_state = tf.placeholder(tf.float32, [None, None, STATE_SIZE])

#pstate_reshaped = tf.reshape(output, [batch_size*time_size, STATE_SIZE])

pstart = start.apply(predicted_state.out)
preshaped_start = tf.reshape(pstart, conv4.get_reshape())

pdeconv1 = deconv1.apply(preshaped_start)
pdeconv2 = deconv2.apply(pdeconv1)
pdeconv3 = deconv3.apply(pdeconv2)
pautoencoder_output = autoencoder_output.apply(pdeconv3)

prediction_loss = tf.reduce_sum(tf.square(tf.reshape(next_state, [-1, STATE_SIZE]) - predicted_state.out))
train_prediction = tf.train.AdamOptimizer(learning_rate=0.001).minimize(prediction_loss)

look_ahead_loss = tf.reduce_sum(tf.square(truth_predictions - look_ahead_predictions))
similarity_loss = -tf.reduce_sum(deviation_in_predictions) #1.0/(prediction_difference + .0000001)

combined_train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(autoencoder_loss + prediction_loss + look_ahead_loss + similarity_loss)


previous_state = tf.placeholder(tf.float32, [None, STATE_SIZE])
imagined_image = autoencoder_output.apply(deconv3.apply(deconv2.apply(deconv1.apply(
        tf.reshape(start.apply(previous_state), conv4.get_reshape())))))
predicted_next_state = predicted_state.apply(fc2.apply(fc1.apply(previous_state)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

vc = cv2.VideoCapture(0) #("star_wars_lightsaber_duel_trimmed.mp4")

time = -1

batch_zero_state = np.zeros([BATCH_SIZE, STATE_SIZE], np.float32)
past_zero_state = np.zeros([OLD_IN_BATCH, STATE_SIZE], np.float32)
recent_zero_state = np.zeros([RECENT_IN_BATCH, STATE_SIZE], np.float32)
zero_state = np.zeros([1, STATE_SIZE], np.float32)

gru_state = np.zeros([1, STATE_SIZE], np.float32)
imagination_state = gru_state.copy()

while True:
    time += 1
    
    for i in range(1):
        ret, frame = vc.read()
    
    image = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)) / 255.
    
    images.append(image)
    while len(images) > MAX_IMAGES_LENGTH:
        images.pop(0)
        
    
    if False and SHOW_FILTERS:
        out_filters = sess.run(conv3.out, feed_dict={input_image: [image]}) 
        display_filter = out_filters[0]
        display_filter = display_filter[:,:,0]
        display_filter /= np.amax(display_filter)
        filter_shape = display_filter.shape
        display_filter = cv2.resize(display_filter, (filter_shape[0] * 16, filter_shape[1] * 16))
        cv2.imshow('Filter', display_filter)
    
    if KEY_INPUT:
        key = cv2.waitKey(1)
        
        hand_open = key == ord('1')
        hand_closed = key == ord('0')

    generated_image, gru_state, predicted_image = sess.run([autoencoder_output.out, output, pautoencoder_output], feed_dict={input_images: [[images[len(images)-1]]], state: gru_state})
    
    gru_state = gru_state[0]
    print(gru_state)
    
    generated_image = generated_image[len(generated_image) - 1]
    predicted_image = predicted_image[len(predicted_image) - 1]
    
    generated_images.append(generated_image)
    while len(generated_images) > MAX_GENERATED_IMAGES_LENGTH:
        generated_images.pop(0)
    
    display_generated = cv2.resize(generated_image, (640,480))
    display_predicted = cv2.resize(predicted_image, (640,480))
    
    
    if time < TRAIN_TIME and time > BATCH_SIZE and time > TIME_SIZE:
        print(time);
        image_batch = []
        
        for i in range(RECENT_IN_BATCH):
            index = len(images) - i
            image_sequence = images[index-TIME_SIZE:index]
            image_batch.append(image_sequence)

        for i in range(OLD_IN_BATCH):
            random_index = random.randint(TIME_SIZE, len(images)-1)
            #image_batch.append(images[random_index-TIME_SIZE:random_index])
            image_sequence = images[random_index-TIME_SIZE:random_index]
            image_batch.append(image_sequence)
        
        
        target_states = sess.run(output, feed_dict={input_images: image_batch, state: batch_zero_state})
        for t in target_states:
            t = t[1:]
        
        for i in image_batch:
            i = i[:len(i)-1]

        sess.run(combined_train, feed_dict={input_images: image_batch, truth_images: image_batch, state: batch_zero_state, next_state: target_states})
        
    elif time == TRAIN_TIME:
        #vc = cv2.VideoCapture(0)
        sleep(0.1)
    
    imagination_state, imagined = sess.run([predicted_next_state, imagined_image], feed_dict={previous_state: imagination_state})
    imagined = imagined[0]
    
    if (time % 50 == 0):
        imagination_state = gru_state.copy()
    
    cv2.imshow('Predicted Image Stream', cv2.resize(imagined, (640,480)))
    cv2.imshow('Generated Image', display_generated)
    cv2.imshow('Predicted Image', display_predicted)
    
    if SHOW_IMAGE:
        display_image = cv2.resize(image, (640,480))
        cv2.imshow('Web Cam Feed', display_image)

    cv2.waitKey(1)

cv2.destroyAllWindows()
