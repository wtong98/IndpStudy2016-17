#!/usr/bin/python

'''
This python script is the meat of the operation. It builds the neural network
and trains it based on the midi files provided in the "train_data" directory, usesa specified primer-melody to build the tune, and spits out the final music
in matrix form. Please feel free to use this script directly as a starting point
for your project and expand on it however you wish.

Authors:
    Austin Choi (achoi1211@gmail.com)
    William Tong (williamtong105@gmail.com)
Date: 16 May 2017
'''

import numpy as np
import tensorflow as tf
import glob
from tqdm import tqdm

import midi_manipulation

###___(^_^)___IMPORTANT VARIABLES___(^_^)___###

#the max/min length of a sample, measured in sixteenth notes
#all songs truncated to this length
#try to keep it smaller than 75, unless you have oodles of RAM
trunc = 50 

#import directories
song_path = r'./train_data/'
primer_path = r'./primer.mid'
output_path = r'./final_tune.txt'

song_length = trunc 
saver = tf.train.Saver()

###___(^_^)___IMPORTANT METHODS___(^_^)___###

'''
Gets the songs from your "train_data" directory and converts them into
matrices for use in training.

Returns: a list of numpy arrays that represent your songs
'''
def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > trunc:
                    songs.append(song[:trunc])
        except Exception as e:
            print('[WARN] Rejecting song %s' % (f))
            raise e           
    return songs

'''
Simpel utility method that initializes weight variables of your neural
network.

Returns: a TensorFlow variable representing a weight
'''
def weight_variable(shape):
    initial_vals = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_vals)

'''
Basically the same as the previous method except it deals with the bias
variable instead (a variable that's added to input, not multiplied)

Returns: a TensorFlow variable representing a bias variable
'''
def bias_variable(shape):
    initial_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_vals)

'''
Builds a 2-dimensional convolution layer using x inputs and W weights

Returns: a TensorFlow node that represents the convolution layer
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
            strides=[1,1,1,1], #Skip nothing, fully connected
            padding='SAME') #Pads with zeros to match input shape

'''
You can define a custom loss function here. Right now, it's implemented 
as a fairly straightforward mean-square error loss calculation, though
alternative methods may prove to be more effective.

Returns: a TensorFlow node that calculates the loss
'''
def custom_loss(predictions, labels):
    return tf.losses.mean_squared_error(labels, predictions)

#---(^_^)---Building model---(^_^)---#

def build_model():
    print('[INFO] Initializing model...')
    
    #Defining inputs
    x = tf.placeholder(tf.float32)
    #Convolution (kinda useless?)
    W_conv = weight_variable([5,5, #5 by 5 patch size
                              1,   #1 channel input
                              4]) #4 channel output
    b_conv = bias_variable([4])
    x_input = tf.reshape(x, [-1, song_length, 156, 1]) #(batch size, length, width, output channel)
    
    hidden_conv = tf.nn.relu(conv2d(x_input, W_conv) + b_conv)
    flattened = tf.reshape(hidden_conv, [-1, 156 * song_length * 4]) #prep for transport
    W_lulz = weight_variable([156 * song_length * 4, song_length * 156]) #max-pooling around here?
    b_lulz = weight_variable([156 * song_length])
    
    rnn_data = tf.reshape(tf.matmul(flattened, W_lulz) + b_lulz, [-1])
    rnn_labels = tf.reshape(x, [-1])
    
    #Recurrent (assuming training one song at a time)
    lstm_size = 10 #bigger = more accurate?
    batch_size = 156 * 5 #amount to be read each time by LSTM
    
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    state = lstm.zero_state(batch_size, tf.float32)
    outputs = []
    loss = 0.0
    
    chunks = int(156 * song_length / batch_size)
    W_almost = weight_variable([lstm.output_size, 1])
    b_there = weight_variable([1])
    
    with tf.variable_scope("world_of_hurt"):
        for chunk in range(chunks): #may need to wrap in a more tractable loop
            inputs = tf.reshape(rnn_data[chunk * batch_size : (chunk + 1) * batch_size], [batch_size, 1])
            if chunk > 0: #ha, I get it (^_^)
                tf.get_variable_scope().reuse_variables()
            else:
                loss = 0.0
            cell_output, state = lstm(inputs, state)
            #loss calculation (using current chunk to predict next one)
            y = tf.reshape(tf.matmul(cell_output, W_almost) + b_there, [-1]) #not sure how correct this is
            if chunk + 1 < chunks:
                target = rnn_labels[(chunk + 1) * batch_size : (chunk + 2) * batch_size]
                target = tf.cast(target, tf.int32)
                loss += custom_loss(y, target)
            outputs.append(y)
            #(^_^) May need to write own loss function
    
    outputs = tf.reshape(tf.concat(outputs, 0), [-1])
    #Bring on the pain
    adventure_step = tf.train.AdamOptimizer().minimize(loss)

    return outputs, adventure_step

#TODO: make sure everything works okay, and finish up the rest of the magic
#TODO: add a method for recovering saves
def train(step, num_eons, num_repeats, lvl_verbosity, lvl_anxiety):
    #Training (may want to mess around with song-order in training scheme)
    sess = tf.InteractiveSession() #because Session is for noobs
    sess.run(tf.global_variables_initializer())

    print('[INFO] Beginning training')
    eons = 12
    repeats = 100

    #inverse qualities (e.g. higher anxiety -> less saves)
    verbosity = 50 
    anxiety = 3
    songs = get_songs(song_path)

    for eon in range(eons):
        i = 1
        for song in songs:
            for repeat in range(repeats):
                if repeat % verbosity == 0:
                    print('[INFO] (%d of %d) steps in (%d of %d) eons: (%d of %d songs) loss: %f' % 
                            (repeat + 1, repeats, eon + 1, eons, i, len(songs), loss.eval(feed_dict={x:song})))
                sess.run(adventure_step, {x:song})
            i += 1
        if eon % anxiety == 0:
            saver.save(sess, r'saves/nintendo_%d.ckpt' % (eon))
    saver.save(sess, r'saves/nintendo_final.ckpt')
        
#Producing output
primer = np.array(midi_manipulation.midiToNoteStateMatrix(primer_path))
music = np.array([])
if len(primer) > song_length:
    primer = primer[:song_length]
else:
    print('[ERROR] Make sure your primer piece is as least %d units long' % (song_length))
    exit()

sets = 2000 #indictates number of iterations piece is processed

print('[INFO] Generating music')
for _ in range(sets):
    if _ % verbosity == 0:
        print('[INFO] Iteration %d of %d' % (_ + 1, sets))
    music = sess.run(outputs, {x:primer})
    primer = music #I think the right word for this is "jank"
final_tune = np.reshape(music, (-1, 156))

#Writing output columns separated by tabs, rows separated by newlines
output_file = open(output_path, 'w')
output_string = ''

for i in range(final_tune.shape[0]):
    for j in range(final_tune.shape[1]):
        output_string += str(final_tune[i][j]) + '\t'
    output_string += '\n'
output_file.write(output_string)
output_file.close() #whew

print('[INFO] Done!')
