#!/usr/bin/python

'''
Converts musical matrix into midi file

Authors:
    Austin Choi (achoi@imsa.edu)
    William Tong (wtong@imsa.edu)
Date: 5 March 2017
'''

import numpy as np
import midi_manipulation

input_path = r'final_tune.txt'
threshold = 1.23
name = 'awesomeness'

music_file = open(input_path)
raw_data = music_file.readlines()
data = np.zeros(shape=(len(raw_data), 156))

for r in range(len(raw_data)):
    row = raw_data[r].split('\t')
    for c in range(156):
        if float(row[c]) > threshold:
            data[r][c] = 1
            print(row[c])
        else:
            data[r][c] = 0
midi_manipulation.noteStateMatrixToMidi(data, name=name)
