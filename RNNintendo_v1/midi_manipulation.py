#!/usr/bin/python
"""
This script was borrowed from a guy's blog, adapted to convert midi files into the
note-state matrices read by compose.py. Unfortunately, I've long since forgotten who or where the
script came from, so many apologies!

Author:
    A really clever dude on the Internet

Date: Some time in the distant, nebulous past
"""

import midi
import numpy as np

lowerBound = 24  # the lowest midi note to be read (dictates size of your matrix)
upperBound = 102  # the highest midi note to be read (also dictates size of your matrix)
span = upperBound-lowerBound
crevice = 10  # changes the matrix to use 10's instead of 1's to indicate note events.
              # I suspect it doesn't actually improve the model that much.

def midiToNoteStateMatrix(midifile, squash=True, span=span):
    """
    Converts the midi file to the note state matrix

    :param midifile: path to your midi file
    :param squash: I have no idea. Give it your best guess.
    :param span: difference between the highest and lowest notes
    :return: numpy array representing your matrix
    """
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [crevice, crevice]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        out =  statematrix
                        condition = False
                        break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix

def noteStateMatrixToMidi(statematrix, name="example", span=span):
    """
    Converts a note state matrix back into a midi file

    :param statematrix: numpy array representing your matrix
    :param name: name to be given to your midi file
    :param span: difference between the highest and lowest notes
    :return: nothing. Writes midi file to disk, using the name you gave it.
    """
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == crevice:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == crevice:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == crevice:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)
