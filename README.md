#Let us quest

## Introduction
Welcome, brave adventurers! Neural nets are a massive component of modern computation, so
it's great you're taking the initiative and getting your feet wet in this independent study.

There's a lot of information here, and there's a load of tiny, intricate moving parts when
it comes to designing a network, so remember to take it slow, learn carefully, and don't
forget about college apps. I recommend reading/scanning through this entire README before
beginning, but if you're eager, feel free to dive right into the code.

## Setting up your machine
Step 0 before you do anything is to make sure your laptop is set up properly with all
the technical goodies you'll need. If you're using Windows, argh!, I recommend
dual-booting with Linux. Not only is GNU/Linux vastly superior to Windows in so many
different ways, it will make installing the software you'll need loads easier, too. For
an easy distribution to start with, I recommend Linux Mint. Full instructions for
its installation are readably available via Google, and if you get stuck, try consulting
with Mike Xu to get everything straigtened out.

Ultimately, here's what you'll need:
<ul>
<li> `Python`: preferably Python 3, but Python 2 will probably work okay if you're on a 
Linux machine (another reason to switch) You'll also need some additional modules listed
below</li>
<ul>
<li> `numpy`: a python module designed for numerical computation </li>
<li> `tqdm`: for attractive loading bars </li>
<li> `scipy`: not, strictly speaking, neccesary, but it's got loads of useful computation </li>
utilities that you might want to try</li>
<li> `matplotlib`: for making pretty plots</li>
</ul>
<li> `TensorFlow`: the neural network framework this project is built on </li>
<li> `Jupyter Notebook`: software that adds Mathematic-style editing to python
</ul>

If you've got python installed, just about all this software can be installed using pip (e.g.
<code>pip install [PackageName]</code>). But especially if you're using Windows + TensorFlow, check
the official instructions online for installing the framework.

## Running the Code
All the files necessary to run the neural net are located in the directory RNNintendo_v1. To do
a simple run with all our current settings intact, navigate into the RNNintendo_v1 directory
and copy all your midi files into the "train_data" directory. Keep ahold of one of your training
samples to use as a primer melody, and rename it to "primer.mid" Assuming you have all
the neccesary software installed, you can then type

`python compose.py`

and off it goes processing all sorts of unbelievably complex calculations, doing the kinds of 
stuff the human brain takes 3 years to learn in just a few days or weeks. When that's finished
and you have a `final_tune.txt` files in the directory, type

`jupyter notebook threshold_finder.ipynb`

which will produce a graph for you at the very bottom along with a recommended value to set as a
threshold. Now open the file `perform.py` and edit the threshold value in that file
accordingly. When you're satisfied, go ahead and type

`python perform.py`

and viola! You have you're very first midi sample of generated music. The `perform.py` script
will also spit out the values of the note-state matrix that it's reading as nonzero.

## Specifics about the code
First off, let me just put it out there, our code is pretty beastly and violates loads of
good coding practices. That being said, we've tried to clean it up a-plenty and make it nice
and pretty and well-documented for you guys. There's plenty of explanations in the comments,
but all the same, we're on the butt-end of senior year and (extremely) ready to graduate,
so if there's anything particularly gruesome or nonsensical, please don't hesitate to reach
out (our emails are at the top of `compose.py`)

First off, if you take a peek into the directories, `samples` contains examples pieces of midi
or note-state matrices produced in a run of our neural network. `primer.mid` is an example midi-clip
used to primer our network. `awesomeness.mid` is an example generated midi file. `final_tune.txt`
is an example note-state matrix depecting a song, with each line corresponding to a row, each column
separated by tab.

A "note-state matrix" itself is the numerical representation of a song in a form that our
network can understand. Essentially, it's composed of a matrix with *t* rows, each row
corresponding to a time-step that's the equivalent of one sixteenth note long, and *2n* columns,
where each *n* represents an arbitrarily defined note spaced one half-note apart from adjacent
values. A note-state matrix is meant to be binary, with a 0 corresponding to nothing happening
and a 1 corresponding to either a note-on (if the 1 occurs in the first *n* columns)
or note-off event (if it happens in the last *n* columns). For the original explanation of the
matrix, check out this [link](https://medium.com/@oktaybahceci/generate-music-with-tensorflow-midis-4bf928a35c3a)

For our note-state matrices in particular, we made a couple modifications so that, instead of reading
a 1 as a trigger, it currently reads a 4 instead. The hope was to allow the network to better
distinguish a note-on from a note-off event, though our efforst don't seemed to have panned out
quite the way we wanted them to. More on that later.

If you take a look at the `RNNintendo_v1` directory, here you'll find all the essential code for
training a running the beast. The single-most important script here, and the one that functions as
the brains of the whole operation, is `compose.py`. This script is a real beast that basically takes
a convolution layer and plops it right on top of an RNN, the idea being to make our network
note and time intransient (meaning that the network won't care about time-shifts or key signatures).
The method in which it does the training and generating is a little different than the ways followed
by GRUV and Magenta. Both of these utilities take a starting melody, then effectively use a system
of probabilities to figure out the next notes. For us, we started out aiming for that as well, but
by a happy accident, found out we had accidently built something that works much more like 
Inceptionism. Instead of going note by note, the network will chunk out a portion of the music, and
train/generate/enhance that portion of music recurrently before moving on. I've been tinkering a
great deal with the code trying to make it more usable, so hopefully everything still works okay,
but in case I accidently broke something, shoot me an email pronto and I'll try to get it patched
back up.

Next up, threshold_finder.ipynb is an iPython notebook that you can only view via a Jupyter server.
Assuming you've also the prerequesite software, you need only type in
`jupyter notebook threshold_finder.ipynb` and away it goes. iPython makes use of mathematica-like
functionality, so you can execute each cell of code independently by pressing shift-enter. The
point of this script is to provide a way to gauge where a "threshold" value for determining
a note-on/off event. Ideally, you'd be able to look at a note-state matrix, pick out the nonzero
values, and say that these are your note-on/off events. But the problem with a machine-learning
generated output is that it spits out results with a load of decimals. Therefore, use this script
to look at your distribution of values. Hopefully, it will have two peaks, one clumped around
0, the other clumped around whatever nonzero value you elected to be your triggering value.
The sample value provided by the script above the graph is a suggested starting point for
choosing a "threshold value," but you can go either higher or lower and see what kinds of results
you get.

Speaking of which, to actually generated these results, turn now to `perform.py`. The purpose of
this script is to take the note-state matrix produced by `compose.py`, the threshold_value you
hardcode manually into the script, and generate a (hopefully) awesome sounding midi. The script
will also spit out a list of all the values that fall above your threshold value. You'll want
this list to be longish, but not so long as to overwhelm your outputs with a barrage of notes and
, of coure, not so short that there's barely any music at all.

And finally, the midi_manipulation.py\* scripts are there as utilities for dealing with note-state
matrices. To modify what nonzero value counts as a "trigger," you'll have to dig in the .py script
and change it. The .pyc script is a compiled version of the original that python produces whenever
its imported.

The `saves` and `train_data` directories are important folders relied upon by compose.py to save
TensorFlow checkpoints to and draw training data from, respectively. A word of warning, I think
the saving mechanism right now is somewhat broken. And at any rate, it takes up a *ridong-culous*
amount of hard-drive space, so consider disabling the feature altogether and hope your machines
don't crash.

## Roadmap
Here are some potential areas you guys can look into for improving the system. If you have
your own plans, feel free to persue them. Please don't feel obligated to follow the below
recommendations.

<ul>
<li><b>Max-pooling:</b> In most (if not all) enterprise-grade convolutional neural networks, a layer
of max-pooling always seems to follow a layer of convolution, apparently as a means of aggregating
data and "looking at the bigger picture." Thus, max-pooling appears to be vital to the process of
achieving note and time intransience whenever convolution is used, though we never took the time to
understand it and implemnt it properly. Colah's blog may be a good place to start with this,</li>
<li><b>Dropout:</b> To prevent oversaturing neurons with input, most networks implement "dropout,"
where values are apparently dropped at random. Similar to max-pooling, dropout seems to be a
vital part to a healthy neural network, but we never bothered to research it further. The TensorFlow
tutorial may be a good place to start with this.</li>
<li><b>Gated Recurrent Unit:</b> Gated Recurrent Units (GRU) are a simpler version of the traditional
Long-Short Term Memory (LSTM) cells used in the network. GRU's are becoming more popular for reasons
beyond me, but they might be worth checking out. Consult Colah's blog for a nice introduction.</li>
<li><b>Multi-layered RNN's:</b> A neat feature that's possible to do (relatively) easily with
TensorFlow is to stack multiple RNN's on top of each other. Perhaps doing so might improve the
network's ability to recognize more subtle patterns in the music. Check the TensorFlow tutorials for
further guidance on this topic.</li>
<li><b>Magenta's MIDI interface:</b> Magenta is a project built on top of TensorFlow, aimed at
technically minded artists to help them start running with neural-network music projects. It also
seems that Magenta has their own library of functions for dealin with midi files. Using Magenta's
midi-awesomeness seems to be far more sophisticated than the current system of simple note-state
matricies used in this project, so you may want to consider switching. Check Magenta's github
for more details.</li>
<li><b>Structure data better:</b> At the moment, the way our music is being represented takes up an
ungodly amount of RAM. To represent just a 50 sixteenth note sequence takes, at one point, memory
on the order of gigabytes. Finding a better way to represent the data is therefore paramount for
scaling the project upwards to larger datasets and longer tracks. One way to achieve this would be
to leverage Magenta's MIDI interface to capture each song in a far more compressed, efficient
form.</li>
</ul>

## Final Thoughts
That's about it, folks. Thanks for sticking around, have fun with the project, and from the bottom
of our hearts, we wish you the best of luck this senior year. Again, if you have any questions,
concerns, or the code just can't seem to work right, please don't hesitate to reach out. Also,
one last pro-tip, to save on RAM when using one of the Linux machines in the IRC, pressing
`[CTRL] + [ALT] + <F2>` will open for you a gui-less virtual terminal, reducing the need to render
centos's fancy desktop environment. Just remember to switch back to the original terminal (I
think it's on `<F1>`) when you're letting the network bake in the background, so that anyone 
snooping around can't mess with your stuff.

Happy coding!
Austin Choi and William Tong

## Some ~~Miscellaneous~~ Helpful Links

### MIDI Utilities
[Music Animation Machine MIDI Player](http://www.musanim.com/player/)
	This is a MIDI player/visualizer that allows you to see when where exactly your notes fall.
	Unfortunately there is only a Windows distribution available, but if you are running Windows
	then this is a fun alternative to Media Player or whatever other piece of software you are using.
	
[Online MIDI Sequencer](https://onlinesequencer.net/)
	Another MIDI player/visualizer, but used in-browser as opposed to downloaded. Has a piano roll
	format, so you can edit/create your own MIDI files here as well.
	
[Anvil Studio](http://www.anvilstudio.com/)
	Kind of like the Online Sequencer, but as a Windows application. Has staff view if you want to
	see the MIDI files in "proper musical notation."

[Magenta's Github](https://github.com/tensorflow/magenta)
	Excellent usage of TensorFlow for music-generating capabilities. Also has a sophisticated
	midi-interface for TensorFlow

[GRUV](https://github.com/MattVitelli/GRUV)
	The original software we used for generating music. It's a little limited, given that it deals
	with raw waveforms instead of midi files, so the music isn't nearly as clean.

### MIDI Libraries/Datasets
[NinSheetMusic](http://www.ninsheetmusic.org/)
	The website we got our Nintendo MIDI files from. With 3300+ sheets hosted on the website, getting
	a proper dataset should be easy. Because all of the MIDI files have the URL format
	http://www.ninsheetmusic.org/download/mid/(a number), we used script to grab a bunch of MIDI
	files at once (script currently not on GitHub, but will try to upload here soon).
	
[Complete Bach MIDI Index](http://www.bachcentral.com/midiindexcomplete.html)
	If you're more into classical music, this site hosts a lot of MIDI transcripts of Bach. A lot.
	
[Piano MIDI German Thing](http://www.piano-midi.de/)
	Another huuuuge set of clasical MIDI files, all as piano solos. Contains works of Chopin, Mozart,
	Beethoven, + others.

### Helpful links to get you running with neural nets
[Colah's Blog](http://colah.github.io/)
	A great introduction to all things neural and awesome

[TensorFlow Tutorial](https://www.tensorflow.org/get_started/)
	Start here when working with TensorFlow
	
### Some articles that show neural nets in action
[The Unreasonable Effectiveness of of Recurrent Neural Networks - Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
	On the longer side, but has a strong explantation of the use of LSTMs in RNNs, this one used for language recognition.

[Who is that Neural Network? - Henrique M. Soares](https://jgeekstudies.org/2017/03/12/who-is-that-neural-network/)
	One of the many applications of neural networks for image recognition, this one being a convolutional neural net.
	
[Recommending music on Spotify with deep learning - Sander Dieleman](http://benanne.github.io/2014/08/05/spotify-cnns.html)
	Another use of convolutional neural networks, this time for providing users of a service with recommendations based on their usage.
