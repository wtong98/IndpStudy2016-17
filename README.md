# Let us quest

## Introduction
Welcome, brave adventurers! Neural nets are a massive element of machine learning and modern computing, 
so it's great you're taking the initiative and getting your feet wet in this independent study.

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
<li> `scipy`: not, strictly speaking, neccesary, but it's got loads of useful computation
utilities that you might want to try</li>
<li> `matplotlib`: for making pretty plots</li>
</ul>
<li> `TensorFlow`: the neural network framework this project is built on </li>
<li> `Jupyter Notebook`: software that adds Mathematic-style editing to python
</ul>

If you've got python installed, just about all this software can be installed using `pip` (e.g.
<code>pip install [PackageName]</code>). But especially if you're using Windows + TensorFlow, check
the official instructions online for installing the framework. If you're running linux, using
`pip` may require installing additional packages. Check with your distro to see how that's done, but
most likely it'll be something along the lines of `sudo apt install python-pip` or `sudo pacman 
-S python-pip`

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
out (our personal emails are at the top of `compose.py`)

First off, if you take a peek into the directories, `samples` contains example pieces of midi
or note-state matrices produced in a run of our neural network. `primer.mid` is an example midi-clip
used to prime our network. `awesomeness.mid` is an example generated midi file. `final_tune.txt`
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
a 1 as a trigger, it currently reads a 10 instead. The hope was to allow the network to better
distinguish a note-on from a note-off event, though our efforts don't seemed to have panned out
quite the way we wanted them to. More on that later.

If you take a look at the `RNNintendo_v1` directory, here you'll find all the essential code for
running the beast. The single-most important script here, and the one that functions as
the brains of the whole operation is `compose.py`. This script is a real monster that basically takes
a convolution layer and plops it right on top of an RNN, Frankenstein style, the idea being to make our network
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

Speaking of which, to actually generate these results, turn now to `perform.py`. The purpose of
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
don't crash. An alternative to constantly saving checkpoints would be to implement a Supervisor instead.
More on that in the next part.

## A brief primer on neural networks
Ultimately, neural networks are magic black boxes in which you feed in data, tell it what you want it to see,
and off it goes learning like a grade-conscious sophomore in MI-3. The way it actually does the learning
is justified by a ton of statistics, calculus, and a dash of linear algebra, though no one really understands it
properly anyway. If you're looking for the details, Colah's blog and the TensorFlow tutorials are an excellent
place to start. Most of the literature is also available freely through arXiv, though if you run into a
pay wall, [this](sci-hub.ac) is an excellent resource (and, incidentally, is also a life-saver in many a 
science classes, too).

Neural networks are often organized into layers, which are collections of neurons that collectively process
data. Networks can be described by "width," the number of neurons in a layer, and "depth," the total number
of layers in a network. Generally, more width equates to greater features of the data capable of being
recognized, whereas more depth equates to more abstract details of the data being learned. The bigger the
network, the more data it needs to train. Too small a dataset and you risk overfitting.

Fundamentally, a single of neuron can be thought of as a simple y = mx + b, where x is your input value, y is
your output, m is called a "weight variable," and b is called a "bias variable." If you're in LinAlg or you've
ever done any work with matrices and vectors, you might remember that y = mx = b can be generalized to
Wx_ + b_ = y_, where W is a matrix of weights, x_ is a vector of inputs, and y_ is a vector of outputs (please
forgive the messy notation--I couldn't figure out how to do math markup in github). Thus, using some clever
notation, we can express an entire layer of neurons with their inputs and outputs using some simple matrix
multiplication.

So, if we stack layer upon layer of neurons together, the output from one layer pouring into the input
of the next, then mathematically, it would look something like this:

y_ = W1(W2(W3(...Wn(x_) + bn_...) + b3_) + b2_) + b1_

Right?

Nope.

The problem with simply stacking neurons like this is that they're all linear. Doing y = mx + b over and
over agains would only modify the m and b coefficients, but the whole darn thing itself remains stubbornly
straight. Don't believe? Try multiplying out y = m(m(mx + b) + b) + b. In effect, the only thing a neural
network wired up like y_ = W1(W2(W3(...Wn(x_)...))) would do is figure out the right m and b to a line that
approximates whatever data you're trying to fit. But what if you're trying to model a curve? Imagine finding 
the right m and b to fit y = mx + b to y = x^2.

To fix the issue, data scientists use tools called "nonlinearities." The point of a *non*linearity is to break
the linearity of y = mx + b by introducing an extra, nonlinear function. An extremely popular choice right now, and
the function we use on our network, is called the "rectified linear unit" (ReLU), which can be expressed as

ReLU(x) = max(0,x)

To stack ReLU into the network, you would thread each layer of neurons (except your final, readout layer!) through
ReLU before passing the output to the next layer of neurons. Thus, it would look something like this:

y_ = W1(
	ReLU(W2(
	ReLU(W3(...
	ReLU(Wn(x_) + bn_...)) + b3_)) + b2_)) + b1_
	
So in effect, you're ditching all the negative values from each layer of neurons as you're going along. It sounds
overly simplistic, but it's been mathematically proven that linear combinations of ReLU can approximate any
function.

To draw from the biological inspiration for neural networks, nonlinearities are analogous to the way synapses
decide whether or not they've been saturated with enough input to fire or activate. For this reason, nonlinearities are
also called actiation functions.

Once you've got your network hooked up, neurons aligned and nonlinearities in place, you need a way to tweak the
weights and biases in such a way that your magic black box spits out the correct answers. That's where all the
"learning" in "machine learning" comes from.

The whole process centers around an important concept called a loss function (you might also see "cost function"
or "utility function"). The loss function's inputs are all the different weights and biases, and it outputs a single
value representing how inaccurate the predictions generated by those weights and biases are. By using a tool called
an "optimizer," you can gradually probe the (likely wickedly complicated and hideously multidimensional) manifold
of your loss function, searching for minima, and then recording the particular weights and biases that generate
such low losses. With luck, those particular weights and biases will generalize to real-world data beyond your
training set, and viola, you have a fully functioning, trained neural network. For our project, our optimizer
relies on the Adam optimizer, the current favorite in machine learning.

## Roadmap
Here are some potential areas you guys can look into for improving the system. If you have
your own plans, feel free to persue them. Please don't feel obligated to follow the below
recommendations.

<ul>
	<li><b>Hyperparameter Optimization</b> There are two kinds of "parameters" talked about when dealing
		with neural networks: parameters and hyperparameters. Parameters are the values assigned to each
		neuron, values that modify whatever inputs you fed the network to get your outputs. Parameters are
		further separated into "weights" and "biases," which you'll learn more about in the TensorFlow
		MNIST tutorials. Hyperparameters, on the other hand, are values that govern the way your network
		behaves. These include the widths and depth of your network, batch-size, the kind of optimizer you're
		using, its learning rate, etc. Hyperparameters are sometimes themselves optimized by neural networks,
		but most people use a simple grid search algorithm instead. In our case, even a grid search may prove
		too computationally expensive to pull off (but definitely give it a shot, if you want), so tuning
		by hand may be the best bet. Check out this paper, especially the
		section on : https://arxiv.org/pdf/1206.5533.pdf </li>
	<li><b>Max-pooling:</b> In most (if not all) enterprise-grade convolutional neural networks, a layer
		of 2-by-2 max-pooling almost always follows a layer (or more) of convolution, as a means of aggregating
		data and "looking at the bigger picture." Essentially, a layer of max pooling looks at successive patches
		of your neuron layer, takes the highest output from that patch, and rejects all other values. In this way,
		data like an image can be "summarized" by a smaller representation, both saving on memory and helping the
		network learn a more abstract representation of your data. Thus, max-pooling appears is vital to
		achieving high quality results in a music-producing network, though we never took the time to
		actually implement it properly. The TensorFlow API should have all the info you'll need.</li>
	<li><b>Dropout:</b> Often times, when you're training a huge network on too little data, the network will
		learn the idiosyncracies of your dataset too well and, in effect, "over-fit," where it matches the
		contours of your data so well that it generalizes poorly to your intended subject. A surprisingly good
		way to fix this is to implement "dropout," where neurons in a layer are randomly dropped from the
		network. Sounds brutal, but it works surprisingly okay. Usually, you'll want to use dropout whenever you're
		playing with fully-connected layers. Convolution layers are typically really resistent to overfitting, for
		reasons I don't understand, so it may not be one-hundred percent essential for you guys to implement
		dropout there, but if you're ever building in some fully-connected layers (NOT including your readout or
		input layers), go ahead and give it a shot. The TensorFlow tutorial may be a good place to start with this.</li>
	<li><b>Gated Recurrent Unit:</b> Gated Recurrent Units (GRU) are a simpler version of the traditional
		Long-Short Term Memory (LSTM) cells used in the network. GRU's are becoming more popular for reasons
		beyond me, but they might be worth checking out. Consult Colah's blog for a nice introduction.</li>
	<li><b>Multi-layered RNN's:</b> A neat feature that's possible to do (relatively) easily with
		TensorFlow is to stack multiple RNN's on top of each other. Perhaps doing so might improve the
		network's ability to recognize more subtle patterns in the music. Check the TensorFlow tutorials for
		further guidance on this topic.</li>
	<li><b>Magenta's MIDI interface:</b> Magenta is a project built on top of TensorFlow, aimed at
		technically minded artists to help them start running with neural-network music projects. It also
		seems that Magenta has their own library of functions for dealing with midi files. Using Magenta's
		midi-awesomeness seems to be far more sophisticated than the current system of simple note-state
		matricies used in this project, so you may want to consider switching. Check Magenta's github
		for more details.</li>
	<li><b>Structure data better:</b> At the moment, the way our music is being represented takes up an
		ungodly amount of RAM. To represent just a 50 sixteenth note sequence takes, at one point, memory
		on the order of gigabytes. Finding a better way to represent the data is therefore paramount for
		scaling the project upwards to larger datasets and longer tracks. One way to achieve this would be
		to leverage Magenta's MIDI interface to capture each song in a far more compressed, efficient
		form.</li>
	<li><b>Architecture</b> There's a load of different ways to design and build neural networks, and often
		the really creative designs yield the most interesting results. When you feel comfortable enough with
		how different types of neurons interact with each other in different types of ways, I encourage you
		guys to play around with different ways to build a network. What if convolution and recurrence happen
		in parallel instead of stacking? What if you have multiple levels of convolution? What if you swapped
		out max-pooling for average pooling? What happens when you play with different strides and patch
		sizes? Check out the architecture for Google's Inception v3 image recognition network for inspiration.
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
