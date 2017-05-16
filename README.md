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
<li> Python: preferably Python 3, but Python 2 will probably work okay if you're on a 
Linux machine (another reason to switch) You'll also need some additional modules listed
below</li>
<ul>
<li> numpy: a python module designed for numerical computation </li>
<li> tqdm: for attractive loading bars </li>
<li> scipy: not, strictly speaking, neccesary, but it's got loads of useful computation </li>
utilities that you might want to try</li>
<li> matplotlib: for making pretty plots</li>
</ul>
<li> TensorFlow: the neural network framework this project is built on </li>
<li> Jupyter Notebook: software that adds Mathematic-style editing to python
</ul>

If you've got python installed, just about all this software can be installed using pip (e.g.
<code>pip install [PackageName]</code>. But especially if you're using Windows + TensorFlow, check
the official instructions online for installing the framework.

## Running the Neural Net
All the files necessary to run the neural net are located in the directory RNNintendo_v1. To 


*stuff goes here*
*get Tensorflow + Magenta* something something


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
[The Unreasonable Effectiveness of of Recurrent Neural Networks - Andrej Karpathy)[http://karpathy.github.io/2015/05/21/rnn-effectiveness/]
	On the longer side, but has a strong explantation of the use of LSTMs in RNNs, this one used for language recognition.

[Who is that Neural Network? - Henrique M. Soares](https://jgeekstudies.org/2017/03/12/who-is-that-neural-network/)
	One of the many applications of neural networks for image recognition, this one being a convolutional neural net.
