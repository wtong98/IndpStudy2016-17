Playing around with ANN's with Tensorflow, mainly for music generation.

This project is the result of our Independent Study for the 2016-2017 school year, in 
which we wrote this RNN to train on MIDI files and generate music using that training.

something something Tensorflow + Magenta something something

## Installation

Currently, all of the Python scripts are located in ~/RNNintendo_v1. Simply place the scripts
in your desired working directory and you should be able to run them just fine (make sure
you have Python on your machine!).

## Running the Neural Net

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
	http://www.ninsheetmusic.org/download/mid/(some number), we used script to grab a bunch of MIDI
	files at once (script currently not on GitHub, but will try to upload here soon).
	
[Complete Bach MIDI Index](http://www.bachcentral.com/midiindexcomplete.html)
	If you're more into classical music, this site hosts a lot of MIDI transcripts of Bach. A lot.
