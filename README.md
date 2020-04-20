# interactiveEEG
collaborative platform for transdisciplinary EEG exploitation. 

**introductory babbling:**

EEG can be a powerful tool for expression, that bridges the artistic and scientific experiences. 
It allows to add a neurocognitive dimension to the domain of interactivity, with the potential to demistify the public understanding of the secret life of the brain. 
The experience of neurofeedback- where internal state, thought, concentration and meditation can control auditory and visual scenes- is inspiring and fascinating. To be able to make it widely accessible is a precious opportunity, which is immediately compatible with the evergrowing scene of multimedia performances and dance.

The starting focus of this project is a simple Python pipeline for motor imagery BCI. This allows us to test the potential and limits of the available device. We also keep an eye on basic visualisations of neural oscillations.

Based on mBrainTrain devices until open source options will become avaible.

**Useful links:**

Device page: https://mbraintrain.com/support/

*CREDENTIALS:

username: User_x73

pass: xQ7b4etgv=6C*

Google doc for quick notes and general dumping: https://docs.google.com/document/d/1gqDmMJUV1vkObeEYlml45LO7qmUgNQ4syoi9aXsdOsI/edit?usp=sharing

Google drive for dataset sharing: https://drive.google.com/drive/folders/1cZ0uvtnLHsLnJOsBhiFi-ILcSbYPya7M?usp=sharing

Openvibe page: http://openvibe.inria.fr/downloads/

Pylsl: https://github.com/labstreaminglayer/liblsl-Python

# Data acquisition pipeline

**Step 0 : Environment setup**

a) Install anaconda with python 3.x

b) Install Openvibe. We are going to use "Designer" and "Acquisition Server" http://openvibe.inria.fr/downloads/

c) Install pylsl using anaconda prompt https://github.com/labstreaminglayer/liblsl-Python

`pip install pylsl`

**Step 1 : Generate and stream fake data**

a) Run open vibe acquisition server

b) From the "Driver" selection menu, select either Oscillator or Time Signal

c) In "Preferences", authorise "LSL_EnableLSLOutput" and change "LSL_SignalStreamName" to **EEG**

d) Press "Connect" and "Play"

**Step 1b : Stream pre-recorded data**

a) Download data file of choice from the drive (link above)

b) Run openvibe Designer. Load "LSLstreamer.mxs", acquired from the Drive folder (link above)

c) Open "Generic stream reader" node. Change path and filename to data file path and filename

d) Open "LSL Export" node. Change "Signal stream" to 'EEG'

e) Start scenario execution (play button)


**Step 2 : Acquire stream on python (REDUNDANT WITH LSL scripts for Data acquisition)** 

 a) With either Designer or Server running, run Python and execute code below, or see pylsls samples (link above)


`from pylsl import StreamInlet, resolve_stream`

**first resolve an EEG stream on the lab network**

`print("looking for an EEG stream...")`

`streams = resolve_stream('name', 'EEG')`

**create a new inlet to read from the stream**

`inlet = StreamInlet(streams[0])`

`while True:`
    
   `sample, timestamp = inlet.pull_sample()`
   
   `print(timestamp, sample)`
   
   




