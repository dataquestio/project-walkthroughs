# Project Overview

In this project, we'll build a system that can record live speech using your microphone, and then transcribe it using speech recognition.  This can be used to automatically record and transcribe meetings, lectures, and other events.

We'll be using Jupyter notebook to write our code, and to build the interactive widgets to start and stop recording.  By the end, you'll be able to click a button, and have speech be automatically recorded and transcribed.

**Project Steps**

* Create Jupyter widgets to record audio and stop recording
* Use pyaudio to record microphone audio
* Create a speech recognition system using vosk
* Add punctuation to the text transcript using recasepunc

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/microphone).

File overview:

* `microphone.ipynb` - the full code from this project

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.8+
* Python packages
    * vosk `pip install vosk`
    * pydub `pip install pydub`
    * transformers `pip install transformers`
    * torch `pip install torch -f https://download.pytorch.org/whl/torch_stable.html`
    * pyaudio `pip install pyaudio`
    * ipywidgets `pip install ipywidgets`

### Vosk

You'll need to download a model file to run vosk properly.  This will automatically download when you run this code:

```
from vosk import Model
Model(model_name="vosk-model-small-en-us-0.15")
```

The full vosk model is large (1+ GB).  If you want to use it, just specify `vosk-model-en-us-0.22` as the model name.

If the models don't automatically download, you can find them [here](https://alphacephei.com/vosk/models).

### Punctuation

By default, vosk will output text with no punctuation.  To add in punctuation, we'll need a different model.  To get this, follow these steps:

* Download the model [here](https://alphacephei.com/vosk/models/vosk-recasepunc-en-0.22.zip) - caution: it's 1 GB+ in size.
* Extract the zip file into the same directory as your code.

### Pyaudio

Pyaudio can be a little tricky to install, since it depends on system packages.  Check the [homepage](http://people.csail.mit.edu/hubert/pyaudio/) for specific instructions for each OS.

You'll also want to figure out the right device to record from.  Run this code to find the index of your microphone:

```
# Find audio device index
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

p.terminate()
```


## Data

All audio will come from your microphone, so you don't need any specific downloads.