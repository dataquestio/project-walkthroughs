# Project Overview

In this project, we'll build a system that can automatically recognize speech and summarize it.  This can be used for automatically transcribing and summarizing lecture recordings, podcasts, or videos.

We'll also include a way to hook up a microphone to automatically record and transcribe audio for live notetaking.  This could be used to record and transcribe meetings in real-time.

By the end of this project, you'll have a speech to text project that you can continue to build on.

**Project Steps**

* Create a speech recognition system using vosk
* Add punctuation to the text transcript using recasepunc
* Summarize the text using a huggingface summarization pipeline
* Create a widget to record and transcribe live audio

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/speech_recognition).

File overview:

* `voice.ipynb` - the code to summarize text
* `marketplace.mp3` - a 45 second audio clip you can use to test the model
* `marketplace_full.mp3` - a 30 minute audio clip you can use to test the model
* `transcript.txt` - the full transcript of `marketplace_full`

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

### Punctuation

By default, vosk will output text with no punctuation.  To add in punctuation, we'll need a different model.  To get this, follow these steps:

* Download the model [here](https://alphacephei.com/vosk/models/vosk-recasepunc-en-0.22.zip) - caution: it's 1 GB+ in size.
* Extract the zip file into the same directory as your code.

### Summarization

To summarize text, we'll need to download a summarization model.  You can download a basic model from huggingface using this code:

```
from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small")
```

### Pyaudio

Pyaudio can be a little tricky to install, since it depends on system packages.  Check the [homepage](http://people.csail.mit.edu/hubert/pyaudio/) for specific instructions for each OS.


## Data

You'll want to download a couple of audio files to test the transcription with:

* [marketplace_full.mp3](https://github.com/dataquestio/project-walkthroughs/raw/master/speech_recognition/marketplace_full.mp3)
* [marketplace.mp3](https://github.com/dataquestio/project-walkthroughs/raw/master/speech_recognition/marketplace.mp3)