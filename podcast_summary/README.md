# Project Overview

In this project, we'll create a data pipeline using Airflow.  The pipeline will download podcast episodes and automatically transcribe them using speech recognition.  We'll store our results in a SQLite database that we can easily query.

**Project Steps**

* Download the podcast metadata xml and parse
* Create a SQLite database to hold podcast metadata
* Download the podcast audio files using requests
* Transcribe the audio files using vosk

By the end, you'll have a good understanding of how to use Airflow, along with a practical project that you can continue to build on.

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/podcast_summary).

File overview:

* `podcast_summary.py` - the code to create a data pipeline

# Local Setup

## Installation

To follow this project, please install the following locally:

* Airflow 2.3+
* Python 3.8+
* Python packages
    * pandas
    * sqlite3
    * xmltodict
    * requests
    * vosk
    * pydub

Please ensure that you have access to the Airflow web interface after installing it.

## Data

We'll download the data we need during this project.  If you want to do the speech 

