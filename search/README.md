# Project Overview

In this project, we'll build a search engine that uses filtering to reorder results.  The engine will get results from the Google API, store them, then rank them based on filters we define.  We'll end up with a basic search page and results list.

We'll use Pycharm, a common python IDE, to write our code and run it.

**Project Steps**

* Setup a programmable search engine [Custom Search API](https://developers.google.com/custom-search/v1/introduction)
  * You can create one [here](https://programmablesearchengine.google.com/controlpanel/all)
* Create an [API key](https://console.cloud.google.com/apis/credentials) for the engine
* Create a module to search using the API
* Create a Flask application to search and render results
* Create filters to re-rank results before displaying them

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/search).

File overview:

* `app.py` - the web interface
* `filter.py` - the code to filter results
* `search.py` - code to get the search results
* `settings.py` - settings needed by the other files
* `storage.py` - code to save the results to a database

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.9+
* Required Python packages (`pip install -r requirements.txt`)

### Other setup

You will need to create a programmable search engine and get an API key by following [these directions](https://developers.google.com/custom-search/v1/introduction).  You will need a Google account, and as part of this you may also need to sign up for Google Cloud.

### Other files

You'll need to download a list of ad and tracker urls from [here](https://raw.githubusercontent.com/notracking/hosts-blocklists/master/dnscrypt-proxy/dnscrypt-proxy.blacklist.txt).  We'll use this to filter out bad domains.  Please save it as `blacklist.txt`.

I also recommend copying the [storage.py](https://github.com/dataquestio/project-walkthroughs/blob/master/search/storage.py) file into your directory before we start the project.

## Run

Run the project with:

* `pip install -r requirements.txt`
* `flask --debug run --port 5001`