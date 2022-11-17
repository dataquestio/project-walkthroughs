# Project Overview

In this project, we'll build a translation API using deep learning.  Using FastAPI, we'll create a web server that exposes a `/translate` route and a `/results` route.  Clients will post their translation request to the `/translate` route, and get the translation results from `/results`.  The server will use a sqlite database to store the translations.  On the backend, we'll use async and a pretrained deep learning language model to run the translation job.

By the end, we'll have a web server that can run translation jobs quickly.  This server can easily be extended to translate more languages, or add more options.

**Project Steps**
* Build API routes
* Add in models to store data to database
* Create tasks to run the translation

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/dl_api)

File overview:

* `requirements.txt` - packages you'll need to install
* `languages.txt` - list of languages that are supported for translation
* `main.py` - defines the web server routes
* `models.py` - defines database models
* `tasks.py` - runs our backend tasks, including the translation

# Prerequisites

To complete this project, you'll need to have a good understanding of:

* Python syntax, including functions, if statements, and data structures
* Data cleaning
* Pandas syntax
* Using Jupyter notebook
* APIs
* The basics of machine learning.

Please make sure you've completed these Dataquest courses (or know the material) before trying this project:

* [Python Introduction](https://www.dataquest.io/course/introduction-to-python/)
* [For Loops and If Statements](https://www.dataquest.io/course/for-loops-and-conditional-statements-in-python/)
* [Dictionaries In Python](https://www.dataquest.io/course/dictionaries-frequency-tables-and-functions-in-python/)
* [Functions and Jupyter Notebook](https://www.dataquest.io/course/python-functions-and-jupyter-notebook/)
* [Python Intermediate](https://www.dataquest.io/course/python-for-data-science-intermediate/)
* [Pandas and NumPy Fundamentals](https://www.dataquest.io/course/pandas-fundamentals/)
* [Data Cleaning](https://www.dataquest.io/course/python-datacleaning/)
* [Machine Learning Fundamentals](https://www.dataquest.io/course/machine-learning-fundamentals/)
* [APIs and Web Scraping](https://www.dataquest.io/course/apis-and-scraping/)

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.8+
* The packages defined in `requirements.txt`
* It's recommended to install PyCharm or VSCode.

## Usage

To run the server locally, run `uvicorn main:app --reload`.

## Data

During this project, we'll download a pretrained language model.

# Deploying

## Deploy with Docker

One strategy is to deploy to the cloud using Docker.  To do this, we first need to build a Docker image.  Then we can deploy the image to the cloud.

### Docker image

To build and test a Docker image, run:

* `docker build -t dlapi .` to build the container.
  * If you're not running on 64-bit linux, instead run `docker buildx build --platform linux/amd64 -t dlapi .`.  This will build the image using the correct architecture for Azure.
* `docker run -d --name dlapi -p 80:80 dlapi` to run the container.
* `docker ps` to view the container information.
* Run `docker logs` to see logs from the container.  You should see `Uvicorn running on http://0.0.0.0:80`.  If you don't see this, wait a bit and try running `docker logs` again.
* Visit `127.0.0.1` or `localhost` to see the API server.  Visit `localhost/docs` to see API docs.
* Run `docker stop dlapi` to stop the container.

### Azure setup

* Create an Azure account.  You can create one [here](https://azure.microsoft.com/en-us/free/search/).
* Create an [Azure subscription](https://portal.azure.com/#view/Microsoft_Azure_Billing/SubscriptionsBlade) if you don't have one.  You can sign up for the [free tier](https://azure.microsoft.com/en-us/free/).
* Install the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/).
* Run `NAME="<name-here>"` to set the name for your resource group.  Replace `<name-here>` with the name you want.
* Create a resource group with `az group create --location eastus --name $NAME`

### Push image to Azure

* Create a container registry with `az acr create --resource-group $NAME --name $NAME --sku Basic`
* Note the `loginServer` in the response.
* Login to the registry with `az acr login --resource-group $NAME --name $NAME`
* Run `docker tag dlapi $NAME.azurecr.io/dlapi`.  Replace `$NAME.azurecr.io` with your container registry login url.
* Push the image to the registry with `docker push $NAME.azurecr.io/dlapi`.

### Run container on Azure

* Run `docker login azure` to log Docker into Azure.
* Create an aci context with `docker context create aci azure`.  Select the resource group to use with your context.
* Run `docker context ls` to view your contexts.
* Switch context with `docker context use azure`
* Run the container with `docker run --name dlapi -p 80:80 -m 1.5G $NAME.azurecr.io/dlapi`
* Run `docker ps` to get the URL of the container.  You should be able to visit the URL shown to use the API.

### Remove container

* Run `docker stop dlapi`
* Run `docker rm dlapi`
* Delete the resource group with `az group delete --name $NAME --no-wait`

## Deploy with Azure App Service

Another strategy is to deploy with Azure App Service.

### Deploy App

* Create an Azure account.
* Create an [Azure subscription](https://portal.azure.com/#view/Microsoft_Azure_Billing/SubscriptionsBlade) if you don't have one.  You can sign up for the [free tier](https://azure.microsoft.com/en-us/free/).
* Install the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
* Login to Azure with `az login`.
* `NAME="dlapi"`
* Create a resource group with `az group create --location eastus --name $NAME`
* Create a plan with `az appservice plan create --name $NAME --resource-group $NAME --sku B2`
* Create app with `az webapp create --name $NAME --resource-group $NAME --plan $NAME --runtime PYTHON:3.9`
* `az webapp up --runtime PYTHON:3.9 --sku B2 --logs`
* Get the resource group name and the app name from the log output.
* `az webapp config appsettings set --resource-group $NAME --name $NAME --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true  POST_BUILD_COMMAND="build.sh" WEBSITE_MEMORY_LIMIT_MB=3500`
* `az webapp config set --resource-group $NAME --name $NAME --startup-file "run.sh"`
* `zip -r app.zip . -x translations.db app.zip __pycache__/*`
* `az webapp deploy --resource-group $NAME --name $NAME --src-path app.zip`
* `az webapp restart --resource-group $NAME --name $NAME`

### Manage App

* View the app using the URL from the logs.
* Delete the app with `az group delete --name $NAME --no-wait`

