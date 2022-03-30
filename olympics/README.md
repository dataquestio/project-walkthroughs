# Project Overview

In this project, we'll learn data visualization and data cleaning using Microsoft Power BI.  We'll cover DAX, the M language, the Power Query editor, and visualization.  This is a great companion to [Dataquest Power BI courses](https://www.dataquest.io/path/analyzing-data-with-microsoft-power-bi-skill-path/).

In this project, you'll gain an understanding of:

* What Power BI is and why you'd use it
* How to import data into Power BI
* How to transform and clean data using Power Query and the M language
* How to model relations
* How to calculate columns using DAX

## Code

You can find the Power BI report for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/olympics).

File overview:

* `Olympics.pbix` - a Power BI report that mirrors what we'll do in this session

# Local Setup

## Installation

To follow this project, please install Power BI.  [These instructions](https://docs.microsoft.com/en-us/power-bi/fundamentals/desktop-get-the-desktop) can help you figure out how.

## Data

You'll need to download three files to follow this project:

* [athlete_events.csv](https://drive.google.com/uc?export=download&id=1Ah4wOyNFMGREq8Yw_Jbv7u2CeI_6tpn5) - contains information on athletes who competed in the Olympics.
* [noc_regions.csv](https://drive.google.com/uc?export=download&id=1aqSmdHo3perJtdduTzCLMB6W5h_KdsOd) - contains information on how national olympic committee codes map to country names.
* [country_population.csv](https://drive.google.com/uc?export=download&id=1jYxLnVMtPdnh3pmJW3OEs6YEk3uTCl1h) - contains information on the population of each country.

The Olympic and NOC data is originally from [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results).  The population data is from the World Bank.

# Project Steps

1. Data exploration - load the data in and explore it by creating visualizations
2. Importing NOC data - load in information matching NOC codes to the actual country name.  We'll then model the relationship.
3. Add in medal columns - use Power Query editor to add some additional columns with medal information.
4. Creating a filled map - create a map that shows the number of medals each country earned.
5. Creating an animated map - make a map that shows how Olympic participation changed over time.
6. Combine population data - load in population data and match it to our existing data.
7. Create rolling medal counts - show a rolling total of medals earned by country.

