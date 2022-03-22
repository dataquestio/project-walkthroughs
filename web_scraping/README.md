# Project Overview

In this project, we'll learn the basics of web scraping in python.  We'll do this by parsing the [New York Times bestseller list](https://www.nytimes.com/books/best-sellers/combined-print-and-e-book-fiction/).  We'll use playwright, a browser automation tool, to automate our scraping.

In this project, you'll gain an understanding of:

* What web scraping is and why you'd do it
* How to extract elements from html using BeautifulSoup
* How to use playwright to automate scraping pages
* How to load scraped data into pandas and analyze it

## Steps

We'll first start with an overview of web scraping, why you'd want to do it, and how to know if you're allowed.

Then we'll explore the NYT bestsellers list and find the elements that we want to extract.  We'll use [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to parse the web page and get the items we want.

Next, we'll use [playwright](https://playwright.dev/), a browser automation tool, to automate getting data from the bestseller list.  We'll also use playwright to click on elements on the page to navigate through the bestseller lists from multiple weeks.

Finally, we'll explore the data a bit in python and talk about how you can use data after scraping it.

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/web_scraping).

File overview:

* `web_scraping.ipynb` - a jupyter notebook where we parse downloaded data
* `single_page/1.py` - a script to visit a single page with playwright and screenshot it
* `single_page/2.py` - a script to visit a single page with playwright and download the articles
* `multi_page/1.py` - a script to visit multiple pages with playwright and download the articles
* `playwright_in_jupyter.ipynb` - an example of using the playwright async API in Jupyter notebook


# Local Setup

## Installation

To follow this project, please install the following:

* Python 3 (at least 3.7)
    * [Mac install instructions](https://www.dataquest.io/blog/installing-python-on-mac/)
    * [Windows instructions](https://www.dataquest.io/blog/installing-python-on-windows/)
* [JupyterLab](https://jupyter.org/install)
* pandas
    * Run `pip install pandas`
* BeautfulSoup
    * Run `pip install beautifulsoup4`
* playwright
    * Run `pip install playwright`
* playwright browsers
    * Run `playwright install`
    
## Data

You won't need to download a specific data set for this project.  The page we'll be scraping is [here](https://www.nytimes.com/books/best-sellers/combined-print-and-e-book-fiction/).
