from settings import *
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
from storage import DBStorage

"""
def search_scrape(query):
    res = list(gsearch(query, num=RESULT_COUNT))
    res_df = pd.DataFrame({"link": res, "rank": list(range(1, len(res) + 1))})
    return res_df
"""

def format_query(query):
    return query.replace(" ", "+")

def search_url(query, start=1):
    query = format_query(query)
    return SEARCH_URL.format(key=SEARCH_KEY, cx=SEARCH_ID, query=query, start=start)

def search_api(query, pages=int(RESULT_COUNT/10)):
    query = format_query(query)

    results = []
    for i in range(0, pages):
        start = i*10+1
        url = search_url(query, start=start)
        response = requests.get(url)
        data = response.json()
        results += data["items"]
    res_df = pd.DataFrame.from_dict(results)
    res_df["rank"] = list(range(1, res_df.shape[0] + 1))
    res_df = res_df[["link", "rank", "snippet", "title"]]
    return res_df

def scrape_page(links):
    html = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for link in links:
            try:
                page.goto(link)
                html.append(page.content())
            except PlaywrightTimeoutError:
                html.append("")
        browser.close()
    return html

class Search():
    min_results = 15

    def __init__(self):
        self.search_func = search_api
        self.storage = DBStorage()
        self.columns = ["query", "rank", "link", "title", "html"]

    def search(self, query):
        stored_results = self.storage.query_results(query)
        if stored_results.shape[0] >= self.min_results:
            return stored_results[self.columns]

        results = self.search_func(query)
        html = scrape_page(results["link"])
        results["html"] = html
        results = results[results["html"].str.len() > 0].copy()
        results["query"] = query
        results = results[self.columns]
        results.apply(lambda x: self.storage.insert_row(x), axis=1)
        print("inserted")
        return results
