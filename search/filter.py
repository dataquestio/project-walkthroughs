from bs4 import BeautifulSoup
from urllib.parse import urlparse
from settings import *

with open("blacklist.txt") as f:
    domains = set(f.read().split("\n"))

def tracker_urls(row):
    soup = BeautifulSoup(row["html"])
    scripts = soup.find_all("script", {"src": True})
    srcs = [s.get("src") for s in scripts]

    links = soup.find_all("a", {"href": True})
    href = [l.get("href") for l in links]

    all_domains = [urlparse(s).hostname for s in srcs + href]
    return len([a for a in all_domains if a in domains])

def get_page_content(row):
    soup = BeautifulSoup(row["html"])
    text = soup.get_text()
    return text

class Filter():
    def __init__(self, results):
        self.filtered = results.copy()

    def tracker_filter(self):
        tracker_count = self.filtered.apply(tracker_urls, axis=1)
        tracker_count[tracker_count > tracker_count.median()] = RESULT_COUNT
        self.filtered["rank"] += tracker_count * 2

    def content_filter(self):
        page_content = self.filtered.apply(get_page_content, axis=1)
        word_count = page_content.apply(lambda x: len(x.split(" ")))

        word_count /= word_count.median()
        word_count[word_count <= .5] = RESULT_COUNT
        word_count[word_count != RESULT_COUNT] = 0
        self.filtered["rank"] += word_count

    def filter(self):
        self.tracker_filter()
        self.content_filter()
        self.filtered = self.filtered.sort_values("rank", ascending=True)
        self.filtered["rank"] = self.filtered["rank"].round()
        return self.filtered