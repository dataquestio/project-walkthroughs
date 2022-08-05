from bs4 import BeautifulSoup
from urllib.parse import urlparse
from settings import *
import re

def count_trackers(row):
    soup = BeautifulSoup(row["html"])
    domain = urlparse(row["link"]).hostname
    scripts = soup.find_all("script", {"src": True})
    srcs = [s.get("src") for s in scripts]
    bad_srcs = [s for s in srcs if ".." not in s and domain not in s and "cdn" not in s]
    return len(bad_srcs)

def get_page_content(row):
    soup = BeautifulSoup(row["html"])
    text = soup.get_text()
    return text

class Filter():
    def __init__(self, results):
        self.filtered = results.copy()

    def js_filter(self):
        tracker_count = self.filtered.apply(count_trackers, axis=1)
        tracker_count[tracker_count > tracker_count.median()] = RESULT_COUNT
        self.filtered["rank"] += tracker_count

    def content_filter(self):
        page_content = self.filtered.apply(get_page_content, axis=1)
        word_count = page_content.apply(lambda x: len(x.split(" ")))
        median = word_count.median()
        word_count /= median
        word_count[word_count <= .5] = RESULT_COUNT
        self.filtered["rank"] += word_count

    def year_filter(self):
        titles = self.filtered["title"]
        year_in_title = titles.apply(lambda x: len(re.findall(r"20\d{2}", x)))
        year_in_title[year_in_title > 0] = RESULT_COUNT
        self.filtered["rank"] += year_in_title

    def filter(self):
        self.js_filter()
        self.content_filter()
        self.year_filter()
        self.filtered = self.filtered.sort_values("rank", ascending=True)
        self.filtered["rank"] = self.filtered["rank"].round()
        return self.filtered