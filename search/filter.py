from bs4 import BeautifulSoup
from urllib.parse import urlparse

def count_trackers(row):
    soup = BeautifulSoup(row["html"])
    domain = urlparse(row["link"]).hostname
    scripts = soup.find_all("script", {"src": True})
    srcs = [s.get("src") for s in scripts]
    bad_srcs = [s for s in srcs if ".." not in s and domain not in s]
    return len(bad_srcs)

class Filter():
    def __init__(self, results):
        self.results = results
        self.filtered = results.copy()

    def js_filter(self):
        tracker_count = self.filtered.apply(count_trackers, axis=1)
        self.filtered = self.filtered[tracker_count < 8]

    def filter(self):
        self.js_filter()
        return self.filtered, self.results[~self.results.index.isin(self.filtered.index)]