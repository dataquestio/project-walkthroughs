# pip install playwright
# playwright install

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://www.nytimes.com/books/best-sellers/combined-print-and-e-book-fiction/")
    page.screenshot(path="nyt.png")
    browser.close()