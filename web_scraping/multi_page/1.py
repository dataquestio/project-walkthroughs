from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://www.nytimes.com/books/best-sellers/combined-print-and-e-book-fiction/")
    # browser.close()
    # Step 2
    html_articles = []
    for i in range(3):
        articles = page.locator("ol[data-testid='topic-list']")
        html_articles.append(articles.inner_html())
        page.locator("nav[aria-labelledby='best-sellers-navigation'] a").nth(0).click()
    
    with open("articles.html", "w+") as f:
        for article in html_articles:
            f.write(article)
    browser.close()
