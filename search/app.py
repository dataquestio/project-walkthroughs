from flask import Flask, request
from search import Search
from filter import Filter
import html

app = Flask(__name__)

styles = """
<style>
    .site {
        font-size: .8rem;
        color: green;
    }
    
    .snippet {
        font-size: .9rem;
        color: gray;
        margin-bottom: 30px;
    }
</style>
"""

search_template = styles + """
     <form action="/" method="post">
      <input type="text" name="query">
      <input type="submit" value="Search">
    </form> 
    """

result_template = """
<p class="site">{rank}: {link}</p>
<a href="{link}">{title}</a>
<p class="snippet">{snippet}</p>
"""

def show_search_form():
    return search_template

def run_search(query):
    se = Search()
    results = se.search(query)
    fi = Filter(results)
    filtered = fi.filter()
    rendered = search_template
    filtered["snippet"] = filtered["snippet"].apply(lambda x: html.escape(x))
    for index, row in filtered.iterrows():
        rendered += result_template.format(**row)
    return rendered

@app.route("/", methods=['GET', 'POST'])
def search_form():
    if request.method == 'POST':
        query = request.form["query"]
        return run_search(query)
    else:
        return show_search_form()