SEARCH_KEY = ""
SEARCH_ID = ""
SEARCH_URL = "https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&q={query}&start={start}&gl=us&num=10"
RESULT_COUNT = 30

import os
if os.path.exists("private.py"):
    from private import *