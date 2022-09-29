FL_API_KEY = ""
FL_API_SECRET = ""

IMAGE_DIR = "images"
MIN_SIZE = 384
FOLDERS = ["happy", "sad", "relaxed"]
VALID_EXTENSIONS = ["jpg", "png", "gif", "jpeg"]
TARGET_EXTENSION = "jpg"

DEVICE = "mps"

import os
if os.path.exists("private.py"):
    from private import *