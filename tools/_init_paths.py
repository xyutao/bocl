"""Set up paths for importing models."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add models to PYTHONPATH
model_path = osp.join(this_dir, '..')
add_path(model_path)