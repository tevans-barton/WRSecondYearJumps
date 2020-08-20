#@Tommy Evans-Barton

import sys
import json
import shutil
import pandas as pd
import os
DATA_PARAMS = '/config/data-params.json'
TOP_PATH = os.environ['PWD']
sys.path.append(TOP_PATH + '/src')
sys.path.append(TOP_PATH + '/src/viz')
import etl
import processing

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def main(targets):
    if 'clean' in targets:
        shutil.rmtree('data/raw', ignore_errors=True)
        shutil.rmtree('data/interim', ignore_errors=True)
        shutil.rmtree('data/final', ignore_errors=True)

    if 'data' in targets:
        cfg = load_params(TOP_PATH + DATA_PARAMS)
        etl.get_data(**cfg)
        
    if 'transform' in targets:
        processing.clean_all_data()
        processing.merge_data()
    return
    

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)