# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:34:57 2022

@author: jlalctar
"""

"""
Script used to build a dataset.
"""

import dsbuild_utils as db
import numpy as np
import time


# Parameters
n_images = 5000
init_page = 1

t0 = time.time()

links,page_end = db.IMG_Scrapper(
    query = 'aerial forest',
    n_images = n_images,
    init_page = init_page,
    early_stop = False)

t1 = time.time()

db.IMG_Download(path = './Data_Raw/', links = links)

t2 = time.time()

db.IMG_Treater(path_base = './Data_Raw/',
               path_end = './Database/',
               eps = 0.5,
               img_name = 'test',
               img_ext = '.png')

t3 = time.time()

print(f"\nTime to scrap {n_images} images: {t1-t0} seconds.\n")
print(f"Time to download the {n_images} images: {t2-t1} seconds.\n")
print(f"Time to process the {n_images} images and build the dataset: {t3-t2}\n")
print(f"Overall time: {t2-t0}\n")
