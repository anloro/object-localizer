# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:38:27 2022

@author: jlalctar
"""

"""
Utilities to build a dataset
"""

# Imports
import requests
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import cv2
import os
import pathlib
from bs4 import BeautifulSoup

# Website scrapper (shutterstock)

def IMG_Scrapper(*,query : str, n_images : int, init_page : int = 1, early_stop : bool = True) -> tuple:
    """
    Scraps images from the Shutterstock website.

    Parameters
    
    path : str = None
        The default is None.
        
        Folder to download the images. If None, images
        will not be downloaded, just the links.
        Path must end by '/'.
        
    query : str
        Search words separated by "-".
            Ex: "forest-aerial".
            
    n_images : int
        Number of images to download, as integer.
        
    init_page : int
        The default is 1.
        
        Selects the result page at which the scrapping will be initiated.
        
    early_stop : bool
        The default is True.
        
        This variable defines whether the program will be stopped by a 429
        HTML error or not (429 - Too many requests).
        
        If set to False, the program will delay itself before requesting again,
        when it faces a 429 error, so that it may continue scrapping.
        
        If True, as default, the program will stop scrapping when encountering
        a denial of type 429.

    Returns
    -------
    links : list
        Fetched images as URLs.

    page : int
        Number of the next page to the one scrapped last.

        Ex. If the last page scrapped is 5, the return will be 6.

    """
    
    # Initial page
    page = init_page
    
    # Images obtained
    img_count = 0
    links = []
    
    # HTML header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
        }
    
    
    while(img_count < n_images):
        # Defining the URL according to the query
        URL = f"https://www.shutterstock.com/search/{query}?page={page}"
        
        # Requesting access to the web
        web = requests.get(URL, headers = headers)
        
        # Validating the status code
        
        if(web.status_code == 429 and not early_stop):
            sleep(30)
        elif(web.status_code != 200):
            print(f"\nWebsite returned code {web.status_code}. {img_count} images fetched.\n")
            break
                          
        # Parsing the webpage content
        content = BeautifulSoup(web.content, 'html.parser')
        
        # Fetching the images
        images = content.find_all('img')[2:]
        
        for img in images:
            links.append(requests.compat.urljoin(URL,img.get('src')))
            img_count += 1
            
            if(img_count == n_images):
                break
        
        # Increasing the page
        page += 1
        
    return (links, page)

# Image downloader

def IMG_Download(path : str, links : list) -> None:
    """
    

    Parameters
    ----------
    path : str
        Path to the folder to store the downloaded images.
        It must end by /.
        
    links : list
        Links to fetched images. Returned by IMG_Scrapper().

    Returns
    -------
    None
        This function returns no value.

    """
    
    if(path[-1] != '/' and path[-1] != '\\'):
        raise ValueError("Path must end by \\ or /")
    
    if(not os.path.isdir(path)):
        os.mkdir(path)
    
    # Downloading the images
        
    for index, url in enumerate(links):
        img_web = requests.get(url)
        
        img = img_web.content
        
        with open(path + "img{}.png".format(index),'wb') as fd:
            fd.write(img)

# Image treater

def IMG_Treater(*,img_name : str, img_ext : str = '.png', path_base : str, path_end : str,eps : int = 0.5) -> None:
    """
    
    Parameters
    ----------
    img_name : str
        Name of the images. This allows to complete a dataset, instead of
        overwriting it.
        
        Note that all the files with the same name will be destroyed.
        
        The program will get img_name and generate names in the format:
            {img_name}(sq or nonsq){ascending counter from 0}{img_ext}
            
    img_ext : str
        The default is '.png'.
        
        Specifies the extension of the treated images.
        
        Please, the point must be included.

    path_base : str
        Folder where base images are placed.
    
    path_end : str
        Folder where treated images will be put.
        
    eps : int
        Probability of an image presenting a square.
    
    Returns
    -------
    None
        This function does not return a value.

    """
    
    if('.' not in img_ext):
        raise ValueError("Extension must have a point (i.e. '.png')")
    
    if(path_base[-1] != '/' and path_base[-1] != '\\'):
        raise ValueError("Path must end by \\ or /")
    
    elif(path_end[-1] != '/' and path_end[-1] != '\\'):
        raise ValueError("Path must end by \\ or /")

    if(not os.path.isdir(path_end)):
        os.mkdir(path_end)
    
    # Creating the dataset subdirectories
    issquare = os.path.isdir(path_end + "Square/")
    isnonsquare = os.path.isdir(path_end + "No_Square/")
    
    if(not issquare and not isnonsquare):
        os.mkdir(path_end + "Square/")
        os.mkdir(path_end + "No_Square/")
    elif(issquare and not isnonsquare):
        os.mkdir(path_end + "No_Square/") 
    elif(not issquare and isnonsquare):
        os.mkdir(path_end + "Square/")
            
    
    pathsq = path_end + "Square/"
    pathnonsq = path_end + "No_Square/"
    
    path = pathlib.Path(path_base)
    dir_elem = list(path.glob('*'))
    sq_count = 0
    nonsq_count = 0
    
    for image in dir_elem:
        
        # Reading the image
        
        img = cv2.imread(str(image.absolute()))
        dim = img.shape
        
        # Cutting the website mark
        img = img[:-20,:,:]
        
        if(np.random.rand() < eps):
            sq_count += 1
            
            # Setting a random white square
            scale = 0.10
            sqdim_val = int(np.round(min(dim[0:1])*scale))
            
            x1coord = np.random.randint(low = sqdim_val, high = dim[1] - sqdim_val*2) # We leave a sqdim_val margin
            y1coord = np.random.randint(low = sqdim_val, high = dim[0] - sqdim_val*2)
            x2coord = x1coord + sqdim_val
            y2coord = y1coord + sqdim_val
            
            cv2.rectangle(img,
                          pt1 = (x1coord,y1coord),
                          pt2 = (x2coord,y2coord),
                          color = (255,255,255),
                          thickness = -1
                          )
            
            # Saving the image
            cv2.imwrite(pathsq + f"{img_name}sq{sq_count}{img_ext}", img)
            
        else:
            nonsq_count += 1
            cv2.imwrite(pathnonsq + f"{img_name}nonsq{nonsq_count}{img_ext}", img)
        

        
        
        