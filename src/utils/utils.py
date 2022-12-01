import os

def getImgpaths(folder: str) -> dict:
    """
    Creates a dictionary with the location of all the images in the similarity
    matrix order.

    Parameters:
        folder: Location of the dataset.
    """

    locs = {}  # Dictionary with the locations

    # Gets all the images knowing all directories have the same structure.
    lvl_cam = {}  # Stores the amount of images in each cam folder
    n = 0
    for heir in sorted(os.walk(folder)):
        if heir[1] == []:  # if you are in the last folder of a tree
            lvl_cam[heir[0]] = len(heir[2])  # save the number of images
            for img in sorted(heir[2]):
                loc = heir[0] + '/' + img  # add to the path the img
                locs[n] = loc  # and store the path
                n += 1
    
    return locs