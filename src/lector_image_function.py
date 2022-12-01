import pathlib
from filtro_mc_function import filtcanny
from filtro_mc_function import filtmask
#from filtro_mc_function import filthres

path = pathlib.Path("./../baseddades/")

images = list(path.glob("*"))

for i in images:
    img = str(i.absolute())
    filtcanny(img)
    filtmask(img)
    #filthres(img)