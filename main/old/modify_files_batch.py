import os
import handytools.navigator as nav

path = "/src/results/"
constraint = lambda f: "CR" in f
res = nav.recursive_search(path, constraint)

for f in res:
    folder = os.path.dirname(f)
    filename = os.path.basename(f)
    os.chdir(folder)
    os.rename(filename, "AN_"+filename)