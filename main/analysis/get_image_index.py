import handytools.navigator as nav

rot_info = nav.pklload(nav.datapath, "stimulus", "shape_info.pkl")["rotation"]

idx = 0
dic = {}
for s in range(51): # there are 51 base shapes
    for r in range(rot_info[s]):
        dic[idx] = (s, r)
        idx += 1
nav.pklsave(dic, nav.datapath, "stimulus", "shape_index.pkl")