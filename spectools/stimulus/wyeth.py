import torch
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
from spectools.stimulus.params import pasu_shape_nrotu, pasu_shape

def fvmax(invec):
    sample = 50.0
    num = np.shape(invec)[0]
    inshft = np.vstack((invec[num-2,:],invec,invec[1,:]))
    ip = np.arange(0,50,1)/50

    vtx = np.empty((1,num*50-49))
    vty = np.empty((1,num*50-49))
    dvtx = np.empty((1,num*50-49))
    dvty = np.empty((1,num*50-49))
  
    for i in range(0,num-1):
        bufvrt = inshft[i:i+4,:]
    
        # Spline equation
        incr = np.empty((4,len(ip)))
        incr[0,:] =   -ip*ip*ip + 3*ip*ip -3*ip + 1
        incr[1,:] =  3*ip*ip*ip - 6*ip*ip +4
        incr[2,:] = -3*ip*ip*ip + 3*ip*ip +3*ip + 1
        incr[3,:] =    ip*ip*ip
        
        dincr = np.empty((4,len(ip)))
        dincr[0,:] = -3*ip*ip +  6*ip - 3
        dincr[1,:] =  9*ip*ip - 12*ip
        dincr[2,:] = -9*ip*ip +  6*ip + 3
        dincr[3,:] =  3*ip*ip
    
        vtx[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,0].reshape(4,1),(1,50))*incr,axis=0)/6.0
        vty[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,1].reshape(4,1),(1,50))*incr,axis=0)/6.0
      
    vtx[0,(num-1)*50] = vtx[0,0]
    vty[0,(num-1)*50] = vty[0,0]
    outvec = np.transpose(np.vstack((vtx, vty)))
    return outvec

def shape51_fine_xy(si, roti, rotf):
    """
    @ Args:
        - si (int): shape index, in [0, 50]
        - roti (int): rotation index, in [0, 7] for 45 deg increments, or -1 to use ``rotf``
        - rotf (float): rotation degree, only used if ``roti`` == -1
    """
    na = np.asarray(pasu_shape[si])       # A 1D list of control points as np array
    invec = na.reshape(int(len(na)/2),2)  # Reshape to an array of (x,y) coords.
    outvec = fvmax(invec)                 # Get a more finely sampled set of points
  
    if (roti == -1):
        rot = rotf*np.pi/180.0    # Rotation in radians
    else:
        rot = roti*45.0*np.pi/180.0    # Rotation in radians
  
    # Get rotated coordinates
    fineX =  np.cos(rot)*outvec[:,0] + np.sin(rot)*outvec[:,1]
    fineY = -np.sin(rot)*outvec[:,0] + np.cos(rot)*outvec[:,1]

    return fineX, fineY

def stimfr_shape51_227(xn, yn, x0, y0, shid, diam, theta, fill, lw, fgval, bgval, pflag):
    """
    @ Args:
        - xn (int): horizontal width of returned array
        - yn (int): vertical height of returned array
        - x0 (float): horizontal offset of center (pix)
        - y0 (float): vertical offset of center (pix)
        - shid (int): shape ID 0 to 50
        - diam (float): diameter (pix) of shape[1] (the large circle)
        - theta (float): rotation (deg)
        - fill (int): 0-outline, 1-fill
        - lw (float): linewidth for outline
        - fgval (float): shape luminance
        - bgval (float): background luminance
        - pflag (bool): plot the stimulus --- FOR TESTING ONLY
    """
  
    imax = 227    # Prepare stimuli so they are centered on a 227 pix canvas
  
    if ((xn > imax) or (yn > imax)):
        print("  *** STIMFR_SHAPE51_227  xn, yn must be <= 227 pix")
        return
    if (diam > imax):
        print("  *** STIMFR_SHAPE51_227  diam must be <= 227 pix")
        return

    margin = 57              #  More generally:  margin ~= round(imax / 4.0)
    nn = imax + 2 * margin
    
    fig = plt.figure(figsize=[nn/100.0,nn/100.0])  # control pixel size
    plt.axis('off')     # No decorations
    
    z = 1.0    # this must be larger than values of scaled shape coords.
    plt.fill([-z, z, z, -z],[-z, -z, z, z],color=[0.0,0.0,0.0])  # 0 background

    fineX, fineY =  shape51_fine_xy(shid,-1,theta)   # Get fine coordinates
    scalef = diam/352.0        # Wyeth found this constant by trial-and-error
    fineX = fineX * scalef
    fineY = fineY * scalef
    
    fineX += x0*0.0083     # x-offset
    fineY += y0*0.0083     # y-offset  *** SHOULD THIS BE + or - ???
    
    # Draw or fill the shape
    if (fill == 0):
        plt.plot(fineX,fineY,linewidth=lw,color=[1,1,1])  # Draw outline
    else:
        plt.fill(fineX,fineY,color=[1,1,1])  # Draw a filled shape
  
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ia = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  
    # Center pixel, determined from examining plot of large and small circles
    xc = 175    # Wyeth found center pixel coordinates by hand
    yc = 173
    xx = xc - round(xn/2)
    yy = yc - round(yn/2)
    s1 = ia[yy:yy+yn, xx:xx+xn, 0]   # NOTE first dim is 'y', keep RED chan
    si = np.array(s1)                # 's1' was READ ONLY, so copy it here
    
    si = si/255.0 * (fgval - bgval) + bgval
  
    if (pflag == 1):
        plt.imshow(si)    # DEBUG - show what will be returned
        plt.show()
  
    plt.close(fig)
  
    return si

def stimset_stim_get_shape_fo(d, fillflag):
    """
    @ Args:
    - d (dict): dictionary with stimulus parameters
    - fillflag (bool): 1-fill, 0-outline, -1-use value in 'd'
    """

    if (fillflag == -1):
        fill = d['fill']
    else:
        fill = fillflag
  
    s = stimfr_shape51_227(d['xn'],d['yn'],d['x0'],d['y0'],d['shid'],d['diam'],
                         d['rot'],fill,d['lw'],d['fgval'],d['bgval'],0)
  
    return s

def stimset_dict_shape_fo_1(xn, sz, lw, fg, bg):
    """
    @ Args:
    - xn (int): stimulus image pixel size
    - sz (int): pixel size of large circle (pasu_shape[1])
    - lw (float): linewidth in pixel, default=1.5
    - fg (float): foreground luminance, in [0, 1]
    - bg (float): background luminance, in [0, 1]
    """
  
    splist = []  # List of dictionary entries, one per stimulus image

    nsh = len(pasu_shape_nrotu)
    dinc = 45.0 #  The increment does not depend on # of unique rotations
  
    for si in range(nsh): # For each rotation
        nrot = pasu_shape_nrotu[si]
        r0 = 0.0
        for i in range(nrot):
            tp = {"xn":xn, "yn":xn, "x0":0, "y0":0, "shid":si, "rot":r0,
                    "diam":sz, "lw":lw, "fgval":fg, "bgval":bg}
            splist.append(tp)
            r0 += dinc
  
    print("  Length of stimulus parameter list:",len(splist))
    return splist

def get_stimulus(fillflag, xn=227, sz=50, lw=1.5, fg=1.0, bg=0.0):
    fname = [nav.datapath, "stimulus_wyeth", f"fill={fillflag}_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"]

    if not nav.exists(*fname):
        splist = stimset_dict_shape_fo_1(xn, sz, lw, fg, bg)

        stims = []
        for d in splist:
            s = stimset_stim_get_shape_fo(d, fillflag)
            stims.append([[s, s, s]])
        stims = np.vstack(stims) # shape=(B,3,227,227)
        nav.npsave(stims, *fname)
        return torch.from_numpy(stims)
    else:
        return torch.from_numpy(nav.npload(*fname))

if __name__ == "__main__":
    get_stimulus(1)