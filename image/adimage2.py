import os
import cv
import numpy as np
import scipy.linalg as linalg
import pydbm.atom
import datetime


def makeDict(image_list, height, width, sizescales, heighthops, widthhops):
    #need to count the number slots in the array

    def iteril(image_list, height, width, sizescales, heighthops, widthhops, arr=None):
        c = 0
        for i, image in enumerate(image_list):
            iheight, iwidth, idepth = np.shape(image)
            for si, s in enumerate(sizescales):
                hc = 0
                while (hc + iheight/s) < height:
                    wc = 0
                    while (wc + iwidth/s) < width:
                        if arr != None:
                            arr[c] = (i, s, hc, wc)
                        c += 1
                        wc += widthhops[si]
                    hc += heighthops[si]
                    
        if arr != None:
            return c, arr
        else:
            return c
            
    count = iteril(image_list, height, width, sizescales, heighthops, widthhops)
    D = np.zeros(count, dtype=[('index', int), ('scale', int), ('honset', int), ('wonset', int)])
    count, D = iteril(image_list, height, width, sizescales, heighthops, widthhops, arr=D)
    return D

def makeScales(image_list, scales):
    allscales = {}
    for scale in scales:
        allscales[scale] = []
        for image in image_list:
            oih, oiw, oip = np.shape(image)
            y = np.zeros((oih/scale, oiw/scale, oip))
            for curh in np.arange(0, oih-(scale-1), scale):
                for curw in np.arange(0, oiw-(scale-1), scale):
                    y[curh/scale, curw/scale, :] = image[curh, curw, :]
            y *= 1./linalg.norm(y)
            allscales[scale].append(y)
    return allscales


def hull(height, width, honset, wonset):
    H = np.zeros((height * width, 2))
    for h in range(height):
        for w in range(width):
            H[h, w] = (h+honset, w+wonset)
    return H

def main(target, corpus, outdir, maxcount, thresh):

    #initialize
    j = cv.LoadImageM(target)
    x = np.asarray(j).astype(float) / 255.
    height, width, planes = np.shape(x)
    res = x.copy()
    mod = np.zeros(np.shape(x))
    count = 0
    m = 1.
    last_m = 1.
    G = pydbm.atom.GaborGen()

    #size decrease factors, add these to arguments for automation
    sdf = [4]
    hhop = [32]
    whop = [58]
    win  = True

    #log in case you care later
    f = open('%s/analysis_params.txt'%outdir, 'w')
    f.write('targ. dims. := %s \nsize dec. factor(s) := %s \nheight hop(s) := %s \nwidth hop(s) := %s \nwindowed = %i'%(np.shape(x), sdf, hhop, whop, win))
    f.close()

    #place to store the model
    mof = open('%s/model_params.txt'%outdir, 'w')

    #read the images into ram (avoid allocating memory in comparison loop if possible)
    D = []
    for i in os.listdir(corpus):
        y = np.asarray(cv.LoadImageM(corpus + '/' + i)).astype(float) / 255.
        mask = np.asarray(cv.LoadImageM(corpus + '_mask/' + os.path.splitext(i)[0] + '-mask.png', 0))

        #get the smallest rectangle that contains all of the points of interest
        mt = mask > 200
        mti = np.argwhere(mt)
        tl = [min(mti[:, 0]), min(mti[:, 1])]
        br = [max(mti[:, 0]), max(mti[:, 1])]
        mask = mask[tl[0]:br[0], tl[1]:br[1]] / 255.
        msh, msw = np.shape(mask)
        alpha = 0.1
        window = G.window(msw, alpha=alpha) * np.vstack(G.window(msh, alpha=alpha))
        y = y[tl[0]:br[0], tl[1]:br[1], :]

        for matric in range(planes):
            y[:, :, matric] *= mask
            if win:
                y[:, :, matric] *= window
        D.append(y)

    #for the comparison
    id = makeDict(D, height, width, sdf, hhop, whop)
    scaled = makeScales(D, sdf)
    coefs = np.zeros((len(id), 3))
    upinds = np.arange(len(id))
    
    #comparison loop
    while (m >= thresh) and (count <= maxcount):
        print(count)
        for ui in upinds:
            y = scaled[id[ui]['scale']][id[ui]['index']]
            yh, yw, yp = np.shape(y)
            for k in range(planes):
                coefs[ui, k] = abs(np.tensordot(y[:, :, k], res[id[ui]['honset']:id[ui]['honset']+yh, id[ui]['wonset']:id[ui]['wonset']+yw, k]))

        #evaluation
        ind = np.argmax([np.sum(coefs[ic, :]) for ic in range(len(coefs))])
        y = scaled[id[ind]['scale']][id[ind]['index']]
        yh, yw, yp = np.shape(y)
        for n in range(planes):
            res[id[ind]['honset']:id[ind]['honset']+yh, id[ind]['wonset']:id[ind]['wonset']+yw, n] -= y[:, :, n] * coefs[ind, n] 
            mod[id[ind]['honset']:id[ind]['honset']+yh, id[ind]['wonset']:id[ind]['wonset']+yw, n] += y[:, :, n] * coefs[ind, n]

        #determine update region
        update = []
        atn = id[ind]
        for ai, atom in enumerate(id):
            ah, aw, ap = np.shape(scaled[atom['scale']][atom['index']])
            if (((atom['honset'] >= atn['honset']) and (atom['honset'] < (atn['honset'] + yh)) or
            (atom['honset'] + ah >= atn['honset']) and (atom['honset'] + ah < (atn['honset'] + yh))) and 
            ((atom['wonset'] >= atn['wonset']) and (atom['wonset'] < (atn['wonset'] + yw)) or
            (atom['wonset'] + ah >= atn['wonset']) and (atom['wonset'] + ah < (atn['wonset'] + yw)))) :
                update.append(ai)
        upinds = np.array(update)

        #write the current, model, residual, atom
        '''
        mat = cv.fromarray(mod * 255)
        cv.SaveImage("%s/iterdir/iteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(res * 255)
        cv.SaveImage("%s/resiterdir/resiteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(y * 255)
        cv.SaveImage("%s/eachiterdir/eachiteration%i.png"%(outdir, count), mat)
        '''

        #store each iteration
        mof.write('[%s, %s, %i, %i, %i]\n'%(os.listdir(corpus)[id[ind]['index']], coefs[ind], id[ind]['scale'], id[ind]['honset'], id[ind]['wonset']))

        #update measure
        m = linalg.norm(res)**2 / linalg.norm(x)**2
        if m > last_m:
            break
        last_m = m
        count += 1
        print m
    
    mat = cv.fromarray(mod * 255)
    cv.SaveImage("%s/model.png"%outdir, mat)
    mat = cv.fromarray(res * 255)
    cv.SaveImage("%s/residual.png"%outdir, mat)
    mof.close()

if __name__ == "__main__":

    now = datetime.datetime.now()
    nows = now.strftime("%Y-%m-%d_%H_%M")
    
    targs = '/home/frobenius/Desktop/Tech-images2'
    corp = ['../corpora/sparse-alpha']
    modp = '../vidtestmodels2'
    
    for corpus in corp:
  
        for t in os.listdir(targs):
            name = os.path.splitext(os.path.basename(t))[0]
            modname = '%s/%s'%(modp, name) + '_' + nows + '_' + os.path.basename(corpus)

            if not os.path.exists(modname):
            
                os.mkdir(modname)
                os.mkdir('%s/iterdir'%modname)
                os.mkdir('%s/resiterdir'%modname)
                os.mkdir('%s/eachiterdir'%modname)
                os.mkdir('%s/vbuild_dir'%modname)
                os.system('cp %s %s'%(targs + '/' + t, modname + '/' + t))
                
            main(targs + '/' + t, corpus, modname, 1000, 0.0001)
            #os.system('python ./fade+movie.py %s %s %s'%('%s/iterdir'%modname, '%s/vbuild_dir'%modname, '%s/model.mp4'%modname))
            #os.system('python ./fade+movie.py %s %s %s'%('%s/resiterdir'%modname, '%s/vbuild_dir'%modname, '%s/residual.mp4'%modname))
