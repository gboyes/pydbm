import os
import cv
import numpy as np
import scipy.linalg as linalg
import pydbm.atom
import datetime

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
    sdf = [2, 4, 8]
    hhop = [64, 32, 16]
    whop = [128, 64, 32]
    win  = False

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
        alpha = 0.1 #+ 1./(np.random.randint(0,1000) + 0.0000000001)
        window = G.window(msw, alpha=alpha) * np.vstack(G.window(msh, alpha=alpha))
        y = y[tl[0]:br[0], tl[1]:br[1], :]

        for matric in range(planes):
            y[:, :, matric] *= mask
            if win:
                y[:, :, matric] *= window
        D.append(y)

    #comparison loop
    while (m >= thresh) and (count <= maxcount):
        print(count)
        C = []

        for oimage in D:
            oih, oiw, oip = np.shape(oimage)
            t = []
            for si, s in enumerate(sdf):
                y = np.zeros((oih/s, oiw/s, oip))
                for curh in np.arange(0, oih-(s-1), s):
                    for curw in np.arange(0, oiw-(s-1), s):
                        y[curh/s, curw/s, :] = oimage[curh, curw, :]
                y *= 1./linalg.norm(y)
                hc = 0
                while (hc + oih/s) < height:
                    wc = 0
                    while (wc + oiw/s) < width:
                        tt = []
                        for k in range(planes):
                            c = np.tensordot(y[:, :, k], res[hc:(hc + oih/s), wc:(wc + oiw/s), k])
                            tt.append(abs(c))
                        t.append(tt)
                        wc += whop[si]
                    hc += hhop[si]
            snap = np.argmax([sum(q) for q in t])
            C.append(t[snap])

        #evaluation
        coef = [sum(q) for q in C]
        ind = np.argmax(coef) #the best overall image 

        #place to store best values
        bsize = 0
        bhhop = 0
        bwhop = 0
        bcoef = 0.

        #re-compute the comparison to get the specifics (this could be a function)
        oimage = D[ind]
        oih, oiw, oip = np.shape(oimage)
        for si, s in enumerate(sdf):
            y = np.zeros((oih/s, oiw/s, oip))
            for curh in np.arange(0, oih-(s-1), s):
                for curw in np.arange(0, oiw-(s-1), s):
                    y[curh/s, curw/s, :] = oimage[curh, curw, :]
            y *= 1./linalg.norm(y)
            hc = 0
            while (hc + oih/s) < height:
                wc = 0
                while (wc + oiw/s) < width:
                    tt = []
                    for k in range(planes):
                        c = np.tensordot(y[:, :, k], res[hc:(hc + oih/s), wc:(wc + oiw/s), k])
                        tt.append(abs(c))
                    if sum(tt) > bcoef:
                       nc = tt
                       bcoef = sum(tt)
                       bsize = s
                       bhhop = hc
                       bwhop = wc
                    wc += whop[si]
                hc += hhop[si]

        y = np.zeros((oih/bsize, oiw/bsize, oip))
        for curh in np.arange(0, oih-(bsize-1), bsize):
            for curw in np.arange(0, oiw-(bsize-1), bsize):
                y[curh/bsize, curw/bsize, :] = oimage[curh, curw, :]
        y *= 1./linalg.norm(y)

        #(finally) subtract the chosen image from the residual etc
        for n in range(planes):
            y[:, :, n] *= nc[n]
            res[bhhop:(bhhop + oih/bsize), bwhop:(bwhop + oiw/bsize), n] -= y[:, :, n] 
            mod[bhhop:(bhhop + oih/bsize), bwhop:(bwhop + oiw/bsize), n] += y[:, :, n]

        #write the current, model, residual, atom
        mat = cv.fromarray(mod * 255)
        cv.SaveImage("%s/iterdir/iteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(res * 255)
        cv.SaveImage("%s/resiterdir/resiteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(y * 255)
        cv.SaveImage("%s/eachiterdir/eachiteration%i.png"%(outdir, count), mat)


        #store each iteration
        mof.write('[%s, %s, %i, %i, %i]\n'%(os.listdir(corpus)[ind], nc, bsize, bhhop, bwhop))

        #update
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
    nows = now.strftime("%Y-%m-%d_%H-%M")
    
    targs = '../targets/true_targets'
    corp = ['../corpora/sparse-alpha-bw']
    modp = '../models'
    
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
                
            main(targs + '/' + t, corpus, modname, 500, 0.0001)
            os.system('python ./fade+movie.py %s %s %s'%('%s/iterdir'%modname, '%s/vbuild_dir'%modname, '%s/model.mp4'%modname))
            #os.system('python ./fade+movie.py %s %s %s'%('%s/resiterdir'%modname, '%s/vbuild_dir'%modname, '%s/residual.mp4'%modname))
