import os
import cv
import numpy as np
import scipy.linalg as linalg

def main(target, corpus, outdir, maxcount, thresh):

    j = cv.LoadImageM(target)
    x = np.asarray(j) / 256.
    res = x.copy()
    mod = np.zeros(np.shape(x))
    y = mod.copy()
    mask = mod.copy()
    d = corpus
    C = []
    m = 1.
    count = 0
    ocount = 0

    while (m >= thresh) and (count <= maxcount):
        print(count)
        
        #comparison loop
        for i in os.listdir(d):
            y = np.asarray(cv.LoadImageM(d + '/' + i)) / 256.
            mask = np.asarray(cv.LoadImageM(d + '_mask/' + os.path.splitext(i)[0] + '-mask.png', 0))

            for matric in range(3):
                y[:, :, matric] *= mask    
                #y[:, :, matric][inds] /= linalg.norm(y[:, :, matric][inds])

            y /= linalg.norm(y)
            t = []
            for k in range(3):
                c = np.tensordot(y[:, :, k], res[:, :, k])
                t.append(c)
            C.append(t)

        coef = [sum(q) for q in C]
        ind = np.argmax(coef)
        nc = C[ind]
        
        y = np.asarray(cv.LoadImageM(d + '/' + os.listdir(d)[ind])) / 256.
        mask = np.asarray(cv.LoadImageM(d + '_mask/' + os.path.splitext(os.listdir(d)[ind])[0] + '-mask.png', 0))
        
        for matric in range(3):
            y[:, :, matric] *= mask
        y /= linalg.norm(y)


        #here make an n frame long fade in of the new atom
        for n in range(3):

            y[:, :, n] *= nc[n]
            res[:, :, n] -= y[:, :, n] 
            mod[:, :, n] += y[:, :, n]

        mat = cv.fromarray(mod * 256)
        cv.SaveImage("%s/iterdir/iteration%i.png"%(outdir, count), mat)

        mat = cv.fromarray(res * 256)
        cv.SaveImage("%s/resiterdir/resiteration%i.png"%(outdir, count), mat)

        mat = cv.fromarray(y * 256)
        cv.SaveImage("%s/eachiterdir/eachiteration%i.png"%(outdir, count), mat)

        m = linalg.norm(res)**2 / linalg.norm(x)**2
        count += 1
        C = []
        print m
    
    mat = cv.fromarray(mod * 256)
    cv.SaveImage("%s/model.png"%outdir, mat)
    mat = cv.fromarray(res * 256)
    cv.SaveImage("%s/residual.png"%outdir, mat)

if __name__ == "__main__":

    for corpus in ['./corpora/mask']:
  
        for t in os.listdir('./targets'):
            name = os.path.splitext(os.path.basename(t))[0]
            modname = './maskmodel/%s'%name + '_' + os.path.basename(corpus)

            if not os.path.exists(modname):
            
                os.mkdir(modname)
                os.mkdir('%s/iterdir'%modname)
                os.mkdir('%s/resiterdir'%modname)
                os.mkdir('%s/eachiterdir'%modname)
                os.system('cp %s %s'%('./targets/' + t, modname + '/' + t))
                

            main('./targets' + '/' + t, corpus, modname, 100, 0.0001)
