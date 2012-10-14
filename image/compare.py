import os
import cv
import numpy as np
import scipy.linalg as linalg

def main(target, corpus, outdir, maxcount, thresh):

    j = cv.LoadImageM(target)
    x = np.asarray(j) / 256.
    res = x.copy()
    mod = np.zeros(np.shape(x))
    d = corpus
    C = []
    m = 1.
    count = 0

    while (m >= thresh) and (count <= maxcount):
        print(count)
        #comparison loop
        for i in os.listdir(d):
            y = np.asarray(cv.LoadImageM(d + '/' + i)) / 256.
            y /= linalg.norm(y)
            t = []
            for k in range(3):
                #c = np.matrix(y[:, :, k]).T * np.matrix(res[:, :, k])
                c = np.tensordot(y[:, :, k], res[:, :, k])
                t.append(c)
            C.append(t)

        #coef = [linalg.norm(q) for q in C]
        coef = [sum(q) for q in C]
        ind = np.argmax(coef)
        nc = C[ind]
        #print(os.listdir(d)[ind])
        y = np.asarray(cv.LoadImageM(d + '/' + os.listdir(d)[ind])) / 256.
        y /= linalg.norm(y)
        Q = np.zeros(np.shape(y))
        for n in range(3):
            #q = np.matrix(y[:, :, n]) * nc[n]
            q = y[:, :, n] * nc[n]
            Q[:, :, n] = q 
        res -= Q
        mod += Q

        mat = cv.fromarray(mod * 256)
        cv.SaveImage("%s/iterdir/iteration%i.png"%(outdir, count), mat)

        m = linalg.norm(res)**2 / linalg.norm(x)**2
        count += 1
        C = []
        print m
    
    mat = cv.fromarray(mod * 256)
    cv.SaveImage("%s/model.png"%outdir, mat)

if __name__ == "__main__":

    for corpus in ['./corpora/360p_sparse']:
  
        for t in os.listdir('./targets'):
            name = os.path.splitext(os.path.basename(t))[0]
            modname = './model/%s'%name + '_' + os.path.basename(corpus)

            if not os.path.exists(modname):
            
                os.mkdir(modname)
                os.mkdir('%s/iterdir'%modname)

            main('./targets' + '/' + t, corpus, modname, 100, 0.0001)
