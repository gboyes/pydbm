import os
import cv
import numpy as np
import scipy.linalg as linalg
import pydbm.atom
import datetime

import threading
from Queue import Queue

class SizeComparisonThread(threading.Thread):
    def __init__(self, image, size, heighthop, widthhop, residual):
        self.image = image
        self.height, self.width, self.depth = np.shape(self.image) 
        self.size = size
        self.residual = residual
        self.heighthop = heighthop
        self.widthhop = widthhop
        self.bestSize = None
        self.bestHHop = None
        self.bestWHop = None
        self.coefs = None
        self.bestCoef = 0.
        threading.Thread.__init__(self)

    def get_result(self):
        return self

    def run(self):
        y = np.zeros((self.height/self.size, self.width/self.size, self.depth))
        for curh in np.arange(0, self.height-(self.size-1), self.size):
            for curw in np.arange(0, self.width-(self.size-1), self.size):
                y[curh/self.size, curw/self.size, :] = self.image[curh, curw, :]
        y *= 1./linalg.norm(y)
        hc = 0
        while (hc + self.height/self.size) < self.height:
            wc = 0
            while (wc + self.width/self.size) < self.width:
                tt = []
                for k in range(self.depth):
                    c = np.tensordot(y[:, :, k], self.residual[hc:(hc + self.height/self.size), wc:(wc + self.width/self.size), k])
                    tt.append(abs(c))
                if sum(tt) > self.bestCoef:
                    self.coefs = tt
                    self.bestCoef = sum(tt)
                    self.bestHHop = hc
                    self.bestWHop = wc
                wc += self.widthhop
            hc += self.heighthop


def get_size(image, sizes, heighthops, widthhops, residual):
    def comparison(q, sizes):
        for si, s in enumerate(sizes):
            thread = SizeComparisonThread(image, s, heighthops[si], widthhops[si], residual)
            thread.start()
            q.put(thread, True)

    result = []
    def sched(q, total_sizes):
        while len(result) < total_sizes:
            thread = q.get(True)
            thread.join()
            result.append(thread.get_result()) 

    q = Queue(1)
    comp_thread = threading.Thread(target=comparison, args=(q, sizes))
    sched_thread = threading.Thread(target=sched, args=(q, len(sizes)))
    comp_thread.start()
    sched_thread.start()
    comp_thread.join()
    sched_thread.join()

    return result[np.argmax([r.bestCoef for r in result])]

class ImageComparisonThread(threading.Thread):

    def __init__(self, image, index, sizes, heighthops, widthhops, residual):
        self.image = image
        self.index = index
        self.height, self.width, self.depth = np.shape(self.image) 
        self.sizes = sizes
        self.residual = residual
        self.heighthops = heighthops
        self.widthhops = widthhops
        self.result = None
        threading.Thread.__init__(self)

    def get_result(self):
        return self.result

    def run(self):
        self.result = get_size(self.image, self.sizes, self.heighthops, self.widthhops, self.residual)
  
def get_images(images, sizes, heighthops, widthhops, residual):
    def comparison(q, images, sizes, heighthops, widthhops):
        for i, image in enumerate(images):
            thread = ImageComparisonThread(image, i, sizes, heighthops, widthhops, residual)
            thread.start()
            q.put(thread, True)

    result = []
    def sched(q, total_images):
        while len(result) < total_images:
            thread = q.get(True)
            thread.join()
            result.append(thread.get_result()) 

    q = Queue(1)
    comp_thread = threading.Thread(target=comparison, args=(q, images, sizes, heighthops, widthhops))
    sched_thread = threading.Thread(target=sched, args=(q, len(images)))
    comp_thread.start()
    sched_thread.start()
    comp_thread.join()
    sched_thread.join()

    return result

def main(target, corpus, outdir, maxcount, thresh):

    #initialize
    j = cv.LoadImageM(target)
    x = np.asarray(j) / 256.
    height, width, planes = np.shape(x)
    res = x.copy()
    mod = np.zeros(np.shape(x))
    count = 0
    m = 1.
    G = pydbm.atom.GaborGen()

    #size decrease factors, add these to arguments for automation
    sdf = [2]
    hhop = [32]
    whop = [64]
    win = False

    #log in case you care later
    f = open('%s/analysis_params.txt'%outdir, 'w')
    f.write('targ. dims. := %s \nsize dec. factor(s) := %s \nheight hop(s) := %s \nwidth hop(s) := %s \nwindowed = %i'%(np.shape(x), sdf, hhop, whop, win))
    f.close()

    #place to store the model
    mof = open('%s/model_params.txt'%outdir, 'w')

    #read the images into ram (avoid allocating memory in comparison loop if possible)
    D = []
    for i in os.listdir(corpus):
        y = np.asarray(cv.LoadImageM(corpus + '/' + i)) / 256.
        mask = np.asarray(cv.LoadImageM(corpus + '_mask/' + os.path.splitext(i)[0] + '-mask.png', 0))

        #get the smallest rectangle that contains all of the points of interest
        mt = mask > 200
        mti = np.argwhere(mt)
        tl = [min(mti[:, 0]), min(mti[:, 1])]
        br = [max(mti[:, 0]), max(mti[:, 1])]
        mask = mask[tl[0]:br[0], tl[1]:br[1]] / 256.
        msh, msw = np.shape(mask)
        alpha = 0.1
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

        #list of 
        C = get_images(D, sdf, hhop, whop, res)
        print(len(C))
        ind = np.argmax([r.bestCoef for r in C])
        atom = C[ind]

        y = np.zeros((atom.height/atom.size, atom.width/atom.size, atom.depth))
        for curh in np.arange(0, atom.height-(atom.size-1), atom.size):
            for curw in np.arange(0, atom.width-(atom.size-1), atom.size):
                y[curh/atom.size, curw/atom.size, :] = atom.image[curh, curw, :]
        y *= 1./linalg.norm(y)

        #(finally) subtract the chosen image from the residual etc
        for n in range(planes):
            y[:, :, n] *= atom.coefs[n]
            res[atom.bestHHop:(atom.bestHHop + atom.height/atom.size), atom.bestWHop:(atom.bestWHop + atom.width/atom.size), n] -= y[:, :, n] 
            mod[atom.bestHHop:(atom.bestHHop + atom.height/atom.size), atom.bestWHop:(atom.bestWHop + atom.width/atom.size), n] += y[:, :, n]

        #write the current, model, residual, atom
        mat = cv.fromarray(mod * 256)
        cv.SaveImage("%s/iterdir/iteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(res * 256)
        cv.SaveImage("%s/resiterdir/resiteration%i.png"%(outdir, count), mat)
        mat = cv.fromarray(y * 256)
        cv.SaveImage("%s/eachiterdir/eachiteration%i.png"%(outdir, count), mat)


        #store each iteration
        mof.write('[%s, %s, %i, %i, %i]\n'%(os.listdir(corpus)[ind], atom.coefs, atom.size, atom.bestHHop, atom.bestWHop))

        m = linalg.norm(res)**2 / linalg.norm(x)**2
        count += 1
        C = []
        print m
    
    mat = cv.fromarray(mod * 256)
    cv.SaveImage("%s/model.png"%outdir, mat)
    mat = cv.fromarray(res * 256)
    cv.SaveImage("%s/residual.png"%outdir, mat)
    mof.close()

if __name__ == "__main__":

    now = datetime.datetime.now()
    nows = now.strftime("%Y-%m-%d_%H-%M")
    
    targs = '../targets/true_targets'
    corp = ['../corpora/sparse-alpha']
    modp = '../thread_mod'
    
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
                
            main(targs + '/' + t, corpus, modname, 10, 0.0001)
            os.system('python ./fade+movie.py %s %s %s'%('%s/iterdir'%modname, '%s/vbuild_dir'%modname, '%s/model.mp4'%modname))
            #os.system('python ./fade+movie.py %s %s %s'%('%s/resiterdir'%modname, '%s/vbuild_dir'%modname, '%s/residual.mp4'%modname))
