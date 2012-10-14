import os
import sys
import numpy as np
import cv

def main(indir, outdir, vidplace):

    nframes = 5
    count = 0
    mod = np.zeros((360, 640, 3))
    
    k = os.listdir(indir)
    kp = np.argsort([eval(os.path.splitext(k_.strip('iteration'))[0]) for k_ in k])

    for i in kp:
        f = k[i]
        x = np.asarray(cv.LoadImageM(indir + '/' + f)) / 256.

        for a in np.linspace(0., 1., nframes):
            mod += x * a * 3.#just a scalar to brighten a bit
            mat = cv.fromarray(mod/(count+1) * 256)
            cv.SaveImage("%s/frame%s.png"%(outdir, str(count).zfill(5)), mat)
            count += 1

    os.system('ffmpeg -r 24 -b 4000 -i %s/frame%%05d.png %s'%(outdir, vidplace))
    os.system('rm %s/*.png'%outdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

    
