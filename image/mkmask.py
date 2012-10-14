import os
import sys

def main(indir, outdir):

    for f in os.listdir(indir):

        os.system('convert %s -channel Alpha -negate -separate %s-mask.png'%(indir + '/' + f, outdir + '/' + os.path.splitext(f)[0]))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
