import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo2d.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov     (optional): use scat_cov instead of scat')
    print('--out     (optional): If not specified save in *_demo_*.')
    print('--map=jet (optional): If not specified use cmap=jet')
    print('--geo     (optional): If specified use cartview')
    print('--vmin|-i    (optional): specify the minimum value')
    print('--vmax|-a    (optional): specify the maximum value')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:gi:a:", ["nside", "cov","out","map","geo","vmin","vmax"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='jet'
    docart=False
    vmin=-3
    vmax=3
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-g","--geo"):
            docart=True
        elif o in ("-m","--map"):
            cmap=a[1:]
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-o", "--out"):
            outname=a[1:]
        elif o in ("-i", "--vmin"):
            vmin=float(a[1:])
        elif o in ("-a", "--vmax"):
            vmax=float(a[1:])
        else:
            print(o,a)
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))):
        print('nside should be a pwer of 2 and in [2,...,256]')
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
    else:
        import foscat.scat as sc

    log= np.load('data/out2dM_%s_log_%d.npy'%(outname,nside))

    plt.figure(figsize=(6,6))
    plt.plot(np.arange(log.shape[0])+1,log,color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Number of iteration')
    
    sst = np.load('./data/sst2dM_%s_map_%d.npy'%(outname,nside))
    im = np.load('./data/in2dM_%s_map_%d.npy'%(outname,nside))
    sm = np.load('./data/st2dM_%s_map_%d.npy'%(outname,nside))
    om = np.load('./data/out2dM_%s_map_%d.npy'%(outname,nside))

    i1=im.shape[0]//2

    start = sc.read('./data/st2dM%d_%s_%d'%(i1,outname,nside))
    out   = sc.read('./data/out2dM%d_%s_%d'%(i1,outname,nside))

    start.plot(name='Input',color='orange')
    out.plot(name='Output',color='red',hold=False)
    
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.imshow((im[i1+1]-im[i1])/sm[0],vmin=-1.0,vmax=1.0,cmap='jet',origin='lower')
    plt.subplot(2,3,2)
    plt.imshow((om[i1+1]-om[i1])/sm[0],vmin=-1.0,vmax=1.0,cmap='jet',origin='lower')
    plt.subplot(2,3,3)
    plt.imshow((om[i1]-im[i1])/sm[0],vmin=-1.0,vmax=1.0,cmap='jet',origin='lower')
    plt.subplot(2,3,4)
    plt.imshow((im[i1+1])/sm[0],vmin=12.0,vmax=18.0,cmap='jet',origin='lower')
    plt.subplot(2,3,5)
    plt.imshow((om[i1+1])/sm[0],vmin=12.0,vmax=18.0,cmap='jet',origin='lower')
    plt.subplot(2,3,6)
    plt.imshow((sst[i1+1])/sm[0],vmin=12.0,vmax=18.0,cmap='jet',origin='lower')
    
    plt.show()
    exit(0)
    plt.figure(figsize=(10,2.5*im.shape[0]))
    for k in range(im.shape[0]):
        
        plt.subplot(im.shape[0],4,1+4*k)
        plt.imshow((im[k]-np.median(im[k]))/sm[k],cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        plt.title('Model')
        plt.subplot(im.shape[0],4,2+4*k)
        plt.imshow((sst[k]-np.median(im[k]))/sm[k],vmin=vmin,vmax=vmax,cmap=cmap,origin='lower',aspect='auto')
        plt.title('Start')
        plt.subplot(im.shape[0],4,3+4*k)
        plt.imshow((om[k]-np.median(im[k]))/sm[k],cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        plt.title('Synthesis')
        plt.subplot(im.shape[0],4,4+4*k)
        plt.imshow((im[k]-om[k])/sm[k],cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        plt.title('Start-Synthesis')
    plt.show()

if __name__ == "__main__":
    main()
