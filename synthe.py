import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt
from scipy.ndimage import gaussian_filter

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo2d.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-g|--gauss][-k|--k5x5][-d|--data][-o|--out]')
    print('-n : is the n of the input map (nxn)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--data  (optional): If not specified use TURBU.npy.')
    print('--out   (optional): If not specified save in *_demo_*.')
    exit(0)

def expand_data(im):
    imout=im.copy()
    nx,ny=imout.shape
    ww=np.array([0.2,0.5,0.2,0.5,1.0,0.5,0.2,0.5,0.2])

    idx=np.where(imout[0,:]<0)[0]
    ii=1
    while idx.shape[0]>0 and ii<imout.shape[0]:
        idx2=np.where((imout[0,:]<0)*(imout[ii,:]>0))[0]
        if idx2.shape[0]>0:
            print(ii,idx2.shape[0])
            imout[0,idx2]=imout[ii,idx2]
        ii=ii+1
        idx=np.where(imout[0,:]<0)[0]
        
    idx=np.where(imout[-1,:]<0)[0]
    ii=-2
    while idx.shape[0]>0 and -ii<imout.shape[0]:
        idx2=np.where((imout[-1,:]<0)*(imout[ii,:]>0))[0]
        if idx2.shape[0]>0:
            print(ii,idx2.shape[0])
            imout[-1,idx2]=imout[ii,idx2]
        ii=ii-1
        idx=np.where(imout[0,:]<0)[0]
        
    idx=np.where(imout[:,0]<0)[0]
    ii=1
    while idx.shape[0]>0 and ii<imout.shape[1]:
        idx2=np.where((imout[:,0]<0)*(imout[:,ii]>0))[0]
        if idx2.shape[0]>0:
            print(ii,idx2.shape[0])
            imout[idx2,0]=imout[idx2,ii]
        ii=ii+1
        idx=np.where(imout[:,0]<0)[0]
        
    idx=np.where(imout[:,-1]<0)[0]
    ii=-2
    while idx.shape[0]>0 and -ii<imout.shape[1]:
        idx2=np.where((imout[:,-1]<0)*(imout[:,ii]>0))[0]
        if idx2.shape[0]>0:
            print(ii,idx2.shape[0])
            imout[idx2,-1]=imout[idx2,ii]
        ii=ii-1
        idx=np.where(imout[:,0]<0)[0]
        
    idx2=np.where(imout>=0)
    mask=np.zeros([nx,ny])
    mask[idx2[0],idx2[1]]=1.0
    idx=np.where((mask[1:-1,1:-1]==0))
    
    while idx[0].shape[0]>0:
        idx2=np.where(imout>=0)
        mask=np.zeros([nx,ny])
        mask[idx2[0],idx2[1]]=1.0
        
        vv = ww[0]*imout[:-2,:-2]*mask[:-2,:-2]   +ww[1]*imout[1:-1,:-2]*mask[1:-1,:-2]   +ww[2]*imout[2:,:-2]*mask[2:,:-2]+ \
             ww[3]*imout[:-2,1:-1]*mask[:-2,1:-1] +ww[4]*imout[1:-1,1:-1]*mask[1:-1,1:-1] +ww[5]*imout[2:,1:-1]*mask[2:,1:-1]+ \
             ww[6]*imout[:-2,2:]*mask[:-2,2:]     +ww[7]*imout[1:-1,2:]*mask[1:-1,2:]     +ww[8]*imout[2:,2:]*mask[2:,2:]

        mv = ww[0]*mask[:-2,:-2] +ww[1]*mask[1:-1,:-2]  +ww[2]*mask[2:,:-2]+ \
             ww[3]*mask[:-2,1:-1]+ww[4]*mask[1:-1,1:-1] +ww[5]*mask[2:,1:-1]+ \
             ww[6]*mask[:-2,2:]  +ww[7]*mask[1:-1,2:]   +ww[8]*mask[2:,2:]
        
        idx=np.where((mv>0)*(mask[1:-1,1:-1]==0))
        print(idx[0].shape[0])
        imout[idx[0]+1,idx[1]+1]=vv[idx[0],idx[1]]/mv[idx[0],idx[1]]

    imout[imout<0]=np.median(imout[im>0])
    return(imout)
    

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xpgkd:o:", \
                                   ["nside", "cov","seed","steps","gauss","k5x5","data","out"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=100
    dop00=False
    dogauss=False
    KERNELSZ=3
    seed=1234
    outname='demo'
    data="wave.npy"
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            print('Use SEED = ',seed)
        elif o in ("-o", "--out"):
            outname=a[1:]
            print('Save data in ',outname)
        elif o in ("-d", "--data"):
            data=a[1:]
            print('Read data from ',data)
        elif o in ("-g", "--gauss"):
            dogauss=True
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>2048:
        print('n should be a power of 2 and in [2,...,2048]')
        usage()
        exit(0)

    print('Work with n=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
        print('Work with ScatCov')
    else:
        import foscat.scat as sc
        print('Work with Scat')
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = '../data'

    #=================================================================================
    # Get data
    #=================================================================================
    im=np.load('/home1/scratch/jmdeloui/TAOS/mars3dT.npy')[48+24,39]
    pim=np.load('/home1/scratch/jmdeloui/TAOS/mars3dP.npy')[48+24]
    tim=np.load('/home1/scratch/jmdeloui/TAOS/mars3dSST.npy')[48+24]
    
    if nside<im.shape[0]:
        im=im[im.shape[0]//2-nside//2:im.shape[0]//2+nside//2,
              im.shape[1]//2-nside//2:im.shape[1]//2+nside//2]
        pim=pim[pim.shape[0]//2-nside//2:pim.shape[0]//2+nside//2,
                pim.shape[1]//2-nside//2:pim.shape[1]//2+nside//2]
        tim=tim[tim.shape[0]//2-nside//2:tim.shape[0]//2+nside//2,
                tim.shape[1]//2-nside//2:tim.shape[1]//2+nside//2]
    mask=np.expand_dims(pim>0,0)
    
    masktim=np.expand_dims((tim>0)*(pim>0),0)
    pim[pim>0]=np.log(pim[pim>0])
    """
    plt.subplot(1,3,1)
    plt.imshow(tim,cmap='jet',vmin=9,vmax=20,origin='lower')
    plt.subplot(1,3,2)
    plt.imshow(expand_data(tim),cmap='jet',vmin=9,vmax=20,origin='lower')
    plt.subplot(1,3,3)
    plt.imshow(expand_data(tim)/masktim[0],cmap='jet',vmin=9,vmax=20,origin='lower')
    plt.show()
    exit(0)
    """
    im=expand_data(im)
    tim=expand_data(tim)

    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  # define the kernel size
                     OSTEP=0,           # get very large scale (nside=1)
                     LAMBDA=lam,
                     TEMPLATE_PATH=scratch_path,
                     use_R_format=True,
                     chans=1,
                     slope=0.5,
                     all_type='float32')
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
    
    def lossX(x,scat_operator,args):
        
        ref = args[0]
        refx = args[1]
        im  = args[2]
        pim = args[3]
        mask = args[4]

        learn=scat_operator.eval(x,mask=mask)
        learnx=scat_operator.eval(x,image2=pim,mask=mask)
            
        loss=scat_operator.reduce_sum(scat_operator.square(ref-learn))
        loss=loss+scat_operator.reduce_sum(scat_operator.square(refx-learnx))

        return(loss*3E-2)

    def lossT(x,scat_operator,args):
        
        tim = args[0]
        masktim = args[1]

        loss=scat_operator.backend.bk_reduce_sum(scat_operator.backend.bk_square(masktim[0]*(x.data[0,2:-2,2:-2]-tim)))
    
        return(loss)

    ref=scat_op.eval(im,mask=mask)
    refX=scat_op.eval(im,image2=pim,mask=mask)

    loss1=synthe.Loss(lossX,scat_op,ref,refX,im,pim,mask)
    loss2=synthe.Loss(lossT,scat_op,tim,masktim)
        
    sy = synthe.Synthesis([loss1,loss2])
    
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================
    np.random.seed(seed)
    imap=im
    
    omap=sy.run(imap,
                EVAL_FREQUENCY = 10,
                do_lbfgs=True,
                NUM_EPOCHS = nstep)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    
    start=scat_op.eval(imap)
    out =scat_op.eval(omap)
    
    np.save('in2d_%s_map_%d.npy'%(outname,nside),im)
    np.save('sst2d_%s_map_%d.npy'%(outname,nside),tim)
    np.save('st2d_%s_map_%d.npy'%(outname,nside),mask[0])
    np.save('stm2d_%s_map_%d.npy'%(outname,nside),masktim[0])
    np.save('out2d_%s_map_%d.npy'%(outname,nside),omap)
    np.save('out2d_%s_log_%d.npy'%(outname,nside),sy.get_history())

    refX.save('in2d_%s_%d'%(outname,nside))
    start.save('st2d_%s_%d'%(outname,nside))
    out.save('out2d_%s_%d'%(outname,nside))

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
