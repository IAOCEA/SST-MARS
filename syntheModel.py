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
            imout[0,idx2]=imout[ii,idx2]
        ii=ii+1
        idx=np.where(imout[0,:]<0)[0]
        
    idx=np.where(imout[-1,:]<0)[0]
    ii=-2
    while idx.shape[0]>0 and -ii<imout.shape[0]:
        idx2=np.where((imout[-1,:]<0)*(imout[ii,:]>0))[0]
        if idx2.shape[0]>0:
            imout[-1,idx2]=imout[ii,idx2]
        ii=ii-1
        idx=np.where(imout[0,:]<0)[0]
        
    idx=np.where(imout[:,0]<0)[0]
    ii=1
    while idx.shape[0]>0 and ii<imout.shape[1]:
        idx2=np.where((imout[:,0]<0)*(imout[:,ii]>0))[0]
        if idx2.shape[0]>0:
            imout[idx2,0]=imout[idx2,ii]
        ii=ii+1
        idx=np.where(imout[:,0]<0)[0]
        
    idx=np.where(imout[:,-1]<0)[0]
    ii=-2
    while idx.shape[0]>0 and -ii<imout.shape[1]:
        idx2=np.where((imout[:,-1]<0)*(imout[:,ii]>0))[0]
        if idx2.shape[0]>0:
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
        imout[idx[0]+1,idx[1]+1]=vv[idx[0],idx[1]]/mv[idx[0],idx[1]]

    imout[imout<0]=np.median(imout[im>0])
    return(imout)
    

def main():
    test_mpi=False
    for ienv in os.environ:
        if 'OMPI_' in ienv:
            test_mpi=True
        if 'PMI_' in ienv:
            test_mpi=True

    size=1
    if test_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

    if size>1:
        print('Use mpi facilities Rk=%d Size=%d'%(rank,size))
        isMPI=True
    else:
        size=1
        rank=0
        isMPI=False

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
    data="data"
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            if rank==0:
                print('Use SEED = ',seed)
        elif o in ("-o", "--out"):
            outname=a[1:]
            if rank==0:
                print('Save data in ',outname)
        elif o in ("-d", "--data"):
            data=a[1:]
            if rank==0:
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

    if rank==0:
        print('Work with n=%d'%(nside))

    sys.stdout.flush()

    if cov:
        import foscat.scat_cov as sc
        if rank==0:
            print('Work with ScatCov')
    else:
        import foscat.scat as sc
        if rank==0:
            print('Work with Scat')
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = '../data'

    n1=1
    n2=8

    #=================================================================================
    # Get data
    #=================================================================================
    im=np.load('%s/mars3dT.npy'%(data))[n1:n2,39]
    pim=np.load('%s/mars3dP.npy'%(data))[n1:n2]
    tim=np.load('%s/mars3dSST.npy'%(data))[n1:n2]
    
    if nside<im.shape[1]:
        im=im[:,im.shape[1]//2-nside//2:im.shape[1]//2+nside//2,
              im.shape[2]//2-nside//2:im.shape[2]//2+nside//2]
        pim=pim[:,pim.shape[1]//2-nside//2:pim.shape[1]//2+nside//2,
                pim.shape[2]//2-nside//2:pim.shape[2]//2+nside//2]
        tim=tim[:,tim.shape[1]//2-nside//2:tim.shape[1]//2+nside//2,
                tim.shape[2]//2-nside//2:tim.shape[2]//2+nside//2]

    i_tim={}
    i_tim[0]=0
    n_tim=0
    for i in range(tim.shape[0]):
        if np.sum(tim[i]-tim[i_tim[n_tim]])!=0:
            i_tim[n_tim+1]=i
            n_tim=n_tim+1
    i_tim[n_tim+1]=tim.shape[0]
    n_tim=n_tim+1

    if rank==0:
        print('SST STEP ',[i_tim[i] for i in range(n_tim+1)])
        sys.stdout.flush()

    mask=np.expand_dims(pim>0,0)
    masktim=(tim>0)*(pim>0)
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
    ntime=im.shape[0]
    for i in range(ntime):
        im[i]=expand_data(im[i])
        tim[i]=expand_data(tim[i])

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
                     isMPI=isMPI,
                     mpi_size=size,
                     mpi_rank=rank,
                     all_type='float32')
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
    
    def lossX(x,scat_operator,args):
        
        ref = args[0]
        refx = args[1]
        reft = args[2]
        im  = args[3]
        pim = args[4]
        mask = args[5]
        maskt= args[6]
        k    = args[7]

        learn=scat_operator.eval(x[k],mask=mask)
        learnx=scat_operator.eval(x[k],image2=pim,mask=mask)
        if k>0:
            learnt=scat_operator.eval(x[k-1],image2=x[k],mask=mask)
            
        loss=scat_operator.reduce_sum(scat_operator.square(ref-learn))
        loss=loss+scat_operator.reduce_sum(scat_operator.square(refx-learnx))
        if k>0:
            loss=loss+scat_operator.reduce_sum(scat_operator.square(reft-learnt))

        return(loss)

    def lossT(x,scat_operator,args):
        
        tim = args[0]
        masktim = args[1]
        k1    = args[2]
        k2    = args[3]

        tmp=x[k1].data[0,2:-2,2:-2]
        for k in range(k1+1,k2):
            tmp=tmp+x[k].data[0,2:-2,2:-2]
        loss=scat_operator.backend.bk_reduce_sum(scat_operator.backend.bk_square(masktim*(tmp-tim)))
        """
        tmp=x[k1]
        for k in range(k1+1,k2):
            tmp=tmp+x[k]

        learn=scat_operator.eval(tmp,mask=masktim)
        
        loss=scat_operator.backend.bk_reduce_sum(scat_operator.backend.bk_square(ref.S1-learn.S1))
        """
        return(loss)

    all_loss=[]
    
    for i in range(ntime):
        if i%size==rank:
            print('Create loss for time step ',i)
            sys.stdout.flush()
            ref=scat_op.eval(im[i],mask=mask[:,i])
            refX=scat_op.eval(im[i],image2=pim[i],mask=mask[:,i])
            if i>0:
                refT=scat_op.eval(im[i-1],image2=im[i],mask=mask[:,i]*mask[:,i-1])
            else:
                refT=None

            loss1=synthe.Loss(lossX,scat_op,ref,refX,refT,im[i],pim[i],mask[:,i],mask[:,i]*mask[:,i-1],i)
            all_loss=all_loss+[loss1]

    for i in range(n_tim):
        if (i+ntime)%size==rank:
            print('TIME STEP ',i_tim[i],i_tim[i+1])
            sys.stdout.flush()
            loss2=synthe.Loss(lossT,scat_op,np.sum(tim[i_tim[i]:i_tim[i+1]],0),masktim[i_tim[i]],i_tim[i],i_tim[i+1])
            all_loss=all_loss+[loss2]

    print('rank=%d NbLoss=%d'%(rank,len(all_loss)))
    sys.stdout.flush()
        
    sy = synthe.Synthesis(all_loss)
    
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================
    np.random.seed(seed)
    imap=im
    
    omap=sy.run(imap,
                EVAL_FREQUENCY = 1,
                do_lbfgs=True,
                NUM_EPOCHS = nstep,
                axis=1)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    
    
    np.save('./data/in2dM_%s_map_%d.npy'%(outname,nside),im)
    np.save('./data/sst2dM_%s_map_%d.npy'%(outname,nside),tim)
    np.save('./data/st2dM_%s_map_%d.npy'%(outname,nside),mask[0])
    np.save('./data/stm2dM_%s_map_%d.npy'%(outname,nside),masktim[0])
    np.save('./data/out2dM_%s_map_%d.npy'%(outname,nside),omap)
    np.save('./data/out2dM_%s_log_%d.npy'%(outname,nside),sy.get_history())

    for i in range(imap.shape[0]):
        start=scat_op.eval(imap[i])
        out =scat_op.eval(omap[i])
        start.save('./data/st2dM%d_%s_%d'%(i,outname,nside))
        out.save('./data/out2dM%d_%s_%d'%(i,outname,nside))
    
    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
