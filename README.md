# TAOS
Synthesis 4D ocean dynamic


# Install foscat library

Before installing, make sure you have python installed in your enviroment.  For mac users it is recomended to use python=3.9*.  

The last version of the foscat library can be installed using PyPi:
```
pip install foscat
```
Load the FOSCAT_DEMO package from github.
```
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git
```


# Run foscat Data preparation


# Run foscat Model

Make sure that you have input data prepared and stored as 
```
./data/mars3dP.npy
./data/mars3dSST.npy
./data/mars3dT.npy
```
Then run following command
```
python syntheModel.py -n=128 -k -c -s=300
```
This process will create following output files
```
out2dM_demo_log_128.npy
*demo_map*.npy
out2dM[0-9]_demo_128_*.npy
st2dM[0-9]_demo_128_*
```

# Verify the Model by plotting
```
python plotresM.py -n=128 -c
```
