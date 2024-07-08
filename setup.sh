cd ../../..;
source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh #SL7
#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh #EL9
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh

source bamboovenv/bin/activate
voms-proxy-init -voms cms -valid 192:00
export X509_USER_PROXY=~/.x509up_u`id -u`
echo $X509_USER_PROXY
export PYTHONPATH="/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboovenv/lib/python3.9/site-packages:$PYTHONPATH"
cd - 
