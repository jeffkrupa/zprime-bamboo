source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh
source bamboodev/bamboovenv/bin/activate
voms-proxy-init -voms cms -valid 192:00
export X509_USER_PROXY=~/.x509up_u`id -u`
