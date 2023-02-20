import yaml
from yaml.loader import SafeLoader
import os,subprocess
os.system("rm tmp_files.txt")
#/store/user/lpcpfnano/jekrupa/postprocess/3Dec22/2017/QCD_HT/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8
#os.system("xrdcp root://cmseos.fnal.gov//store/user/jkrupa/zprlegacy/file_database.yaml ./file_database.yaml")

tag="9Feb22_2"
year = "2017"
process = ["SingleTop","ZJetsToQQ","WJetsToQQ","QCD_HT","TTToSemiLeptonic","JetHT","SingleMuon","WJetsToLNu","VectorZPrime","TTToHadronic","TTTo2L2Nu"]
#with open("file_database.yaml") as f:
#   data = yaml.load(f,Loader=SafeLoader)
#   #print(data)
#for key,val in data.items():
#    for subkey, subprocess in val[year].items():
#        print(subkey,subprocess)

for p in process:
    x = subprocess.check_output(f"xrdfs root://cmseos.fnal.gov/ ls /store/user/lpcpfnano/jekrupa/postprocess/{tag}/{year}/{p}", shell=True,encoding='utf-8')
    subprocesses = x.split("\n")[:-1]
    print(subprocesses)
    for sp in subprocesses:
        sp = sp.split("/")[-1]
        x = os.system(f"xrdfs root://cmseos.fnal.gov/ ls /store/user/lpcpfnano/jekrupa/postprocess/{tag}/{year}/{p}/{sp} > file_paths/{year}/{sp}.txt",)#encoding='utf-8')
        os.system(f"sed -i 's,^,root://cmseos.fnal.gov/,' file_paths/{year}/{sp}.txt")
        #print([y for y in x.split("\n")])
