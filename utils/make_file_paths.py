import yaml
from yaml.loader import SafeLoader
import os,subprocess
os.system("rm tmp_files.txt")
#/store/user/lpcpfnano/jekrupa/postprocess/3Dec22/2017/QCD_HT/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8
#os.system("xrdcp root://cmseos.fnal.gov//store/user/jkrupa/zprlegacy/file_database.yaml ./file_database.yaml")

tag="v2_3"
year = "2017"
#process = ["SingleTop","ZJetsToQQ","WJetsToQQ","QCDb","TTToSemiLeptonic",f"JetHT{year}","JetHT","SingleMuon","WJetsToLNu","VectorZPrime","TTToHadronic","TTTo2L2Nu","DYJetsToLL","VectorZPrime_MIT",]

process = [

	"/store/user/lpcpfnano/yihan/v2_2/2017/TTbar/",
	"/store/user/lpcpfnano/drankin/v2_2/2017/TTbar/",
	"/store/user/lpcpfnano/jekrupa/v2_2/2017/TTbar/",
	"/store/group/lpcpfnano/cmantill/v2_3/2017/ZJetsToQQ/",
	"/store/group/lpcpfnano/cmantill/v2_3/2017/WJetsToQQ/",
	"/store/group/lpcpfnano/cmantill/v2_3/2017/JetHT2017/",
	"/store/group/lpcpfnano/cmantill/v2_3/2017/QCD/",
]

#with open("file_database.yaml") as f:
#   data = yaml.load(f,Loader=SafeLoader)
#   #print(data)
#for key,val in data.items():
#    for subkey, subprocess in val[year].items():
#        print(subkey,subprocess)

for p in process:
  try:
    x = subprocess.check_output(f"xrdfs root://cmseos.fnal.gov/ ls {p}", shell=True,encoding='utf-8')
    subprocesses = x.split("\n")[:-1]
    print( subprocesses)
    year = p.split("/")[-2]
    print(year)
    for sp in subprocesses:
        sp = sp.split("/")[-1]
        print(f"xrdfs root://cmseos.fnal.gov/ ls {p}/{sp} > ../file_paths/{year}/{sp}.txt",)
        x = os.system(f"xrdfs root://cmseos.fnal.gov/ ls {p}/{sp} > ../file_paths/{year}/{sp}.txt",)#encoding='utf-8')
        os.system(f"sed -i 's,^,root://cmseos.fnal.gov/,' ../file_paths/{year}/{sp}.txt")
        #print([y for y in x.split("\n")])
  except:
    print(f"cannot find {p}--skipping")
