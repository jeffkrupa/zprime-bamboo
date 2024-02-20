import yaml
from yaml.loader import SafeLoader
import os,subprocess
os.system("rm tmp_files.txt")

file_database = "file_database.yaml"
processes = ["DYJetsToLL"]#WJetsToLNu","SingleTop","Higgs",] #["SingleMuon","SingleTop","WJetsToLNu","DYJetsToLL","JetHT","QCD_HT","WJetsToQQ", "ZJetsToQQ","VectorZPrime","TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu",]
year = "2017"

if __name__ == "__main__":
    with open(file_database,"r") as file:
      paths = yaml.safe_load(file)
      for process in processes: 
        entries = paths[process][year]
        for subp, path in entries.items():
            sp_name = subp
            os.system(f"rm ../file_paths/{year}/{sp_name}.txt",)
            for p in path:
              
                try:
                    x = subprocess.check_output(f"echo xrdfs root://cmseos.fnal.gov/ ls {p}", shell=True,encoding='utf-8')
                    #print(f"xrdfs root://cmseos.fnal.gov/ ls {p} > ../file_paths/{year}/{sp_name}.txt",)
                    x = os.system(f"xrdfs root://cmseos.fnal.gov/ ls {p} >> ../file_paths/{year}/{sp_name}.txt",)#encoding='utf-8')
                except:
                    print(f"cannot find {p}--skipping")
            os.system(f"sed -i 's,^,root://cmseos.fnal.gov/,' ../file_paths/{year}/{sp_name}.txt")
            os.system(f"sed -i '/log/d' ../file_paths/{year}/{sp_name}.txt")
            #if "QCD" in sp_name:
            #    #os.system(f"sed -i '0~9p' ../file_paths/{year}/{sp_name}.txt")
            #    os.system(f"mv ../file_paths/{year}/{sp_name}.txt ../file_paths/{year}/{sp_name}_tmp.txt")
            #    os.system(f"awk \'NR%10==0\' ../file_paths/{year}/{sp_name}_tmp.txt > ../file_paths/{year}/{sp_name}.txt")
            #    os.system(f"rm ../file_paths/{year}/{sp_name}_tmp.txt")
            os.system(f"sed -i '/SingleMuon_Run2017E\/220108_001102\/0000\/nano_data2017_184.root/d' ../file_paths/{year}/{sp_name}.txt")
