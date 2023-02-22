import pandas as pd
import argparse,sys
import mplhep as mpl
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser("ZPrime analyzer")
parser.add_argument("-pq",dest="parquet",type=str,required=True)
parser.add_argument("-y",dest="year",type=str,required=True)


args = parser.parse_args()

master_data = pd.read_parquet(args.parquet)

def axis_settings(ax):
    import matplotlib.ticker as plticker
    #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    ax.xaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.tick_params(direction='in', axis='both', which='major', labelsize=24, length=12)#, labelleft=False )
    ax.tick_params(direction='in', axis='both', which='minor' , length=6)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')    
    #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
    ax.grid(which='major', alpha=0.9, linestyle='dotted')
    return ax

def make_roc(signame,bkgname,score,title="",kinsel=1,flavorsel=1):
    cut = kinsel

    sigsel = kinsel&flavorsel&(master_data["process"]==signame)
    bkgsel = kinsel&(master_data["process"]==bkgname)
    sig = master_data[sigsel]
    bkg = master_data[bkgsel]

    if type(score) == str:
      if score in master_data.columns:
        sig_score = sig[score]
        bkg_score = bkg[score]
      else:
        raise RuntimeError(f"Don't recognize score {score}") 
    else:
        sig_score = score[sigsel]
        bkg_score = score[bkgsel]
    print(sig_score,bkg_score)

    plt.clf()
    fig,ax = plt.subplots()
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax = axis_settings(ax)
    #bins=None

make_roc("ZJetsToQQ","QCD","particleNetMD_QCD","",flavorsel=((master_data["q1_flavor"]).abs()==5)&(abs(master_data["q2_flavor"]).abs()==5))
make_roc("ZJetsToQQ","QCD",master_data["particleNetMD_Xbb"]/(master_data["particleNetMD_Xbb"]+master_data["particleNetMD_QCD"]))
