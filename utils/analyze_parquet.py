import matplotlib
matplotlib.use('Agg')
from scipy.spatial import distance
import pandas as pd
import argparse,sys,os
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import auc
from matplotlib import rcParams
from scipy import integrate
plt.style.use([hep.style.ROOT, hep.style.firamath])
parser = argparse.ArgumentParser("ZPrime analyzer")
parser.add_argument("-pq",dest="parquet",type=str,required=True)
parser.add_argument("-y",dest="year",type=str,required=True)
parser.add_argument("-o",dest="opath",type=str,required=True)
args = parser.parse_args()
os.system(f"mkdir -p {args.opath}")
rlabel = f"{args.year} (13 TeV)"

_ptlow = 400
_pthigh = 1200
_msdlow = 40 
_msdhigh = 350
inlay_font = {
        #'fontfamily' : 'arial',
        #'weight' : 'normal',
        'size'   : 14
}
axis_font = {
        #'fontfamily' : 'arial',
        #'weight' : 'normal',
        'size'   : 24
}
legend_font = {
        #'family' : 'sans-serif',
        #'weight' : 'normal',
        'size'   : 20
}
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=14)
plt.rc('legend', fontsize=10)

###Global settings
nn_bins = np.linspace(-0.01,1.01,1000)
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

shortname = { "ZJetsToQQ" : "Z", "QCD" : "QCD" }


def make_response(procname,score,kinsel=1,flavorsel=1,flavorname=""):
    cut = kinsel

    sel = kinsel&flavorsel&(master_data["process"]==procname)
    if "Vector" in procname or "ZJets" in procname or "WJets" in procname: sel &= (master_data["is_Vmatched"]) 
    sample = master_data[sel]

    if type(score) == str:
      if score in master_data.columns:
        sig_score = sample[score]
      else:
        raise RuntimeError(f"Don't recognize score {score}") 
    else:
        sig_score = score[sel]

  
    response, bins, _ = plt.hist(sig_score,bins=nn_bins,density=True,weights=sample["weight"])#label=shortname[signame]+f" ({flavorname})",histtype="step",linewidth=2.,density=True)

    return response, bins, sample 

def plot_sculpting_curves(score, kinsel, title, qcdname="QCD"):
    sel = kinsel&(master_data["process"]==qcdname)
    sample = master_data[sel]

    if type(score) == str:
      if score in master_data.columns:
        sig_score = sample[score]
      else:
        raise RuntimeError(f"Don't recognize score {score}") 
    else:
        sig_score = score[sel]
    response, bins, _ = plt.hist(sig_score,bins=np.linspace(-0.01,1.01,1000),density=True,weights=sample["weight"])
    scorepdf = np.cumsum(response)*(bins[1]-bins[0])
    pctls = [.97,.95,.93,.9,.8,.5,.0]
    cuts = np.searchsorted(scorepdf,pctls)
    fig,ax = plt.subplots()
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax = axis_settings(ax)
    ax.text(0.7,0.5,title,transform=ax.transAxes,**inlay_font)
    ax.text(0.7,0.45,qcdname,transform=ax.transAxes,**inlay_font)

    inclusive,_ = np.histogram(sample["msd"],density=True,weights=sample["weight"],bins=np.linspace(40,300,15))
    for c in cuts: 
        cut_value = bins[c]
        passing_cut = (sig_score>=cut_value)
        passing_cut_hist,_ = np.histogram(sample[passing_cut]["msd"],density=True,weights=sample[passing_cut]["weight"],bins=np.linspace(40,300,15))
        print(distance.jensenshannon(inclusive,passing_cut_hist))
        ax.hist(sample[passing_cut]["msd"],density=True,histtype="step",linewidth=1.5,weights=sample[passing_cut]["weight"],
                label="$\epsilon\mathrm{_{QCD}}$="+str(int(round(passing_cut.sum()/sel.sum()*100,0)))+"%"+ "(JSD=%.3f)"%distance.jensenshannon(inclusive,passing_cut_hist),
                bins=np.linspace(40,300,15)
        )
        #print(c,cut_value,passing_cut.sum()/sel.sum(),sample[passing_cut]["msd"])
    ax.set_xlabel("Jet $m\mathrm{_{SD}}$ (GeV)", horizontalalignment='right',x=1.0,**axis_font)
    ax.set_ylabel("Events (normalized)",horizontalalignment='right',y=1.0,**axis_font)
    ax.legend(loc="upper right",prop=legend_font)
    plt.tight_layout()
    plt.savefig(args.opath+f"/{title}_mass_{qcdname}_sculpt.png")
    plt.savefig(args.opath+f"/{title}_mass_{qcdname}_sculpt.pdf")

def plot_response_and_roc(score_list,label_list,bins,xtitle="",reverse=False,truthlabel_list=[]):

    fig,ax = plt.subplots()
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax = axis_settings(ax)
    for score,label in zip(score_list,label_list):
        ax.stairs(score,bins,label=label,linewidth=2.5 )

    ax.set_yscale('log')
    ax.set_xlabel(xtitle, horizontalalignment='right',x=1.0,**axis_font)
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(1e-2,1e2)
    ax.set_ylabel("Events (normalized)",horizontalalignment='right',y=1.0,**axis_font)
    ax.legend(loc="upper right",prop=legend_font)
    plt.tight_layout()
    plt.savefig(args.opath+f"/{xtitle}_response.png")
    plt.savefig(args.opath+f"/{xtitle}_response.pdf")
    plt.clf()
    from collections import defaultdict
    tprs = defaultdict(list)
    fprs = defaultdict(list)
    fig,ax = plt.subplots()
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax = axis_settings(ax)
    for score,label,truthlabel in zip(score_list,label_list,truthlabel_list):
        if not reverse:
            if truthlabel:
                tprs[label] = [np.sum(score[ib:])/np.sum(score) for ib in range(len(bins),0,-1)]  
            else:
                fprs[label] = [np.sum(score[ib:])/np.sum(score) for ib in range(len(bins),0,-1)]  
        else:
            if truthlabel:
                tprs[label] = [np.sum(score[:ib])/np.sum(score) for ib in range(0,len(bins),1)]  
            else:
                fprs[label] = [np.sum(score[:ib])/np.sum(score) for ib in range(0,len(bins),1)] 
    for sig_key,tpr in tprs.items():
        for bkg_key,fpr in fprs.items():
             
             rounded_auc = round(integrate.trapz(tpr,fpr),2) #round(auc(fpr, tpr),2)
             ax.plot(fpr, tpr, 
                    label = f"{sig_key} vs {bkg_key} (AUC={rounded_auc})",
                    lw=2.5,
             )
    #ax.set_xscale('log')
    ax.text(0.55,0.32,f"{xtitle} score",transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.28,"Truth matched AK8 jets",transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.24,"%.0f < $p_{T}$ < %.0f"%(_ptlow,_pthigh),transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.2,"%.0f < $m_{SD}$ < %.0f"%(_msdlow,_msdhigh),transform=ax.transAxes,**inlay_font)
    ax.set_ylabel("True positive rate",horizontalalignment='right',y=1.0,**axis_font)
    ax.set_xlabel("False positive rate", horizontalalignment='right',x=1.0,**axis_font)
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    ax.legend(loc="lower right",prop=legend_font)
    plt.tight_layout() 
    print("PLOT", f"/{xtitle}_roc.png")
    plt.savefig(args.opath+f"/{xtitle}_roc.png")
    plt.savefig(args.opath+f"/{xtitle}_roc.pdf")

kinsel = (master_data["rho"] < -2)&(master_data["rho"]>-5.5)&(master_data["msd"] > _msdlow)&(master_data["msd"] < _msdhigh)& (master_data["pt"] > _ptlow)&(master_data["pt"] < _pthigh)&(master_data["nelectrons"]==0)&(master_data["nmuons"]==0)&(master_data["ntaus"]==0)

bb = ((master_data["q1_flavor"]).abs()==5)&(abs(master_data["q2_flavor"]).abs()==5)
cc = ((master_data["q1_flavor"]).abs()==4)&(abs(master_data["q2_flavor"]).abs()==4)
qq = ((master_data["q1_flavor"]).abs()<4)&(abs(master_data["q2_flavor"]).abs()<4)

#zbb_score,_    = make_response("ZJetsToQQ","particleNetMD_QCD",kinsel=kinsel,flavorsel=bb,flavorname="bb")
#zcc_score,_    = make_response("ZJetsToQQ","particleNetMD_QCD",kinsel=kinsel,flavorsel=cc,flavorname="cc")
#zqq_score,_    = make_response("ZJetsToQQ","particleNetMD_QCD",kinsel=kinsel,flavorsel=qq,flavorname="qq")

#zqq_hist,bins,sample = make_response("ZJetsToQQ","particleNetMD_Xqq",kinsel=kinsel,flavorsel=qq,flavorname="qq")
#qcd_hist,bins,sample = make_response("QCD","particleNetMD_Xqq",kinsel=kinsel,)
#plot_sculpting_curves("particleNetMD_Xqq",kinsel,"particleNetMD_Xqq")  


#plot_sculpting_curves(master_data["particleNetMD_Xqq"]/(master_data["particleNetMD_Xqq"]+master_data["particleNetMD_QCD"]),kinsel,"particleNet-MD\nqq vs QCD","QCD_HT1500to2000")  
#plot_sculpting_curves(master_data["particleNetMD_Xbb"]/(master_data["particleNetMD_Xbb"]+master_data["particleNetMD_QCD"]),kinsel,"particleNet-MD\nbb vs QCD","QCD_HT1500to2000")  
#
#plot_sculpting_curves(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel,"IN+Disco\nqq vs QCD","QCD_HT1500to2000")
#plot_sculpting_curves(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel,"IN+Disco\nbb vs QCD","QCD_HT1500to2000")

#plot_response_and_roc([zqq_hist,qcd_hist], ["Z(qq)","QCD"], bins, xtitle="particleNetMD-Xqq",reverse=False, truthlabel_list=[1,0])

#zbb_score,_    = make_response("ZJetsToQQ","zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime",kinsel=kinsel,flavorsel=bb,flavorname="bb")
#zcc_score,_    = make_response("ZJetsToQQ","zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime",kinsel=kinsel,flavorsel=cc,flavorname="cc")
#zqq_score,_    = make_response("ZJetsToQQ","zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime",kinsel=kinsel,flavorsel=qq,flavorname="qq")
#qcd_score,bins = make_response("QCD","zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime",kinsel=kinsel,)
#plot_response_and_roc([zbb_score,zcc_score,zqq_score,qcd_score], ["Z(bb)","Z(cc)","Z(qq)","QCD"], bins, xtitle="PN+Disco",reverse=False, truthlabel_list=[1,1,1,0])

#sys.exit(1)

#zcc_score,_    = make_response("ZJetsToQQ",master_data["particleNetMD_Xcc"]/(master_data["particleNetMD_Xcc"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,flavorsel=cc,flavorname="cc")
#qcd_score,bins    = make_response("QCD",master_data["particleNetMD_Xcc"]/(master_data["particleNetMD_Xcc"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,)

#plot_response_and_roc([zcc_score,qcd_score], ["Z(cc)","QCD",], bins, xtitle="PN (cc-score)",reverse=False, truthlabel_list=[1,0])
zqq_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["particleNetMD_Xqq"]/(master_data["particleNetMD_Xqq"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,flavorsel=qq,flavorname="qq")
qcd_score,bins,_    = make_response("QCD",master_data["particleNetMD_Xqq"]/(master_data["particleNetMD_Xqq"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,)
plot_response_and_roc([zqq_score,qcd_score], ["Z'(qq) m=200 GeV","QCD",], bins, xtitle="PNqq",reverse=False, truthlabel_list=[1,0])

zbb_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["particleNetMD_Xbb"]/(master_data["particleNetMD_Xbb"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,flavorsel=bb,flavorname="bb")
qcd_score,bins,_    = make_response("QCD",master_data["particleNetMD_Xbb"]/(master_data["particleNetMD_Xbb"]+master_data["particleNetMD_QCD"]),kinsel=kinsel,)
plot_response_and_roc([zbb_score,qcd_score], ["Z'(bb) m=200 GeV","QCD",], bins, xtitle="PNbb",reverse=False, truthlabel_list=[1,0])
#plot_response_and_roc([zcc_score,qcd_score], ["Z(cc)","QCD",], bins, xtitle="PN (cc-score)",reverse=False, truthlabel_list=[1,0])

zqq_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel=kinsel,flavorsel=qq,flavorname="qq")
qcd_score,bins,_    = make_response("QCD",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel=kinsel,)
plot_response_and_roc([zqq_score,qcd_score], ["Z'(qq) m=200 GeV","QCD",], bins, xtitle="INqq",reverse=False, truthlabel_list=[1,0])

zbb_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel=kinsel,flavorsel=bb,flavorname="bb")
qcd_score,bins,_    = make_response("QCD",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]/(master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"]+master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"]),kinsel=kinsel,)
plot_response_and_roc([zbb_score,qcd_score], ["Z'(bb) m=200 GeV","QCD",], bins, xtitle="INbb",reverse=False, truthlabel_list=[1,0])

zqq_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["n2b1"],kinsel=kinsel)
qcd_score,_,_    = make_response("QCD",master_data["n2b1"],kinsel=kinsel,)
plot_response_and_roc([zqq_score,qcd_score], ["Z'(qq) m=200 GeV","QCD",], bins, xtitle="n2b1",reverse=True, truthlabel_list=[1,0])


zbb_score,_,_    = make_response("VectorZPrimeToQQ_M200.root",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"],kinsel=kinsel,)
qcd_score,bins,_    = make_response("QCD",master_data["zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"],kinsel=kinsel,)
plot_response_and_roc([zbb_score,qcd_score], ["Z'(bb) m=200 GeV","QCD",], bins, xtitle="IN",reverse=True, truthlabel_list=[1,0])
