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
import tqdm
import scipy.ndimage as sc
plt.style.use([hep.style.ROOT, hep.style.firamath])
parser = argparse.ArgumentParser("ZPrime analyzer")
parser.add_argument("-pq",dest="parquet",type=str,required=True)
parser.add_argument("-y",dest="year",type=str,required=True)
parser.add_argument("-o",dest="opath",type=str,required=True)
args = parser.parse_args()
os.system(f"mkdir -p {args.opath}")
os.system(f"mkdir -p {args.opath}/distributions/")
rlabel = f"{args.year} (13 TeV)"

_ptlow = 200
_pthigh = 1200
_msdlow = 1 
_msdhigh = 400
_rholow = -7
_rhohigh = -1.5
_nbins = 81

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
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=18)
plt.rc('legend', fontsize=10)

###Global settings
nn_bins = np.linspace(-0.01,1.01,10000)
print('pre-import')
import pyarrow.parquet as pq


parquet_file = pq.ParquetFile(args.parquet,memory_map=True)
master_data = parquet_file.read().to_pandas()

#master_data = pq.read_table(args.parquet,)
#master_data = [pd.read_parquet(p,engine="fastparquet") for p in args.parquet]
#master_data = master_data.to_pandas()
print("imported ", master_data)
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
    #print(sample.pt)
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
    #print(sample)
    if type(score) == str:
      if score in master_data.columns:
        sig_score = sample[score]
      else:
        raise RuntimeError(f"Don't recognize score {score}") 
    else:
        sig_score = score[sel]
    print("weight",sample["weight"])
   
    plt.clf()
    fig,ax=plt.subplots() 
    response, bins, _ = ax.hist(sig_score,bins=np.linspace(-0.01,1.01,1000),density=True,weights=sample["weight"])
    ax.set_xlabel("tagger score")
    ax.set_ylabel("Events")
    plt.savefig(args.opath+f"/{title}_mass_{qcdname}_scores.png")
    plt.savefig(args.opath+f"/{title}_mass_{qcdname}_scores.pdf")
    plt.clf()
    response = response/np.sum(response)/np.diff(bins)
    scorepdf = np.cumsum(response)*np.diff(bins)
    print("scorepdf",scorepdf)
    pctls = [.97,.95,.93,.9,.8,.5,.0]
    cuts = np.searchsorted(scorepdf,pctls)
    print("cuts",cuts)
    fig,ax = plt.subplots()
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax = axis_settings(ax)
    ax.text(0.7,0.5,title,transform=ax.transAxes,**inlay_font)
    ax.text(0.7,0.45,qcdname,transform=ax.transAxes,**inlay_font)

    inclusive,_ = np.histogram(sample["msd"],density=True,weights=sample["weight"],bins=np.linspace(40,300,15))
    for c in cuts: 
        cut_value = bins[c]
        passing_cut = (sig_score>=cut_value)
        print("c,bins[c],cut_value,len(passing_cut)",c,bins[c],cut_value,len(passing_cut))
        passing_cut_hist,_ = np.histogram(sample[passing_cut]["msd"],density=True,weights=sample[passing_cut]["weight"],bins=np.linspace(40,300,15))
        #print(passing_cut)
        #print(distance.jensenshannon(inclusive,passing_cut_hist))
        print(sample[passing_cut]["weight"],)
        print(sample["weight"])
        ax.hist(sample[passing_cut]["msd"],density=True,histtype="step",linewidth=1.5,weights=sample[passing_cut]["weight"],
                label="$\epsilon\mathrm{_{QCD}}$="+str(int(round(sample[passing_cut]["weight"].sum()/sample["weight"].sum()*100,0)))+"%"+ "(JSD=%.3f)"%np.nan_to_num(distance.jensenshannon(inclusive,passing_cut_hist)),
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
    ax.set_ylim(1e-5,1e2)
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
    #print(tprs,fprs) 
    for sig_key,tpr in tprs.items():
        if 'qq' in sig_key: proc = 'qq'
        if 'cc' in sig_key: proc = 'cc'
        if 'bb' in sig_key: proc = 'bb'
        for bkg_key,fpr in fprs.items():
           if ("PN" in sig_key and "PN" in bkg_key):
               leg_text = "ParticleNetMD_"+proc
           elif ("T" in sig_key and "T" in bkg_key):
               leg_text = "Transformer_"+proc 
           else:
               continue
           rounded_auc = round(integrate.trapz(tpr,fpr),2) #round(auc(fpr, tpr),2)
           ax.plot(fpr, tpr, 
                    label = f"{leg_text} (AUC={rounded_auc})",
                    lw=2.5,
           )
    #ax.set_xscale('log')
    ax.text(0.55,0.32,f"{xtitle} score",transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.28,"Truth matched AK8 jets",transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.24,"%.0f < $p_{T}$ < %.0f"%(_ptlow,_pthigh),transform=ax.transAxes,**inlay_font)
    ax.text(0.55,0.2,"%.0f < $m_{SD}$ < %.0f"%(_msdlow,_msdhigh),transform=ax.transAxes,**inlay_font)
    ax.set_ylabel("True positive rate",horizontalalignment='right',y=1.0,**axis_font)
    ax.set_xlabel("False positive rate", horizontalalignment='right',x=1.0,**axis_font)
    ax.set_xlim(0.0,1.0)
    ax.set_ylim(0.0,1.0)
    ax.legend(loc="lower right",prop=legend_font)
    plt.tight_layout() 
    plt.savefig(args.opath+f"/{xtitle}_roc.png")
    plt.savefig(args.opath+f"/{xtitle}_roc.pdf")

    return tprs,fprs


def find_pctl(score,weights,ibin,pctl=0.05,reverse=False,):
    pdf,bins = np.histogram(score, bins=np.linspace(0,0.5,100), weights=weights,density=True,)
    #fig,ax = plt.subplots()
    #ax = axis_settings(ax)
    cdf = np.cumsum(pdf)*np.diff(bins)
    pctl_bin = np.searchsorted(cdf, [pctl if reverse else 1.-pctl])
    #ax.stairs(pdf, bins)
    #ax.set_xlabel(r"Jet $N_2$")
    #ax.set_ylabel(r"Normalized events")
    #ax.axvline(bins[pctl_bin[0]],linestyle="--",color="red",lw=2,)
    #plt.savefig(args.opath+f"/distributions/rho_pt_{ibin}.png")
    #plt.savefig(args.opath+f"/distributions/rho_pt_{ibin}.pdf")
    return bins[pctl_bin[0]]


def make_ddt_map(qcdname, score, msd, pt, weights, taggername, reverse=False):

    rho = 2*np.log(msd/pt)

    plt.clf()
    fig,ax=plt.subplots()
    ax = axis_settings(ax)
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax.hist(pt,bins=np.linspace(_ptlow,_pthigh,_nbins),weights=weights,histtype="step",lw=2,label="QCD")
    ax.set_xlabel("Jet $p_{T}$ (GeV)")   
    ax.set_ylabel(r"Events")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.opath+f"/pt.png")
    plt.savefig(args.opath+f"/pt.pdf")

      
    plt.clf()
    fig,ax=plt.subplots()
    ax = axis_settings(ax)
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    ax.hist(rho,bins=np.linspace(_rholow,_rhohigh,_nbins),weights=weights,histtype="step",lw=2,label="QCD")
    ax.set_xlabel(r"Jet $\rho$ (GeV)")   
    ax.set_ylabel(r"Events")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.opath+f"/rho.png")
    plt.savefig(args.opath+f"/rho.pdf")


    plt.clf()
    fig,ax=plt.subplots()
    ax = axis_settings(ax)

    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    h2, rhoedges, ptedges, im = plt.hist2d(rho, pt,
                                         bins=[np.linspace(_rholow, _rhohigh, _nbins), np.linspace(_ptlow,_pthigh,_nbins)] ,
                                         weights=weights, density=False, 
                                         norm=matplotlib.colors.LogNorm(),
                                         )
    ax.set_xlim(_rholow, _rhohigh)
    ax.set_ylim(_ptlow,_pthigh)
    ax.set_xlabel(r"Jet $\rho$")
    ax.set_ylabel("Jet $p_{T}$ (GeV)")   
    ax.set_aspect(abs(_rhohigh-_rholow)/(_pthigh-_ptlow))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.opath+f"/rho_pt.png") 
    plt.savefig(args.opath+f"/rho_pt.pdf")

    #sys.exit()
    ddt_map = np.zeros(shape=(h2.shape[0],h2.shape[1]))
    counter = 0
    for irho in tqdm.tqdm(range(len(rhoedges)-1)):
        for ipt in range(len(ptedges)-1):
            sel = (rho <= rhoedges[irho+1]) & (rho > rhoedges[irho]) & (pt <= ptedges[ipt+1]) & (pt > ptedges[ipt])
            score_tmp,weights_tmp = score[sel], weights[sel]
            #print(score_tmp,weights_tmp)
            ddt_map[irho, ipt] = find_pctl(score_tmp,weights_tmp,"pt_{}_{}_rho_{}_{}".format(round(rhoedges[irho],2),round(rhoedges[irho+1],2),round(ptedges[ipt],2),round(ptedges[ipt+1],2)),reverse=reverse)
            counter += 1 
    plt.clf()
    fig,ax = plt.subplots()
    ax = axis_settings(ax)
    plt.imshow(ddt_map.T,origin="lower",extent=[_rholow,_rhohigh,_ptlow,_pthigh,],aspect=abs(_rhohigh-_rholow)/(_pthigh-_ptlow),interpolation='none',)
    ax.set_xlabel(r"Jet $\rho$")
    ax.set_ylabel("Jet $p_{T}$ (GeV)")
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.opath+f"/{taggername}_ddtmap_rho_pt.png")
    plt.savefig(args.opath+f"/{taggername}_ddtmap_rho_pt.pdf")

    fig,ax = plt.subplots()
    ax = axis_settings(ax)
    ddt_map_smoothed = sc.gaussian_filter(ddt_map.T,1)
    np.savez(args.opath+f"/ddt_map_smoothed",ddt_map=ddt_map_smoothed,rhoedges=rhoedges,ptedges=ptedges)
    plt.imshow(ddt_map_smoothed,origin="lower",extent=[_rholow,_rhohigh,_ptlow,_pthigh,],aspect=abs(_rhohigh-_rholow)/(_pthigh-_ptlow),interpolation='none',)
    ax.set_xlabel(r"Jet $\rho$")
    ax.set_ylabel("Jet $p_{T}$ (GeV)")
    hep.cms.label("Preliminary",rlabel=rlabel, data=False)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.opath+f"/{taggername}_ddtmap_rho_pt_smoothed.png")
    plt.savefig(args.opath+f"/{taggername}_ddtmap_rho_pt_smoothed.pdf")
 
    import hist
    import json
    import correctionlib.convert
    h = (
        hist.Hist.new
        .Reg(_nbins-1, _rholow, _rhohigh, name="rho")
        .Reg(_nbins-1, _ptlow, _pthigh, name="pt")
    )
    sfhist = hist.Hist(*h.axes[:], data=ddt_map)
    plt.clf()
    fig,ax=plt.subplots()
    sfhist.plot2d()
    plt.savefig(args.opath+f"/{taggername}_ddtmap_rho_pt_closure.pdf")
    plt.clf()


    sfhist.name = "ddtmap_5pct_n2"
    sfhist.label = "out"

    

    sfhist_correctionlib = correctionlib.convert.from_histogram(sfhist)
    sfhist_correctionlib.data.flow = "clamp"
    cset = correctionlib.schemav2.CorrectionSet(
      schema_version=2,
      description="my N2DDT correction",
      corrections=[
        sfhist_correctionlib,
      ],
    )
    with open(args.opath+f"/{taggername}_ddtmap_rho_pt.json","w") as outfile:
        outfile.write(cset.json(exclude_unset=True))

    sfhist_smoothed = hist.Hist(*h.axes[:], data=ddt_map_smoothed)
    sfhist_smoothed.name = "ddtmap_5pct_n2_smoothed"
    sfhist_smoothed.label = "out"
    sfhist_correctionlib = correctionlib.convert.from_histogram(sfhist_smoothed)
    sfhist_correctionlib.data.flow = "clamp"
    cset = correctionlib.schemav2.CorrectionSet(
      schema_version=2,
      description="my N2DDT correction",
      corrections=[
        sfhist_correctionlib,
      ],
    ) 
    with open(args.opath+f"/{taggername}_ddtmap_rho_pt_smoothed.json","w") as outfile:
        outfile.write(cset.json(exclude_unset=True))
    return

#for ipq in range(len(master_data)):
#   sel = (master_data[ipq]["rho"] < -2) & (master_data[ipq]["rho"] > -5.5) & (master_data[ipq]["msd"] > _msdlow) & (master_data[ipq]["pt"] > _ptlow)
#   print(master_data[ipq])
#   master_data[ipq] = master_data[ipq][sel]
#   print(master_data[ipq])
#master_data = master_data[0]
#kinsel = (master_data["rho"] < -1)&(master_data["rho"]>-4.7)
#kinsel = (master_data["msd"] > _msdlow)&(master_data["msd"] < _msdhigh)& (master_data["pt"] > _ptlow)&(master_data["pt"] < _pthigh)

#master_data = master_data[kinsel]

#bb = ((master_data["q1_flavor"]).abs()==5)&(abs(master_data["q2_flavor"]).abs()==5)
#cc = ((master_data["q1_flavor"]).abs()==4)&(abs(master_data["q2_flavor"]).abs()==4)
#qq = ((master_data["q1_flavor"]).abs()<4)&(abs(master_data["q2_flavor"]).abs()<4)



def plot_zprime_roc(signal,background,score_T,score_PN,flavorsel,flavorname,signalnicename):
    T_zqq_score,bins,_    = make_response(signal,score_T,kinsel=kinsel,flavorsel=flavorsel,flavorname=flavorname)
    T_qcd_score,_,_       = make_response(background,score_T,kinsel=kinsel)
    
    PN_zqq_score,bins,_    = make_response(signal,score_PN,kinsel=kinsel,flavorsel=flavorsel,flavorname=flavorname)
    PN_qcd_score,_,_       = make_response(background,score_PN,kinsel=kinsel)
    
    tpr, fpr = plot_response_and_roc([T_zqq_score,T_qcd_score, PN_zqq_score, PN_qcd_score],[f"{signalnicename} Transformer","QCD Transformer",f"{signalnicename} PN","QCD PN"],bins,xtitle=signalnicename,reverse=False, truthlabel_list=[1,0,1,0],)

#twoProngPN = master_data["particleNetMD_Xqq"] + master_data["particleNetMD_Xcc"] + master_data["particleNetMD_Xbb"]
#make_ddt_map("QCD", twoProngPN, master_data["msd"], master_data["pt"], master_data["weight"],"2prongPN")
make_ddt_map("QCD", master_data["n2b1"], master_data["msd_corrected"], master_data["pt"], master_data["weight"],"n2b1",reverse=True)

sys.exit(1)
plot_zprime_roc("VectorZPrimeToQQ_M75.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_qq"],master_data["particleNetMD_Xqq"],qq,"qq","75 GeV Z\'(qq)")
plot_zprime_roc("VectorZPrimeToQQ_M75.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_bb"],master_data["particleNetMD_Xbb"],bb,"bb","75 GeV Z\'(bb)")
plot_zprime_roc("VectorZPrimeToQQ_M75.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_cc"],master_data["particleNetMD_Xcc"],cc,"cc","75 GeV Z\'(cc)")

plot_zprime_roc("VectorZPrimeToQQ_M200.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_qq"],master_data["particleNetMD_Xqq"],qq,"qq","200 GeV Z\'(qq)")
plot_zprime_roc("VectorZPrimeToQQ_M200.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_bb"],master_data["particleNetMD_Xbb"],bb,"bb","200 GeV Z\'(bb)")
plot_zprime_roc("VectorZPrimeToQQ_M200.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_cc"],master_data["particleNetMD_Xcc"],cc,"cc","200 GeV Z\'(cc)")

plot_zprime_roc("VectorZPrimeToQQ_flat.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_qq"],master_data["particleNetMD_Xqq"],qq,"qq","flat Z\'(qq)")
plot_zprime_roc("VectorZPrimeToQQ_flat.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_bb"],master_data["particleNetMD_Xbb"],bb,"bb","flat Z\'(bb)")
plot_zprime_roc("VectorZPrimeToQQ_flat.root","QCD",master_data["zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_cc"],master_data["particleNetMD_Xcc"],cc,"cc","flat Z\'(cc)")



plot_sculpting_curves(master_data["particleNetMD_Xqq"],kinsel,"particleNet-MD\nqq vs QCD",)  
plot_sculpting_curves(master_data["particleNetMD_Xbb"],kinsel,"particleNet-MD\nbb vs QCD",)  
plot_sculpting_curves(master_data["zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_qq"],kinsel,"transformer\nqq vs QC")
plot_sculpting_curves(master_data["zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_bb"],kinsel,"transformer\nbb vs QC")


