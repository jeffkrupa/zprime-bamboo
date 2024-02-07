#!/usr/bin/env python
import ROOT
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os,argparse
#hep.style.use("CMS") # string aliases work too
import CMS_lumi, tdrstyle
import array
tdrstyle.setTDRStyle()

#CMS_lumi.lumi_13TeV = "41.1 fb^{-1}"
#CMS_lumi.writeExtraText = 1
#CMS_lumi.extraText = "Preliminary"
iPeriod = 3
iPos = 11

parser = argparse.ArgumentParser("ZPrime analyzer")
parser.add_argument("-pq",dest="parquet",type=str,required=True)
parser.add_argument("-y",dest="year",type=str,required=True)
parser.add_argument("-o",dest="opath",type=str,required=True)
parser.add_argument("-c",dest="cut",type=float,required=True)
parser.add_argument("-e",dest="efficiency",type=float,required=True)
parser.add_argument("--erf",dest="erf",action="store_true",)
parser.add_argument("--cmsbkg",dest="cmsbkg",action="store_true",)
args = parser.parse_args()
os.system(f"mkdir -p {args.opath}")

efficiency = float(args.efficiency)
cut = float(args.cut)
fit_min = 40
fit_max = 180
nbins = 24
# In[2]:


jmsd = ROOT.RooRealVar("jmsd", "Jet mass", fit_min, fit_max, "GeV")
c = ROOT.RooCategory("c", "c")
c.defineType("data_pass", +1)
c.defineType("data_fail", -1)
#jmsd.setBins(25,"cache")
param = ROOT.RooArgSet(jmsd)
param_ = ROOT.RooArgSet(jmsd)
rlabel = "Preliminary"


# In[3]:
#param.setBins(20)

data_pass = ROOT.RooDataSet("data_pass","data_pass",{jmsd, c})
data_fail = ROOT.RooDataSet("data_fail","data_fail",{jmsd, c})


# In[4]:

dataframe = pd.read_parquet(args.parquet)
#dataframe = pd.read_parquet("/eos/user/j/jekrupa/plots/bamboo/9Feb22_jobs2_2017_CR2_v3/results/wtagging_region.parquet")
#dataframe = pd.read_parquet("/eos/project/c/contrast/public/cl/www/zprime/bamboo/3Nov23-2prongarbitration-CR2-4/results/CR2.parquet")
dataframe["particleNetMD_2prong"] = dataframe["particleNetMD_Xbb"] + dataframe["particleNetMD_Xcc"] + dataframe["particleNetMD_Xqq"]


# In[5]:


histo_min_max = ROOT.TH1D("histo_min_max", "histo_min_max", nbins, fit_min, fit_max)
histo_min_max_all = ROOT.TH1D("histo_min_max_all", "histo_min_max_all", nbins, fit_min, fit_max)
histo_min_max_nopass = ROOT.TH1D("histo_min_max_nopass", "histo_min_max_nopass", nbins, fit_min, fit_max)


# In[6]:


for i,row in dataframe.iterrows():
#    if i>100000: break
    if row.process=="data" and row.msd<fit_max and row.msd>fit_min:
        histo_min_max_all.Fill(row.msd)
        #if row.particleNetMD_2prong > 0.9:
        #if row.particleNetMD_2prong >= cut:
        if row.pnmd2prong_ddt >= 0.:

            jmsd.setVal(row.msd)
            c.setLabel("data_pass")
            #jmsd = row.msd
            #param = row.msd
            #print(param)
            #data_pass.add(row.msd)
            data_pass.add({jmsd, c})
            #histo_min_max.Fill(param)

        else:
            jmsd.setVal(row.msd)
            c.setLabel("data_fail")
            #jmsd = row.msd
            #param = row.msd
            data_fail.add({jmsd, c})
            #histo_min_max_nopass.Fill(param)

print('data pass', data_pass.Print("v"))

all_selection = (dataframe.process=="data") & (dataframe.msd < fit_max) & (dataframe.msd > fit_min) 

# In[10]:


pass_selection = all_selection & (dataframe.particleNetMD_2prong >= 0.95) #rough cut-5% QCD pass PN (inclusive)
fail_selection = all_selection & (dataframe.particleNetMD_2prong < 0.95) #rough cut-95% QCD fail PN (inclusive)


# In[11]:


print("Passing histogram: ",dataframe[pass_selection].weight.sum())
print("Failing histogram: ",dataframe[fail_selection].weight.sum())
print("Pass/fail = ",dataframe[pass_selection].weight.sum()/dataframe[fail_selection].weight.sum())
entries_signal = histo_min_max_all.Integral() #dataframe[pass_selection].weight.sum() + dataframe[fail_selection].weight.sum()
#entries_signal = dataframe[pass_selection].weight.sum() + dataframe[fail_selection].weight.sum()
# In[12]:


fig,ax=plt.subplots()
hep.cms.label("Preliminary",data=True, lumi=41, year=2017)
(fail_h, pass_h), bins, _ = ax.hist([dataframe[fail_selection].msd,dataframe[pass_selection].msd,],weights=[dataframe[fail_selection].weight,dataframe[pass_selection].weight,],bins=np.linspace(fit_min,fit_max,nbins), label=["Data fail","Data pass"],histtype="step",stacked=False,)
ax.legend()
ax.set_xlabel("Jet mass (GeV)")
ax.set_ylabel("Entries")
plt.tight_layout()

plt.savefig(args.opath+"/mass_dist.png")
plt.savefig(args.opath+"/mass_dist.pdf")
#plt.show()


# In[13]:


# In[14]:


##define the parameters and functions
#starting one
#mean1 = ROOT.RooRealVar("mean1","mean1", 80, 70, 90)
#gamma1 = ROOT.RooRealVar("gamma1", "gamma1", 2.495, -10, 10)
#sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 2, -10, 10)

#nice 1
#mean1 = ROOT.RooRealVar("mean1","mean1", 80, 70, 90)
#gamma1 = ROOT.RooRealVar("gamma1", "gamma1", 2.495, -30, 30)
#sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 2, -30, 30)

#nice 2
#mean1 = ROOT.RooRealVar("mean1","mean1", 80, 50, 110)
#gamma1 = ROOT.RooRealVar("gamma1", "gamma1", 5, -30, 30)
#sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 2, -30, 30)

offset1 = ROOT.RooRealVar("offset1", "Offset1", 70, 50,80)
offset2 = ROOT.RooRealVar("offset2", "Offset2", 70, 50,80)
width1 = ROOT.RooRealVar("width1", "Width1", 10, 1, 20)
width2 = ROOT.RooRealVar("width2", "Width2", 10, 1, 20)
# Create the error function component using RooGenericPdf
erf_pdf1 = ROOT.RooGenericPdf("erf_pdf1", "Error Function",
                             "0.5 * (1 + TMath::Erf((jmsd - offset1) / (width1 * sqrt(2))))",
                             ROOT.RooArgList(jmsd, offset1, width1))
erf_pdf2 = ROOT.RooGenericPdf("erf_pdf2", "Error Function",
                             "0.5 * (1 + TMath::Erf((jmsd - offset2) / (width2 * sqrt(2))))",
                             ROOT.RooArgList(jmsd, offset2, width2))



Nerf1  = ROOT.RooRealVar("Nerf1", "Bkg pass erf", entries_signal*0.1, 1., entries_signal*1)
Nerf2  = ROOT.RooRealVar("Nerf2", "Bkg fail erf", entries_signal*0.1, 1., entries_signal*1)


alpha1_cms = ROOT.RooRealVar("alpha1_cms", "Turn-on threshold", 60, 40, 90)
beta1_cms = ROOT.RooRealVar("beta1_cms", "Exponential fall-off", -0.1,-10,0)
gamma1_cms = ROOT.RooRealVar("gamma1_cms", "Turn-on width", 10,0.001,50)
alpha2_cms = ROOT.RooRealVar("alpha2_cms", "Turn-on threshold", 60, 40, 90)
beta2_cms = ROOT.RooRealVar("beta2_cms", "Exponential fall-off", -0.1,-10,0)
gamma2_cms = ROOT.RooRealVar("gamma2_cms", "Turn-on width", 10,0.001,50)


mean1 = ROOT.RooRealVar("mean1","mean1", 90, fit_min, fit_max)#80, 70, 90
mean2 = ROOT.RooRealVar("mean2","mean2", 90, fit_min, fit_max)#80, 70, 90
#cms_shape1 = ROOT.RooCMSShape("cms_shape1", "CMS Shape", jmsd, alpha1_cms, beta1_cms, gamma1_cms, mean1)
#cms_shape2 = ROOT.RooCMSShape("cms_shape2", "CMS Shape", jmsd, alpha2_cms, beta2_cms, gamma2_cms, mean2)

cms_shape1 = ROOT.RooGenericPdf("cms_shape1","cms_shape1",
 "RooMath::erfc((alpha1_cms-jmsd)*beta1_cms) * TMath::Exp(-(jmsd-mean1)*gamma1_cms)", 
 ROOT.RooArgList(alpha1_cms,jmsd,beta1_cms,gamma1_cms,mean1)
)
cms_shape2 = ROOT.RooGenericPdf("cms_shape2","cms_shape2",
 "RooMath::erfc((alpha2_cms-jmsd)*beta2_cms) * TMath::Exp(-(jmsd-mean2)*gamma2_cms)", 
 ROOT.RooArgList(alpha2_cms,jmsd,beta2_cms,gamma2_cms,mean2)
)

gamma1 = ROOT.RooRealVar("gamma1", "gamma1", 5, -100, 100)#2.495, -10, 10#10, -30, 30
gamma2 = ROOT.RooRealVar("gamma2", "gamma2", 5, -100, 100)#2.495, -10, 10#10, -30, 30
sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 2, -30, 30)#2, -10, 10#2, -30, 30
sigma2 = ROOT.RooRealVar("sigma2", "sigma2", 2, -30, 30)#2, -10, 10#2, -30, 30
alphaL_B = ROOT.RooRealVar("alphaL_B","alphaL_B",1.6,0.5,3)
nL_B = ROOT.RooRealVar("nL_B","nL",5,0.001,50)
signal1 = ROOT.RooVoigtian("signal1","signal1",jmsd,mean1,gamma1,sigma1)
signal2 = ROOT.RooVoigtian("signal2","signal2",jmsd,mean2,gamma2,sigma2)
CB_B_S = ROOT.RooCBShape("CB_B_S","CB_B_S",jmsd,mean1,sigma1,alphaL_B,nL_B)
me = ROOT.RooRealVar("ml", "ml", -0.1, -10, 10)
me2 = ROOT.RooRealVar("ml2", "ml2", -0.1, -10, -0.01)
exp = ROOT.RooExponential("exp", "exponential PDF x", jmsd, me)
exp2 = ROOT.RooExponential("exp2", "exponential PDF x", jmsd, me2)
bkgB_1 = ROOT.RooRealVar("bkgB1","bkgB_1",-0.1,-10,10)
cheb = ROOT.RooChebychev("cheb","cheb",jmsd,ROOT.RooArgList(bkgB_1))

gaus = ROOT.RooGaussian("gaus","gaus",jmsd, mean1, sigma1)
gaus2 = ROOT.RooGaussian("gaus2","gaus2",jmsd, mean2, sigma2)

#define normalization of sig, bkg passing, bkg failing
nSig = ROOT.RooRealVar("nSig", "Number of signal candidates ", entries_signal*0.99, 1., entries_signal)
nBkg_pass = ROOT.RooRealVar("nBkg_pass", "Bkg pass component", entries_signal*0.1, 1., entries_signal*1)
nBkg_fail = ROOT.RooRealVar("nBkg_fail", "Bkg fail component", entries_signal*0.1, 1., entries_signal*1)
eff = ROOT.RooRealVar("eff", "eff", 0.9, 0, 1);

#break sig down into passing and failing ratio is the (efficiency)
nSig_pass = ROOT.RooFormulaVar("nSig_pass", "eff*nSig", ROOT.RooArgSet(eff, nSig))
nSig_fail = ROOT.RooFormulaVar("nSig_fail", "(1-eff)*nSig", ROOT.RooArgSet(eff, nSig))


# In[15]:
print('before workspace')

#build the workspace
w = ROOT.RooWorkspace("w","w")
pars_w = ROOT.RooArgSet()
pars_w.add(alpha1_cms)
pars_w.add(beta1_cms)
pars_w.add(gamma1_cms)
pars_w.add(alpha2_cms)
pars_w.add(beta2_cms)
pars_w.add(gamma2_cms)
pars_w.add(cms_shape1)
pars_w.add(cms_shape2)
pars_w.add(Nerf1)
pars_w.add(Nerf2)
pars_w.add(width1)
pars_w.add(width2)
pars_w.add(offset1)
pars_w.add(offset2)
pars_w.add(erf_pdf1)
pars_w.add(erf_pdf2)
pars_w.add(mean1)
pars_w.add(mean2)
pars_w.add(gamma1)
pars_w.add(gamma2)
pars_w.add(sigma1)
pars_w.add(sigma2)
pars_w.add(signal1)
pars_w.add(signal2)
pars_w.add(me)
pars_w.add(me2)
pars_w.add(bkgB_1)
pars_w.add(exp)
pars_w.add(exp2)
pars_w.add(cheb)
#signal bkg pure numbers
pars_w.add(nSig)
#pars_w.add(nBkg);
#signal bkg pass and fail numbers
pars_w.add(nSig_pass)
pars_w.add(nBkg_pass)
pars_w.add(nSig_fail)
pars_w.add(nBkg_fail)
pars_w.add(CB_B_S)
pars_w.add(alphaL_B)
pars_w.add(nL_B)
pars_w.add(gaus)
pars_w.add(gaus2)
#efficiency
pars_w.add(eff)
w.Import(pars_w,ROOT.RooFit.RecycleConflictNodes());


# In[16]:

model_r0="SUM::model_r0(nSig_pass*signal1, nBkg_pass*exp)"
model_r1="EDIT::model_r1(model_r0, nSig_pass=nSig_fail, nBkg_pass=nBkg_fail, signal1=signal2, cms_shape1=exp2)"

if args.erf:
    model_r0=model_r0.replace(")",", Nerf1*erf_pdf1)")
    model_r1=model_r1.replace(")",", Nerf1=Nerf2,erf_pdf1=erf_pdf2)")
    print("Adding erf...", model_r0, model_r1)
elif args.cmsbkg:
    model_r0="SUM::model_r0(nSig_pass*signal1, nBkg_pass*cms_shape1)"
    model_r1="EDIT::model_r1(model_r0, nSig_pass=nSig_fail, nBkg_pass=nBkg_fail, signal1=signal2, cms_shape1=cms_shape2)"
else:
    model_r0="SUM::model_r0(nSig_pass*signal1, nBkg_pass*exp)"
    model_r1="EDIT::model_r1(model_r0, nSig_pass=nSig_fail, nBkg_pass=nBkg_fail, signal1=signal2, exp=exp2)"
  
w.factory(model_r0)
w.factory(model_r1)
#w.factory("SUM::model_r0(nSig_pass*signal1, nBkg_pass*exp)");
#w.factory("EDIT::model_r1(model_r0, nSig_pass=nSig_fail, nBkg_pass=nBkg_fail, signal1=signal2, exp=exp2)");


# In[17]:


data_category = ROOT.RooCategory("data_category","data_category");
data_category.defineType("pass_probe");
data_category.defineType("fail_probe");


# In[18]:


#print('param cosa ', param.Print("v"))

#combData = ROOT.RooDataSet("combData", "combined data", param, ROOT.RooFit.Index(data_category), ROOT.RooFit.Import("pass_probe", histo_min_max), ROOT.RooFit.Import("fail_probe", histo_min_max_nopass))
combData = ROOT.RooDataSet("combData", "combined data", {jmsd}, ROOT.RooFit.Index(data_category), ROOT.RooFit.Import("pass_probe", data_pass), ROOT.RooFit.Import("fail_probe", data_fail))

#combData = ROOT.RooDataSet(
#    "combData",
#    "combined data",
#    {x},
#    Index=sample,
#    Import={"physics": data, "control": data_ctl},
#)

simPdf = ROOT.RooSimultaneous("simPdf","simultaneous pdf",data_category);
simPdf.addPdf(w.pdf("model_r0"),"pass_probe");
simPdf.addPdf(w.pdf("model_r1"),"fail_probe");

print('ciao ', combData.Print())


# In[20]:


results = simPdf.fitTo(combData, ROOT.RooFit.Range(fit_min,fit_max),ROOT.RooFit.PrintLevel(3),ROOT.RooFit.Save(), ROOT.RooFit.SumW2Error(ROOT.kTRUE));
fit_params = results.floatParsFinal()  # Get the final parameters of the fit
print(results.Print("v"))

print(results.Print("v"))
# In[ ]:


canvas = ROOT.TCanvas("canvas3", "canvas3", 800, 800, 800, 800);
#canvas.Divide(1,2);
#canvas.cd(1);
#ROOT.gPad.SetPad(0.,0.3,1.,1.);


xframe = jmsd.frame();
combData.plotOn(xframe, ROOT.RooFit.Cut("data_category==data_category::pass_probe"),ROOT.RooFit.Binning(nbins));
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "pass_probe"), ROOT.RooFit.ProjWData(data_category,combData),)#ROOT.RooFit.LineColor(ROOT.kBlue));
chi2 = xframe.chiSquare()
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "pass_probe"), ROOT.RooFit.Components(signal1), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kRed));
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "pass_probe"), ROOT.RooFit.Components(exp), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kGreen));
if args.erf:
   simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "pass_probe"), ROOT.RooFit.Components(erf_pdf1), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kOrange));
elif args.cmsbkg:
   simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "pass_probe"), ROOT.RooFit.Components(cms_shape1), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kOrange));
n_floating_params = results.floatParsFinal().getSize()
dof = nbins - n_floating_params
chi2_value = chi2 * dof
from scipy.stats import chi2 as chi2_distribution
p_value = chi2_distribution.sf(chi2_value, dof)

xframe.Draw()

xframe.GetXaxis().SetLabelSize(0.035)
xframe.GetYaxis().SetLabelSize(0.035)

# In[ ]:
# Position the text and add the p-value
x_text = 0.22  # X position of the text (left)
y_text = 0.85  # Y position of the text (top)
latex = ROOT.TLatex()
latex.SetNDC()  # Use normalized coordinates for positioning
latex.SetTextSize(0.03)  # Set the text size
latex.SetTextAlign(11)  # Align at top left


latex.DrawLatex(x_text, y_text, f"#chi^{{2}}/NDOF = {chi2_value:.2f}/{dof}")
latex.DrawLatex(x_text, y_text-0.03, f"p-value = {p_value:.2f}")
latex.DrawLatex(x_text, y_text-0.06, f"QCD eff={efficiency}")
latex.DrawLatex(x_text, y_text-0.09, f"Data Pass")



canvas.Draw()


csv_data = f"param,value,err \np_value,{p_value},0. \ndof,{dof},0. \nchi2,{chi2_value},0. \nefficiency,{fit_params.find('eff').getVal()},{fit_params.find('eff').getError()}"

'''
canvas.cd(2);
ROOT.gPad.SetPad(0.,0.,1.,0.3);

xframe_pull = jmsd.frame(ROOT.RooFit.Title(" "),ROOT.RooFit.Range(50,110),ROOT.RooFit.Bins(25));
hpull = xframe.pullHist();
xframe_pull.addPlotable(hpull,"EP");
xframe_pull.SetTitle("Pulls bin by bin");
xframe_pull.addObject(xframe.pullHist(), "ep");
xframe_pull.SetMaximum(30);
xframe_pull.SetMinimum(-30);
xframe_pull.Draw("same")
'''
# In[ ]:


canvas.SaveAs(f"./nov23/pass_Data_{efficiency}.pdf")
canvas.SaveAs(f"./nov23/pass_Data_{efficiency}.png")


canvas1 = ROOT.TCanvas("canvas4", "canvas4", 800, 800, 800, 800);
#canvas1.Divide(1,2);
#canvas1.cd(1);
#ROOT.gPad.SetPad(0.,0.3,1.,1.);

xframe = jmsd.frame();
combData.plotOn(xframe, ROOT.RooFit.Cut("data_category==data_category::fail_probe"),ROOT.RooFit.Binning(nbins));
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "fail_probe"), ROOT.RooFit.ProjWData(data_category,combData),)#ROOT.RooFit.LineColor(ROOT.kBlue));
fail_chi2 = xframe.chiSquare()
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "fail_probe"), ROOT.RooFit.Components(signal2), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kRed));
simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "fail_probe"), ROOT.RooFit.Components(exp2), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kGreen));
if args.erf:
   simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "fail_probe"), ROOT.RooFit.Components(erf_pdf1), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kOrange));
elif args.cmsbkg:
   simPdf.plotOn(xframe, ROOT.RooFit.Slice(data_category, "fail_probe"), ROOT.RooFit.Components(cms_shape2), ROOT.RooFit.ProjWData(data_category,combData), ROOT.RooFit.LineColor(ROOT.kOrange));
xframe.GetXaxis().SetLabelSize(0.035)
xframe.GetYaxis().SetLabelSize(0.035)
xframe.Draw()
# In[ ]:
'''
canvas1.cd(2);
ROOT.gPad.SetPad(0.,0.,1.,0.3);

xframe_pull = jmsd.frame(ROOT.RooFit.Title(" "),ROOT.RooFit.Range(50,110),ROOT.RooFit.Bins(25));
hpull = xframe.pullHist();
xframe_pull.addPlotable(hpull,"EP");
xframe_pull.SetTitle("Pulls bin by bin");
xframe_pull.addObject(xframe.pullHist(), "ep");
xframe_pull.SetMaximum(30);
xframe_pull.SetMinimum(-30);
xframe_pull.Draw("same")
'''
xframe.GetXaxis().SetLabelSize(0.035)
xframe.GetYaxis().SetLabelSize(0.035)
n_floating_params = results.floatParsFinal().getSize()
dof = nbins - n_floating_params
fail_chi2_value = fail_chi2 * dof
from scipy.stats import chi2 as chi2_distribution
fail_p_value = chi2_distribution.sf(fail_chi2_value, dof)


xframe.GetXaxis().SetLabelSize(0.035)
xframe.GetYaxis().SetLabelSize(0.035)

latex = ROOT.TLatex()
latex.SetNDC()  # Use normalized coordinates for positioning
latex.SetTextSize(0.03)  # Set the text size
latex.SetTextAlign(11)  # Align at top left

# Position the text and add the p-value
x_text = 0.22  # X position of the text (left)
y_text = 0.85  # Y position of the text (top)

latex.DrawLatex(x_text, y_text, f"#chi^{{2}}/NDOF = {fail_chi2_value:.2f}/{dof}")
latex.DrawLatex(x_text, y_text-0.03, f"p-value = {fail_p_value:.2f}")
latex.DrawLatex(x_text, y_text-0.06, f"QCD eff={efficiency}")
latex.DrawLatex(x_text, y_text-0.09, f"Data Fail")
canvas1.Draw()
canvas1.SaveAs(f"./nov23/fail_Data_{efficiency}.pdf")
canvas1.SaveAs(f"./nov23/fail_Data_{efficiency}.png")


#                   eff    9.0000e-01    2.7637e-01 +/-  1.40e-02  <none>
#                gamma1    5.0000e+00   -2.2448e+01 +/-  3.80e+00  <none>
#                gamma2    5.0000e+00    3.2277e+01 +/-  3.28e+00  <none>
#                 mean1    9.0000e+01    9.2758e+01 +/-  2.88e-01  <none>
#                 mean2    9.0000e+01    9.2780e+01 +/-  3.15e-01  <none>
#                    ml   -1.0000e-01   -7.9799e-03 +/-  7.04e-04  <none>
#                   ml2   -1.0000e-01   -2.1685e-02 +/-  1.06e-03  <none>
#             nBkg_fail    3.0336e+03    1.0986e+04 +/-  5.90e+02  <none>
#             nBkg_pass    3.0336e+03    2.2514e+03 +/-  2.35e+02  <none>
#                  nSig    3.0033e+04    1.7098e+04 +/-  6.36e+02  <none>
#                sigma1    2.0000e+00    2.8848e+00 +/-  3.58e+00  <none>
#                sigma2    2.0000e+00    4.2700e+00 +/-  2.48e+00  <none>

csv_data += f"\nfail_p_value,{fail_p_value},0. \nfail_chi2,{fail_chi2_value},0. "#\nefficiency,{fit_params.find('eff').getVal()},{fit_params.find('eff').getError()} \n"

csv_data += f"\gamma,{fit_params.find('gamma1').getVal()},{fit_params.find('gamma1').getError()}"
csv_data += f"\fail_gamma,{fit_params.find('gamma2').getVal()},{fit_params.find('gamma2').getError()}"

print("csv_data",csv_data)
with open(f"nov23/{efficiency}_data.csv","w") as file:
   file.write(csv_data)

print("csv_data",csv_data)

print("done...")

