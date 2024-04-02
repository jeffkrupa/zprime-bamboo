#!/usr/bin/python -tt
from bamboo.analysismodules import NanoAODHistoModule, HistogramsModule
from bamboo.treedecorators import NanoAODDescription
from bamboo.scalefactors import binningVariables_nano,lumiPerPeriod_default
from bamboo.scalefactors import makeBtagWeightItFit, get_scalefactor, get_correction, get_bTagSF_itFit
from bamboo.analysisutils import loadPlotIt
from bamboo.plots import Plot
from bamboo.plots import EquidistantBinning as EqB
from bamboo import treedecorators as btd
from bamboo import treefunctions as op
from bamboo import scalefactors

from itertools import chain
import math
import numpy as np

import logging
logger = logging.getLogger(__name__)

v_PDGID = {
    "HiggsToBB" : 25,
    "GluGluHToBB" : 25,
    "WJetsToLNu" : 24,
    "WJetsToQQ" : 24,
    "DYJetsToLL" : 23,
    "ZJetsToQQ" : 23,
    "ZJetsToCC" : 23,
    "ZJetsToBB" : 23,
    "VectorZPrimeToQQ" : 55,
    "VectorZPrimeToBB" : 55,
    #"TTbar" : 24,
    #"SingleTop" : 24,
    #"TTbar_matched" : 24,
    #"TTbar_unmatched" : 24,
    #"SingleTop_matched" : 24,
    #"SingleTop_unmatched" : 24,
}

syst_file = {
    "triggerweights" :  f"/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/fatjet_triggerSF.json",
    "MUO" : { 
        "2016APV" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016preVFP_UL/muon_Z.json.gz",
        "2016" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json.gz",
        "2017" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2017_UL/muon_Z.json.gz",
        "2018" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_Z.json.gz",
    },
    "BTV" : {
        "2016APV" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz",
        "2016" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz",
        "2017" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz",
        "2018" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz",
    },
    "pileup" : {
        "2016APV" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2016preVFP_UL/puWeights.json.gz",
        "2016" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2016postVFP_UL/puWeights.json.gz",
        "2017" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2017_UL/puWeights.json.gz",
        "2018" : "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2018_UL/puWeights.json.gz",
    },
    "NLOVkfactors" : "/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/ULvjets_corrections.json",
    "EWHiggsCorrections" : "/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/EWHiggsCorrections.json",
    "msdcorr" : f"/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/msdcorr.json",
}

pnmdWPs = {
    "2016APV" : {
        'L': 0.942,
        'T': 0.985,
    },
    "2016" : {
        'L': 0.942,
        'T': 0.987,
    },
    "2017" : {
        'L': 0.942,
        'T': 0.985,
    },
    "2018" : {
        'L': 0.942,
        'T': 0.985,
    },
        
}
btagWPs = {
    '2016APV': {
        'L': 0.0508,
        'M': 0.2598,
        'T': 0.6502,
    },
    '2016': {
        'L': 0.0480,
        'M': 0.2489,
        'T': 0.6377,
    },
    '2017': {
        'L': 0.0532,
        'M': 0.3040,
        'T': 0.7476,
    },
    '2018': {
        'L': 0.0490,
        'M': 0.2783,
        'T': 0.7100,
    }
}
GEN_FLAGS = {
    "IsPrompt": 0,
    "IsDecayedLeptonHadron": 1,
    "IsTauDecayProduct": 2,
    "IsPromptTauDecayProduct": 3,
    "IsDirectTauDecayProduct": 4,
    "IsDirectPromptTauDecayProduct": 5,
    "IsDirectHadronDecayProduct": 6,
    "IsHardProcess": 7,
    "FromHardProcess": 8,
    "IsHardProcessTauDecayProduct": 9,
    "IsDirectHardProcessTauDecayProduct": 10,
    "FromHardProcessBeforeFSR": 11,
    "IsFirstCopy": 12,
    "IsLastCopy": 13,
    "IsLastCopyBeforeFSR": 14,
}

class makeYieldPlots:
    def __init__(self):
        self.calls = 0
        self.plots = []
    def addYields(self, sel, name, title):
        """
            Make Yield plot and use it also in the latex yield table
            sel     = refine selection
            name    = name of the PDF to be produced
            title   = title that will be used in the LateX yield table
        """
        self.plots.append(Plot.make1D("Yield_"+name,   
                        op.c_int(0),
                        sel,
                        EqB(1, 0., 1.),
                        title = title + " Yield",
                        plotopts = {"for-yields":True, "yields-title":title, 'yields-table-order':self.calls}))
        self.calls += 1
    def returnPlots(self):
        return self.plots

def goodFlag(p):
    return op.AND(p.statusFlags & 2**GEN_FLAGS["IsLastCopy"], p.statusFlags & 2**GEN_FLAGS["FromHardProcess"])


class zprlegacy(NanoAODHistoModule):
    def addArgs(self, parser):
        super().addArgs(parser)
        parser.add_argument("--mvaSkim", action="store_true", help="Produce MVA training skims")
        parser.add_argument("--mvaEval", action="store_true", help="Import MVA model and evaluate it on the dataframe")
        parser.add_argument("--SR", action="store_true", default=False, help="Make SR")
        parser.add_argument("--CR1", action="store_true", default=False, help="Make CR1")
        parser.add_argument("--CR2", action="store_true", default=False, help="Make CR2")
        parser.add_argument("--arbitration", action="store", required=True, help="Arbitration of jets.")
        parser.add_argument("--split_signal_region", action="store_true", required=False, help="Flag to split SR into high and low bvl lscore.")
        #### Till now we don't need --mvaEval since we don't have a MVA model ####

    def __init__(self, args):
        super(zprlegacy, self).__init__(args)
        #if not (args.SR or args.CR1 or args.CR2):
        #    return RuntimeError("Need to run on SR/CR1/CR2")


    def prepareTree(self, tree, sample=None, sampleCfg=None, description=None, backend=None):
        ## initializes tree.Jet.calc so should be called first (better: use super() instead)
        # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC#Recommended_for_MC
        from bamboo.treedecorators import nanoRochesterCalc, nanoJetMETCalc, CalcCollectionsGroups, nanoFatJetCalc
        from bamboo.analysisutils import configureJets, configureType1MET
        from bamboo.analysisutils import configureRochesterCorrection
        era = sampleCfg["era"]
        tree,noSel,be,lumiArgs = NanoAODHistoModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg,description=NanoAODDescription.get('v7', year="2016" if "APV" in era else era,isMC=self.isMC(sample), systVariations=[nanoRochesterCalc,nanoFatJetCalc,nanoJetMETCalc]),backend=backend)

        addition = ""
        self.yield_object = makeYieldPlots()

        if self.isMC(sample):
            jesUncertaintySources = ["Total"]
            JECs = {
                '2016APV'     : "Summer19UL16APV_V7_MC",
                '2016'        : "Summer19UL16_V7_MC",
                '2017'        : "Summer19UL17_V5_MC", 
                '2018'        : "Summer19UL18_V5_MC",
            }
            JERs = {
                '2016APV'     : "Summer20UL16APV_JRV3_MC",
                '2016'        : "Summer20UL16_JRV3_MC",
                '2017'        : "Summer19UL17_JRV3_MC",
                '2018'        : "Summer19UL18_JRV2_MC",
            }
            mcYearForFatJets=era
        
        else:
            jesUncertaintySources = None
            JECs = {
                '2016APV'     : "Summer19UL16APV_RunBCDEF_V7_DATA",
                '2016'        : "Summer19UL16_RunFGH_V7_DATA",
                '2017B'       : "Summer19UL17_RunB_V5_DATA",
                '2017C'       : "Summer19UL17_RunC_V5_DATA",
                '2017D'       : "Summer19UL17_RunD_V5_DATA",
                '2017E'       : "Summer19UL17_RunE_V5_DATA",
                '2017F'       : "Summer19UL17_RunF_V5_DATA",
                '2018A'       : "Summer19UL18_RunA_V5_DATA",
                '2018B'       : "Summer19UL18_RunB_V5_DATA",
                '2018C'       : "Summer19UL18_RunC_V5_DATA",
                '2018D'       : "Summer19UL18_RunD_V5_DATA",
            }
            
            JERs = { 
                '2016APV'     : None,
                '2016'        : None,
                '2017'        : None,
                '2018'        : None,
            }
            mcYearForFatJets=None

            if era in ["2017","2018"]:
                  addition = sample.split(era)[1]
        print(addition)
        print (JECs[era+addition])
        print (self.isMC(sample))
        print("ERA", era)
        cmJMEArgs = {
                "jec": JECs[era+addition],
                "smear": JERs[era],
#                "splitJER": True,
                "jesUncertaintySources": jesUncertaintySources,
                #"jecLevels":[], #  default : L1FastJet, L2Relative, L3Absolute, and also L2L3Residual for data
                "regroupTag": "V2",
                "addHEM2018Issue": (era == "2018"),
                "mayWriteCache": (self.args.distributed != "worker"),
                "isMC": self.isMC(sample),
                "backend": be,
                "uName": sample,
                "genMatchDR":0.4,
                #"mcYearForFatJets": mcYearForFatJets
                }

#        print (cmJMEArgs["smear"])
#        print (cmJMEArgs["mcYearForFatJets"])
#        print (cmJMEArgs)
#        if self.isMC(sample):
#              cmJMEArgs["jesUncertaintySources"] = ["Total"]
#              configureJets(tree._FatJet, "AK8PFPuppi", mcYearForFatJets=era, **cmJMEArgs_mc)
#        else:
#              cmJMEArgs["jesUncertaintySources"] = ["Total"]
        if self.isMC(sample):
            configureJets(tree._FatJet, "AK8PFPuppi",  **cmJMEArgs, mcYearForFatJets="2016" if "APV" in mcYearForFatJets else mcYearForFatJets)
        else:
            configureJets(tree._FatJet, "AK8PFPuppi",  **cmJMEArgs,) 
        #configureType1MET(tree._MET, **cmJMEArgs)


        return tree,noSel,be,lumiArgs


    def definePlots(self,t, noSel, sample=None, sampleCfg=None):
        from bamboo.plots import Plot, EquidistantBinning, CutFlowReport
        from bamboo import treefunctions as op
        try:
            do_genmatch = any(sampleCfg["group"] in x for x in v_PDGID.keys())
            if "VectorZPrime" in sampleCfg["group"]: do_genmatch = True
 
        except:
            do_genmatch = True
        print("do_genmatch",do_genmatch,"sample",sample)
        era = sampleCfg["era"]
        jettrigger = []
        muontrigger = []
        if "18" in era:
            jettrigger = [ t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, t.HLT.AK8PFHT800_TrimMass50, t.HLT.AK8PFJet400_TrimMass30,t.HLT.AK8PFJet420_TrimMass30, ]#t.HLT.AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4 ] this last one is not in all runs??            
            muontrigger= [ t.HLT.Mu50, t.HLT.OldMu100, t.HLT.TkMu100 ] 
            sr_pt_cut = 500.
            pt_bins = ((500,550),(550,600),(600,700),(700,800),(800,1200))
        elif "17" in era:
            if "Run2017B" in sample:
                jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, ] #t.HLT.AK8PFHT800_TrimMass50]
                muontrigger = [t.HLT.Mu50, ]
            else:
                jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, t.HLT.AK8PFHT800_TrimMass50, t.HLT.AK8PFJet400_TrimMass30,t.HLT.AK8PFJet420_TrimMass30]
                muontrigger = [t.HLT.Mu50, t.HLT.OldMu100, t.HLT.TkMu100]
            sr_pt_cut = 525.
            pt_bins = ((525,575),(575,625),(625,700),(700,800),(800,1200))
        elif "16" in era:
            jettrigger = [t.HLT.PFHT900, t.HLT.AK8PFJet360_TrimMass30, t.HLT.AK8PFHT700_TrimR0p1PT0p03Mass50, t.HLT.PFHT650_WideJetMJJ950DEtaJJ1p5, t.HLT.PFHT650_WideJetMJJ900DEtaJJ1p5, t.HLT.AK8DiPFJet280_200_TrimMass30_BTagCSV_p20, t.HLT.PFJet450]
            if "Run2016H" not in sample:
                jettrigger += [t.HLT.PFHT800]
            muontrigger = [ t.HLT.Mu50, ] #t.HLT.TkMu50 ]
            if sample not in ["SingleMuon_Run2016B_ver2_HIPM","JetHT_Run2016B_ver2_HIPM"]:
                muontrigger += [ t.HLT.TkMu50 ]
            sr_pt_cut = 500.
            pt_bins = ((500,550),(550,600),(600,700),(700,800),(800,1200))
            


        filters = [t.Flag.goodVertices,t.Flag.globalSuperTightHalo2016Filter,t.Flag.HBHENoiseFilter,t.Flag.HBHENoiseIsoFilter,t.Flag.EcalDeadCellTriggerPrimitiveFilter, t.Flag.BadPFMuonFilter, t.Flag.BadPFMuonDzFilter, t.Flag.eeBadScFilter, t.Flag.ecalBadCalibFilter]
        #isoMuFilterMask = 0xA

        if self.isMC(sample):
            noSel = noSel.refine("mcWeight", weight=t.genWeight, autoSyst=True)
        else:
            ##blinding data for now
            if self.args.SR:
                noSel = noSel.refine("blinded",cut=t.event%10==0)
            jetSel = noSel.refine("passJetHLT", cut=op.OR(*(jettrigger))) 
            filterJetSel = jetSel.refine("passFilterJet",cut=op.AND(*(filters)))
            muSel  = noSel.refine("passMuHLT", cut=op.OR(*(muontrigger))) 
            filterMuSel = muSel.refine("passFilterMu",cut=op.AND(*(filters)))

        plots = []

        #triggerObj = op.select(t.TrigObj, lambda trgObj: op.AND( trgObj.id == 13,
        #			(trgObj.filterBits & isoMuFilterMask) )) 


        loose_muons = op.sort(op.select(t.Muon, lambda mu : op.AND(
					mu.pt > 10.,
					#mu.looseId,
					op.abs(mu.eta) < 2.4,
					op.abs(mu.pfRelIso04_all) < 0.25,
					)), lambda mu : -mu.pt)


#        candidatemuons = op.sort(op.select(t.Muon, lambda mu : op.AND(
#					mu.pt > 53.,
#					mu.looseId,
#					op.abs(mu.eta) < 2.1,
#					op.abs(mu.pfRelIso04_all) < 0.25,
	#				)), lambda mu : -mu.pt)
        candidatemuons = op.select(t.Muon, lambda mu : op.AND(
					mu.pt > 53.,
					mu.looseId,
					op.abs(mu.eta) < 2.1,
					op.abs(mu.pfRelIso04_all) < 0.25,
					)
        )
        electrons = op.sort(op.select(t.Electron, lambda el : op.AND(
					el.pt > 10.,
					#el.mvaFall17V2Iso_WPL,
					el.cutBased >= 1,
					op.abs(el.eta) < 2.5,
					)), lambda el : -el.pt)

        taus = op.sort(op.select(t.Tau, lambda tau : op.AND(
					tau.pt > 20.,
					tau.decayMode >= 0,
					op.abs(tau.eta) < 2.3,
					tau.idDeepTau2017v2p1VSe >= 2,
					tau.idDeepTau2017v2p1VSjet >= 16,
					tau.idDeepTau2017v2p1VSmu >= 8,
					)), lambda tau : -tau.pt)

	#AK8 (highest pt or second highest pt, depending on number of fatjets left)
        if self.args.arbitration == "pt":
            fatjets = op.sort(op.select(t.FatJet, lambda fj : op.AND(
					fj.pt > 200.,
					op.abs(fj.eta) < 2.5,
                                        #2*op.log(fj.msoftdrop/fj.pt)>-8, 
                                        #2*op.log(fj.msoftdrop/fj.pt)<-1,
                                        #fj.subJetIdx1 > -1,
					)), lambda fj : -fj.pt)

        elif self.args.arbitration == "2prong":
            fatjets = op.sort(op.select(t.FatJet, lambda fj : op.AND(
					fj.pt > 200.,
					op.abs(fj.eta) < 2.5,
                                        #2*op.log(fj.msoftdrop/fj.pt)>-8, 
                                        #2*op.log(fj.msoftdrop/fj.pt)<-1,
                                        #fj.subJetIdx1 > -1,
					)), lambda fj : -(fj.particleNetMD_Xqq+fj.particleNetMD_Xcc+fj.particleNetMD_Xbb))
        jidx = fatjets[0].idx


        ##Case 1: subJet1 is valid and subJet2 is not valid (it's never the case that subJet2 is valid and subJet1 is invalid)
        ##Case 2: both subJet1 and subJet2 are valid
        ##Case 3: neither subJet is valid
        msdraw = op.multiSwitch(
             (op.AND(fatjets[0].subJet1.isValid,op.NOT(fatjets[0].subJet2.isValid)), op.sqrt(op.invariant_mass(fatjets[0].subJet1.p4*(1-fatjets[0].subJet1.rawFactor), fatjets[0].subJet1.p4*(1-fatjets[0].subJet1.rawFactor) ))),
             (op.AND(fatjets[0].subJet1.isValid,fatjets[0].subJet2.isValid), op.invariant_mass(fatjets[0].subJet1.p4 * (1-fatjets[0].subJet1.rawFactor), fatjets[0].subJet2.p4 * (1-fatjets[0].subJet2.rawFactor))), 
             op.c_float(-99.)
        )
        msoftdrop = fatjets[0].msoftdrop
        msdfjcorr = msdraw / (1 - fatjets[0].rawFactor)
        from bamboo import treedecorators as btd


        msdbranch = op.map(t.FatJet, lambda fj: op.switch(op.rng_len(fatjets) > 0 , msdfjcorr, -99))
        t.FatJet.valueType.msdfjcorr = btd.itemProxy(msdbranch) 
        from bamboo.scalefactors import get_scalefactor, get_correction
        massSF = get_correction(syst_file["msdcorr"],
            "msdfjcorr",
            #params = {"mdivpt" : lambda fj : fj.msdfjcorr/fj.pt, 
            params = {"mdivpt" : lambda fj : msdbranch[fj.idx]/fj.pt, 
                      "logpt" : lambda fj : op.log(fj.pt),
                      "eta"   : lambda fj : fj.eta, 
                     },
            sel = noSel,
        )
        corrected_msoftdrop = op.product(msdfjcorr, massSF(fatjets[0]))
        corrected_msd = op.map(t.FatJet, lambda fj: op.switch(op.rng_len(fatjets) > 0 , corrected_msoftdrop, -99))

	#btagged AK4
        jets = op.sort(op.select(t.Jet, lambda j : op.AND(
					j.pt > 50.,
					op.abs(j.eta) < 2.5,
					#j.btagCSVV2 > 0.8838,
					)), lambda j : -j.pt)[:4]

        jets_away = op.select(jets, lambda j : op.deltaR(fatjets[0].p4, j.p4)  > 0.8) 

        #https://github.com/jennetd/hbb-coffea/blob/UL2022/boostedhiggs/vbfprocessor.py#L263-L268
        ak4_jets = op.sort(op.select(t.Jet, lambda j : op.AND(
                                        j.pt > 30.,
                                        op.abs(j.eta) < 2.5,
                                        j.jetId & (1 << 1) != 0, #tight id
                                        j.puId > 0,
                                        )), lambda j : -j.pt)[:]

        

        ak4_jet_opp_hemisphere = op.sort(op.select(ak4_jets[:4], lambda j : op.deltaPhi(j.p4,fatjets[0].p4) > np.pi/2), lambda j : -j.btagDeepB)[0]
          
        
        if "TT" in sample or "SingleTop" in sample:
            print("Running ", sample)
            top_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 6,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"])), lambda p: -p.status)

        if do_genmatch:
            genQuarks = op.select(t.GenPart, lambda q: op.AND(op.abs(q.pdgId) >= 1, op.abs(q.pdgId) <= 5))
            if "VectorZPrime" in sample:
                w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 55,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"],p.statusFlags & 2**GEN_FLAGS["FromHardProcess"])),lambda p: -p.status)
            elif "ZJetsTo" in sample or "DYJets" in sample:
                w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 23,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"],p.statusFlags & 2**GEN_FLAGS["FromHardProcess"])),lambda p: -p.status)
            elif "WJetsTo" in sample or "WJetsLNu" in sample or "ST" in sample or "TTTo" in sample:
                w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 24,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"],p.statusFlags & 2**GEN_FLAGS["FromHardProcess"])),lambda p: -p.status)
            elif "HiggsToBB" in sampleCfg["group"]:
                w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 25,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"],p.statusFlags & 2**GEN_FLAGS["FromHardProcess"])), lambda p: -p.status)
            #else: 
            #    m_pdgid = v_PDGID[sampleCfg["group"]]
            #print("m_pdgid",m_pdgid)
            #w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.OR(op.abs(p.pdgId) == 23,op.abs(p.pdgId) == 24,op.abs(p.pdgId) == 25, op.abs(p.pdgId) == 55),p.statusFlags & 2**GEN_FLAGS["IsLastCopy"])), lambda p: -p.status)
            #w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 24,p.statusFlags & 2**GEN_FLAGS["IsLastCopy"])),
            #                      lambda p: -p.status)
            q_from_w = op.select(genQuarks, lambda q : q.parent.idx == w_by_status[0].idx)
            Vgen_matched = op.rng_count(q_from_w, lambda q: op.deltaR(q.p4, fatjets[0].p4) < 0.8) == 2 
            if self.args.split_signal_region:
                if "ZJetsToBB" in sampleCfg["group"] or "VectorZPrimeToBB" in sampleCfg["group"]:
                    Vgen_matched = op.AND(Vgen_matched, op.rng_count(q_from_w, lambda q: op.abs(q.pdgId) == 5) == 2)
                elif "ZJetsToCC" in sampleCfg["group"] or "VectorZPrimeToCC" in sampleCfg["group"]:
                    Vgen_matched = op.AND(Vgen_matched, op.rng_count(q_from_w, lambda q: op.abs(q.pdgId) == 4) == 2)
                elif "ZJetsToQQ" in sampleCfg["group"] or "VectorZPrimeToQQ" in sampleCfg["group"]:
                    Vgen_matched = op.AND(Vgen_matched, op.rng_count(q_from_w, lambda q: op.abs(q.pdgId)  < 4) == 2)
            dr_to_q1 = op.deltaR(q_from_w[0].p4, fatjets[0].p4)
            dr_to_q2 = op.deltaR(q_from_w[1].p4, fatjets[0].p4)
            Vgen_quality_criterion_pt = ((fatjets[0].pt - w_by_status[0].pt)/w_by_status[0].pt) < 0.5
            Vgen_quality_criterion_msd = ((corrected_msd[fatjets[0].idx] - w_by_status[0].mass)/w_by_status[0].mass) < 0.3
            

        ddtmap_file = f"/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/ddt_maps.json"

        #pnmd2prong_ddtmap = get_correction(ddtmap_file,
        #    f"ddtmap_PNMD_pctl0.05_QCD_{era}" ,
        #    params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(corrected_msd[fj.idx]/fj.pt) },
        #    sel=noSel,
        #)
        jidx = fatjets[0].idx
        #pnmd2prong_ddt = t._FatJet.orig[jidx].particleNetMD_Xbb + t._FatJet.orig[jidx].particleNetMD_Xcc + t._FatJet.orig[jidx].particleNetMD_Xqq - pnmd2prong_ddtmap(fatjets[0])
        pnmd2prong = t._FatJet.orig[jidx].particleNetMD_Xbb + t._FatJet.orig[jidx].particleNetMD_Xcc + t._FatJet.orig[jidx].particleNetMD_Xqq
        pnmdbvl = t._FatJet.orig[jidx].particleNetMD_Xbb/(t._FatJet.orig[jidx].particleNetMD_Xbb + t._FatJet.orig[jidx].particleNetMD_Xcc + t._FatJet.orig[jidx].particleNetMD_Xqq)

        print("Collisions%s_UltraLegacy_goldenJSON"%("".join(era[2:4])))
        puReweight = get_correction(syst_file["pileup"][era], 
            #"Collisions%s_UltraLegacy_goldenJSON"%("".join(era[-2:])),
            f"Collisions{era[2:4]}_UltraLegacy_goldenJSON",
            params={"NumTrueInteractions": lambda nTrueInt : nTrueInt},
            systParam="weights", 
            systNomName="nominal", systName="pu", systVariations=("up", "down"),
            sel=noSel,
        ) 
        if self.args.SR:
            jettriggerSF = get_correction(syst_file["triggerweights"],
                f"fatjet_triggerSF{era[:4]}",
                params = {"pt": lambda fj : fj.p4.Pt(), "msd" : lambda fj : corrected_msd[fj.idx] , },#"systematic" : "key",},
                systParam="systematic",
                systNomName="nominal", systVariations=("stat_up","stat_dn"),
                systName="triggerSF",
                sel=noSel,
            )
        #print(era+"_UL")
        if era in ["2017", "2018"]:
            year_key = era+"_UL"
        elif era == "2016APV":
            year_key = "2016preVFP_UL"
        elif era == "2016":
            year_key = "2016postVFP_UL"
        if self.args.CR1 or self.args.CR2:
            muoTrigSF = get_correction(syst_file["MUO"][era],
                "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose" if era in ["2017","2018"] else "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
                params={"pt": lambda mu: mu.pt,
                        "abseta": lambda mu: op.abs(mu.eta),
                        "year":year_key,
                        #"year":era+"_UL",
                        #"ValType":"syst"
                },
                systNomName="sf",
                systParam="ValType",
                systVariations={f"muotrigup": "systup", f"muotrigdown": "systdown"},
                systName="muotrig",
                sel=noSel,
            )
            muoIDSF = get_correction(syst_file["MUO"][era],
                "NUM_MediumPromptID_DEN_TrackerMuons",
                params={"pt": lambda mu: mu.pt,
                        "abseta": lambda mu: op.abs(mu.eta),
                        "year":year_key,
                        #"year":era+"_UL",
                        #"ValType":"sf"
                },
                systNomName="sf",
                systParam="ValType",
                systVariations={f"muoidup": "systup", f"muoiddown": "systdown"},
                systName="muoid",
                sel=noSel,
            ) 
            muoIsoSF = get_correction(syst_file["MUO"][era],
                "NUM_LooseRelIso_DEN_MediumPromptID",
                params={"pt": lambda mu: mu.pt,
                        "abseta": lambda mu: op.abs(mu.eta),
                        "year":year_key,
                        #"year":era+"_UL",
                        #"ValType":"sf"
                },
                systNomName="sf",
                systParam="ValType",
                systVariations={f"muoisoup": "systup", f"muoisodown": "systdown"},
                systName="muoiso",
                sel=noSel,
            ) 

                    



        if "WJetsToQQ" in sample or "WJetsToLNu" in sample:
            
            qcdWkfactor = get_correction(syst_file["NLOVkfactors"],
                "ULW_MLMtoFXFX",
                 params={"vpt" : lambda w : w.pt,},
                 systNomName="nominal",
                 sel=noSel,
            )

            nloWkfactor = get_correction(syst_file["NLOVkfactors"],
                "W_FixedOrderComponent",
                 params={"vpt" : lambda w : w.pt,},
                 systParam="systematic",
                 systNomName="nominal", systVariations=("d1K_NLO_up","d2K_NLO_up","d3K_NLO_up","d1kappa_EW_up","W_d2kappa_EW_up","W_d3kappa_EW_up","d1K_NLO_down","d2K_NLO_down","d3K_NLO_down","d1kappa_EW_down","W_d2kappa_EW_down","W_d3kappa_EW_down"),
                 sel=noSel,
            )

        if "ZJets" in sample or "DYJetsToLL" in sample:

            qcdZkfactor = get_correction(syst_file["NLOVkfactors"],
                "ULZ_MLMtoFXFX",
                 params={"vpt" : lambda w : w.pt,},
                 systNomName="nominal",
                 sel=noSel,
            )

            nloZkfactor = get_correction(syst_file["NLOVkfactors"],
                "Z_FixedOrderComponent",
                 params={"vpt" : lambda w : w.pt,},
                 systParam="systematic",
                 systNomName="nominal", systVariations=("d1K_NLO_up","d2K_NLO_up","d3K_NLO_up","d1kappa_EW_up","Z_d2kappa_EW_up","Z_d3kappa_EW_up","d1K_NLO_down","d2K_NLO_down","d3K_NLO_down","d1kappa_EW_down","Z_d2kappa_EW_down","Z_d3kappa_EW_down"),
                 sel=noSel,
            )

        if "HiggsToBB" in sampleCfg["group"]:
            
            nloVBFkfactor = get_correction(syst_file["EWHiggsCorrections"],
                 "VBF_EW",
                 params={"hpt" : lambda w : w.pt,},
                 sel=noSel,
            )

            nloVHkfactor = get_correction(syst_file["EWHiggsCorrections"],
                 "VH_EW",
                 params={"hpt" : lambda w : w.pt,},
                 sel=noSel,
            )
            nlottHkfactor = get_correction(syst_file["EWHiggsCorrections"],
                 "ttH_EW",
                 params={"hpt" : lambda w : w.pt,},
                 sel=noSel,
            )


        if self.isMC(sample):
            if self.args.SR:
                noSel = noSel.refine("trigweight",weight=jettriggerSF(fatjets[0]))
            elif self.args.CR1 or self.args.CR2:
                noSel = noSel.refine("mutrigweight",cut=op.rng_len(candidatemuons)>0,weight=muoTrigSF(candidatemuons[0]))
                noSel = noSel.refine("muIsoweight",cut=op.rng_len(candidatemuons)>0,weight=muoIsoSF(candidatemuons[0]))
                noSel = noSel.refine("muIDweight",cut=op.rng_len(candidatemuons)>0,weight=muoIDSF(candidatemuons[0]))
            noSel = noSel.refine("puweight",weight=puReweight(t.Pileup_nTrueInt))
            normalizationSyst = op.systematic(op.c_float(1.), name="testNormalizationSyst", up=op.c_float(1.25), down=op.c_float(0.75))
            noSel = noSel.refine("testSystematic", weight=normalizationSyst)
            #noSel = noSel.refine("puweight",weight=op.systematic(puReweightNom(t.Pileup_nTrueInt), name="puReweight",up=puReweightUp(t.Pileup_nTrueInt),down=puReweightDown(t.Pileup_nTrueInt)))

            if "18" not in era:
                noSel = noSel.refine("L1prefireweight",op.systematic(t.L1PreFiringWeight_Nom, name="L1PreFiring", up=t.L1PreFiringWeight_Up, down=t.L1PreFiringWeight_Dn))

            if "WJetsToQQ" in sample or "WJetsToLNu" in sample:
                noSel = noSel.refine("nloWkfactor",weight=nloWkfactor(w_by_status[0]))
                noSel = noSel.refine("qcdWkfactor",weight=qcdWkfactor(w_by_status[0]))
            if "ZJets" in sample or "DYJetsToLL" in sample:
                noSel = noSel.refine("nloZkfactor",weight=nloZkfactor(w_by_status[0]))
                noSel = noSel.refine("qcdZkfactor",weight=qcdZkfactor(w_by_status[0]))
            if "VBF" in sample:
                noSel = noSel.refine("nloHkfactor",weight=nloVBFkfactor(w_by_status[0])) 
            if "WplusH" in sample or "WminusH" in sample or "ZH" in sample:
                noSel = noSel.refine("nloHkfactor",weight=nloVHkfactor(w_by_status[0]))  
            if "ttH" in sample:
                noSel = noSel.refine("nloHkfactor",weight=nlottHkfactor(w_by_status[0]))  
            if "TT" in sample:
                toppt_weight1 = op.exp(0.0615-0.0005*op.min(op.max(top_by_status[0].pt,0.),500.))
                toppt_weight2 = op.exp(0.0615-0.0005*op.min(op.max(top_by_status[1].pt,0.),500.))
                noSel = noSel.refine("ttreweight",weight=op.sqrt(op.product(toppt_weight1,toppt_weight2)))

            jetSel = noSel.refine("passJetHLT",)
            filterJetSel = jetSel.refine("passFilterJet",)
            muSel = noSel.refine("passMuHLT",)
            filterMuSel = muSel.refine("passFilterMu",)

            if self.args.SR:
                btvSF = lambda flav : get_bTagSF_itFit(syst_file["BTV"][era],"deepJet", "btagDeepFlavB", flav, noSel, syst_prefix="btagSF_",
                    decorr_eras=True, era=year_key,
                )
                btvWeight = makeBtagWeightItFit(ak4_jets, btvSF)
            else:
                btvSF = lambda flav : get_bTagSF_itFit(syst_file["BTV"][era],"deepJet", "btagDeepFlavB", flav, noSel, syst_prefix="btagSF_",
                    decorr_eras=True, era=year_key,
                )
                btvWeight = makeBtagWeightItFit(jets_away, btvSF)
  
	########################
	###   SR selection   ###
	########################

        #sr_cut_dict = {
        #    "fj_pt_cut" : fatjets[0].p4.Pt() > sr_pt_cut,
        #    "fj_eta_cut" : op.abs(fatjets[0].p4.Eta()) < 2.5)
        #
        #}
        SR_pt_cut = filterJetSel.refine("fj_pt_cut",cut=fatjets[0].p4.Pt() > sr_pt_cut)
        SR_eta_cut = SR_pt_cut.refine("fj_eta_cut",cut=op.abs(fatjets[0].p4.Eta()) < 2.5)
        SR_msd_cut = SR_eta_cut.refine("fj_msd_cut",cut=corrected_msd[fatjets[0].idx]>40)
        SR_rho_cut = SR_msd_cut.refine("fj_rho_cut",cut=op.AND(2*op.log(corrected_msd[fatjets[0].idx]/fatjets[0].pt) > -8,2*op.log(corrected_msd[fatjets[0].idx]/fatjets[0].pt) <-1.))
        SR_id_cut = SR_rho_cut.refine("id_cut",cut=[fatjets[0].jetId & (1 << 1) !=0])
        if self.isMC(sample):
            SR_antiak4btagMediumOppHem_cut = SR_id_cut.refine("SR_antiak4btagMediumOppHem_cut",cut=[ak4_jet_opp_hemisphere.btagDeepB < btagWPs[era]["M"]], weight=btvWeight)
        else: 
            SR_antiak4btagMediumOppHem_cut = SR_id_cut.refine("SR_antiak4btagMediumOppHem_cut",cut=[ak4_jet_opp_hemisphere.btagDeepB < btagWPs[era]["M"]])
        SR_electron_cut = SR_antiak4btagMediumOppHem_cut.refine("el_cut",cut=[op.rng_len(electrons) == 0])
        SR_muon_cut = SR_electron_cut.refine("mu_cut",cut=[op.rng_len(loose_muons) == 0])
        SR_tau_cut = SR_muon_cut.refine("tau_cut",cut=[op.rng_len(taus) == 0]) 
        if "ZJetsToQQ" in sample or "ZJetsToBB" in sample or "WJetsToQQ" in sample or "VectorZPrime" in sample or "HiggsToBB" in sampleCfg["group"]:
            print("adding matching to cut")
            SR_Vmatched = SR_tau_cut.refine("fj_Vmatched",cut=op.AND(Vgen_matched,Vgen_quality_criterion_pt,Vgen_quality_criterion_msd))
        else:
            SR_Vmatched = SR_tau_cut.refine("fj_Vmatched",cut=1) 
        SR_MET = SR_Vmatched.refine("MET_cut",cut=[t.MET.pt < 140.])
        SR_nak4jets = SR_MET.refine("SR_nak4jets",cut=[op.rng_len(ak4_jets) < 6]) 
        #self.yield_object.addYields(filterJetSel,"starting","test")
        #self.yield_object.addYields(SR_pt_cut,"pt_cut","test")
 
        SR_cut = SR_nak4jets
        #SR_cut = SR_Vmatched
        #######################
        ###  CR1 selection  ###
        #######################
        CR1_jpt_cut = filterMuSel.refine("CR1_jpt_cut",cut=fatjets[0].p4.Pt() > 400)
        CR1_jeta_cut = CR1_jpt_cut.refine("CR1_jeta_cut",cut=op.abs(fatjets[0].p4.Eta())<2.5)
        CR1_jmsd_cut = CR1_jeta_cut.refine("CR1_jmsd_cut",cut=corrected_msd[fatjets[0].idx] > 40)
        CR1_jrho_cut = CR1_jmsd_cut.refine("CR1_jrho_cut",cut=op.AND(2*op.log(fatjets[0].msoftdrop/fatjets[0].pt) > -7,2*op.log(fatjets[0].msoftdrop/fatjets[0].pt) <-1.))
        CR1_id_cut = CR1_jrho_cut.refine("CR1_id_cut",cut=[fatjets[0].jetId & (1 << 1) !=0])
        #CR1_mu_cut = CR1_id_cut.refine("CR1_mu_cut",cut=[op.rng_len(loose_muons) == 1])
        CR1_muonDphiAK8 = CR1_id_cut.refine("CR1_muonDphiAK8",cut=op.deltaPhi(candidatemuons[0].p4, fatjets[0].p4) > 2.*np.pi/3.)
        #CR1_muonDphiAK8 = CR1_mu_cut.refine("CR1_muonDphiAK8",cut=op.rng_any(t.Muon[0], lambda mu : op.abs(op.deltaPhi(mu.p4, fatjets[0].p4))>2*np.pi/3))
        if self.isMC(sample):
            CR1_ak4btagMedium08 = CR1_muonDphiAK8.refine("CR1_ak4btagMedium08",cut=op.rng_any(jets_away, lambda j : j.btagDeepB>btagWPs[era]["M"]),weight=btvWeight)
        else:
            CR1_ak4btagMedium08 = CR1_muonDphiAK8.refine("CR1_ak4btagMedium08",cut=op.rng_any(jets_away, lambda j : j.btagDeepB>btagWPs[era]["M"]))
        CR1_electron_cut = CR1_ak4btagMedium08.refine("CR1_electron_cut",cut=[op.rng_len(electrons) == 0])
        CR1_mu_ncut = CR1_electron_cut.refine("CR1_mu_ncut",cut=[op.rng_len(candidatemuons) == 1])
        CR1_tau_cut = CR1_mu_ncut.refine("CR1_tau_cut",cut=[op.rng_len(taus) == 0]) 
        CR1_cut = CR1_tau_cut

        #######################
        ###  CR2 selection  ###
        #######################
        Wleptonic_candidate = op.sum(candidatemuons[0].p4,t.MET.p4)
        CR2_jpt_cut = filterMuSel.refine("CR2_jpt_cut",cut=fatjets[0].p4.Pt() > 200)
        CR2_jeta_cut = CR2_jpt_cut.refine("CR2_jeta_cut",cut=op.abs(fatjets[0].p4.Eta())<2.5)
        CR2_jmsd_cut = CR2_jeta_cut.refine("CR2_jmsd_cut",cut=corrected_msd[fatjets[0].idx] > 40)
        CR2_mu_pt_cut = CR2_jmsd_cut.refine("CR2_mu_pt_cut",cut=candidatemuons[0].pt > 53)
        CR2_mu_eta_cut = CR2_mu_pt_cut.refine("CR2_mu_eta_cut",cut=op.abs(candidatemuons[0].p4.Eta()) < 2.1)
        CR2_mu_pfRelIso04_all_cut = CR2_mu_eta_cut.refine("CR2_mu_pfRelIso04_all_cut",cut=candidatemuons[0].pfRelIso04_all<0.15)
        CR2_mu_tightId_cut = CR2_mu_pfRelIso04_all_cut.refine("CR2_mu_tightId_cut",cut=candidatemuons[0].tightId)
        CR2_muonDphiAK8 = CR2_mu_tightId_cut.refine("CR2_muonDphiAK8",cut=op.abs(op.deltaPhi(candidatemuons[0].p4, fatjets[0].p4))>2*np.pi/3)
        CR2_ak4btagMedium08 = CR2_muonDphiAK8.refine("CR2_ak4btagMedium08",cut=op.rng_any(jets_away, lambda j : j.btagDeepB>btagWPs[era]["M"]))
        CR2_MET = CR2_ak4btagMedium08.refine("CR2_MET",t.MET.pt>40)
        CR2_electron_cut = CR2_MET.refine("CR2_electron_cut",cut=[op.rng_len(electrons) == 0])
        CR2_tau_cut = CR2_electron_cut.refine("CR2_tau_cut",cut=[op.rng_len(taus) == 0])
        CR2_mu_cutloose = CR2_tau_cut.refine("CR2_mu_cutloose",cut=[op.rng_len(candidatemuons) == 1])
        CR2_Wleptonic_cut = CR2_mu_cutloose.refine("CR2_Wleptonic_cut",cut=Wleptonic_candidate.Pt()>200)
        print(sampleCfg) 
        if self.args.CR2 and "subprocess" in sampleCfg.keys():
            #Find lastcopy w closest to candidate fatjet and count quarks in 0.8
            w = op.sort(op.select(t.GenPart, lambda p : op.AND(op.abs(p.pdgId) == 24, p.statusFlags & 2**GEN_FLAGS["IsLastCopy"])),lambda p: op.deltaR(p.p4, fatjets[0].p4))[0]
            genQuarks = op.select(t.GenPart, lambda q: op.AND(op.abs(q.pdgId) >= 1, op.abs(q.pdgId) <= 5))
            q_from_w = op.select(genQuarks, lambda q : q.parent.idx == w.idx)
            Vgen_matched = op.rng_count(q_from_w, lambda q: op.deltaR(q.p4, fatjets[0].p4) < 0.8) == 2

            Vgen_quality_criterion_pt = ((fatjets[0].pt - w.pt)/w.pt) < 0.5
            Vgen_quality_criterion_msd = ((corrected_msd[fatjets[0].idx] - w.mass)/w.mass) < 0.3

            #Vgen_matched = op.deltaR(w_from_top.p4, fatjets[0].p4) < 0.8
            subProc = sampleCfg["subprocess"]
            if "_matched" in subProc: 
                CR2_matching_cut = CR2_Wleptonic_cut.refine("CR2_matching",cut=op.AND(Vgen_matched,Vgen_quality_criterion_pt,Vgen_quality_criterion_msd))
            elif "_unmatched" in subProc:
                CR2_matching_cut = CR2_Wleptonic_cut.refine("CR2_matching",cut=op.NOT(op.AND(Vgen_matched,Vgen_quality_criterion_pt,Vgen_quality_criterion_msd)))
            CR2_cut = CR2_matching_cut
        else:
            CR2_cut = CR2_Wleptonic_cut
        



        #this should be the last cut defined above
        # CR_cut = muonDphiAK8.refine("CR_cut", cut=[op.AND(hasTriggerObj,hasBtaggedAK4,noEle,oneMuon,muonDphiAK8)] ) 

        ### these commands create the .tex file with the efficiency table ###

        if self.args.CR2:
            CR2_yields = CutFlowReport("CR2_yields", printInLog=True,)# recursive=True)
            plots.append(CR2_yields)
            CR2_yields.add(noSel, title ='TEST')
            CR2_yields.add(muSel, title= 'trigger')
            CR2_yields.add(filterMuSel, title= 'filters')
            CR2_yields.add(CR2_jpt_cut, title= 'CR2_jpt_cut')
            CR2_yields.add(CR2_jeta_cut, title= 'CR2_jeta_cut')
            CR2_yields.add(CR2_jmsd_cut, title= 'CR2_jmsd_cut')
            CR2_yields.add(CR2_mu_pt_cut, title= 'CR2_mu_pt_cut')
            CR2_yields.add(CR2_mu_eta_cut, title= 'CR2_mu_eta_cut')
            CR2_yields.add(CR2_mu_pfRelIso04_all_cut, title= 'CR2_mu_pfRelIso04_all_cut')
            CR2_yields.add(CR2_mu_tightId_cut, title= 'CR2_mu_tightId_cut')
            #CR2_yields.add(CR2_jrho_cut, title= 'CR2_jrho_cut')
            #CR2_yields.add(CR2_mu_cut, title= 'CR2_mu_cut')
            CR2_yields.add(CR2_muonDphiAK8, title= 'CR2_muonDphiAK8')
            CR2_yields.add(CR2_ak4btagMedium08, title= 'CR2_ak4btagMedium08')
            CR2_yields.add(CR2_MET, title= 'CR2_MET')
            CR2_yields.add(CR2_electron_cut, title= 'CR2_electron_cut')
            CR2_yields.add(CR2_mu_cutloose, title= 'CR2_mu_cutloose')
            CR2_yields.add(CR2_Wleptonic_cut, title= 'CR2_Wleptonic_cut')
            #print("sampleCfg[subprocess]",sampleCfg["subprocess"])
            if "subprocess" in sampleCfg.keys():  
                CR2_yields.add(CR2_matching_cut, title= 'CR2_matching')

        if self.args.CR1:
            CR1_yields = CutFlowReport("CR1_yields", printInLog=True,)# recursive=True)
            plots.append(CR1_yields)
            CR1_yields.add(noSel, title ='TEST')
            CR1_yields.add(muSel, title= 'trigger')
            CR1_yields.add(filterMuSel, title= 'filters')
            CR1_yields.add(CR1_jpt_cut, title= 'CR1_jpt_cut')
            CR1_yields.add(CR1_jeta_cut, title= 'CR1_jeta_cut')
            CR1_yields.add(CR1_jmsd_cut, title= 'CR1_jmsd_cut')
            CR1_yields.add(CR1_jrho_cut, title= 'CR1_jrho_cut')
            CR1_yields.add(CR1_id_cut, title= 'CR1_id_cut')
            #CR1_yields.add(CR1_mu_cut, title= 'CR1_mu_cut')
            CR1_yields.add(CR1_muonDphiAK8, title= 'CR1_muonDphiAK8')
            CR1_yields.add(CR1_ak4btagMedium08, title= 'CR1_ak4btagMedium08')
            CR1_yields.add(CR1_electron_cut, title= 'CR1_electron_cut')
            CR1_yields.add(CR1_mu_ncut, title= 'CR1_mu_ncut')
            CR1_yields.add(CR1_tau_cut, title= 'CR1_tau_cut')
 
        if self.args.SR:
            SR_yields = CutFlowReport("SR_yields", printInLog=True,)# recursive=True)
            plots.append(SR_yields)
            SR_yields.add(noSel, title ='inclusive')
            SR_yields.add(filterJetSel, title= 'filters')
            #SR_yields.add(trigweightedJetSel, title= 'trigweightedJetSel')
            SR_yields.add(jetSel, title ='trigger')
            SR_yields.add(SR_pt_cut, title= 'fj_pt')
            SR_yields.add(SR_eta_cut, title= 'fj_eta')
            SR_yields.add(SR_msd_cut, title= 'fj_msd')
            SR_yields.add(SR_rho_cut, title= 'fj_rho')
            SR_yields.add(SR_antiak4btagMediumOppHem_cut, title ='SR_antiak4btagMediumOppHem_cut')
            SR_yields.add(SR_electron_cut, title= 'no_electron')
            SR_yields.add(SR_muon_cut, title= 'no_muon')
            SR_yields.add(SR_tau_cut, title= 'no_tau')
            SR_yields.add(SR_Vmatched, title= 'fj_vmatched')
            SR_yields.add(SR_MET, title= 'met')
            SR_yields.add(SR_nak4jets, title= 'met')

         
        mvaVariables = {
            "weight"            :   noSel.weight,
            "pt"                :   fatjets[0].p4.Pt(),
            "eta"               :   fatjets[0].p4.Eta(),
            "msd"               :   fatjets[0].msoftdrop,
            "corr_msd"          :   corrected_msd[fatjets[0].idx],
            "jetId"             :   fatjets[0].jetId,
            "n2b1"              :   fatjets[0].n2b1,
            #"pnmd2prong_ddt"    :   pnmd2prong_ddt,
            "particleNetMD_Xqq" :   fatjets[0].particleNetMD_Xqq,
            "particleNetMD_Xcc" :   fatjets[0].particleNetMD_Xcc,
            "particleNetMD_Xbb" :   fatjets[0].particleNetMD_Xbb,
            "particleNetMD_QCD" :   fatjets[0].particleNetMD_QCD,
        }
        if do_genmatch:
            mvaVariables["is_Vmatched"]    = op.AND(Vgen_matched, Vgen_quality_criterion_pt, Vgen_quality_criterion_msd)
            mvaVariables["q1_flavor"]      = q_from_w[0].pdgId
            mvaVariables["q2_flavor"]      = q_from_w[1].pdgId
        #if self.isMC(sample):
        #    mvaVariables["msd_jesUp"]   =  t._FatJet["jesTotalup"]  [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jesDown"] =  t._FatJet["jesTotaldown"][fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jerUp"]   =  t._FatJet["jerup"]       [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jerDown"] =  t._FatJet["jerdown"]     [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jmsUp"]   =  t._FatJet["jmsup"]       [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jmsDown"] =  t._FatJet["jmsdown"]     [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jmrUp"]   =  t._FatJet["jmrup"]       [fatjets[0].idx].msoftdrop
        #    mvaVariables["msd_jmrDown"] =  t._FatJet["jmrdown"]     [fatjets[0].idx].msoftdrop


        ### Save mvaVariables to be retrieved later in the postprocessor and saved in a parquet file ###
       
        if self.args.SR:
            selection = SR_cut
            prefix="SR_"
            if self.args.mvaSkim:
                from bamboo.plots import Skim
                
                parquet_cut = noSel.refine("parquet_cut", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt>170,corrected_msd[fatjets[0].idx]>5.,op.rng_len(fatjets)>0), ak4_jet_opp_hemisphere.btagDeepB < btagWPs[era]["M"]])
                plots.append(Skim("signal_region", mvaVariables, parquet_cut))

        elif self.args.CR1:
            selection = CR1_cut
            prefix="CR1_" 

        elif self.args.CR2:
            selection = CR2_cut
            prefix="CR2_"
            if self.args.mvaSkim:
                from bamboo.plots import Skim
                plots.append(Skim("CR2", mvaVariables, selection))

        #PNBVL CUTS
        pn_bvl_cut = 0.5

        ##### PT-BINNED PLOTS 
        for iptbin, (pt_low, pt_high) in enumerate(pt_bins):
            pt_sel = op.AND(fatjets[0].pt > pt_low, fatjets[0].pt <= pt_high)


            ### One SR
            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p05_pass", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["L"])), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong loose (%i < p_{T} < %i)"%(pt_low, pt_high)))
            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p05_fail", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_fail",cut=op.AND(pt_sel,pnmd2prong<=pnmdWPs[era]["L"])), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_fail", xTitle="Jet m_{SD} (GeV) Fail PN-MD 2prong loose (%i < p_{T} < %i)"%(pt_low, pt_high)))


            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p01_pass", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass_0p01",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["T"])), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong tight (%i < p_{T} < %i)"%(pt_low, pt_high)))
            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p01_fail", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_fail_0p01",cut=op.AND(pt_sel,pnmd2prong<=pnmdWPs[era]["T"])), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_fail", xTitle="Jet m_{SD} (GeV) Fail PN-MD 2prong tight (%i < p_{T} < %i)"%(pt_low, pt_high)))


            if self.args.split_signal_region:
                ### Two SRs
            
                plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p05_pass_lowbvl", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass_lowbvl",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["L"],pnmdbvl<=pn_bvl_cut)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass_lowbvl", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong loose, Low bvl (%i < p_{T} < %i)"%(pt_low, pt_high)))
                plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p05_pass_highbvl", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass_highbvl",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["L"],pnmdbvl>pn_bvl_cut)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass_highbvl", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong loose, High bvl (%i < p_{T} < %i)"%(pt_low, pt_high)))
                plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p01_pass_lowbvl", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass_lowbvl_0p01",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["T"],pnmdbvl<=pn_bvl_cut)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass_0p01_lowbvl", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong tight, Low bvl (%i < p_{T} < %i)"%(pt_low, pt_high)))
                plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_pnmd2prong_0p01_pass_highbvl", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_pnmd2prong_pass_highbvl_0p01",cut=op.AND(pt_sel,pnmd2prong>pnmdWPs[era]["T"],pnmdbvl>pn_bvl_cut)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_pnmd2prong_pass_0p01_highbvl", xTitle="Jet m_{SD} (GeV) Pass PN-MD 2prong tight, High bvl (%i < p_{T} < %i)"%(pt_low, pt_high)))


            ### N2
            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_n2_0p05_pass", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_n2_pass",cut=op.AND(pt_sel,fatjets[0].n2b1<0.19)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_n2_pass", xTitle="Jet m_{SD} (GeV) Pass N_{2} (%i < p_{T} < %i)"%(pt_low, pt_high)))
            plots.append(Plot.make1D(prefix+f"ptbin{iptbin}_n2_0p05_fail", corrected_msd[fatjets[0].idx], selection.refine(f"ptbin{iptbin}_n2_fail",cut=op.AND(pt_sel,fatjets[0].n2b1>=0.19)), EquidistantBinning(62,40.,350.), title=f"ptbin{iptbin}_n2_fail", xTitle="Jet m_{SD} (GeV) Fail N_{2} (%i < p_{T} < %i)"%(pt_low, pt_high)))
   
     
        #####INCLUSIVE PLOTS 
        inclusive_pt_sel = op.AND(fatjets[0].pt > sr_pt_cut, fatjets[0].pt <= 1200)
        #pnMD_2prong = fatjets[0].particleNetMD_Xqq + fatjets[0].particleNetMD_Xcc + fatjets[0].particleNetMD_Xbb
        #### ParticleNet-MD plots
        plots.append(Plot.make1D(prefix+"particlenet_2prong_MD", pnmd2prong, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD ZPrime binary score", xTitle="ParticleNet-MD 2prong score"))
        #plots.append(Plot.make1D(prefix+"particlenet_2prong_MD_ddt", pnmd2prong_ddt, selection, EquidistantBinning(25,-1.,0.1), title="ParticleNet-MD ZPrime binary score ddt", xTitle="ParticleNet-MD 2prong score (DDT)"))
        plots.append(Plot.make1D(prefix+"particlenet_bb_MD", fatjets[0].particleNetMD_Xbb, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD bb score", xTitle="ParticleNet-MD bb score"))
        plots.append(Plot.make1D(prefix+"particlenet_cc_MD", fatjets[0].particleNetMD_Xcc, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD cc score", xTitle="ParticleNet-MD cc score"))
        plots.append(Plot.make1D(prefix+"particlenet_qq_MD", fatjets[0].particleNetMD_Xqq, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD qq score", xTitle="ParticleNet-MD qq score"))
        plots.append(Plot.make1D(prefix+"particlenet_bb_MD_vs_QCD", fatjets[0].particleNetMD_Xbb/(fatjets[0].particleNetMD_Xbb+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD bb score", xTitle="ParticleNet-MD bb vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_cc_MD_vs_QCD", fatjets[0].particleNetMD_Xcc/(fatjets[0].particleNetMD_Xcc+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD cc score", xTitle="ParticleNet-MD cc vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_qq_MD_vs_QCD", fatjets[0].particleNetMD_Xqq/(fatjets[0].particleNetMD_Xqq+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD qq score", xTitle="ParticleNet-MD qq vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_QCD_MD", fatjets[0].particleNetMD_QCD, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD QCD score", xTitle="ParticleNet-MD QCD score"))
        #### Jet kinematics 
        plots.append(Plot.make1D(prefix+"FatjetMsd_corrected", corrected_msd[fatjets[0].idx], selection, EquidistantBinning(25,40.,400.), title="FatJet pT", xTitle="FatJet m_{SD} corrected (GeV)"))
        if self.args.split_signal_region:

            plots.append(Plot.make1D(prefix+f"pnmd2prong_0p01_pass_highbvl", corrected_msd[fatjets[0].idx], selection.refine("tight_lowbvl",cut=op.AND(pnmd2prong>pnmdWPs[era]["T"],pnmdbvl>0.5)), EquidistantBinning(62,40.,350.), title="FatJet corrected msd low bvl", xTitle="FatJet m_{SD} corrected tight High bvl (GeV)"))
            plots.append(Plot.make1D(prefix+f"pnmd2prong_0p01_pass_lowbvl", corrected_msd[fatjets[0].idx], selection.refine("tight_highbvl",cut=op.AND(pnmd2prong>pnmdWPs[era]["T"],pnmdbvl<=0.5)), EquidistantBinning(62,40.,350.), title="FatJet corrected msd high bvl", xTitle="FatJet m_{SD} corrected tight Low bvl (GeV)"))



            plots.append(Plot.make1D(prefix+f"pnmd2prong_0p05_pass_highbvl", corrected_msd[fatjets[0].idx], selection.refine("loose_lowbvl",cut=op.AND(pnmd2prong>pnmdWPs[era]["L"],pnmdbvl>0.5)), EquidistantBinning(62,40.,350.), title="FatJet corrected msd low bvl", xTitle="FatJet m_{SD} corrected loose High bvl (GeV)"))
            plots.append(Plot.make1D(prefix+f"pnmd2prong_0p05_pass_lowbvl", corrected_msd[fatjets[0].idx], selection.refine("loose_highbvl",cut=op.AND(pnmd2prong>pnmdWPs[era]["L"],pnmdbvl<=0.5)), EquidistantBinning(62,40.,350.), title="FatJet corrected msd high bvl", xTitle="FatJet m_{SD} corrected loose Low bvl (GeV)"))

        plots.append(Plot.make1D(prefix+"FatjetMsd", fatjets[0].msoftdrop, selection, EquidistantBinning(25,40.,400.), title="FatJet pT", xTitle="FatJet m_{SD} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetPt", fatjets[0].p4.Pt(), selection, EquidistantBinning(25,200.,1400.) if self.args.CR2 else EquidistantBinning(25,400,1400.), title="FatJet pT", xTitle="FatJet p_{T} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetEta", fatjets[0].p4.Eta(), selection, EquidistantBinning(25,-2.5,2.5), title="FatJet #eta", xTitle="FatJet #eta"))
        plots.append(Plot.make1D(prefix+"FatJetRho", 2*op.log(corrected_msd[fatjets[0].idx]/fatjets[0].pt), selection, EquidistantBinning(25,-5.5,-2), title="FatJet #rho", xTitle="FatJet #rho"))
        plots.append(Plot.make1D(prefix+"FatJetN2",  fatjets[0].n2b1, selection, EquidistantBinning(25,0,0.5), title="FatJet N2", xTitle="FatJet N_{2}"))
        plots.append(Plot.make1D(prefix+"MET",  t.MET.pt, selection, EquidistantBinning(25,0,200), title="MET", xTitle="MET (GeV)"))
        plots.append(Plot.make1D(prefix+"nAK4",  op.rng_len(ak4_jets), selection, EquidistantBinning(10,0,10), title="nAK4jets", xTitle="Number of AK4 jets"))
 

        if self.args.CR1 or self.args.CR2: 
            #### Muon kinematics 
            plots.append(Plot.make1D(prefix+"muonpt",candidatemuons[0].pt, selection, EquidistantBinning(50,0.,1000.),title= "Candidate muon pt", xTitle="Muon p_{T} (GeV)" ))
            plots.append(Plot.make1D(prefix+"pfRelIso04_all",candidatemuons[0].pfRelIso04_all, selection, EquidistantBinning(20,0.,.4),title= "MuonpfRelIso04_all", xTitle="Muon relative isolation (0.4)" ))
            plots.append(Plot.make1D(prefix+"_mujetpt",fatjets[0].p4.Pt(), selection,EquidistantBinning(20,500.,1000.),title= "Jet p_{T} (GeV)", xTitle="Jet p_{T} (GeV)" ))
            plots.append(Plot.make1D(prefix+"_mujetmsd",fatjets[0].msoftdrop, selection,EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV)", xTitle="Jet m_{SD} (GeV)" ))
            plots.append(Plot.make1D(prefix+"_mujetmsd_corrected",corrected_msd[fatjets[0].idx], selection,EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV)", xTitle="Jet m_{SD} (GeV)" ))
            #plots.append(Plot.make1D(prefix+"_pnmd2prong_ddt_pass",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_ddt_pass",cut=op.AND(selection,pnmd2prong_ddt>0.)),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Pass PN-MD DDT)", xTitle="Jet m_{SD} (GeV) (Pass PN-MD DDT)" ))
            #plots.append(Plot.make1D(prefix+"_pnmd2prong_ddt_fail",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_ddt_fail",cut=op.AND(selection,pnmd2prong_ddt<=0.)),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Fail PN-MD DDT)", xTitle="Jet m_{SD} (GeV) (Fail PN-MD DDT)" ))
            plots.append(Plot.make1D(prefix+"pnmd2prong_0p01_pass",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_0p01_pass",cut=op.AND(selection,pnmd2prong>pnmdWPs[era]["T"])),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Pass PN-MD 1%)", xTitle="Jet m_{SD} (GeV) (Pass PN-MD 1%)" ))
            plots.append(Plot.make1D(prefix+"pnmd2prong_0p01_fail",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_0p01_fail",cut=op.AND(selection,pnmd2prong<=pnmdWPs[era]["T"])),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Fail PN-MD 1%)", xTitle="Jet m_{SD} (GeV) (Fail PN-MD w%)" ))


            plots.append(Plot.make1D(prefix+"_pnmd2prong_0p05_pass",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_0p05_pass",cut=op.AND(selection,pnmd2prong>pnmdWPs[era]["L"])),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Pass PN-MD 5%)", xTitle="Jet m_{SD} (GeV) (Pass PN-MD 5%)" ))
            plots.append(Plot.make1D(prefix+"_pnmd2prong_0p05_fail",corrected_msd[fatjets[0].idx], selection.refine("pnmd2prong_0p05_fail",cut=op.AND(selection,pnmd2prong<=pnmdWPs[era]["L"])),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Fail PN-MD 5%)", xTitle="Jet m_{SD} (GeV) (Fail PN-MD 5%)" ))
            plots.append(Plot.make1D(prefix+"_n2_0p05_pass",corrected_msd[fatjets[0].idx], selection.refine("n2_0p05_pass",cut=op.AND(selection,fatjets[0].n2b1<0.19)),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Pass N_{2} 5%)", xTitle="Jet m_{SD} (GeV) (Pass N_{2} 5%)" ))
            plots.append(Plot.make1D(prefix+"_n2_0p05_fail",corrected_msd[fatjets[0].idx], selection.refine("n2_0p05_fail",cut=op.AND(selection,fatjets[0].n2b1>=0.19)),EquidistantBinning(40,40.,240.),title= "Jet m_{SD} (GeV) (Fail N_{2} 5%)", xTitle="Jet m_{SD} (GeV) (Fail N_{2} 5%)" ))
        #print("helloxx")
        #plots.extend(self.yield_object.returnPlots())

        return plots

    def postProcess(self, taskList, config=None, workdir=None, resultsdir=None):
        super(zprlegacy, self).postProcess(taskList, config=config, workdir=workdir, resultsdir=resultsdir)
        from bamboo.plots import Plot, DerivedPlot
        plotList = [ ap for ap in self.plotList if ( isinstance(ap, Plot) or isinstance(ap, DerivedPlot) ) ]
        from bamboo.analysisutils import loadPlotIt
        p_config, samples, plots, systematics, legend = loadPlotIt(config, plotList, eras=self.args.eras[1], workdir=workdir, resultsdir=resultsdir, readCounters=self.readCounters, vetoFileAttributes=self.__class__.CustomSampleAttributes, plotDefaults=self.plotDefaults)

        from bamboo.root import gbl
        import math


        ########################################################################################################
        ######## save a parquet file with the variables needed for MVA training ################################
        ########################################################################################################
        #mvaSkim
        from bamboo.plots import Skim
        skims = [ap for ap in self.plotList if isinstance(ap, Skim)]
        if self.args.mvaSkim and skims:
            from bamboo.analysisutils import loadPlotIt
            p_config, samples, _, systematics, legend = loadPlotIt(config, [], eras=self.args.eras[1], workdir=workdir, resultsdir=resultsdir, readCounters=self.readCounters, vetoFileAttributes=self.__class__.CustomSampleAttributes)
            try:

                
                import os
                import pandas as pd
                import pyarrow as pa
                import pyarrow.parquet as pq
                from bamboo.root import gbl
                
                # Assuming `skims`, `samples`, `resultsdir` are defined
                
                for skim in skims:
                    for smp_idx, smp in enumerate(samples):
                        for cb_idx, cb in enumerate(smp.files if hasattr(smp, "files") else [smp]):
                            tree = cb.tFile.Get(skim.treeName)
                            if not tree:
                                print(f"KEY TTree {skim.treeName} does not exist, we are gonna skip this {smp}\n")
                            else:
                                cols = gbl.ROOT.RDataFrame(cb.tFile.Get(skim.treeName)).AsNumpy()
                                cols["weight"] *= cb.scale
                                cols["process"] = [smp.name] * len(cols["weight"])
                                df_chunk = pd.DataFrame(cols)
                                df_chunk["process"] = pd.Categorical(df_chunk["process"], categories=pd.unique(df_chunk["process"]), ordered=False)
                
                                # Generate a unique filename for each sample and callback
                                pq_filename = os.path.join(resultsdir, f"{skim.name}_{smp.name}_{cb_idx}.parquet")
                                df_chunk.to_parquet(pq_filename, index=False)
                
                                print(f"Written {pq_filename}")
                                del df_chunk  # Free memory immediately after writing
                
            except ImportError as ex:
                logger.error("Could not import pandas, no dataframes will be saved")

        
