#!/usr/bin/python -tt
from bamboo.analysismodules import NanoAODHistoModule, HistogramsModule
from bamboo.treedecorators import NanoAODDescription
from bamboo.scalefactors import binningVariables_nano,lumiPerPeriod_default
from bamboo.analysisutils import loadPlotIt

from bamboo import treefunctions as op
from bamboo import scalefactors

from itertools import chain
import math
import numpy as np

import logging
logger = logging.getLogger(__name__)
v_PDGID = {
    "ZJetsToQQ" : 24,
    "WJetsToQQ" : 23,
    "VectorZPrime" : 55,
    "TTbar" : 23,
    "SingleTop" : 23,
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

def goodFlag(p):
    #return op.AND([p.statusFlags & 2**GEN_FLAGS[FLAG] for FLAG in FLAGS],p.parent.idx >= 0)
    return op.AND(p.statusFlags & 2**GEN_FLAGS["IsLastCopy"], p.statusFlags & 2**GEN_FLAGS["FromHardProcess"],)# p.parent.idx >= 0)
    #return op.AND(p.statusFlags & 2**GEN_FLAGS["IsLastCopy"], p.parent.idx >= 0)


class zprlegacy(NanoAODHistoModule):
    def addArgs(self, parser):
        super().addArgs(parser)
        parser.add_argument("--mvaSkim", action="store_true", help="Produce MVA training skims")
        parser.add_argument("--mvaEval", action="store_true", help="Import MVA model and evaluate it on the dataframe")
        parser.add_argument("--SR", action="store_true", default=False, help="Make SR")
        parser.add_argument("--CR1", action="store_true", default=False, help="Make CR1")
        parser.add_argument("--CR2", action="store_true", default=False, help="Make CR2")
        parser.add_argument("--arbitration", action="store", required=True, help="Arbitration of jets.")
        #### Till now we don't need --mvaEval since we don't have a MVA model ####

    def __init__(self, args):
        super(zprlegacy, self).__init__(args)
        #if not (args.SR or args.CR1 or args.CR2):
        #    return RuntimeError("Need to run on SR/CR1/CR2")


    def prepareTree(self, tree, sample=None, sampleCfg=None, description=None, backend=None):
        ## initializes tree.Jet.calc so should be called first (better: use super() instead)
        # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC#Recommended_for_MC
#        tree,noSel,be,lumiArgs = NanoAODHistoModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg, description=NanoAODDescription.get('v7', year='2018', isMC=True), backend=backend)#, calcToAdd=["nMuon"])
        from bamboo.treedecorators import nanoRochesterCalc, nanoJetMETCalc, CalcCollectionsGroups, nanoFatJetCalc
        from bamboo.analysisutils import configureJets
        from bamboo.analysisutils import configureRochesterCorrection
        era = sampleCfg["era"]
        tree,noSel,be,lumiArgs = NanoAODHistoModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg,description=NanoAODDescription.get('v7', year=era,isMC=self.isMC(sample), systVariations=[nanoRochesterCalc,nanoJetMETCalc,nanoFatJetCalc]),backend=backend)
        #tree,noSel,be,lumiArgs = HistogramsModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg,)
        FatJetMETCalc = CalcCollectionsGroups(FatJet=("pt", "mass", "msoftdrop",),MET=("pt","mass"))
        #if era == "2018":
        #    configureRochesterCorrection(tree._Muon, "/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/fromYihan/RoccoR2018UL.txt", isMC=self.isMC(sample), backend=be)
        #    return RuntimeError("Need to run on some region!")

        addition = ""
        if self.isMC(sample):
            jesUncertaintySources = ["Total"]
            JECs = {'2017'        : "Summer19UL17_V5_MC", 
                    '2018'        : "Summer19UL18_V5_MC",
                    }
            JERs = {'2017'        : "Summer19UL17_JRV3_MC",
                    '2018'        : "Summer19UL18_JRV2_MC",
                    }
            mcYearForFatJets=era
        
        else:
            jesUncertaintySources = None
            JECs = {'2017B'       : "Summer19UL17_RunB_V5_DATA",
                    '2017C'       : "Summer19UL17_RunC_V5_DATA",
                    '2017D'       : "Summer19UL17_RunD_V5_DATA",
                    '2017E'       : "Summer19UL17_RunE_V5_DATA",
                    '2017F'       : "Summer19UL17_RunF_V5_DATA",
                    '2018A'       : "Summer19UL18_RunA_V5_DATA",
                    '2018B'       : "Summer19UL18_RunB_V5_DATA",
                    '2018C'       : "Summer19UL18_RunC_V5_DATA",
                    '2018D'       : "Summer19UL18_RunD_V5_DATA",
                    }
            
            JERs = {'2017'        : None,
                    '2018'        : None,
                    }
            mcYearForFatJets=None

            if era in ["2017","2018"]:
                  addition = sample.split(era)[1]
        print(addition)
        print (JECs[era+addition])
        print (self.isMC(sample))

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
                "mcYearForFatJets": mcYearForFatJets
                }

        print (cmJMEArgs["smear"])
        print (cmJMEArgs["mcYearForFatJets"])
        print (cmJMEArgs)
#        if self.isMC(sample):
#              cmJMEArgs["jesUncertaintySources"] = ["Total"]
#              configureJets(tree._FatJet, "AK8PFPuppi", mcYearForFatJets=era, **cmJMEArgs_mc)
#        else:
#              cmJMEArgs["jesUncertaintySources"] = ["Total"]
        configureJets(tree._FatJet, "AK8PFPuppi",  **cmJMEArgs)


        return tree,noSel,be,lumiArgs


    def definePlots(self,t, noSel, sample=None, sampleCfg=None):

        from bamboo.plots import Plot, EquidistantBinning, CutFlowReport
        from bamboo import treefunctions as op
        try: 
            do_genmatch = any(sampleCfg["group"] in x for x in v_PDGID.keys())
        except:
            do_genmatch = True
            #hotfix for Z' not having "group"
        print("do genmatch? ", do_genmatch)
        print("sample: ", sample)
        era = sampleCfg["era"]
        jettrigger = []
        muontrigger = []
        #print(t.HLT.PFHT1050)
        #print(t.HLT["PFHT1050"])
        if era == "2018":
            jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, t.HLT.AK8PFHT800_TrimMass50, t.HLT.AK8PFJet400_TrimMass30,t.HLT.AK8PFJet420_TrimMass30, t.HLT.AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02]            
            #jettrigger = [ t.HLT.PFHT1050, t.HLT.AK8PFJet400_TrimMass30, t.HLT.AK8PFHT800_TrimMass50, t.HLT.PFJet500, t.HLT.AK8PFJet500, ]#t.HLT.AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4]
            muontrigger= [ t.HLT.Mu50 ] # Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
            sr_pt_cut = 500.
        if era == "2017":
            sr_pt_cut = 525.
            if "Run2017B" in sample:
                jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, ] #t.HLT.AK8PFHT800_TrimMass50]
            if "Run2017C" in sample or "Run2017D" in sample or "Run2017E" in sample:
                jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, t.HLT.AK8PFHT800_TrimMass50, t.HLT.AK8PFJet400_TrimMass30,t.HLT.AK8PFJet420_TrimMass30]
            elif "Run2017F" in sample:
                jettrigger = [t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500, t.HLT.AK8PFHT800_TrimMass50, t.HLT.AK8PFJet400_TrimMass30,t.HLT.AK8PFJet420_TrimMass30, ]#t.HLT.AK8PFJet330_BTagCSV_p17]
            muontrigger = [t.HLT.Mu50,]# t.HLT.OldMu100, t.HLT.TkMu100]

        filters = [t.Flag.goodVertices,t.Flag.globalSuperTightHalo2016Filter,t.Flag.HBHENoiseFilter,t.Flag.HBHENoiseIsoFilter,t.Flag.EcalDeadCellTriggerPrimitiveFilter, t.Flag.BadPFMuonFilter, t.Flag.BadPFMuonDzFilter, t.Flag.eeBadScFilter, t.Flag.ecalBadCalibFilter]
        isoMuFilterMask = 0xA

        if self.isMC(sample):
            noSel = noSel.refine("mcWeight", weight=t.genWeight, autoSyst=True)
            jetSel = noSel.refine("passJetHLT", cut=1, autoSyst=True)
            filterJetSel = jetSel.refine("passFilterJet",cut=1)
            muSel = noSel.refine("passMuHLT", cut=1, autoSyst=True)
            filterMuSel = muSel.refine("passFilterMu",cut=1)
            #jetSel = noSel.refine("passJetHLT", cut="(1==1)")
            #muSel = noSel.refine("passMuHLT", cut="(1==1)")
        else:
            noSel = noSel.refine("None",)
             
            blindedSel = noSel.refine("blinded",cut=t.event%10==0) 
            jetSel = blindedSel.refine("passJetHLT", cut=op.OR(*(jettrigger))) 
            filterJetSel = jetSel.refine("passFilterJet",cut=op.AND(*(filters)))
            muSel  = noSel.refine("passMuHLT", cut=op.OR(*(muontrigger))) 
            filterMuSel = muSel.refine("passFilterMu",cut=op.AND(*(filters)))

        plots = []

        triggerObj = op.select(t.TrigObj, lambda trgObj: op.AND( trgObj.id == 13,
			(trgObj.filterBits & isoMuFilterMask) )) 
        veto_muons = op.sort(op.select(t.Muon, lambda mu : op.AND(
					mu.pt > 5.,
					)), lambda mu : -mu.pt)


        loose_muons = op.sort(op.select(t.Muon, lambda mu : op.AND(
					mu.pt > 10.,
					mu.looseId,
					op.abs(mu.eta) < 2.1,
					op.abs(mu.pfRelIso04_all) < 0.4,
					)), lambda mu : -mu.pt)


        candidatemuons = op.sort(op.select(t.Muon, lambda mu : op.AND(
					mu.pt > 55.,
					mu.looseId,
					op.abs(mu.eta) < 2.1,
					op.abs(mu.pfRelIso04_all) < 0.1,
					)), lambda mu : -mu.pt)

        electrons = op.sort(op.select(t.Electron, lambda el : op.AND(
					el.pt > 10.,
					#el.mvaFall17V2Iso_WPL,
					el.cutBased >= 1,
					op.abs(el.eta) < 2.5,
					)), lambda el : -el.pt)

        taus = op.sort(op.select(t.Tau, lambda tau : op.AND(
					tau.pt > 20.,
					tau.decayMode,
					tau.rawIso < 5,
					op.abs(tau.eta) < 2.3,
					tau.idDeepTau2017v2p1VSe >= 2,
					tau.idDeepTau2017v2p1VSjet >= 16,
					tau.idDeepTau2017v2p1VSmu >= 8,
					)), lambda tau : -tau.pt)

	#AK8 (highest pt or second highest pt, depending on number of fatjets left)
        if self.args.arbitration == "pt":
            fatjets = op.sort(op.select(t.FatJet, lambda fj : op.AND(
					fj.pt > 100.,
					op.abs(fj.eta) < 2.5,
                                        2*op.log(fj.msoftdrop/fj.pt)>-8, 
                                        2*op.log(fj.msoftdrop/fj.pt)<-1,
					)), lambda fj : -fj.pt)

        elif self.args.arbitration == "2prong":
            fatjets = op.sort(op.select(t.FatJet, lambda fj : op.AND(
					fj.pt > 100.,
					op.abs(fj.eta) < 2.5,
                                        2*op.log(fj.msoftdrop/fj.pt)>-8, 
                                        2*op.log(fj.msoftdrop/fj.pt)<-1,
					)), lambda fj : -(fj.particleNetMD_Xqq+fj.particleNetMD_Xcc+fj.particleNetMD_Xbb))


	#btagged AK4
        jets = op.sort(op.select(t.Jet, lambda j : op.AND(
					j.pt > 50.,
					op.abs(j.eta) < 2.5,
					#j.btagCSVV2 > 0.8838,
					)), lambda j : -j.pt)[:4]
        jets_away = op.select(jets, lambda j : op.deltaR(fatjets[0].p4, j.p4)  > 0.8) 


        if do_genmatch:
            genQuarks = op.select(t.GenPart, lambda q: op.AND(op.abs(q.pdgId) >= 1, op.abs(q.pdgId) <= 5))
            w_by_status = op.sort(op.select(t.GenPart, lambda p : op.AND(op.OR(op.abs(p.pdgId) == 23,op.abs(p.pdgId) == 24, op.abs(p.pdgId) == 55),p.statusFlags & 2**GEN_FLAGS["IsLastCopy"])),
                                  lambda p: -p.status)
            q_from_w = op.select(genQuarks, lambda q : q.parent.idx == w_by_status[0].idx)
            Vgen_matched = op.rng_count(q_from_w, lambda q: op.deltaR(q.p4, fatjets[0].p4) < 0.8) == 2
            dr_to_q1 = op.deltaR(q_from_w[0].p4, fatjets[0].p4)
            dr_to_q2 = op.deltaR(q_from_w[1].p4, fatjets[0].p4)

           


        from bamboo.scalefactors import get_scalefactor, get_correction
        ddtmap_file = f"/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/ddt_maps.json"
        jettriggerSF_file = f"/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/corrections/fatjet_triggerSF.json"
        pnmd2prong_ddtmap = get_correction(ddtmap_file,
            f"ddtmap_PNMD_pctl0.05_QCD_{era}" ,
            params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(fj.msoftdrop/fj.pt) },
            sel=noSel,
        )
        pnmd2prong_ddt = t._FatJet.orig[fatjets[0].idx].particleNetMD_Xbb + t._FatJet.orig[fatjets[0].idx].particleNetMD_Xcc + t._FatJet.orig[fatjets[0].idx].particleNetMD_Xqq - pnmd2prong_ddtmap(t._FatJet.orig[fatjets[0].idx])

        jettriggerSF = get_correction(jettriggerSF_file,
            f"fatjet_triggerSF{era}",
            params = {"pt": lambda fj : fj.p4.Pt(), "msd" : lambda fj : fj.msoftdrop, "systematic" : "nominal", },
            sel=noSel,
        )

        if self.isMC(sample):
            trigweightedJetSel = filterJetSel.refine("weightedJetSel",cut=1, weight=jettriggerSF(fatjets[0]))
        else:
            trigweightedJetSel = filterJetSel.refine("weightedJetSel",cut=1, weight=1)
        #pnmd2prong_ddt = (fatjets[0].particleNetMD_Xbb + fatjets[0].particleNetMD_Xcc + fatjets[0].particleNetMD_Xqq) - pnmd2prong_ddtmap(fatjets[0])
 
        '''
        #sf = get_correction("msdcorr.json","msdraw_onebin", )#)params={"pt": lambda obj : obj.pt,})
        n2b1ddt = get_correction("/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/runs/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt.json",
            "ddtmap_5pct_n2",
            params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(fj.msoftdrop/fj.pt) }, 
            sel=noSel 
        )
        n2b1ddt_smoothed = get_correction("/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/runs/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt_smoothed.json",
            "ddtmap_5pct_n2_smoothed",
            params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(fj.msoftdrop/fj.pt) }, 
            sel=noSel 
        )
        '''

        #n2b1ddt = get_scalefactor("jet","/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt_smoothed.json","ddtmap_5pct_n2_smoothed",params = {"pt": lambda fj : fj[0].p4.Pt(), "rho" : lambda fj : 2*:)
        hasTriggerObj  = noSel.refine("hasTriggerObj", cut=[op.rng_len(triggerObj) > 0] )
        hasBtaggedAK4 =  hasTriggerObj.refine("hasBtaggedAK4", cut=[op.rng_len(jets) > 0] )  
        ''' 
        noEle     = hasBtaggedAK4.refine("noEle", cut=[op.rng_len(electrons) == 0] )  
        noTau     = noEle.refine("noTau", cut=[op.rng_len(taus) == 0] )  

        oneMuon     = noTau.refine("oneMuon", cut=[op.rng_len(muons) == 1] )  

        deltaRfj = oneMuon.refine("deltaRfj",cut=op.deltaR(jets[0].p4,fatjets[0].p4) > 0.8)
        ptcut = deltaRfj.refine("ptcut",cut=fatjets[0].p4.Pt() > 300)
        CR_cut = ptcut.refine("CR_cut", cut=[op.deltaR(muons[0].p4,fatjets[0].p4) > 2*np.pi/3] )  
        '''


	########################
	###   SR selection   ###
	########################
        SR_pt_cut = trigweightedJetSel.refine("fj_pt_cut",cut=fatjets[0].p4.Pt() > sr_pt_cut)
        SR_eta_cut = SR_pt_cut.refine("fj_eta_cut",cut=op.abs(fatjets[0].p4.Eta()) < 2.5)
        SR_msd_cut = SR_eta_cut.refine("fj_msd_cut",cut=fatjets[0].msoftdrop>40)
        SR_rho_cut = SR_msd_cut.refine("fj_rho_cut",cut=op.AND(2*op.log(fatjets[0].msoftdrop/fatjets[0].pt) > -5.5,2*op.log(fatjets[0].msoftdrop/fatjets[0].pt) <-2.))
        SR_id_cut = SR_rho_cut.refine("id_cut",cut=[fatjets[0].jetId & (1 << 1) !=0])
        SR_electron_cut = SR_id_cut.refine("el_cut",cut=[op.rng_len(electrons) == 0])
        SR_muon_cut = SR_electron_cut.refine("mu_cut",cut=[op.rng_len(loose_muons) == 0])
        SR_tau_cut = SR_muon_cut.refine("tau_cut",cut=[op.rng_len(taus) == 0]) 
        if "ZJetsToQQ" in sample or "WJetsToQQ" in sample or "VectorZPrime" in sample:
            SR_Vmatched = SR_tau_cut.refine("fj_Vmatched",cut=Vgen_matched)
        else:
            SR_Vmatched = SR_tau_cut.refine("fj_Vmatched",cut=1) 
 
        SR_cut = SR_Vmatched

        #######################
        ###  CR1 selection  ###
        #######################
        CR1_jpt_cut = filterMuSel.refine("CR1_jpt_cut",cut=fatjets[0].p4.Pt() > 400)
        CR1_jeta_cut = CR1_jpt_cut.refine("CR1_jeta_cut",cut=op.abs(fatjets[0].p4.Eta())<2.5)
        CR1_jmsd_cut = CR1_jeta_cut.refine("CR1_jmsd_cut",cut=fatjets[0].msoftdrop > 40)
        CR1_jrho_cut = CR1_jmsd_cut.refine("CR1_jrho_cut",cut=-5.5 < 2*op.log(fatjets[0].msoftdrop/fatjets[0].pt) < -2.)
        CR1_mu_cut = CR1_jrho_cut.refine("CR1_mu_cut",cut=[op.rng_len(candidatemuons) == 1])
        CR1_muonDphiAK8 = CR1_mu_cut.refine("CR1_muonDphiAK8",cut=op.rng_any(t.Muon, lambda mu : op.abs(op.deltaPhi(mu.p4, fatjets[0].p4))>2*np.pi/3))
        CR1_ak4btagMedium08 = CR1_muonDphiAK8.refine("CR1_ak4btagMedium08",cut=op.rng_any(jets_away, lambda j : j.btagCSVV2>0.8838))
        CR1_electron_cut = CR1_ak4btagMedium08.refine("CR1_electron_cut",cut=[op.rng_len(electrons) == 0])
        CR1_mu_cutloose = CR1_electron_cut.refine("CR1_mu_cutloose",cut=[op.rng_len(loose_muons) == 1])
        CR1_tau_cut = CR1_mu_cutloose.refine("CR1_tau_cut",cut=[op.rng_len(taus) == 0]) 
        CR1_cut = CR1_tau_cut

        #######################
        ###  CR2 selection  ###
        #######################
        Wleptonic_candidate = op.sum(loose_muons[0].p4,t.MET.p4)
        CR2_jpt_cut = filterMuSel.refine("CR2_jpt_cut",cut=fatjets[0].p4.Pt() > 200)
        CR2_jeta_cut = CR2_jpt_cut.refine("CR2_jeta_cut",cut=op.abs(fatjets[0].p4.Eta())<2.5)
        CR2_jmsd_cut = CR2_jeta_cut.refine("CR2_jmsd_cut",cut=fatjets[0].msoftdrop > 40)
        CR2_mu_pt_cut = CR2_jmsd_cut.refine("CR2_mu_pt_cut",cut=loose_muons[0].pt > 53)
        CR2_mu_eta_cut = CR2_mu_pt_cut.refine("CR2_mu_eta_cut",cut=op.abs(loose_muons[0].p4.Eta()) < 2.1)
        CR2_mu_pfRelIso04_all_cut = CR2_mu_eta_cut.refine("CR2_mu_pfRelIso04_all_cut",cut=loose_muons[0].pfRelIso04_all<0.15)
        CR2_mu_tightId_cut = CR2_mu_pfRelIso04_all_cut.refine("CR2_mu_tightId_cut",cut=loose_muons[0].tightId)
        CR2_muonDphiAK8 = CR2_mu_tightId_cut.refine("CR2_muonDphiAK8",cut=op.abs(op.deltaPhi(loose_muons[0].p4, fatjets[0].p4))>2*np.pi/3)
        CR2_ak4btagMedium08 = CR2_muonDphiAK8.refine("CR2_ak4btagMedium08",cut=op.rng_any(jets_away, lambda j : j.btagCSVV2>0.8838))
        CR2_MET = CR2_ak4btagMedium08.refine("CR2_MET",t.MET.pt>40)
        CR2_electron_cut = CR2_MET.refine("CR2_electron_cut",cut=[op.rng_len(electrons) == 0])
        CR2_mu_cutloose = CR2_electron_cut.refine("CR2_mu_cutloose",cut=[op.rng_len(loose_muons) == 1])
        CR2_Wleptonic_cut = CR2_mu_cutloose.refine("CR2_Wleptonic_cut",cut=Wleptonic_candidate.Pt()>200)
        CR2_cut = CR2_Wleptonic_cut



        #this should be the last cut defined above
        # CR_cut = muonDphiAK8.refine("CR_cut", cut=[op.AND(hasTriggerObj,hasBtaggedAK4,noEle,oneMuon,muonDphiAK8)] ) 

        ### these commands create the .tex file with the efficiency table ###
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
        


        CR1_yields = CutFlowReport("CR1_yields", printInLog=True,)# recursive=True)
        plots.append(CR1_yields)
        CR1_yields.add(noSel, title ='TEST')
        CR1_yields.add(muSel, title= 'trigger')
        CR1_yields.add(filterMuSel, title= 'filters')
        CR1_yields.add(CR1_jpt_cut, title= 'CR1_jpt_cut')
        CR1_yields.add(CR1_jeta_cut, title= 'CR1_jeta_cut')
        CR1_yields.add(CR1_jmsd_cut, title= 'CR1_jmsd_cut')
        CR1_yields.add(CR1_jrho_cut, title= 'CR1_jrho_cut')
        CR1_yields.add(CR1_mu_cut, title= 'CR1_mu_cut')
        CR1_yields.add(CR1_muonDphiAK8, title= 'CR1_muonDphiAK8')
        CR1_yields.add(CR1_ak4btagMedium08, title= 'CR1_ak4btagMedium08')
        CR1_yields.add(CR1_electron_cut, title= 'CR1_electron_cut')
        CR1_yields.add(CR1_mu_cutloose, title= 'CR1_mu_cutloose')
        CR1_yields.add(CR1_tau_cut, title= 'CR1_tau_cut')

        SR_yields = CutFlowReport("SR_yields", printInLog=True,)# recursive=True)
        plots.append(SR_yields)
        SR_yields.add(noSel, title ='inclusive')
        SR_yields.add(filterJetSel, title= 'filters')
        SR_yields.add(trigweightedJetSel, title= 'trigweightedJetSel')
        SR_yields.add(jetSel, title ='trigger')
        SR_yields.add(SR_pt_cut, title= 'fj_pt')
        SR_yields.add(SR_eta_cut, title= 'fj_eta')
        SR_yields.add(SR_msd_cut, title= 'fj_msd')
        SR_yields.add(SR_rho_cut, title= 'fj_rho')
        SR_yields.add(SR_electron_cut, title= 'no_electron')
        SR_yields.add(SR_muon_cut, title= 'no_muon')
        SR_yields.add(SR_tau_cut, title= 'no_tau')
        SR_yields.add(SR_Vmatched, title= 'fj_vmatched')

         

        ### Here you can specify the variables that you want to save in the .parquet file, you need to add --mvaSkim to the command line ###
        if self.args.arbitration == "pt":
            loose_fatjets = op.sort(
                op.select(
                t.FatJet, lambda fj : op.AND(fj.msoftdrop > 10.,2*op.log(fj.msoftdrop/fj.pt)>-8, 2*op.log(fj.msoftdrop/fj.pt)<-1, fj.pt>170, op.abs(fj.eta)<2.5, fj.jetId & (1 << 1) !=0 ) 
                ), lambda fj : -fj.pt, 
            )
        elif self.args.arbitration == "2prong":
            loose_fatjets = op.sort(
                op.select(
                t.FatJet, lambda fj : op.AND(fj.msoftdrop > 10.,2*op.log(fj.msoftdrop/fj.pt)>-8, 2*op.log(fj.msoftdrop/fj.pt)<-1,fj.pt>170., op.abs(fj.eta)<2.5, fj.jetId & (1 << 1) !=0 ) 
                ), lambda fj : -(fj.particleNetMD_Xqq+fj.particleNetMD_Xcc+fj.particleNetMD_Xbb), 
            )
        ###new way: use fatjets, not loose_fatjets 
        mvaVariables = {
                "weight"            :   noSel.weight,
                "pt"                :   fatjets[0].p4.Pt(),
                "eta"               :   fatjets[0].p4.Eta(),
                "msd"               :   fatjets[0].msoftdrop,
                "jetId"             :   fatjets[0].jetId,
                "n2b1"              :   fatjets[0].n2b1,
                "pnmd2prong_ddt"    :   pnmd2prong_ddt,
                "particleNetMD_Xqq" :   fatjets[0].particleNetMD_Xqq,
                "particleNetMD_Xcc" :   fatjets[0].particleNetMD_Xcc,
                "particleNetMD_Xbb" :   fatjets[0].particleNetMD_Xbb,
                "particleNetMD_QCD" :   fatjets[0].particleNetMD_QCD,
        }
        if do_genmatch:
            mvaVariables["is_Vmatched"]    = Vgen_matched
            mvaVariables["q1_flavor"]      = q_from_w[0].pdgId
            mvaVariables["q2_flavor"]      = q_from_w[1].pdgId
        if self.isMC(sample):
            mvaVariables["msd_jesUp"]   =  t._FatJet["jesTotaldown"][fatjets[0].idx].msoftdrop
            mvaVariables["msd_jesDown"] =  t._FatJet["jesTotalup"]  [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jerUp"]   =  t._FatJet["jerup"]       [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jerDown"] =  t._FatJet["jerdown"]     [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmsUp"]   =  t._FatJet["jmsup"]       [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmsDown"] =  t._FatJet["jmsdown"]     [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmrUp"]   =  t._FatJet["jmrup"]       [fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmrDown"] =  t._FatJet["jmrdown"]     [fatjets[0].idx].msoftdrop
        '''
        ###old way: use loose_fatjets
        mvaVariables = {
                "weight"            :   noSel.weight,
                "pt"                :   loose_fatjets[0].p4.Pt(),
                "eta"               :   loose_fatjets[0].p4.Eta(),
                "msd"               :   loose_fatjets[0].msoftdrop,
                "jetId"             :   loose_fatjets[0].jetId,
                "n2b1"              :   loose_fatjets[0].n2b1,
                "pnmd2prong_ddt"    :   loose_fatjets_pnmd2prong_ddt,
                #"n2b1_ddt"          :   loose_fatjets[0].n2b1 - n2b1ddt(loose_fatjets[0]) ,
                #"n2b1_ddt_smoothed" :   loose_fatjets[0].n2b1 - n2b1ddt_smoothed(loose_fatjets[0]) ,
                #"rho"               :   2*op.log(loose_fatjets[0].msoftdrop/loose_fatjets[0].pt),
                "particleNetMD_Xqq" :   loose_fatjets[0].particleNetMD_Xqq,
                "particleNetMD_Xcc" :   loose_fatjets[0].particleNetMD_Xcc,
                "particleNetMD_Xbb" :   loose_fatjets[0].particleNetMD_Xbb,
                "particleNetMD_QCD" :   loose_fatjets[0].particleNetMD_QCD,
        }
        if do_genmatch:
            mvaVariables["is_Vmatched"]    = Vgen_matched
            mvaVariables["q1_flavor"]      = q_from_w[0].pdgId
            mvaVariables["q2_flavor"]      = q_from_w[1].pdgId
        if self.isMC(sample):
            mvaVariables["msd_jesUp"]   =  t._FatJet["jesTotaldown"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jesDown"] =  t._FatJet["jesTotalup"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jerUp"]   =  t._FatJet["jerup"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jerDown"] =  t._FatJet["jerdown"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmsUp"]   =  t._FatJet["jmsup"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmsDown"] =  t._FatJet["jmsdown"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmrUp"]   =  t._FatJet["jmrup"][loose_fatjets[0].idx].msoftdrop
            mvaVariables["msd_jmrDown"] =  t._FatJet["jmrdown"][loose_fatjets[0].idx].msoftdrop
        '''
        ### Save mvaVariables to be retrieved later in the postprocessor and saved in a parquet file ###
       
        if self.args.SR:
            selection = SR_cut
            prefix="SR_"
            if self.args.mvaSkim:
                from bamboo.plots import Skim
                parquet_cut = noSel.refine("parquet_cut", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt>500,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region1", mvaVariables, parquet_cut))
                parquet_cut2 = noSel.refine("parquet_cut2", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=500,fatjets[0].pt>400,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region2", mvaVariables, parquet_cut2))
                parquet_cut3 = noSel.refine("parquet_cut3", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=400,fatjets[0].pt>350,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region3", mvaVariables, parquet_cut3))
                parquet_cut4 = noSel.refine("parquet_cut4", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=350,fatjets[0].pt>300,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region4", mvaVariables, parquet_cut4))
                parquet_cut5 = noSel.refine("parquet_cut5", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=300,fatjets[0].pt>250,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region5", mvaVariables, parquet_cut5))
                parquet_cut6 = noSel.refine("parquet_cut6", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=250,fatjets[0].pt>210,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region6", mvaVariables, parquet_cut6))
                parquet_cut7 = noSel.refine("parquet_cut7", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,fatjets[0].pt<=210,fatjets[0].pt>180,fatjets[0].msoftdrop>10.,op.rng_len(fatjets)>0)])
                plots.append(Skim("signal_region7", mvaVariables, parquet_cut7))
        elif self.args.CR1:
            selection = CR1_cut
            prefix="CR1_" 
        elif self.args.CR2:
            selection = CR2_cut
            prefix="CR2_"
            if self.args.mvaSkim:
                from bamboo.plots import Skim
                plots.append(Skim("CR2", mvaVariables, selection))
 
        selection_pass = selection.refine("pass_pnmd2prong_1pctddt",cut = [pnmd2prong_ddt >= 0.])
        selection_fail = selection.refine("fail_pnmd2prong_1pctddt",cut = [pnmd2prong_ddt  < 0.])

        pnMD_2prong = fatjets[0].particleNetMD_Xqq + fatjets[0].particleNetMD_Xcc + fatjets[0].particleNetMD_Xbb
        #### ParticleNet-MD plots
        plots.append(Plot.make1D(prefix+"particlenet_2prong_MD", pnMD_2prong, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD ZPrime binary score", xTitle="ParticleNet-MD 2prong score"))
        plots.append(Plot.make1D(prefix+"particlenet_2prong_MD_ddt", pnmd2prong_ddt, selection, EquidistantBinning(25,-1.,0.1), title="ParticleNet-MD ZPrime binary score ddt", xTitle="ParticleNet-MD 2prong score (DDT)"))
        plots.append(Plot.make1D(prefix+"particlenet_bb_MD", fatjets[0].particleNetMD_Xbb, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD bb score", xTitle="ParticleNet-MD bb score"))
        plots.append(Plot.make1D(prefix+"particlenet_cc_MD", fatjets[0].particleNetMD_Xcc, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD cc score", xTitle="ParticleNet-MD cc score"))
        plots.append(Plot.make1D(prefix+"particlenet_qq_MD", fatjets[0].particleNetMD_Xqq, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD qq score", xTitle="ParticleNet-MD qq score"))
        plots.append(Plot.make1D(prefix+"particlenet_bb_MD_vs_QCD", fatjets[0].particleNetMD_Xbb/(fatjets[0].particleNetMD_Xbb+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD bb score", xTitle="ParticleNet-MD bb vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_cc_MD_vs_QCD", fatjets[0].particleNetMD_Xcc/(fatjets[0].particleNetMD_Xcc+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD cc score", xTitle="ParticleNet-MD cc vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_qq_MD_vs_QCD", fatjets[0].particleNetMD_Xqq/(fatjets[0].particleNetMD_Xqq+fatjets[0].particleNetMD_QCD), selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD qq score", xTitle="ParticleNet-MD qq vs QCD"))
        plots.append(Plot.make1D(prefix+"particlenet_QCD_MD", fatjets[0].particleNetMD_QCD, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD QCD score", xTitle="ParticleNet-MD QCD score"))
        #### Jet kinematics 
        plots.append(Plot.make1D(prefix+"FatjetMsd", fatjets[0].msoftdrop, selection, EquidistantBinning(25,40.,400.), title="FatJet pT", xTitle="FatJet m_{SD} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatjetMsd_pass", fatjets[0].msoftdrop, selection_pass, EquidistantBinning(25,40.,400.), title="FatJet msd", xTitle="FatJet m_{SD} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatjetMsd_fail", fatjets[0].msoftdrop, selection_fail, EquidistantBinning(25,40.,400.), title="FatJet msd", xTitle="FatJet m_{SD} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetPt", fatjets[0].p4.Pt(), selection, EquidistantBinning(25,200.,1400.) if self.args.CR2 else EquidistantBinning(25,400,1400.), title="FatJet pT", xTitle="FatJet p_{T} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetEta", fatjets[0].p4.Eta(), selection, EquidistantBinning(25,-2.5,2.5), title="FatJet #eta", xTitle="FatJet #eta"))
        plots.append(Plot.make1D(prefix+"FatJetRho", 2*op.log(fatjets[0].msoftdrop/fatjets[0].pt), selection, EquidistantBinning(25,-5.5,-2), title="FatJet #rho", xTitle="FatJet #rho"))
        plots.append(Plot.make1D(prefix+"FatJetN2",  fatjets[0].n2b1, selection, EquidistantBinning(25,0,0.5), title="FatJet N2", xTitle="FatJet N_{2}"))
 

        if self.args.CR1 or self.args.CR2: 
            #### Muon kinematics 
            plots.append(Plot.make1D(prefix+"nmuons",op.rng_len(loose_muons), selection, EquidistantBinning(5,0.,5.),title= "Number of Muons", xTitle="Number of muons" ))
            plots.append(Plot.make1D(prefix+"muonpt",loose_muons[0].pt, selection, EquidistantBinning(20,51.,300.),title= "Candidate muon pt", xTitle="Muon p_{T} (GeV)" ))
            plots.append(Plot.make1D(prefix+"muoneta",loose_muons[0].p4.Eta(), selection, EquidistantBinning(20,-2.1,2.1),title= "Candidate muon eta", xTitle="Muon #eta" ))
            plots.append(Plot.make1D(prefix+"pfRelIso04_all",loose_muons[0].pfRelIso04_all, selection, EquidistantBinning(20,0.,.4),title= "MuonpfRelIso04_all", xTitle="Muon relative isolation (0.4)" ))
        print("helloxx")
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
        ######## save a TTree with invariant mass of gg and bb for the different categories#####################
        ########################################################################################################
        import os.path
        from bamboo.plots import Skim
        skims = [ap for ap in self.plotList if isinstance(ap, Skim)]
        if self.args.mvaEval and skims:
            from bamboo.analysisutils import loadPlotIt
            p_config, samples, _, systematics, legend = loadPlotIt(config, [], eras=self.args.eras[1], workdir=workdir, resultsdir=resultsdir, readCounters=self.readCounters, vetoFileAttributes=self.__class__.CustomSampleAttributes)
            try:
                from bamboo.root import gbl
                import pandas as pd
                for skim in skims:
                    frames = []
                    for smp in samples:
                        for cb in (smp.files if hasattr(smp, "files") else [smp]):  # could be a helper in plotit
                            tree = cb.tFile.Get(skim.treeName)
                            if not tree:
                               print( f"KEY TTree {skim.treeName} does not exist, we are gonna skip this {smp}\n")
                            else:
                               cols = gbl.ROOT.RDataFrame(cb.tFile.Get(skim.treeName)).AsNumpy()
                               cols["weight"] *= cb.scale
                               cols["process"] = [smp.name]*len(cols["weight"])
                               frames.append(pd.DataFrame(cols))
                    df = pd.concat(frames)
                    print(df)
                    for col in df.columns:
                        if "process" in col: continue
                        df[col] = df[col].astype('float16')
                    df["process"] = pd.Categorical(df["process"], categories=pd.unique(df["process"]), ordered=False)
                    pqoutname = os.path.join(resultsdir, f"{skim.name}.parquet")
                    df.to_parquet(pqoutname)
                    del df
                    logger.info(f"Dataframe for skim {skim.name} saved to {pqoutname}")
            except ImportError as ex:
                logger.error("Could not import pandas, no dataframes will be saved")

        ########################################################################################################
        ######## save a parquet file with the variables needed for MVA training ################################
        ########################################################################################################
        #mvaSkim
        if self.args.mvaSkim and skims:
            from bamboo.analysisutils import loadPlotIt
            p_config, samples, _, systematics, legend = loadPlotIt(config, [], eras=self.args.eras[1], workdir=workdir, resultsdir=resultsdir, readCounters=self.readCounters, vetoFileAttributes=self.__class__.CustomSampleAttributes)
            try:
                from bamboo.root import gbl
                import pandas as pd
                for skim in skims:
                    frames = []
                    for smp in samples:
                        print("cacca")
                        #if not "QCD" in smp.name: continue
                        print(smp.name)
                        for cb in (smp.files if hasattr(smp, "files") else [smp]):  # could be a helper in plotit
                            tree = cb.tFile.Get(skim.treeName)
                            if not tree:
                               print( f"KEY TTree {skim.treeName} does not exist, we are gonna skip this {smp}\n")
                            else:
                               cols = gbl.ROOT.RDataFrame(cb.tFile.Get(skim.treeName)).AsNumpy()
                               #for key,val in cols.items():
                               #    if "process" in key: continue
                               #    cols[key] = val.astype(np.float16)
                               cols["weight"] *= cb.scale
                               cols["process"] = [smp.name]*len(cols["weight"])
                               frames.append(pd.DataFrame(cols))
                               #print("cols",type(cols["pt"]))
                               print("cols",type(cols["pt"][0]))
                            #break
                    df = pd.concat(frames)
                    #for col in df.columns:
                    #    if "process" in col: continue
                    #    df[col] = df[col].astype('float16')
                    #print(df)
                    #print(df["pt"].dtype)
                    df["process"] = pd.Categorical(df["process"], categories=pd.unique(df["process"]), ordered=False)
                    #csvoutname = os.path.join(resultsdir, f"{skim.name}.csv.gzip")
                    #df.to_csv(csvoutname,)
                    pqoutname = os.path.join(resultsdir, f"{skim.name}.parquet")
                    df.to_parquet(pqoutname,)
                    del df
                    logger.info(f"Dataframe for skim {skim.name} saved to {pqoutname}")
            except ImportError as ex:
                logger.error("Could not import pandas, no dataframes will be saved")

        
