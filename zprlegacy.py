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
    "TTbar" : 23
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
        #### Till now we don't need --mvaEval since we don't have a MVA model ####

    def __init__(self, args):
        super(zprlegacy, self).__init__(args)
        #if not (args.SR or args.CR1 or args.CR2):
        #    return RuntimeError("Need to run on SR/CR1/CR2")


    def prepareTree(self, tree, sample=None, sampleCfg=None, description=None, backend=None):
        ## initializes tree.Jet.calc so should be called first (better: use super() instead)
        # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC#Recommended_for_MC
#        tree,noSel,be,lumiArgs = NanoAODHistoModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg, description=NanoAODDescription.get('v7', year='2018', isMC=True), backend=backend)#, calcToAdd=["nMuon"])
        from bamboo.treedecorators import nanoRochesterCalc
        from bamboo.analysisutils import configureJets
        from bamboo.analysisutils import configureRochesterCorrection
        tree,noSel,be,lumiArgs = NanoAODHistoModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg,description=NanoAODDescription.get('v7', year='2018',isMC=True, systVariations=[nanoRochesterCalc]))
        #tree,noSel,be,lumiArgs = HistogramsModule.prepareTree(self, tree, sample=sample, sampleCfg=sampleCfg,)
        era = sampleCfg["era"]
        #if era == "2018":
        #    configureRochesterCorrection(tree._Muon, "/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/fromYihan/RoccoR2018UL.txt", isMC=self.isMC(sample), backend=be)
        #    return RuntimeError("Need to run on some region!")

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
        if era == "2018":
            jettrigger = [ t.HLT.PFHT1050, t.HLT.AK8PFJet400_TrimMass30, t.HLT.AK8PFHT800_TrimMass50, t.HLT.PFJet500, t.HLT.AK8PFJet500]
            muontrigger= [ t.HLT.Mu50 ] # Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
        if era == "2017":
            #jettrigger = [t.HLT.AK8PFJet330_PFAK8BTagCSV_p17, t.HLT.PFHT1050, t.HLT.AK8PFJet400_TrimMass30, t.HLT.AK8PFHT800_TrimMass50, t.HLT.PFJet500, t.HLT.AK8PFJet500]
            jettrigger = [ t.HLT.PFHT1050, t.HLT.PFJet500, t.HLT.AK8PFJet500]
            muontrigger = [t.HLT.Mu50,]# t.HLT.OldMu100, t.HLT.TkMu100]
        isoMuFilterMask = 0xA

        if self.isMC(sample):
            noSel = noSel.refine("mcWeight", weight=t.genWeight, autoSyst=True)
            jetSel = noSel.refine("passJetHLT", cut=1, autoSyst=True)
            muSel = noSel.refine("passMuHLT", cut=1, autoSyst=True)
            #jetSel = noSel.refine("passJetHLT", cut="(1==1)")
            #muSel = noSel.refine("passMuHLT", cut="(1==1)")
        else:
            noSel = noSel.refine("None",) 
            jetSel = noSel.refine("passJetHLT", cut=op.OR(*(jettrigger))) 
            muSel  = noSel.refine("passMuHLT", cut=op.OR(*(muontrigger))) 

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
        fatjets = op.sort(op.select(t.FatJet, lambda fj : op.AND(
					fj.pt > 100.,
					op.abs(fj.eta) < 2.5,
					)), lambda fj : -fj.pt)

	#fatjets = op.select(t.FatJet, lambda fj : op.AND(fj.pt > 500.,op.abs(fj.eta) < 2.5,))

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
        #sf = get_correction("msdcorr.json","msdraw_onebin", )#)params={"pt": lambda obj : obj.pt,})
        n2b1ddt = get_correction("/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt.json",
            "ddtmap_5pct_n2",
            params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(fj.msoftdrop/fj.pt) }, 
            sel=noSel 
        )
        n2b1ddt_smoothed = get_correction("/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt_smoothed.json",
            "ddtmap_5pct_n2_smoothed",
            params = {"pt": lambda fj : fj.p4.Pt(), "rho" : lambda fj : 2*op.log(fj.msoftdrop/fj.pt) }, 
            sel=noSel 
        )
        #n2b1ddt = get_scalefactor("jet","/afs/cern.ch/work/j/jekrupa/public/bamboodev/bamboo/examples/zprlegacy/24May23_v1/results_parquet_correctionlib/n2b1_ddtmap_rho_pt_smoothed.json","ddtmap_5pct_n2_smoothed",params = {"pt": lambda fj : fj[0].p4.Pt(), "rho" : lambda fj : 2*:)
        print(n2b1ddt)
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
        SR_pt_cut = jetSel.refine("fj_pt_cut",cut=fatjets[0].p4.Pt() > 500)
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
        CR1_jpt_cut = muSel.refine("CR1_jpt_cut",cut=fatjets[0].p4.Pt() > 400)
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
        CR2_jpt_cut = muSel.refine("CR2_jpt_cut",cut=fatjets[0].p4.Pt() > 200)
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

        loose_fatjets = op.sort(
          op.select(
             t.FatJet, lambda fj : op.AND(fj.pt>170, op.abs(fj.eta)<2.5, fj.jetId & (1 << 1) !=0 ) 
          ), lambda fj : -fj.pt, 
        ) 
        mvaVariables = {
                "weight"           :   noSel.weight,
                "pt"               :   loose_fatjets[0].p4.Pt(),
                "msd"              :   loose_fatjets[0].msoftdrop,
                "msd_corrected"    :   loose_fatjets[0].zpr_FatJet_corrected_mass,
                "n2b1"             :   loose_fatjets[0].n2b1,
                "n2b1_ddt"         :   loose_fatjets[0].n2b1 - n2b1ddt(loose_fatjets[0]) ,
                "n2b1_ddt_smoothed":   loose_fatjets[0].n2b1 - n2b1ddt_smoothed(loose_fatjets[0]) ,
                "rho"              :   2*op.log(loose_fatjets[0].msoftdrop/loose_fatjets[0].pt),
                #"nelectrons"       :   op.rng_len(electrons),
                #"nmuons"           :   op.rng_len(loose_muons),
                #"ntaus"            :   op.rng_len(taus),
                #"zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_bb" : loose_fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_bb,
                #"zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_cc" : loose_fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_cc,
                #"zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_qq" : loose_fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_qq,
                #"zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_QCD" : loose_fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_QCD,
                #"zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_bb" : loose_fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_bb,
                #"zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_cc" : loose_fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_cc,
                #"zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_qq" : loose_fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_qq,
                #"zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD" : loose_fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD,
                #"zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_bb" : loose_fatjets[0].zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_bb,
                #"zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_cc" : loose_fatjets[0].zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_cc,
                #"zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_qq" : loose_fatjets[0].zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_qq,
                #"zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_QCD" : loose_fatjets[0].zpr_TRANSFORMER_9APR23_V1_CATEGORICAL_QCD,
                #"particleNetMD_Xqq"                        : loose_fatjets[0].particleNetMD_Xqq,
                #"particleNetMD_Xcc"                        : loose_fatjets[0].particleNetMD_Xcc,
                #"particleNetMD_Xbb"                        : loose_fatjets[0].particleNetMD_Xbb,
                #"particleNetMD_QCD"                        : loose_fatjets[0].particleNetMD_QCD,
        }
        ''' 
                "zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime" : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime,
                "zpr_PN_PFSVE_DISCO200_FLAT_BINARY_QCD"    : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_BINARY_QCD,
                "zpr_PN_PFSVE_DISCO200_FLAT_3CAT_bbvQCD"   : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_bbvQCD, 
                "zpr_PN_PFSVE_DISCO200_FLAT_3CAT_ccvQCD"   : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_ccvQCD, 
                "zpr_PN_PFSVE_DISCO200_FLAT_3CAT_qqvQCD"   : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_qqvQCD, 
                "zpr_PN_PFSVE_DISCO200_FLAT_CAT_bb"        : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_bb,
                "zpr_PN_PFSVE_DISCO200_FLAT_CAT_cc"        : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_cc,
                "zpr_PN_PFSVE_DISCO200_FLAT_CAT_qq"        : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_qq,
                "zpr_PN_PFSVE_DISCO200_FLAT_CAT_QCD"       : loose_fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_QCD,
                "zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb"        : loose_fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb,
                "zpr_IN_PFSVE_DISCO200_FLAT_CAT_cc"        : loose_fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_cc,
                "zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq"        : loose_fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq,
                "zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD"       : loose_fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD,
                "zpr_INv1_PFSVE_DISCO200_FLAT_CAT_bb"      : loose_fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_bb,
                "zpr_INv1_PFSVE_DISCO200_FLAT_CAT_cc"      : loose_fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_cc,
                "zpr_INv1_PFSVE_DISCO200_FLAT_CAT_qq"      : loose_fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_qq,
                "zpr_INv1_PFSVE_DISCO200_FLAT_CAT_QCD"     : loose_fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_QCD,
                "zpr_PN_PFSVE_noDISCO_FLAT_CAT_bb"         : loose_fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_bb,
                "zpr_PN_PFSVE_noDISCO_FLAT_CAT_cc"         : loose_fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_cc,
                "zpr_PN_PFSVE_noDISCO_FLAT_CAT_qq"         : loose_fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_qq,
                "zpr_PN_PFSVE_noDISCO_FLAT_CAT_QCD"        : loose_fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_QCD,
        '''
        if do_genmatch:
            #mvaVariables["W_pdgId"]        = w_by_status[0].pdgId
            #mvaVariables["W_status"]       = w_by_status[0].status
            #mvaVariables["W_pt"]           = w_by_status[0].pt
            #mvaVariables["is_Vmatched"]    = Vgen_matched
            #mvaVariables["q1_dr_jet"]        = dr_to_q1 
            #mvaVariables["q2_dr_jet"]        = dr_to_q2 
            mvaVariables["q1_flavor"]      = q_from_w[0].pdgId
            mvaVariables["q2_flavor"]      = q_from_w[1].pdgId
            #mvaVariables["q1_status"]      = q_from_w[0].statusFlags
            #mvaVariables["q2_status"]      = q_from_w[1].statusFlags
        ### Save mvaVariables to be retrieved later in the postprocessor and saved in a parquet file ###
        if self.args.mvaSkim:
            from bamboo.plots import Skim
            #parquet_cut = noSel.refine("parquet_cut", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,loose_fatjets[0].pt>200, loose_fatjets[0].msoftdrop>10,op.rng_len(loose_fatjets)>0)])
            parquet_cut = noSel.refine("parquet_cut", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,loose_fatjets[0].pt>500,loose_fatjets[0].msoftdrop>20.,op.rng_len(loose_fatjets)>0)])
            plots.append(Skim("signal_region1", mvaVariables, parquet_cut))
            parquet_cut2 = noSel.refine("parquet_cut2", cut=[op.AND(op.rng_len(electrons) == 0,op.rng_len(loose_muons) == 0,op.rng_len(taus) == 0,loose_fatjets[0].pt<500,loose_fatjets[0].msoftdrop>20.,op.rng_len(loose_fatjets)>0)])
            plots.append(Skim("signal_region2", mvaVariables, parquet_cut2))
        pnMD_2prong = fatjets[0].particleNetMD_Xqq + fatjets[0].particleNetMD_Xcc + fatjets[0].particleNetMD_Xbb
       
        if self.args.SR:
            selection = SR_cut
            prefix="SR_"
        elif self.args.CR1:
            selection = CR1_cut
            prefix="CR1_" 
        elif self.args.CR2:
            selection = CR2_cut
            prefix="CR2_" 

        #### ParticleNet-MD plots
        plots.append(Plot.make1D(prefix+"particlenet_2prong_MD", pnMD_2prong, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD ZPrime binary score", xTitle="ParticleNet-MD 2prong score"))
        plots.append(Plot.make1D(prefix+"particlenet_bb_MD", fatjets[0].particleNetMD_Xbb, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD bb score", xTitle="ParticleNet-MD bb score"))
        plots.append(Plot.make1D(prefix+"particlenet_cc_MD", fatjets[0].particleNetMD_Xcc, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD cc score", xTitle="ParticleNet-MD cc score"))
        plots.append(Plot.make1D(prefix+"particlenet_qq_MD", fatjets[0].particleNetMD_Xqq, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD qq score", xTitle="ParticleNet-MD qq score"))
        plots.append(Plot.make1D(prefix+"particlenet_QCD_MD", fatjets[0].particleNetMD_QCD, selection, EquidistantBinning(25,0.,1.), title="ParticleNet-MD QCD score", xTitle="ParticleNet-MD QCD score"))
        #### TRANSFORMER+DISCO (CAT) plots
        '''

        plots.append(Plot.make1D(prefix+"TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD", fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD",xTitle="Transformer QCD score"))
        plots.append(Plot.make1D(prefix+"TRANSFORMER_25MAR23_V3_CATEGORICAL_bb", fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_bb, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_25MAR23_V3_CATEGORICAL_bb",xTitle="Transformer bb score"))
        plots.append(Plot.make1D(prefix+"TRANSFORMER_25MAR23_V3_CATEGORICAL_cc", fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_cc, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_25MAR23_V3_CATEGORICAL_cc",xTitle="Transformer cc score"))
        plots.append(Plot.make1D(prefix+"TRANSFORMER_25MAR23_V3_CATEGORICAL_qq", fatjets[0].zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_qq, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_25MAR23_V3_CATEGORICAL_qq",xTitle="Transformer qq score"))
        #plots.append(Plot.make1D(prefix+"TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_bb", fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_bb, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_bb",xTitle="Transformer+DISCO bb score"))
        #plots.append(Plot.make1D(prefix+"TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_cc", fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_cc, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_cc",xTitle="Transformer+DISCO cc score"))
        #plots.append(Plot.make1D(prefix+"TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_qq", fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_qq, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_qq",xTitle="Transformer+DISCO qq score"))
        #plots.append(Plot.make1D(prefix+"TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_QCD", fatjets[0].zpr_TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_QCD, selection, EquidistantBinning(25,0.,1.), title="TRANSFORMER_2MAR23_V3_DISCO500ALLSIGBKG_CATEGORICAL_QCD",xTitle="Transformer+DISCO QCD score"))
zpr_TRANSFORMER_25MAR23_V3_CATEGORICAL_QCD

        #### IN+DISCO (CAT) plots
        plots.append(Plot.make1D(prefix+"IN_PFSVE_DISCO200_FLAT_CAT_bb", fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_bb, selection, EquidistantBinning(25,0.,1.), title="IN_DISCO_bb", xTitle="InteractionNet+DISCO bb score"))
        plots.append(Plot.make1D(prefix+"IN_PFSVE_DISCO200_FLAT_CAT_cc", fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_cc, selection, EquidistantBinning(25,0.,1.), title="IN_DISCO_cc", xTitle="InteractionNet+DISCO cc score"))
        plots.append(Plot.make1D(prefix+"IN_PFSVE_DISCO200_FLAT_CAT_qq", fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_qq, selection, EquidistantBinning(25,0.,1.), title="IN_DISCO_qq", xTitle="InteractionNet+DISCO qq score"))
        plots.append(Plot.make1D(prefix+"IN_PFSVE_DISCO200_FLAT_CAT_QCD", fatjets[0].zpr_IN_PFSVE_DISCO200_FLAT_CAT_QCD, selection, EquidistantBinning(25,0.,1.), title="IN_DISCO_QCD", xTitle="InteractionNet+DISCO QCD score"))

        #### INv1+DISCO (CAT) plots
        plots.append(Plot.make1D(prefix+"INv1_PFSVE_DISCO200_FLAT_CAT_bb", fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_bb, selection, EquidistantBinning(25,0.,1.), title="INv1_DISCO_bb", xTitle="InteractionNet(v1)+DISCO bb score"))
        plots.append(Plot.make1D(prefix+"INv1_PFSVE_DISCO200_FLAT_CAT_cc", fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_cc, selection, EquidistantBinning(25,0.,1.), title="INv1_DISCO_cc", xTitle="InteractionNet(v1)+DISCO cc score"))
        plots.append(Plot.make1D(prefix+"INv1_PFSVE_DISCO200_FLAT_CAT_qq", fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_qq, selection, EquidistantBinning(25,0.,1.), title="INv1_DISCO_qq", xTitle="InteractionNet(v1)+DISCO qq score"))
        plots.append(Plot.make1D(prefix+"INv1_PFSVE_DISCO200_FLAT_CAT_QCD", fatjets[0].zpr_INv1_PFSVE_DISCO200_FLAT_CAT_QCD, selection, EquidistantBinning(25,0.,1.), title="INv1_DISCO_QCD", xTitle="InteractionNet(v1)+DISCO QCD score"))



        #### ParticleNet+DISCO (BINARY) plots
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_BINARY_zprime", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_BINARY_zprime, selection, EquidistantBinning(25,0.,1.), title="ParticleNetDISCO", xTitle="ParticleNet+DISCO 2prong score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_BINARY_QCD", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_BINARY_QCD, selection, EquidistantBinning(25,0.,1.), title="ParticleNetDISCO", xTitle="ParticleNet+DISCO QCD score"))
        #### ParticleNet+DISCO each vs QCD plots
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_3CAT_bbvQCD", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_bbvQCD, selection, EquidistantBinning(25,0.,1.), title="zpr_PN_PFSVE_DISCO200_FLAT_3CAT_bbvQCD", xTitle="ParticleNet+DISCO bb vs QCD score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_3CAT_ccvQCD", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_ccvQCD, selection, EquidistantBinning(25,0.,1.), title="zpr_PN_PFSVE_DISCO200_FLAT_3CAT_ccvQCD", xTitle="ParticleNet+DISCO cc vs QCD score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_3CAT_qqvQCD", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_3CAT_qqvQCD, selection, EquidistantBinning(25,0.,1.), title="zpr_PN_PFSVE_DISCO200_FLAT_3CAT_qqvQCD", xTitle="ParticleNet+DISCO qq vs QCD score"))
 
        #### ParticleNet+DISCO (CAT) plots
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_CAT_bb", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_bb, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_DISCO200_FLAT_CAT_bb", xTitle="ParticleNet+DISCO bb score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_CAT_cc", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_cc, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_DISCO200_FLAT_CAT_cc", xTitle="ParticleNet+DISCO cc score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_CAT_qq", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_qq, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_DISCO200_FLAT_CAT_qq", xTitle="ParticleNet+DISCO qq score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_DISCO200_FLAT_CAT_QCD", fatjets[0].zpr_PN_PFSVE_DISCO200_FLAT_CAT_QCD, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_DISCO200_FLAT_CAT_QCD", xTitle="ParticleNet+DISCO QCD score"))


        #### ParticleNet+NODISCO (CAT) plots
        plots.append(Plot.make1D(prefix+"PN_PFSVE_noDISCO_FLAT_CAT_bb", fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_bb, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_noDISCO_FLAT_CAT_bb", xTitle="ParticleNet (no Disco) bb score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_noDISCO_FLAT_CAT_cc", fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_cc, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_noDISCO_FLAT_CAT_cc", xTitle="ParticleNet (no Disco) cc score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_noDISCO_FLAT_CAT_qq", fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_qq, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_noDISCO_FLAT_CAT_qq", xTitle="ParticleNet (no Disco) qq score"))
        plots.append(Plot.make1D(prefix+"PN_PFSVE_noDISCO_FLAT_CAT_QCD", fatjets[0].zpr_PN_PFSVE_noDISCO_FLAT_CAT_QCD, selection, EquidistantBinning(25,0.,1.), title="PN_PFSVE_noDISCO_FLAT_CAT_QCD", xTitle="ParticleNet (no Disco) QCD score"))
        #### Jet kinematics 
        plots.append(Plot.make1D(prefix+"FatjetMsd", fatjets[0].msoftdrop, selection, EquidistantBinning(25,40.,400.), title="FatJet pT", xTitle="FatJet m_{SD} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetPt", fatjets[0].p4.Pt(), selection, EquidistantBinning(25,200.,1400.) if self.args.CR2 else EquidistantBinning(25,450.,1400.), title="FatJet pT", xTitle="FatJet p_{T} (GeV)"))
        plots.append(Plot.make1D(prefix+"FatJetEta", fatjets[0].p4.Eta(), selection, EquidistantBinning(25,-2.5,2.5), title="FatJet #eta", xTitle="FatJet #eta"))
        plots.append(Plot.make1D(prefix+"FatJetRho", 2*op.log(fatjets[0].msoftdrop/fatjets[0].pt), selection, EquidistantBinning(25,-5.5,-2), title="FatJet #rho", xTitle="FatJet #rho"))
        plots.append(Plot.make1D(prefix+"FatJetN2",  fatjets[0].n2b1, selection, EquidistantBinning(25,0,0.5), title="FatJet N2", xTitle="FatJet N_{2}"))
 

        '''
        
        #### Muon kinematics 
        plots.append(Plot.make1D(prefix+"nmuons",op.rng_len(loose_muons), selection, EquidistantBinning(5,0.,5.),title= "Number of Muons", xTitle="Number of muons" ))
        plots.append(Plot.make1D(prefix+"muonpt",loose_muons[0].pt, selection, EquidistantBinning(20,51.,300.),title= "Candidate muon pt", xTitle="Muon p_{T} (GeV)" ))
        plots.append(Plot.make1D(prefix+"muoneta",loose_muons[0].p4.Eta(), selection, EquidistantBinning(20,-2.1,2.1),title= "Candidate muon eta", xTitle="Muon #eta" ))
        plots.append(Plot.make1D(prefix+"pfRelIso04_all",loose_muons[0].pfRelIso04_all, selection, EquidistantBinning(20,0.,.4),title= "MuonpfRelIso04_all", xTitle="Muon relative isolation (0.4)" ))

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
                    pqoutname = os.path.join(resultsdir, f"{skim.name}.parquet.gzip")
                    df.to_parquet(pqoutname,compression="gzip")
                    del df
                    logger.info(f"Dataframe for skim {skim.name} saved to {pqoutname}")
            except ImportError as ex:
                logger.error("Could not import pandas, no dataframes will be saved")

        
