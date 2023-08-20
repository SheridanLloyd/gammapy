from gammapy.modeling.models import BHJetSpectralModel
import astropy.units.quantity

def run_6():


# PKS 2155-304 - my RUN 6
# BHJetparams={
#        "Mbh":1e9,
#        "theta":7.5,#2.5
#        "dist":459902,#459902
#        "redsh":0.1,#0.1
#        "jetrat":.05,#0.5 0.174 for opt x-ray
#        "r_0":10,#10
#        "z_diss":1000,#1000
#        "z_acc":1000,#1000
#        "z_max":1e8,#e8
#        "t_e": 800,#free 960
#        "f_nth":1e-4, # 1e-4 free big effect on norm
#        "f_pl": 0.01, #0.01
#        "pspec": 2, #3
#        "f_heat":20, #free 20 x t_e heating factor at z diss
#        "f_beta":10,#1
#        "f_sc":0.9,# free 5.e-6 #0.9
#        "p_beta":1e-2, #fixed in BHJet #0.01
#        "sig_acc":0.05, #0.1
#        "l_disk":0.01,#0.8
#        "r_in":1e4,# 1e4
#        "r_out":1e5,
#        "compar1":2,
#        "compar2":2,
#        "compar3":2,
#        "compsw":1,
#        "velsw":20,
#        "infosw":0
#        }
    return

def run_8():
    # PKS 2155-304 using canonical Blazar params as Table 4 BHJet paper 2108.12011
    BHJetparams = {
        "Mbh": 1e9,
        "theta": 4,  # 2.5
        "dist": 459902,  # 459902
        "redsh": 0.1,  # 0.1
        "jetrat": 0.4,  # 0.5 0.174 for opt x-ray, synonym Nj (L EDD)
        "r_0": 10,  # 10
        "z_diss": 1000,  # 1000
        "z_acc": 1000,  # 1000
        "z_max": 1e8,  # e8
        "t_e": 512,  # free 960
        "f_nth": 0.11,  # 1e-4 free big effect on norm
        "f_pl": 0,  # 0.01
        "pspec": 2,  # 3
        "f_heat": 2,  # free 20 x t_e heating factor at z diss
        "f_beta": 0.1,  # 1
        "f_sc": 2e-6,  # free 5.e-6 #0.9
        "p_beta": 1e-2,  # fixed in BHJet #0.01
        "sig_acc": 0.01,  # 0.1
        "l_disk": 0.8,  # 0.8
        "r_in": 1,  # 1e4
        "r_out": 1e5,
        "compar1": 0.05,
        "compar2": 0.5,
        "compar3": 2,
        "compsw": 2,
        "velsw": 20,  # synonym gamma acc
        "infosw": 0
    }

    # Mrk 501
    # BHJetparams={
    #        "Mbh":1e9,
    #        "theta":2.5,
    #        "dist":543e3,
    #        "redsh":0.1,
    #        "jetrat":.9e-2,
    #        "r_0":18,
    #        "z_diss":600,
    #        "z_acc":600,
    #        "z_max":6.6e5,
    #        "t_e": 1296.58173356, # free
    #        "f_nth":0.02168926, # free big effect on norm
    #        "f_pl":0,
    #        "pspec":1.95,
    #        "f_heat": 17.67855256, #free
    #        "f_beta":10.,
    #        "f_sc":6.00919172e-06,# free
    #        "p_beta":0.001, #fixed in BHJet
    #        "sig_acc":0.1,
    #        "l_disk":0.11,
    #        "r_in":1e4,
    #        "r_out":1e5,
    #        "compar1":2,
    #        "compar2":0,
    #        "compar3":100,
    #        "compsw":0,
    #        "velsw":59.99999318,
    #        "infosw":0
    #        }

    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.t_e.frozen = False
    modelBHJet.t_e.min = 511
    modelBHJet.t_e.max = 1500

    modelBHJet.f_nth.frozen = False
    modelBHJet.f_nth.min = 0.1  # v low compared to before
    modelBHJet.f_nth.max = 0.5

    modelBHJet.f_sc.frozen = False
    modelBHJet.f_sc.min = 1e-6  # v low compared to before
    modelBHJet.f_sc.max = 5e3

    modelBHJet.f_heat.frozen = False
    modelBHJet.f_heat.min = 1
    modelBHJet.f_heat.max = 20
    #
    modelBHJet.jetrat.frozen = False
    modelBHJet.jetrat.min = 0.01
    modelBHJet.jetrat.max = 0.999
    #
    # modelBHJet.pspec.frozen=False
    # modelBHJet.pspec.min=0.1
    # modelBHJet.pspec.max=3
    #
    modelBHJet.f_beta.frozen = False
    modelBHJet.f_beta.min = 0.1
    modelBHJet.f_beta.max = 20
    #
    # modelBHJet.Mbh.frozen=True
    # modelBHJet.Mbh.min=1e5
    # modelBHJet.Mbh.max=1e9
    #
    modelBHJet.theta.frozen = False
    modelBHJet.theta.min = 2.5
    modelBHJet.theta.max = 10
    #
    # modelBHJet.compar1.frozen=False
    # modelBHJet.compar1.min=1
    # modelBHJet.compar1.max=1e6
    #
    #
    # modelBHJet.compar2.frozen=False
    # modelBHJet.compar2.min=1
    # modelBHJet.compar2.max=1e6
    #
    #
    # modelBHJet.compar3.frozen=False
    # modelBHJet.compar3.min=1
    # modelBHJet.compar3.max=1e6
    #

    # rando
    # modelBHJet.velsw.value=62.55981319364623
    # modelBHJet.t_e.value=5261.51036583522
    # modelBHJet.f_nth.value=0.7825211555646718
    # modelBHJet.f_sc.value=0.00016955949833024969
    # modelBHJet.f_heat.value=30.116402138745123
    #

    # Values before NaN croak
    # modelBHJet.velsw.value=18.30352343
    # modelBHJet.t_e.value=92.87860123
    # modelBHJet.f_nth.value=2.05154563
    # modelBHJet.f_sc.value=0.00016956
    # modelBHJet.f_heat.value=30.11640214
    return modelBHJet

def fermi_neg_stat():
    BHJetparams ={'Mbh': 1.e+09, 'theta': 2.5, 'dist': 543400., 'redsh': 0.116, 'jetrat': 0.02879969, 'r_0': 21.68521006,
     'z_diss': 500.62874181, 'z_acc': 1000., 'z_max': 660000., 't_e': 1000., 'f_nth': 0.1, 'f_pl': 0., 'pspec': 1.7,
     'f_heat': 2.49999771, 'f_beta': 40., 'f_sc': 2.e-06, 'p_beta': 0., 'sig_acc': 0.03000317, 'l_disk': 0.8,
     'r_in': 40., 'r_out': 100000., 'compar1': 0.05, 'compar2': 0.5, 'compar3': 2., 'compsw': 2., 'velsw': 15.,
     'infosw': 0., 'dummy_norm': 1.}

    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.f_nth.frozen = True

    modelBHJet.jetrat.frozen = False
    modelBHJet.jetrat.min = 1e-4
    modelBHJet.jetrat.max = 1e-1

    modelBHJet.r_0.frozen = False
    modelBHJet.r_0.min = 1
    modelBHJet.r_0.max = 200

    modelBHJet.z_diss.frozen = False
    modelBHJet.z_diss.min = 500
    modelBHJet.z_diss.max = 2500

    modelBHJet.sig_acc.frozen = False
    modelBHJet.sig_acc.min = 1e-3
    #modelBHJet.sig_acc.max = 1e-1
    modelBHJet.sig_acc.max = 100

    modelBHJet.pspec.frozen = False
    modelBHJet.pspec.min = 1
    modelBHJet.pspec.max = 4

    modelBHJet.f_heat.frozen = False
    modelBHJet.f_heat.min = 0.1
    modelBHJet.f_heat.max = 2.5

    modelBHJet.f_beta.frozen = False
    modelBHJet.f_beta.min = 10
    modelBHJet.f_beta.max = 100

    modelBHJet.f_sc.frozen = False
    modelBHJet.f_sc.min = 1e-7
    #modelBHJet.f_sc.max = 1e-5 # very big ranges can be 5e3 in BHJet paper
    modelBHJet.f_sc.max = 100

    modelBHJet.r_in.frozen = False
    modelBHJet.r_in.min = 1
    modelBHJet.r_in.max = 200 # very big ranges can be 5e3 in BHJet paper

    modelBHJet.t_e.frozen = False
    modelBHJet.t_e.min = 511
    modelBHJet.t_e.max = 2000



    return modelBHJet

def joint_fit_model_norm_weight():
    # from RUN 17 - achieved stat of 177 (weighted so that HESS x10 favoured)
    BHJetparams = {'Mbh': 1.e+09, 'theta': 2.5, 'dist': 543400., 'redsh': 0.116, 'jetrat': 0.00449045, 'r_0': 11.26482938,
     'z_diss': 1878.4594537, 'z_acc': 1000., 'z_max': 660000., 't_e': 511.00532878, 'f_nth': 0.1, 'f_pl': 0.,
     'pspec': 2.20984303, 'f_heat': 0.57659825, 'f_beta': 87.75222989, 'f_sc': 0.12426452, 'p_beta': 0.,
     'sig_acc': 0.86340635, 'l_disk': 0.8, 'r_in': 38.01976458, 'r_out': 100000., 'compar1': 0.05, 'compar2': 0.5,
     'compar3': 2., 'compsw': 2., 'velsw': 15., 'infosw': 0., 'dummy_norm': 1.}
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    return modelBHJet
def Matteo_1810_11341():
    # PKS 2155-304 using canonical Blazar params as Table 4 BHJet paper 2108.12011
    # bit of a mash-up with 2108.12011 as some parameters have changed names/meaning
    BHJetparams = {
        "Mbh": 1e9,
        "theta": 2.5,  # 2.5
        "dist": 543400,  # 459902
        "redsh": 0.116,  # 0.1
        "jetrat": 1e-2,  # 0.5 0.174 for opt x-ray, synonym Nj (L EDD)
        "r_0": 20,  # 10 jet radius
        "z_diss": 1000,  # 1000
        "z_acc": 1000,  # 1000
        "z_max": 6.6e5,  # e8
        "t_e": 1000,  # free 960
        "f_nth": 0.1,  # epsilon_pl in 1810.11341
        "f_pl": 0,  # 0.01
        "pspec": 1.7,  # 3 aka p parameter - slope of non thermal distribution
        "f_heat": 10,  # free 20 x t_e heating factor at z diss
        "f_beta": 40,  # 22-86 prev known as f_b free param of dynamical timescale fixes cooling break energy
        "f_sc": 2e-6,  # free 5.e-6 #0.9
        "p_beta": 0,  # fixed in BHJet #0.01
        "sig_acc": 0.03,  # confusingly prev called sigma diss
        "l_disk": 0.8,  # 0.8
        "r_in": 40,  # 1e4
        "r_out": 1e5,
        "compar1": 0.05,
        "compar2": 0.5,
        "compar3": 2,
        "compsw": 2,
        "velsw": 15,  # synonym gamma acc , invokes BLJET flavor of model
        "infosw": 0
    }
    # Change mins as a result of good fit hitting the minimums
    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.f_nth.frozen = True

    modelBHJet.jetrat.frozen = False
    modelBHJet.jetrat.min = 1e-4
    modelBHJet.jetrat.max = 1e-1

    modelBHJet.r_0.frozen = False
    modelBHJet.r_0.min = 1
    modelBHJet.r_0.max = 200

    modelBHJet.z_diss.frozen = False
    #modelBHJet.z_diss.min = 500
    modelBHJet.z_diss.min = 10
    modelBHJet.z_diss.max = 2500

    modelBHJet.sig_acc.frozen = False
    modelBHJet.sig_acc.min = 1e-3
    #modelBHJet.sig_acc.max = 1e-1
    #modelBHJet.sig_acc.max = 100 # run 27
    modelBHJet.sig_acc.max = 500

    modelBHJet.pspec.frozen = False
    modelBHJet.pspec.min = 1
    modelBHJet.pspec.max = 4

    modelBHJet.f_heat.frozen = False
    #modelBHJet.f_heat.min = 0.1
    modelBHJet.f_heat.min = 0.01 # reduce min
    modelBHJet.f_heat.max = 2.5

    modelBHJet.f_beta.frozen = False
    modelBHJet.f_beta.min = 10
    #modelBHJet.f_beta.max = 100
    modelBHJet.f_beta.max = 1000 # increase as was hitting max

    modelBHJet.f_sc.frozen = False
    modelBHJet.f_sc.min = 1e-7
    #modelBHJet.f_sc.max = 1e-5 # very big ranges can be 5e3 in BHJet paper
    modelBHJet.f_sc.max = 100

    modelBHJet.r_in.frozen = False
    modelBHJet.r_in.min = 1
    modelBHJet.r_in.max = 200 # very big ranges can be 5e3 in BHJet paper

    modelBHJet.t_e.frozen = False
    #modelBHJet.t_e.min = 511
    modelBHJet.t_e.min = 10 # was hitting limit so change
    #modelBHJet.t_e.max = 2000 #run 27
    modelBHJet.t_e.max = 4000

    return modelBHJet


def run25_final():
    BHJetparams={'Mbh': 1.e+09, 'theta': 2.5, 'dist': 543400., 'redsh': 0.116, 'jetrat': 0.0030991, 'r_0': 10.57909962,
     'z_diss': 506.37911635, 'z_acc': 1000., 'z_max': 660000., 't_e': 511.00003397, 'f_nth': 0.1, 'f_pl': 0.,
     'pspec': 2.09074565, 'f_heat': 0.10000027, 'f_beta': 99.99999795, 'f_sc': 0.00782045, 'p_beta': 0.,
     'sig_acc': 0.11494939, 'l_disk': 0.8, 'r_in': 33.87755264, 'r_out': 100000., 'compar1': 0.05, 'compar2': 0.5,
     'compar3': 2., 'compsw': 2., 'velsw': 15., 'infosw': 0., 'dummy_norm': 1.}

    # Change mins as a result of good fit hitting the minimums
    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.f_nth.frozen = True

    modelBHJet.jetrat.frozen = False
    modelBHJet.jetrat.min = 1e-4
    modelBHJet.jetrat.max = 1e-1

    modelBHJet.r_0.frozen = False
    modelBHJet.r_0.min = 1
    modelBHJet.r_0.max = 200

    modelBHJet.z_diss.frozen = False
    #modelBHJet.z_diss.min = 500
    modelBHJet.z_diss.min = 10
    modelBHJet.z_diss.max = 2500

    modelBHJet.sig_acc.frozen = False
    modelBHJet.sig_acc.min = 1e-3
    #modelBHJet.sig_acc.max = 1e-1
    #modelBHJet.sig_acc.max = 100 # run 27
    modelBHJet.sig_acc.max = 500

    modelBHJet.pspec.frozen = False
    modelBHJet.pspec.min = 1
    modelBHJet.pspec.max = 4

    modelBHJet.f_heat.frozen = False
    #modelBHJet.f_heat.min = 0.1
    modelBHJet.f_heat.min = 0.01 # reduce min
    modelBHJet.f_heat.max = 2.5

    modelBHJet.f_beta.frozen = False
    modelBHJet.f_beta.min = 10
    #modelBHJet.f_beta.max = 100
    modelBHJet.f_beta.max = 1000 # increase as was hitting max

    modelBHJet.f_sc.frozen = False
    modelBHJet.f_sc.min = 1e-7
    #modelBHJet.f_sc.max = 1e-5 # very big ranges can be 5e3 in BHJet paper
    modelBHJet.f_sc.max = 100

    modelBHJet.r_in.frozen = False
    modelBHJet.r_in.min = 1
    modelBHJet.r_in.max = 200 # very big ranges can be 5e3 in BHJet paper

    modelBHJet.t_e.frozen = False
    #modelBHJet.t_e.min = 511
    modelBHJet.t_e.min = 10 # was hitting limit so change
    #modelBHJet.t_e.max = 2000 #run 27
    modelBHJet.t_e.max = 4000
    return modelBHJet

def run_30_final():
    print("Use run_30_final)")
    BHJetparams={'Mbh':  1.e+09, 'theta':  2.5, 'dist':  543400., 'redsh':  0.116, 'jetrat':  0.00612095, 'r_0':  10.5869359, 'z_diss':  287.21589913, 'z_acc':  1000., 'z_max':  660000., 't_e':  322.57727993, 'f_nth':  0.1, 'f_pl':  0., 'pspec':  2.21242564, 'f_heat':  0.01000024, 'f_beta':  999.99982078, 'f_sc':  0.00172837, 'p_beta':  0., 'sig_acc':  0.23902557, 'l_disk':  0.8, 'r_in':  33.91879099, 'r_out':  100000., 'compar1':  0.05, 'compar2':  0.5, 'compar3':  2., 'compsw':  2., 'velsw':  15., 'infosw':  0., 'dummy_norm':  1.}

    # Change mins as a result of good fit hitting the minimums
    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.f_nth.frozen = True

    modelBHJet.jetrat.frozen = False
    modelBHJet.jetrat.min = 1e-4
    modelBHJet.jetrat.max = 1e-1

    modelBHJet.r_0.frozen = False
    modelBHJet.r_0.min = 1
    modelBHJet.r_0.max = 200

    modelBHJet.z_diss.frozen = False
    #modelBHJet.z_diss.min = 500
    modelBHJet.z_diss.min = 10
    modelBHJet.z_diss.max = 2500

    modelBHJet.sig_acc.frozen = False
    modelBHJet.sig_acc.min = 1e-3
    #modelBHJet.sig_acc.max = 1e-1
    #modelBHJet.sig_acc.max = 100 # run 27
    modelBHJet.sig_acc.max = 500

    modelBHJet.pspec.frozen = False
    modelBHJet.pspec.min = 1
    modelBHJet.pspec.max = 4

    modelBHJet.f_heat.frozen = False
    #modelBHJet.f_heat.min = 0.1
    modelBHJet.f_heat.min = 0.001 # reduce min
    modelBHJet.f_heat.max = 2.5

    modelBHJet.f_beta.frozen = False
    modelBHJet.f_beta.min = 10
    #modelBHJet.f_beta.max = 100
    modelBHJet.f_beta.max = 10000 # increase as was hitting max

    modelBHJet.f_sc.frozen = False
    modelBHJet.f_sc.min = 1e-7
    #modelBHJet.f_sc.max = 1e-5 # very big ranges can be 5e3 in BHJet paper
    modelBHJet.f_sc.max = 100

    modelBHJet.r_in.frozen = False
    modelBHJet.r_in.min = 1
    modelBHJet.r_in.max = 200 # very big ranges can be 5e3 in BHJet paper

    modelBHJet.t_e.frozen = False
    #modelBHJet.t_e.min = 511
    modelBHJet.t_e.min = 10 # was hitting limit so change
    #modelBHJet.t_e.max = 2000 #run 27
    modelBHJet.t_e.max = 4000
    return modelBHJet
def run_9_low_hess_stats():


    BHJetparams={'Mbh':  1.e+09, 'theta':  10., 'dist':  459902., 'redsh':  0.1, 'jetrat':  0.01, 'r_0':  10., 'z_diss':  1000., 'z_acc':  1000., 'z_max':  1.e+08, 't_e':  511.00004138, 'f_nth':  0.10105465, 'f_pl':  0., 'pspec':  2., 'f_heat':  1.00000103, 'f_beta':  0.10000015, 'f_sc':  4999.99983226, 'p_beta':  0.01, 'sig_acc':  0.01, 'l_disk':  0.8, 'r_in':  1., 'r_out':  100000., 'compar1':  0.05, 'compar2':  0.5, 'compar3':  2., 'compsw':  2., 'velsw':  20., 'infosw':  0., 'dummy_norm':  1.}
    # Plot the separate contributions from each seed photon field
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    modelBHJet.t_e.frozen=False
    modelBHJet.t_e.min=511
    modelBHJet.t_e.max=1500

    modelBHJet.f_nth.frozen=False
    modelBHJet.f_nth.min=0.1 #v low compared to before
    modelBHJet.f_nth.max=0.5

    modelBHJet.f_sc.frozen=False
    modelBHJet.f_sc.min=1e-6 #v low compared to before
    modelBHJet.f_sc.max=5e3

    modelBHJet.f_heat.frozen=False
    modelBHJet.f_heat.min=1
    modelBHJet.f_heat.max=20
    #
    modelBHJet.jetrat.frozen=False
    modelBHJet.jetrat.min=0.01
    modelBHJet.jetrat.max=0.999
    #
    #modelBHJet.pspec.frozen=False
    #modelBHJet.pspec.min=0.1
    #modelBHJet.pspec.max=3
    #
    modelBHJet.f_beta.frozen=False
    modelBHJet.f_beta.min=0.1
    modelBHJet.f_beta.max=20
    #
    # modelBHJet.Mbh.frozen=True
    # modelBHJet.Mbh.min=1e5
    # modelBHJet.Mbh.max=1e9
    #
    modelBHJet.theta.frozen=False
    modelBHJet.theta.min=2.5
    modelBHJet.theta.max=10
    #
    # modelBHJet.compar1.frozen=False
    # modelBHJet.compar1.min=1
    # modelBHJet.compar1.max=1e6
    #
    #
    # modelBHJet.compar2.frozen=False
    # modelBHJet.compar2.min=1
    # modelBHJet.compar2.max=1e6
    #
    #
    # modelBHJet.compar3.frozen=False
    # modelBHJet.compar3.min=1
    # modelBHJet.compar3.max=1e6
    #

    #rando
    # modelBHJet.velsw.value=62.55981319364623
    # modelBHJet.t_e.value=5261.51036583522
    # modelBHJet.f_nth.value=0.7825211555646718
    # modelBHJet.f_sc.value=0.00016955949833024969
    # modelBHJet.f_heat.value=30.116402138745123
    #

    # Values before NaN croak
    #modelBHJet.velsw.value=18.30352343
    #modelBHJet.t_e.value=92.87860123
    #modelBHJet.f_nth.value=2.05154563
    #modelBHJet.f_sc.value=0.00016956
    #modelBHJet.f_heat.value=30.11640214
    return modelBHJet

def good_fermi():
    BHJetparams = {'Mbh':  1.e+09, 'theta':  2.5, 'dist':  543400., 'redsh':  0.116, 'jetrat':  0.0001, 'r_0':  9.52387442, 'z_diss':  859.22839479, 'z_acc':  1000., 'z_max':  660000., 't_e':  519.60478658, 'f_nth':  0.1, 'f_pl':  0., 'pspec':  1.32196564, 'f_heat':  0.41270582, 'f_beta':  99.99883881, 'f_sc':  0.00295681, 'p_beta':  0., 'sig_acc':  99.99983251, 'l_disk':  0.8, 'r_in':  52.1173464, 'r_out':  100000., 'compar1':  0.05, 'compar2':  0.5, 'compar3':  2., 'compsw':  2., 'velsw':  15., 'infosw':  0., 'dummy_norm':  1.}
    # 69.10141828085294 Fermi cash SpectrumDataset
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    return modelBHJet

def good_hess():
    BHJetparams = {'Mbh': 1.e+09, 'theta': 2.5, 'dist': 543400., 'redsh': 0.116, 'jetrat': 0.09997112, 'r_0': 29.14614532, 'z_diss': 501.16633994, 'z_acc': 1000., 'z_max': 660000., 't_e': 1784.74598726, 'f_nth': 0.1, 'f_pl': 0., 'pspec': 1.87981654, 'f_heat': 2.49997355, 'f_beta': 13.02990564, 'f_sc': 2.13163341e-06, 'p_beta': 0., 'sig_acc': 0.03989187, 'l_disk': 0.8, 'r_in': 13.77120156, 'r_out': 100000., 'compar1': 0.05, 'compar2': 0.5, 'compar3': 2., 'compsw': 2., 'velsw': 15., 'infosw': 0., 'dummy_norm': 1.}
    # stat was 11.79757607110937 hess_reduced wstat SpectrumDatasetOnOff

    modelBHJet = BHJetSpectralModel(**BHJetparams)

    return modelBHJet

def good_xmm():
    BHJetparams = {'Mbh': 1.e+09, 'theta': 2.5, 'dist': 543400., 'redsh': 0.116, 'jetrat': 0.0999852, 'r_0': 64.04195713, 'z_diss': 515.0753283, 'z_acc': 1000., 'z_max': 660000., 't_e': 511.12289802, 'f_nth': 0.1, 'f_pl': 0., 'pspec': 2.36836652, 'f_heat': 0.49420928, 'f_beta': 98.52350802, 'f_sc': 0.00053134, 'p_beta': 0., 'sig_acc': 0.40609144, 'l_disk': 0.8, 'r_in': 39.9235586, 'r_out': 100000., 'compar1': 0.05, 'compar2': 0.5, 'compar3': 2., 'compsw': 2., 'velsw': 15., 'infosw': 0., 'dummy_norm': 1.}
    # 272200.64755578 yRMZjmIP wstat StandardOGIPDataset
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    return modelBHJet

def good_joint_fit():
    BHJetparams = {'Mbh':  1.e+09, 'theta':  2.5, 'dist':  543400., 'redsh':  0.116, 'jetrat':  0.0030991, 'r_0':  10.57909962, 'z_diss':  506.37911635, 'z_acc':  1000., 'z_max':  660000., 't_e':  511.00003397, 'f_nth':  0.1, 'f_pl':  0., 'pspec':  2.09074565, 'f_heat':  0.10000027, 'f_beta':  99.99999795, 'f_sc':  0.00782045, 'p_beta':  0., 'sig_acc':  0.11494939, 'l_disk':  0.8, 'r_in':  33.87755264, 'r_out':  100000., 'compar1':  0.05, 'compar2':  0.5, 'compar3':  2., 'compsw':  2., 'velsw':  15., 'infosw':  0., 'dummy_norm':  1.}
    # Total Stat 101.93753929945794
    modelBHJet = BHJetSpectralModel(**BHJetparams)

    return modelBHJet
