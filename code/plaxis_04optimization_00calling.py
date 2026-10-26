# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:20:29 2026

@author: annaleonie

"""

# for ISRES change in
# C:\\ProgramData\\Seequent\\PLAXIS Python Distribution V3\\python\\\Lib\\site-packages\\pymoo\algorithms\\soo\nonconvex\\sres.py
# necessary is: J = np.arange(len(phi)) -> J = np.arange(len(phi), dtype=np.int64) (open as admin is often necessary depending on the computer settings)



def fct_opt(study           = 0, # 0 = no output
            seed            = 1,
            ALGORITHMs      = (
                              'GA',
                              'DE',
                              # 'ISRES',
                              # 'MixedVariableGA',
                              'PSO',
                              # 'PatternSearch',
                              # 'grid',
                              ),
            RUN_DEVICE      = 0,
            dimension       = 'unitcell2',
            RUN_COMPalgo    = False,
            RUN_COMPplot    = False,
            RUN_GRID        = False,
            RUN_GRIDplot    = False,
            RUN_PLOTcost    = False,
            RUN_PLOTcons    = False,
            RUN_PLOTboth    = False,
            RUN_PLOTpara    = False,
            RUN_PLOTgrid    = False,
            RUN_PLOTstat    = False,
            ):

    import numpy as np
    from plaxis_04optimization_01main import opt_algo
    from plaxis_04optimization_03plots import (plot_convergence,
                                               plot_evolution_parameter,
                                               plot_evolution_cost_settlement,
                                               plot_best_feasible,
                                               plot_best_feasible_points,
                                               plot_optimization_statistics,
                                               )
    from plaxis_00calling_02helpers import open_plaxisData
    # print(f'opti {seed} and {study}')

    if study == 0: return

    elif study in [1,2,3,5]:
        loadDEFAULT       = 100
        slabDEFAULT       = 24
        soilDEFAULT       = 5
        settlDEFAULT      = 0.04
        # testing with Radius und DISTANCE -> COMMON RATIO
        DESIGN1 = dict(RUN_PATH_MAIN= rf'Opt_2026-09-28-v1-DR-seed{seed}',
                       stop_window  = 20, # at least 2 to 4 times pop_size
                       pop_size     = 10,
                       calcSize     = 80, # popsize+2*stopwin
                       seed         = seed,
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_num",),
                       bounds       = dict(si_R      = (0.30, 0.45),
                                           si_num    = (5,  30),
                                           ),
                       scales       = dict(si_R      = 100,
                                           ),
                       fixed        = dict(si_method = 'VR',
                                           si_L      = 9,
                                           slab      = 1.0,
                                           ltp       = 0.5,
                                           si_E      = 100e3,
                                           ),
                       )
        # testing with Radius und Length
        DESIGN2 = dict(RUN_PATH_MAIN= rf'Opt_2026-09-28-v2-RL-seed{seed}',
                       stop_window  = 20, # at least 2 to 4 times pop_size
                       pop_size     = 10,
                       seed         = seed,
                       calcSize     = 80, # popsize+2*stopwin
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_L",),
                       bounds       = dict(si_R      = (0.30, 0.45),
                                           si_L      = (2.0, 12.5),
                                           ),
                       scales       = dict(si_R      = 100,
                                           si_L      = 4,
                                           ),
                       fixed        = dict(si_method = 'VR',
                                           si_num    = 15,
                                           slab      = 1.0,
                                           ltp       = 0.5,
                                           si_E      = 100e3,
                                           ),
                       )
        # testing with Distance und Length
        DESIGN3 = dict(RUN_PATH_MAIN= rf'Opt_2026-09-28-v3-DL-seed{seed}',
                       stop_window  = 20,
                       pop_size     = 10,
                       seed         = seed,
                       calcSize     = 80, # popsize+2*stopwin
                       user_methods = ("VR",),
                       vary         = ("si_L", "si_num"),
                       bounds       = dict(
                                           si_L      = (2.0, 12.5),
                                           si_num    = (10,  20),
                                           ),
                       scales       = dict(
                                           si_L      = 10,      #every 0.25m
                                           ),
                       fixed        = dict(
                                           si_method = "VR",
                                           slab      = 1,
                                           ltp       = 0.5,
                                           si_E      = 100e3,
                                           si_R      = 0.4, # from DESIGN1
                                           ),
                       )
        DESIGN5 = dict(RUN_PATH_MAIN= rf'Opt_2026-09-29-v5-RLNPS-seed{seed}',
                       stop_window  = 80,
                       pop_size     = 30,
                       calcSize     = 240,
                       seed         = 3,
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_L", "si_num", "ltp", "slab"),
                       bounds       = dict(
                                           ltp       = ( 0.00, 1.50),#16
                                           slab      = ( 0.30, 2.50),#23
                                           si_R      = ( 0.30, 0.45),#16                                     
                                           si_L      = ( 2.0, 12.50),#53
                                           si_num    = (10.0, 20.00),#11 = 3.4Mio
                                           ),
                       scales       = dict(
                                           ltp       = 10,      #every 0.10m
                                           slab      = 10,      #every 0.10m
                                           si_R      = 100,     #every 0.01m
                                           si_L      = 5,       #every 0.20m
                                           ),
                       fixed        = dict(
                                           si_method = "VR",
                                           si_E      = 100e3,
                                           ),
                       )
        if   study == 1:        DESIGN = DESIGN1
        elif study == 2:        DESIGN = DESIGN2
        elif study == 3:        DESIGN = DESIGN3
        elif study == 5:        DESIGN = DESIGN5
    elif study in [100,101,105]:
        loadDEFAULT       = 100
        slabDEFAULT       = 24
        soilDEFAULT = 'Refmodel'
        settlDEFAULT = 0.1
        if seed == 101:
            seed = 1
            study = 101
        # testing with Radius und Length and NUM
        DESIGN100 = dict(RUN_PATH_MAIN= rf'Opt_2026-10-02-v100-RL-seed{seed}',
                       stop_window  = 20, # at least 2 to 4 times pop_size
                       pop_size     = 10,
                       calcSize     = 80, # popsize+2*stopwin
                       seed         = seed,
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_L",),
                       bounds       = dict(si_R      = (0.30, 0.45),
                                           si_L      = (2.0, 12.5),
                                           si_num    = (10, 32),
                                           ),
                       scales       = dict(si_R      = 100,
                                           si_L      = 4,
                                           ),
                       fixed        = dict(si_method = 'VR',
                                           slab      = 2.5,
                                           ltp       = 0.0,
                                           si_E      = 100e3,
                                           ),
                       )
        # testing with Radius und Length and NUM - see if same seed = same result
        DESIGN101 = dict(RUN_PATH_MAIN= rf'Opt_2026-10-02-v100-RL-seed10{seed}',
                       stop_window  = 20, # at least 2 to 4 times pop_size
                       pop_size     = 10,
                       calcSize     = 80, # popsize+2*stopwin
                       seed         = seed,
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_L",),
                       bounds       = dict(si_R      = (0.30, 0.45),
                                           si_L      = (2.0, 12.5),
                                           si_num    = (10, 32),
                                           ),
                       scales       = dict(si_R      = 100,
                                           si_L      = 4,
                                           ),
                       fixed        = dict(si_method = 'VR',
                                           slab      = 2.5,
                                           ltp       = 0.0,
                                           si_E      = 100e3,
                                           ),
                       )
        DESIGN105 = dict(RUN_PATH_MAIN= rf'Opt_2026-10-02-v105-RL-seed{seed}',
                       stop_window  = 80, # at least 2 to 4 times pop_size
                       pop_size     = 30,
                       calcSize     = 240, # popsize+2*stopwin
                       seed         = seed,
                       user_methods = ("VR",),
                       vary         = ("si_R", "si_L",),#forgot to add it here
                       bounds       = dict(
                                           ltp       = ( 0.00, 1.50),#16
                                           slab      = ( 0.30, 3.00),#23
                                           si_R      = ( 0.30, 0.45),#16
                                           si_L      = ( 2.0, 12.50),#53
                                           si_num    = (10.0, 20.00),#11 = 3.4Mio
                                           ),
                       scales       = dict(
                                           ltp       = 10,      #every 0.10m
                                           slab      = 10,      #every 0.10m
                                           si_R      = 100,     #every 0.01m
                                           si_L      = 5,       #every 0.20m
                                           ),
                       fixed        = dict(si_method = 'VR',
                                           si_E      = 100e3,
                                           ),
                       )
        if   study == 100:      DESIGN = DESIGN100
        elif study == 101:      DESIGN = DESIGN101 # same conditions
        elif study == 105:      DESIGN = DESIGN105
    elif study in (200, 201,                                   # Nguyen
                   210,211,212,213,214,                        # Yasser
                   220,221,222,223,224,225,226,227,228,229,    # Stacho
                   230,231,232,233,234,235,236,237,238,239     # Bouassida
                   ):
        if study == 200:
            loadDEFAULT       = 120
            slabDEFAULT       = 43.5
            soilDEFAULT = 'Nguyen_SC'
            settlDEFAULT = 0.15710 # see paper
            DESIGN = dict(RUN_PATH_MAIN = f'Opt_Nguyen_SC_seed{seed}',
                          pop_size      = 20,   # assumption: trial budget
                          calcSize      = 400,  # assumption: max evaluations
                          stop_window   = 80,   # assumption: eval
                          user_methods  = ('VR_Nguyen',),
                          seed          = seed,
                          vary          = ('si_R',
                                           'si_num',
                                           'si_L'),
                          bounds        = dict(si_R      = (0.4-0.1, 0.6+0.1),
                                               si_L      = (12-1, 17+1),
                                               si_num    = (14, 28),
                                               # spacing = 43.5/(si_num+1):
                                               # limits 1.50 bis 2.90 m
                                               ),
                          scales        = dict(si_R=100, # alle 0.1m wie page8
                                               si_L=20), # alle 0.2m
                          fixed         = dict(si_method    ='VR_Nguyen',
                                               slab=0.5,  # assumption
                                               ltp=0.0,  # no SC LTP
                                               si_E=100000.0 * 1.3 * 0.4 / 0.7),
                          )
        elif study == 201:
            loadDEFAULT  = 120
            slabDEFAULT  = 43.5
            soilDEFAULT  = 'Nguyen_RI'
            settlDEFAULT = 0.03357  # Nguyen: RI solution 3
            settlDEFAULT = 0.15710  # Nguyen: standardfall auswertung see SC
            si_E_ref = 4700.0 * (10e3 / 1000.0)**0.5 * 1000.0
            DESIGN = dict(RUN_PATH_MAIN = f'Opt_Nguyen_RI_seed{seed}',
                          pop_size      = 20,   # assumption: trial budget
                          calcSize      = 800,  # assumption: max evaluations
                          stop_window   = 80,   # assumption: evaluations
                          user_methods  = ('RI_Nguyen',),
                          seed          = seed,
                          vary          = ('si_R',
                                           'si_num',
                                           'si_L'),
                          bounds        = dict(si_R   = (0.22-0.1, 0.31+0.1),
                                               si_L   = (12-1, 17+1),
                                               si_num = (14, 28),
                                               # spacing: 1.50 bis 2.90 m
                                               ),
                          scales        = dict(si_R = 100,  # alle 0.01 m
                                               si_L = 20),  # alle 0.05 m
                          fixed         = dict(si_method = 'RI_Nguyen',
                                               slab      = 0.5,   # assumption
                                               ltp       = 0.4,   # Nguyen
                                               si_E = si_E_ref),
                          )
        elif study in (210, 211, 212, 213, 214):
            loadDEFAULT  = 34
            soilDEFAULT  = 'Study_Yasser'
            SLABthickness = 0.6
            settlDEFAULT = 0.0886  # page 7
            si_num_ref    = 9
            slabDEFAULT   = 2.2 * (si_num_ref + 1)  # ≈ 24.2 m
            si_E_ref     = 100000
            DESIGN = dict(pop_size     = 20,
                          calcSize     = 400,
                          stop_window  = 80,
                          user_methods = ('VR_Yasser',),
                          seed         = seed,
                          )
            # 214: nur Säulenlänge
            if study == 214:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Yasse214_seed{seed}',
                              vary = ('si_L',),
                              bounds = dict(si_L = (6.0, 25.0)),
                              scales = dict(si_L = 20,),  # Schrittweite 0.05 m
                              fixed = dict(si_method = 'VR_Yasser',
                                           si_num    = si_num_ref,
                                           si_R      = 0.50,
                                           slab      = SLABthickness,
                                           ltp       = 0.0,
                                           si_E      = si_E_ref,
                                           ),
                              )
            # 211: nur LTP-Dicke
            elif study == 211:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Yasser211_seed{seed}',
                              vary = ('ltp',),
                              bounds = dict(ltp = (0.0, 1.75),),
                              scales = dict(ltp = 40,),
                              fixed = dict(si_method = 'VR_Yasser',
                                           si_num    = si_num_ref,
                                           si_R      = 0.50,
                                           si_L      = 15.0,
                                           slab      = SLABthickness,
                                           si_E      = si_E_ref,
                                           ),
                              )
            # 212: Säulenlänge und LTP-Dicke
            elif study == 212:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Yasser212_seed{seed}',
                              vary = ('si_L', 'ltp'),
                              bounds = dict(si_L = (6.0, 25.0),
                                            ltp  = (0.0, 1.75),
                                            ),
                              scales = dict(si_L = 20,
                                            ltp  = 40,
                                            ),
                              fixed = dict(si_method = 'VR_Yasser',
                                           si_num    = si_num_ref,
                                           si_R      = 0.50,
                                           slab      = 0.60,
                                           si_E      = si_E_ref,
                                           ),
                              )
            # 213: Säulenlänge und Radius, ohne LTP
            elif study == 213:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Yasser213_seed{seed}',
                              vary = ('si_L', 'si_R'),
                              bounds = dict(si_L = (6.0, 25.0),
                                            si_R = (0.50, 0.75),
                                            ),
                              scales = dict(si_L = 20,  si_R = 100,),
                              fixed = dict(si_method = 'VR_Yasser',
                                           si_num    = si_num_ref,
                                           slab      = SLABthickness,
                                           ltp       = 0.0,
                                           si_E      = si_E_ref,
                                           ),
                              )
            # 210: Säulenlänge, Radius und LTP-Dicke
            elif study == 210:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Yasser210_seed{seed}',
                              vary = ('si_L', 'si_R', 'ltp'),
                              bounds = dict(si_L = (6.0,  25.0),
                                            si_R = (0.50, 0.75),
                                            ltp  = (0.0,  1.75),
                                            ),
                              scales = dict(si_L = 20,
                                            si_R = 100,
                                            ltp  = 40,),
                              fixed = dict(si_method = 'VR_Yasser',
                                           si_num    = si_num_ref,
                                           slab      = SLABthickness,
                                           si_E      = si_E_ref,
                                           ),
                              )
            else: print(f'study {study} not defined in Study_Yasser')

        elif study in (220, 221, 222, 223, 224, 225, 226, 227, 228, 229):
            loadDEFAULT   = 80
            slabDEFAULT   = 80
            soilDEFAULT   = 'Study_Stacho'
            settlDEFAULT  = 0.045 # min
            DESIGN = dict(RUN_PATH_MAIN = f'Opt_Stacho{study}_seed{seed}',
                          seed         = seed,
                          )
            # 220 -> wrong : pubble statt pebblw
            if study == 220:
                DESIGN = dict(pop_size     = 20,
                              calcSize     = 800,
                              stop_window  = 80,
                              user_methods = ('VR_quarry',
                                              'VR_crushedpebble',
                                              'VR_pubble',
                                              ),
                              seed         = seed,
                              )
                DESIGN.update(vary = ('si_num',
                                      'si_method'),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            # 221
            elif study == 221:
                DESIGN.update(
                              pop_size     = 10,
                              calcSize     = 800,
                              stop_window  = 30,
                              user_methods = ('VR_quarry',
                                              ),
                              vary = ('si_num',
                                      ),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_method = 'VR_quarry',
                                           si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            
            elif study == 222:
                DESIGN.update(
                              pop_size     = 10,
                              calcSize     = 800,
                              stop_window  = 30,
                              user_methods = ('VR_crushedpebble',
                                             ),
                              vary = ('si_num',
                                      ),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_method = 'VR_crushedpebble',
                                           si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            
            elif study == 223:
                DESIGN.update(
                              pop_size     = 10,
                              calcSize     = 800,
                              stop_window  = 30,
                              user_methods = (
                                             'VR_pebble',
                                             ),
                              vary = ('si_num',
                                      ),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_method = 'VR_pebble',
                                           si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            elif study == 224:
                settlDEFAULT  = 0.045 # min
                DESIGN.update(
                              pop_size     = 10,
                              calcSize     = 800,
                              stop_window  = 60,
                              user_methods = ('VR_quarry',
                                             'VR_crushedpebble',
                                             'VR_pebble',
                                             ),
                              vary = ('si_num',
                                      'si_method'),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            elif study == 225:
                settlDEFAULT  = 0.062 # min
                DESIGN.update(
                              pop_size     = 10,
                              calcSize     = 800,
                              stop_window  = 60,
                              user_methods = ('VR_quarry',
                                             'VR_crushedpebble',
                                             'VR_pebble',
                                             ),
                              vary = ('si_num',
                                      'si_method'),
                              bounds = dict(si_num = (40, 53)),
                              scales = None,
                              fixed = dict(si_L      = 5,
                                           si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            # 222
            elif study == 226:
                DESIGN.update(
                              pop_size     = 20,
                              calcSize     = 800,
                              stop_window  = 80,
                              user_methods = ('VR_quarry',
                                             'VR_crushedpebble',
                                             'VR_pebble',
                                             ),
                              vary = ('si_num',
                                      'si_method',
                                      'si_L'),
                              bounds = dict(si_num = (40, 53),
                                            si_L = (2, 50)),
                              scales = dict(si_L = 20,),
                              fixed = dict(si_R      = 0.3,
                                           slab      = 0,
                                           ltp       = 1,
                                           si_E      = 100000,#not used
                                           ),
                              )
            else: print(f'study {study} not defined regaridng Study_Stacho')
            # print(DESIGN)

        elif study in (230, 231):
            loadDEFAULT  = 80
            soilDEFAULT  = 'Study_Bouassida'
            SLABthickness = 0.5
            LTPthickness  = 0.5
            settlDEFAULT  = 0.220 # see paper
            slabDEFAULT   = 15
            DESIGN = dict(pop_size     = 20,
                          calcSize     = 800,
                          stop_window  = 80,
                          user_methods = ('VR_Bouassida',
                                          ),
                          seed         = seed,
                          )
            # 220
            if study == 230:
                DESIGN.update(RUN_PATH_MAIN = f'Opt_Bouassida230_seed{seed}',
                              vary = ('si_num',
                                      'si_L'),
                              bounds = dict(si_num = (4, 14),
                                            si_L = (8, 18)),
                              scales = dict(si_L = 0.5,), # every 2 m
                              fixed = dict(si_R      = 0.4,
                                           slab      = SLABthickness,
                                           ltp       = LTPthickness,
                                           si_E      = 30e3,#not used
                                           ),
                              )
            else: print(f'study {study} not defined regaridng Study_Bouassida')

    else: return

    if RUN_COMPalgo or RUN_COMPplot:
        paths = []
        for algo in ALGORITHMs:
            pathFolder = f'{DESIGN["RUN_PATH_MAIN"]}_{algo}'
            output = []
            if RUN_COMPalgo: output.extend(('run',
                                            'console'))
            if RUN_COMPplot: output.extend(('convergence',
                                            'landscape_bars',
                                            'landscape_surface',
                                            'landscape_feasible',
                                            'evolution',
                                            'cost_settlement',
                                            'parallel',
                                            'method_ranges'))
            output = tuple(output)
            _, pathAlgo, _ = opt_algo(pathFolder  = pathFolder,
                                      dimension    = dimension,
                                      device       = RUN_DEVICE,
                                      user_methods = DESIGN["user_methods"],
                                      user_load    = loadDEFAULT,
                                      user_slab    = slabDEFAULT,
                                      user_soil    = soilDEFAULT,
                                      algorithm    = algo,
                                      seed         = DESIGN["seed"],
                                      vary         = DESIGN["vary"],
                                      scales       = DESIGN["scales"],
                                      fixed        = DESIGN["fixed"],
                                      bounds       = DESIGN["bounds"],
                                      limits=dict(u          = settlDEFAULT,
                                                  u_diff     = np.inf,
                                                  distortion = np.inf,
                                                  Mperh      = np.inf),
                                      constraint_mode = 'constraints',
                                      pop_size        = DESIGN["pop_size"],
                                      calcSize        = DESIGN["calcSize"],
                                      stop_window     = DESIGN["stop_window"],
                                      stop_cost_rel   = 0.001,
                                      output          = output
                                      )

    if RUN_GRID or RUN_GRIDplot:
        algo = 'grid'
        pathFolder = f'{DESIGN["RUN_PATH_MAIN"]}_{algo}'
        output = []
        if RUN_GRID: output.extend(('run',
                                    'console'))
        if RUN_GRIDplot: output.extend(('convergence',
                                        'landscape_bars',
                                        'landscape_surface',
                                        'landscape_feasible',
                                        'evolution',
                                        'cost_settlement',
                                        'parallel',
                                        'method_ranges'))
        output = tuple(output)
        # print(output)
        res, path, plots = opt_algo(pathFolder  = pathFolder,
                                   dimension    = dimension,
                                   device       = RUN_DEVICE,
                                   user_methods = DESIGN["user_methods"],
                                   user_load    = loadDEFAULT,
                                   user_slab    = slabDEFAULT,
                                   user_soil    = soilDEFAULT,
                                   algorithm    = algo,
                                   seed         = DESIGN["seed"],
                                   vary         = DESIGN["vary"],
                                   scales       = DESIGN["scales"],
                                   fixed        = DESIGN["fixed"],
                                   bounds       = DESIGN["bounds"],
                                   limits=dict(u          = settlDEFAULT,
                                               u_diff     = np.inf,
                                               distortion = np.inf,
                                               Mperh      = np.inf),
                                   constraint_mode        = 'constraints',
                                   output                 = output,
                                   )
        if 'run' in output:
            print('Best grid design:',                  res.X)
            print('Best feasible cost:',                res.F)
            print('Grid folder for later comparison:',  path)

    if (RUN_PLOTcost or
        RUN_PLOTcons or
        RUN_PLOTboth or
        RUN_PLOTpara):

        paths = []
        for algo in ALGORITHMs:
            pathFolder = f'{DESIGN["RUN_PATH_MAIN"]}_{algo}'
            _, _, path = open_plaxisData(RUN_DEVICE, 'opt', pathFolder,
                                         create = False)
            import os
            if os.path.isdir(path):
                paths.append(path)
        if not paths:
            print('No existing optimization folders found for'
                  f'{DESIGN["RUN_PATH_MAIN"]}_{ALGORITHMs}.')
            return
        _, _, savepath = open_plaxisData(RUN_DEVICE,
                                         'opt',
                                         f'{DESIGN["RUN_PATH_MAIN"]}_comparison')
        reference_kwargs = {}
        if RUN_PLOTgrid:
            import os
            _, _, reference_path = open_plaxisData(RUN_DEVICE,
                                           'opt',
                                           f'{DESIGN["RUN_PATH_MAIN"]}_grid',
                                           create=False
                                           )
            if os.path.isdir(reference_path):
                reference_kwargs["reference_path"] = reference_path
            else:
                print(f'Grid folder not found: {reference_path}')
        if RUN_PLOTcost:
            plot_convergence(paths,
                             save_path=savepath,
                             metrics=(
                                      'cost',
                                      # 'constraints',
                                      ),
                             **reference_kwargs
                             )
        if RUN_PLOTcons:
            plot_convergence(paths,
                             save_path=savepath,
                             metrics=(
                                      # 'cost',
                                      'constraints',
                                      ),
                             # **reference_kwargs
                             )
        if RUN_PLOTboth:
            plot_convergence(paths,
                             save_path=savepath,
                             metrics=(
                                      'cost',
                                      'constraints',
                                      ),
                             **reference_kwargs
                             )
        if RUN_PLOTpara:
            plot_evolution_parameter(paths, save_path=savepath)
            plot_evolution_cost_settlement(paths, save_path=savepath)
            _, groups = plot_best_feasible(paths, savepath)
            for idx, group in enumerate(groups, start=1):
                if len(group) < 2: continue
                table_filename = (f"optimization_best_feasible_{idx}.tex"
                                  if len(groups) > 1
                                  else "optimization_best_feasible.tex"
                                  )
                filename = f"optimization_best_feasible_{idx}_points"
                plot_best_feasible_points(paths          =[savepath],
                                          save_path      =savepath,
                                          x              =group[0],
                                          y              =group[1],
                                          labels         =None,
                                          table_filename =table_filename,
                                          filename       =filename,
                                          save           =True,
                )
    if RUN_PLOTstat and seed == 3:
        import re
        from itertools import combinations
        folder_comparison = re.sub(r"(seed)\d+",
                                   r"\1",
                                   DESIGN["RUN_PATH_MAIN"],
                                   flags=re.IGNORECASE,
                                   )
        _, _, savepath = open_plaxisData(RUN_DEVICE,
                                         "opt",
                                         f"{folder_comparison}_comparison",
                                         )
        paths = []
        for seed_nr in [1, 2, 3]:
            folder = re.sub(r"(seed)\d+",
                            lambda match: f"{match.group(1)}{seed_nr}",
                            DESIGN["RUN_PATH_MAIN"],
                            flags=re.IGNORECASE,
                            )
            _, _, path = open_plaxisData(RUN_DEVICE,
                                         "opt",
                                         f"{folder}_comparison",
                                         create=False,
                                         )
            paths.append(path)
        # if len(DESIGN["vary"]) == 2:
            for idx, (x, y) in enumerate(combinations(DESIGN["vary"], 2),
                                         start=1):
                table_filename = "optimization_best_feasible_all.tex"
                filename = f"optimization_all_best_feasible_{idx}_points"
                plot_best_feasible_points(paths             = paths,
                                          save_path         = savepath,
                                          x                 = x,
                                          y                 = y,
                                          table_filename    = table_filename,
                                          filename          = filename,
                                          save              = True,
                                          )
        plot_optimization_statistics(RUN_DEVICE, DESIGN["RUN_PATH_MAIN"])

if __name__ == "__main__":
    fct_opt(device=89)
