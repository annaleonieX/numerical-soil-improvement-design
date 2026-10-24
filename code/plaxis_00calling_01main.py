# -*- coding: utf-8 -*-
'''
Created on Tue Feb 24 12:18:18 2026
@author: annaleoniesalzmann

At the beginning, PLAXIS 2D must be connected to server port 10000,
and PLAXIS 3D to port 10002. The corresponding output window is always 
opened via port 10001.
You can then either select the device or adjust all the parameters:
path, plaxis3Dpw, plaxis2Dpw, study type, output and dimension.
'''

###############################################################################
#------------------------------- USER INPUTS ----------------------------------
###############################################################################

### Modelsetup within 00calling_02helper
device     = 27
pathFolder = r'Optimization - 2026-09-26 - Sensitivity Study'

# pip install plxscripting
# pip install -U pymoo

from plaxis_00calling_02helpers      import (open_plaxisData,
                                             fct_getLines)
from plaxis_01model_01main           import (fct_model,
                                             fct_modelParams)
from plaxis_02eval_01main            import (fct_eval)
from plaxis_03plot_01main            import (fct_plot)
from plaxis_04optimization_00calling import (fct_opt)
import traceback
import os.path

###############################################################################
#------------------------------- FOR LATEX FILE -------------------------------
###############################################################################

datasets = ['plaxis_01model_01main', 'plaxis_01model_02helpers',
            'plaxis_02eval_01main',  'plaxis_00calling_02helpers',
            'plaxis_03plot_01main',  'plaxis_03plot_02helpers']
keywords = [('all',"Soil improvement interface")]

fct_getLines(datasets, keywords)

###############################################################################
#---------------------------------- CALLING -----------------------------------
###############################################################################

def save_error(path_data, folder, stage, error):
    path_error = os.path.join(path_data, f'ERROR_{stage}.txt')

    with open(path_error, 'w', encoding='utf-8') as f:
        f.write(f'{folder} not possible\n')
        f.write(f'{stage} Error: {error}\n\n')
        f.write('Traceback:\n')
        f.write(traceback.format_exc())

def run_study(device,
              output,
              pathFolder,
              dimensions,
              variations,
              folder_func,
              model_kwargs_func = lambda variation: {},
              eval_testing      = False,
              folders           = [],
              foldersComp       = [],
              model             = r'Model'):
    for dimension in dimensions:
        port_ip, port_pw, path = open_plaxisData(device,
                                                 dimension,
                                                 pathFolder)
        for variation in variations:
            folder = folder_func(variation, dimension)
            folders.append(folder)
            path_data = os.path.join(path, folder)
            # --------------------------------------------------------------
            # MODEL
            # --------------------------------------------------------------
            if 1 in output:
                try:
                    params = fct_modelParams(**model_kwargs_func(variation))
                    fct_model(path_data,
                              model,
                              dimension,
                              port_ip,
                              port_pw,
                              *params)
                except Exception as e:
                    print(f'{folder} not possible')
                    print(f'MODEL Error: {e}')
                    traceback.print_exc()
                    save_error(path_data, folder, 'MODEL', e)
            # --------------------------------------------------------------
            # EVALUATION
            # --------------------------------------------------------------
            if 2 in output:
                try:
                    fct_eval(path_data,
                             model,
                             dimension,
                             port_ip,
                             port_pw,
                             testing=eval_testing)
                except Exception as e:
                    print(f'{folder} not possible')
                    print(f'EVAL Error: {e}')
                    traceback.print_exc()
                    save_error(path_data, folder, 'EVAL', e)
            # --------------------------------------------------------------
            # PLOT
            # --------------------------------------------------------------
            if 3 in output:
                try:
                    fct_plot(path, [folder])
                except Exception as e:
                    print(f'{folder} not possible')
                    print(f'PLOT Error: {e}')
                    traceback.print_exc()
                    save_error(path_data, folder, 'PLOT', e)
    # Plot all models together
    _, _, path = open_plaxisData(device, 'spacer', pathFolder)
    if 4 in output:
        try:
            # print(f'{folders} get printed')
            folders_plot = [folder for folder in folders
                            if os.path.isfile(os.path.join(path,
                                              folder,
                                              'input_parameters.txt'))
                            ]
            missing = [folder
                       for folder in folders
                       if folder not in folders_plot
                       ]
            if missing: print(f'Missing models: {missing}')
            fct_plot(path,
                     folders_plot,
                     foldersComp = foldersComp)
        except Exception as e:
            print(f'{folders} not possible')
            print(f'PLOT Error: {e}')
            traceback.print_exc()
            try:
                save_error(path_data, folder, 'mulitPLOT', e)
            except: print('Error within model error plotting')
    return folders

###############################################################################
#-------------------------------- USED INPUTS ---------------------------------
###############################################################################

### BASEMODEL
if   device in [89]: output = []
elif device in [27]: output = []
elif device in [0]:  output = []
else:                output = []
dimensions  = ['unitcell2', 'unitcell3', 'quarter3', 'full3']
variations  = ['spacer']
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        eval_testing      = False, # only if needed for graphs
                        folder_func       = lambda v,d: f'modeller_base_{d}',
                        model_kwargs_func = lambda v:   {},
                        folders           = [],
                        )

### surrounding test
if   device == 0:  output = []
elif device == 27: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [0.25, 0.5, 1, 2, 3, 4]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'lambda_{v}_{d}',
                        model_kwargs_func = lambda v:   {'toweroverFac': v},
                        folders           = ['lambda_0_unitcell2'],
                        )

### overall mesh variation
if   device == 0:  output = []
elif device == 27: output = []
else:              output = []
dimensions  = ['unitcell2', 'unitcell3']
variations  = ['coarse', 'mediumCoarse', 'medium', 'fineMedium', 'fine']
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        eval_testing      = True,
                        folder_func       = lambda v,d: f'meshing_{v}_{d}',
                        model_kwargs_func = lambda v:   {'meshRef': v},
                        folders           = [],
                        )
if   device == 0:  output = []
elif device == 27: output = [4]
else:              output = []
dimensions  = ['unitcell2','unitcell3']
variations  = ['medium', 'fineMedium', 'fine']
if 1 in output: raise KeyboardInterrupt # copy model from above
if 2 in output: raise KeyboardInterrupt # copy model from above
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        eval_testing      = True,
                        folder_func       = lambda v,d: f'meshingFiner_{v}_{d}',
                        model_kwargs_func = lambda v:   {'meshRef': v},
                        folders           = [],
                        )
# raise KeyboardInterrupt
### single mesh variation
if   device == 0:  output = []
elif device == 27: output = []
else:              output = []
dimensions  = ['unitcell2', 'unitcell3']
variations  = ['fineFoundation',
               'fineImprovement',
               'fineSoillayer',
               'fineMeshing',
               # 'fineLTP',
               'fine', # COPY Model from above
               ]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        eval_testing      = False,
                        folder_func       = lambda v,d: f'meshParts_{v}_{d}',
                        model_kwargs_func = lambda v:   {'meshRef': v},
                        folders           = []
                        )

### group effect
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['siNum_1_unitcell2']
                        )
### group effect -> bigger model
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'lambda3siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v,
                                                         'toweroverFac': 3},
                        folders           = [],
                        foldersComp = (['siNum_1_unitcell2']
                                       + [f'siNum_{n}_quarter3'
                                          for n in range(1, 30, 2)]),
                        )
### group effect -> finer mesh
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [3,7,11,15,19] # wont work 11+
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNumFine_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v,
                                                         'meshRef':    'fine'},
                        folders           = [],
                        foldersComp       = (['siNum_1_unitcell2']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(1, 30, 2)]),
                        )


### group effect -> first approach with OLD MODEL
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27]
if 1 in output: raise KeyboardInterrupt # old model
if 2 in output: raise KeyboardInterrupt # old eval
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNumRef_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['siNumRef_1_unitcell2'],
                        )
### first approach with OLD MODEL -> loading
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['unitcell3',
               'unitcell2']
if 1 in output: raise KeyboardInterrupt # already calc
variations  = [450, 700]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,#siNumRefeval
                        variations,
                        folder_func       = lambda v,d: f'loadRef_{v}_{d}',
                        model_kwargs_func = lambda v:   {'soil': 'Refmodel',
                                                         'line_load': v},
                        folders           = [],
                        )
### group effect -> first approach -> slab thinner
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [3,7,11,15,19,23,27,31,34]
if 1 in output: raise KeyboardInterrupt # already calc
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,#siNumRefeval
                        variations,
                        folder_func       = lambda v,d: f'slab05siNumRef_{v}_{d}',
                        model_kwargs_func = lambda v:   {'soil': 'Refmodel',
                                                         'slab_height': 0.5,
                                                         'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['slab05siNumRef_1_unitcell2'],
                        foldersComp       = ([]
                                             + ['siNumRef_1_unitcell2']
                                             + ['siNumRef_1_quarter3']
                                             + [f'siNumRef_{n}_quarter3'
                                                for n in range(3, 28, 4)]
                                             ),
                        )
### first approach with OLD MODEL -> ltp thicker
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3','unitcell2']
if 1 in output: raise KeyboardInterrupt # already calc
variations  = [0, 0.5, 1]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,#siNumRefeval
                        variations,
                        folder_func       = lambda v,d: f'ltpRef_{v}_{d}',
                        model_kwargs_func = lambda v:   {'soil': 'Refmodel',
                                                         'ltp_height': v},
                        folders           = [],
                        )

### group effect -> first approach with OLD MODEL and NEW EVAL
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27] # old model
if 1 in output: raise KeyboardInterrupt # already calc
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,#siNumRefeval
                        variations,
                        folder_func       = lambda v,d: f'siNumEval_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['siNumEval_1_unitcell2'],
                        foldersComp       = [],
                        )
### group effect -> first approach with NEW MODEL and NEW EVAL
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNumModel_{v}_{d}',
                        model_kwargs_func = lambda v:   {'soil': 'Refmodel',
                                                         'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['siNumModel_1_unitcell2'],
                        foldersComp       = [],
                        )
### group effect -> PLOTTING
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [] # only for plotting
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNumRefmodel_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = [],
                        foldersComp       = ([]
                                             # + ['siNumRefmodelNeg_1_unitcell2']
                                             # + ['siNumRefmodelPosNeg_1_unitcell2']
                                             + ['siNumRef_1_unitcell2']
                                             + ['siNumRef_1_quarter3']
                                             + [f'siNumRef_{n}_quarter3'
                                                for n in range(3, 28, 4)]
                                             + ['siNumEval_1_unitcell2']
                                             + ['siNumEval_1_quarter3']
                                             + [f'siNumEval_{n}_quarter3'
                                                for n in range(3, 28, 4)]
                                             + ['siNumModel_1_unitcell2']
                                             + ['siNumModel_1_quarter3']
                                             + [f'siNumModel_{n}_quarter3'
                                                for n in range(3, 24, 4)]
                                             ),
                        )

### loading
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3','unitcell2']
variations  = [100, 200]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'load_{v}_{d}',
                        model_kwargs_func = lambda v:   {'line_load': v},
                        folders           = []
                        )
### loading - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27]
load        = 200
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'load{load}siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'line_load': load,
                                                         'si_numberx':  v,
                                                         'si_numbery':  v},
                        folders           = [f'load{load}siNum_1_unitcell2'],
                        foldersComp       = ([]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 28, 4)]),
                        )
### loading - GROUP
if   device == 27: output = []
elif device == 0:  output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['full3']
load        = 200
variations  = [1,3,7,11,15,19]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'F3load{load}siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'line_load': load,
                                                         'si_numberx':  v,
                                                         'si_numbery':  v},
                        folders           = [f'F3load{load}siNum_1_unitcell2'],
                        foldersComp       = [],
                        )
### loading - GROUP
if   device == 27: output = []
elif device == 0:  output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3','full3']
# dimensions  = ['unitcell2']
load        = 200
variations  = [1,3,7,11,15,19] # 15 und 19 full3 not calculated
# variations  = [1]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'noExcload{load}siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'line_load': load,
                                                         'si_top': 0,# no excav.
                                                         'si_numberx':  v,
                                                         'si_numbery':  v},
                        folders           = [f'noExcload{load}siNum_1_unitcell2'],
                        foldersComp       = [],
                        )
### loading - GROUP -> for plotting
if   device == 27: output = []
elif device == 0:  output = []
elif device == 89: output = []
else:              output = []
load        = 200
types = [
         'noExc',
         '',
         ]
for typ in types:
    folders     = run_study(device,
                            output,
                            pathFolder,
                            ['unitcell2'],
                            [],
                            folder_func = lambda v,d: f'F3{typ}load{load}siNum_{v}_{d}',
                            model_kwargs_func = lambda v:   {'line_load': load,
                                                             'si_numberx':  v,
                                                             'si_numbery':  v},
                            folders     = [],
                            foldersComp = [f'{prefix}{typ}load{load}siNum_{n}_{model}'
                                           for prefix, dim in [('F3', 'full3'),
                                                               ('', 'quarter3')]
                                           for n, model in ([(1, 'unitcell2')]
                                                            + [(n, dim)
                                                               for n in [1,
                                                              *range(3, 20, 4)]]
                                                            )],
                            )
# raise ValueError

### slab variation
if   device == 89: output = []
elif device == 27: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 08 - slab'
dimensions  = ['unitcell2', 'quarter3']
variations  = [0.3, 0.65, 1.0, 1.35, 1.7, 2.05, 2.4] # 0.05 not possible
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'slab_{v}_{d}',
                        model_kwargs_func = lambda v:   {'slab_height': v},
                        folders           = []
                        )
### slab variation - group effect
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 08 - slab - group effect'
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'slab17siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'slab_height': 1.7,
                                                         'si_numberx':  v,
                                                         'si_numbery':  v},
                        folders           = ['slab17siNum_1_unitcell2'], #copy
                        foldersComp       = ([]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 24, 4)]),
                        )

### ltp variation
if   device == 89: output = []
elif device == 27: output = []
elif device == 0:  output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 09 - ltp'
dimensions  = ['unitcell2', 'quarter3']
variations  = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        eval_testing      = False,
                        folder_func       = lambda v,d: f'ltp_{v}_{d}',
                        model_kwargs_func = lambda v:   {'ltp_height': v,},
                        folders           = []
                        )
### ltp variation
if   device == 89: output = []
elif device == 27: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 09 - ltp'
dimensions  = ['quarter3', 'unitcell2'] # uc gleich wie oben
variations  = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'siNum23ltp_{v}_{d}',
                        model_kwargs_func = lambda v:   {'ltp_height': v,
                                                         'si_numberx': 23,
                                                         'si_numbery': 23},
                        folders           = []
                        )
### ltp variation t=0 - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 09 - ltp - group effect'
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19] # 23,27 took too long
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'ltp0siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'ltp_height': 0.0,
                                                         'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['ltp0siNum_1_unitcell2'] #copy
                        )
### ltp variation t=1 - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 09 - ltp - group effect'
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'ltp1siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'ltp_height':1.0,
                                                         'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['ltp1siNum_1_unitcell2'], #copy
                        foldersComp       = ([]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 24, 4)]
                                             + ['ltp0siNum_1_unitcell2']
                                             + ['ltp0siNum_1_quarter3']
                                             + [f'ltp0siNum_{n}_quarter3'
                                                for n in range(3, 20, 4)]
                                             ),
                        )

### spacing variation
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 10 - spacing'
dimensions  = ['unitcell2', 'quarter3']
variations  = [1.0, 1.5, 2.0, 2.5, 3.0]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'dis_{v}_{d}',
                        model_kwargs_func = lambda v:   {'disx': v,
                                                         'disy': v},
                        folders           = []
                        )
### spacing variation - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 10 - spacing - group effect'
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23,27]
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'dis2siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'disx': 2.0,
                                                         'disy': 2.0,
                                                         'si_numberx': v,
                                                         'si_numbery': v},
                        folders           = ['dis2siNum_1_unitcell2'], #copy
                        foldersComp       = ([]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 24, 4)]),
                        )

# si_stiffness due to stiff soil
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 11 - stiffness SI'
dimensions  = ['unitcell2', 'quarter3']
dimensions  = ['unitcell2']
variations  = [30e6, 100e3] # not working with MC for LTP!!!
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'e_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_stiffness': v},
                        folders           = ['e_100e3_unitcell2',
                                             'e_100e3_quarter3',
                                             'e_30e6_unitcell2',
                                             'e_30e6_quarter3']
                        )
### si stiffness - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 12 - stiffness SI - group effect'
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23] # UC2 not working with MC for LTP!!!
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'e30e6siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_stiffness': 30e6,
                                                         'si_numberx':   v,
                                                         'si_numbery':   v},
                        folders           = ['e30e6siNum_1_unitcell2'], #copy
                        foldersComp       = ([]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 24, 4)]),
                        )

# si_stiffness due to stiff soil # TO COMPARE LTP with LE und MC
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 11 - stiffness SI'
dimensions  = ['unitcell2']
variations  = []
folders     = run_study(device,
                        output,
                        pathFolder,
                        dimensions,
                        variations,
                        folder_func       = lambda v,d: f'ltpLEe_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_stiffness': v,
                                                         'ltp_mat':
                                                         {'Model': 'LE'}},
                        foldersComp       = ([]
                                             + ['ltpLEsiNum_1_unitcell2']
                                             + ['ltpLEsiNum_1_quarter3']
                                             + [f'ltpLEsiNum_{n}_quarter3'
                                                for n in range(3, 20, 4)]
                                             + ['siNum_1_unitcell2']
                                             + ['siNum_1_quarter3']
                                             + [f'siNum_{n}_quarter3'
                                                for n in range(3, 20, 4)]),
                        )
# si_stiffness due to stiff soil
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = [
               'unitcell2',
               'quarter3',
               ]
variations  = [
               30e6,
               800e3,
               400e3,
               100e3,
               ]
folders     = run_study(device, output, pathFolder, dimensions, variations,
                        eval_testing      = False,
                        folder_func       = lambda v,d: f'ltpLEe_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_stiffness': v,
                                                         'ltp_mat':
                                                             {'Model': 'LE'}},
                        )
### si stiffness - GROUP
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions  = ['quarter3']
variations  = [1,3,7,11,15,19,23]
stiffnesses = [
               30e6,
               800e3,
               400e3,
               100e3,
               ]
for stiffness in stiffnesses:
    folders     = run_study(device, output, pathFolder, dimensions, variations,
                            folder_func = lambda v,d: f'ltpLEe{stiffness}siNum_{v}_{d}',
                            model_kwargs_func = lambda v: {'si_stiffness': stiffness,
                                                           'si_numberx': v,
                                                           'si_numbery': v,
                                                           'ltp_mat':
                                                               {'Model': 'LE'}},
                            folders = [f'ltpLEe{stiffness}siNum_1_unitcell2'],
                            )
### si stiffness - GROUP - FOR PLOTTING
if   device == 27: output = []
elif device == 89: output = []
else:              output = []
dimensions      = ['quarter3']
siNums          = [1, 3, 7, 11, 15, 19, 23]
stiffnesses     = [30e6, 800e3, 400e3, 100e3]
folders     = run_study(device, output, pathFolder, dimensions, [],
                        folder_func = lambda v,d: f'ltpLEe0siNum_{v}_{d}',
                        model_kwargs_func = lambda v:   {'si_stiffness': 0,
                                                         'si_numberx':   v,
                                                         'si_numbery':   v,
                                                         'ltp_mat':
                                                             {'Model': 'LE'}},
                        folders           = [],
                        foldersComp       = [f'ltpLEe{e}siNum_1_unitcell2'
                                             for e in stiffnesses]
                                            +
                                            [f'ltpLEe{e}siNum_{i}_{d}'
                                             for e in stiffnesses
                                             for d in dimensions
                                             for i in siNums],
                        )


### si_length
# if   device == 27: output = []
# elif device == 89: output = []
# else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 13 - radius'
# dimensions  = ['unitcell2', 'quarter3']
# variations  = [3,6,9,12]
# folders     = run_study(device,
#                         output,
#                         pathFolder,
#                         dimensions,
#                         variations,
#                         folder_func       = lambda v,d: f'l_{v}_{d}',
#                         model_kwargs_func = lambda v:   {'si_length': v},
#                         folders           = []
#                         )
### si_length - GROUP
# if   device == 27: output = []
# elif device == 89: output = []
# else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 13 - radius - group effect'
# dimensions  = ['unitcell2', 'quarter3']
# variations  = [1,3,7,11,15,19,23]
# folders     = run_study(device,
#                         output,
#                         pathFolder,
#                         dimensions,
#                         variations,
#                         folder_func       = lambda v,d: f'l6siNum_{v}_{d}',
#                         model_kwargs_func = lambda v:   {'si_length': 6,
#                                                          'si_numberx': v,
#                                                          'si_numbery': v},
#                         folders           = ['l6siNum_1_unitcell2'] #copy
#                         foldersComp       = ([]
#                                              + ['siNum_1_unitcell2']
#                                              + ['siNum_1_quarter3']
#                                              + [f'siNum_{n}_quarter3'
#                                                 for n in range(3, 24, 4)]),
#                         )

### si_radius
# if   device == 27: output = []
# elif device == 89: output = []
# else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 13 - radius'
# dimensions  = ['unitcell2', 'quarter3']
# variations  = [0.3, 0.5]
# folders     = run_study(device,
#                         output,
#                         pathFolder,
#                         dimensions,
#                         variations,
#                         folder_func       = lambda v,d: f'r_{v}_{d}',
#                         model_kwargs_func = lambda v:   {'si_radius': v},
#                         folders           = []
#                         )
### si_radius - GROUP
# if   device == 27: output = []
# elif device == 89: output = []
# else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 13 - radius - group effect'
# dimensions  = ['quarter3']
# variations  = [1,3,7,11,15,19,23]
# folders     = run_study(device,
#                         output,
#                         pathFolder,
#                         dimensions,
#                         variations,
#                         folder_func       = lambda v,d: f'r05siNum_{v}_{d}',
#                         model_kwargs_func = lambda v:   {'si_radius': 0.5,
#                                                          'si_numberx': v,
#                                                          'si_numbery': v},
#                         folders           = ['r05siNum_1_unitcell2'] #copy
#                         foldersComp       = ([]
#                                              + ['siNum_1_unitcell2']
#                                              + ['siNum_1_quarter3']
#                                              + [f'siNum_{n}_quarter3'
#                                                 for n in range(3, 24, 4)]),
#                         )

# friction of SI
# if   device == 27: output = []
# elif device == 89: output = []
# else:              output = []
# pathFolder  = r'Optimization - 2026-09-14-v1 - 11 - friction SI'
# dimensions  = ['unitcell2', 'quarter3']
# variations  = [0.8, 1]
# folders     = run_study(device,
#                         output,
#                         pathFolder,
#                         dimensions,
#                         variations,
#                         folder_func       = lambda v,d: f'Rinter_{v}_{d}',
#                         model_kwargs_func = lambda v:   {'si_Rinter': v},
#                         folders           = []
#                         )

### UC test regarding error
if device == 0 :
    output           = []
    variations_dis   = [1.5]
    variations_ltpH  = [0.5]
    variations_ltpMc = [5]
    variations_ltpMp = [42]
    variations_ltpMm = ['MC']
    variations_meshRef = ['fine']
elif device == 27 :
    output           = []
    variations_dis   = [1, 1.5, 2, 2.5, 3]
    variations_ltpH  = [0,0.5,1,1.5]
    variations_ltpH  = [0]
    variations_ltpH  = [0.5,1,1.5]
    variations_ltpMc = [0.1,1,5,10]
    variations_ltpMp = [38,40,42,45]
    variations_ltpMm = ['HS','MC']
    variations_meshRef = ['mediumCoarse','fine']
else: output      = []
if len(output) > 0:
    pathFolder  = r'Optimization - 2026-09-14-v1 - 01 - uc2 validation'
    dimensions  = ['unitcell2']
    from itertools import product
    variations  = product(variations_dis,
                          variations_ltpH,
                          variations_ltpMc,
                          variations_ltpMp,
                          variations_ltpMm,
                          variations_meshRef)
    def model_kwargs_uc2_validation(variation):
        (variation_dis,
         variation_ltpH,
         variation_ltpMc,
         variation_ltpMp,
         variation_ltpMm,
         variation_meshRef) = variation
        return {'meshRef': variation_meshRef,
                'disx': variation_dis,
                'disy': variation_dis,
                'ltp_height': variation_ltpH,
                'ltp_mat': {'Model': variation_ltpMm,
                            'cRef': variation_ltpMc,
                            'phi': variation_ltpMp
                            }
                }
    def folder_uc2_validation(variation, dimension):
        (variation_dis,
         variation_ltpH,
         variation_ltpMc,
         variation_ltpMp,
         variation_ltpMm,
         variation_meshRef) = variation
        folder = (f'disIhIcIphiImImesh_'
                  f'{variation_dis}I{variation_ltpH}I'
                  f'{variation_ltpMc}I{variation_ltpMp}I'
                  f'{variation_ltpMm}I{variation_meshRef}_'
                  f'{dimension}')
        return folder.replace('.', 'p')
    folders     = run_study(device,
                            output,
                            pathFolder,
                            dimensions,
                            variations,
                            folder_func = lambda v,d:
                                folder_uc2_validation(v,d),
                            model_kwargs_func = lambda v:
                                model_kwargs_uc2_validation(v),
                            )

if 5 in output:
    import pandas as pd
    results = []
    for folder in folders:
        parts  = folder.split('_')
        values = parts[1].split('I')
        dis    = values[0].replace('p', '.')
        LTP    = values[1].replace('p', '.')
        c      = values[2].replace('p', '.')
        phi    = values[3].replace('p', '.')
        matmodel  = values[4]
        mesh   = values[5]
        dimension = parts[2]
        port_ip, port_pw, path = open_plaxisData(device,
                                                 dimension,
                                                 pathFolder)
        path_data = os.path.join(path, folder)
        if not os.path.exists(path_data):
            results.append({'dimension': dimension,
                            'dis': dis,
                            'h': LTP,
                            'c': c,
                            'phi': phi,
                            'matmodel': matmodel,
                            'mesh': mesh,
                            'phase': '',
                            'calc': 'no'})
            continue
        from plaxis_02eval_01main       import (fct_evalCHECK)
        calc = fct_evalCHECK('Model', path_data, dimension, port_ip, port_pw)
        print(calc)
        results.append({'dimension': dimension,
                        'dis': dis,
                        'h': LTP,
                        'c': c,
                        'phi': phi,
                        'matmodel': matmodel,
                        'mesh': mesh,
                        'phase': 10,
                        'calc': calc})
    df_results = pd.DataFrame(results,
        columns=['dimension',
                 'dis',
                 'h',
                 'c',
                 'phi',
                 'matmodel',
                 'mesh'
                 'phase',
                 'calc'])
    csv_output = os.path.join(path,
                              'calculation_status.csv')
    df_results.to_csv(csv_output,
                      sep=';',
                      index=False)
    print(f'Calculation status saved to:\n{csv_output}')

if 6 in output:
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A3
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate,
                                    Table,
                                    TableStyle,
                                    Paragraph,
                                    Spacer,
                                    PageBreak)
    # CSV EINLESEN
    csv_output = os.path.join(path, 'calculation_statusv2.csv')
    df = pd.read_csv(csv_output, sep=';')
    # Zahlen wieder in numerische Werte umwandeln
    df['dis'] = df['dis'].astype(float)
    df['h']   = df['h'].astype(float)
    df['c']   = df['c'].astype(float)
    df['phi'] = df['phi'].astype(float)
    # PDF
    doc = SimpleDocTemplate(os.path.join(path, 'calculation_status.pdf'),
                            pagesize     = landscape(A3),
                            rightMargin  = 10 * mm,
                            leftMargin   = 10 * mm,
                            topMargin    = 10 * mm,
                            bottomMargin = 10 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title',
                                 parent     = styles['Title'],
                                 fontSize   = 18,
                                 leading    = 22,
                                 spaceAfter = 5 * mm)
    heading_style = ParagraphStyle('heading',
                                   parent      = styles['Heading2'],
                                   fontSize    = 12,
                                   leading     = 15,
                                   spaceBefore = 4 * mm,
                                   spaceAfter  = 3 * mm)
    header_style = ParagraphStyle('header',
                                  parent       = styles['Normal'],
                                  fontSize     = 7,
                                  leading      = 8,
                                  alignment    = TA_CENTER)
    cell_style = ParagraphStyle('cell',
                                parent         = styles['Normal'],
                                fontSize       = 12,
                                leading        = 12,
                                alignment      = TA_CENTER)
    story = []
    # TABELLEN ERSTELLEN
    for dimension in sorted(df['dimension'].unique()):
        df_dim = df[df['dimension'] == dimension].copy()
        story.append(Paragraph(f'Calculation status – {dimension}',
                               title_style))
        story.append(Paragraph('<font color="#198754"><b>●</b></font> '
                               'calculation successful &nbsp;&nbsp;&nbsp; '
                               '<font color="#DC3545"><b>●</b></font> '
                               'calculation failed &nbsp;&nbsp;&nbsp; '
                               'empty = calculation not available',
                               styles['Normal']))
        story.append(Spacer(1, 4 * mm))
        # Eine Tabelle für jedes LTP + MESH + MODEL
        for h in sorted(df_dim['h'].unique()):
            df_ltp = df_dim[df_dim['h'] == h].copy()
            story.append(Paragraph(f'h = {h:g}', heading_style))
            for mesh in sorted(df_ltp['mesh'].unique()):
                df_mesh = df_ltp[df_ltp['mesh'] == mesh].copy()
                story.append(Paragraph(f'meshing = {mesh}', heading_style))
                for matmodel in sorted(df_mesh['matmodel'].unique()):
                    df_model = df_mesh[df_mesh['matmodel'] == matmodel].copy()
                    story.append(Paragraph(f'matmodel = {matmodel}', heading_style))
                    # Spaltenkombinationen c | phi | mm
                    configurations = (
                        df_ltp[['c', 'phi',]].drop_duplicates()
                        .sort_values(['c', 'phi']))
                    configurations = list(
                        configurations.itertuples(index=False, name=None))
                    header = [Paragraph('<b>dis ↓</b>', header_style)]
                    for c, phi in configurations:
                        c_text = (f'{c:g}'
                                  if float(c).is_integer() else f'{c:g}')
                        phi_text = (f'{phi:g}'
                                    if float(phi).is_integer() else f'{phi:g}')
                        header.append(Paragraph(f'<b>c{c_text}|φ{phi_text}</b>',
                                                header_style))
                    data = [header]
                    dis_values = sorted(df_ltp['dis'].unique())
                    for dis in dis_values:
                        row = [Paragraph(f'<b>{dis:g}</b>', cell_style)]
                        for c, phi, in configurations:
                            selection = df_ltp[(df_ltp['dis']   == dis) &
                                               (df_ltp['c']     == c)   &
                                               (df_ltp['phi']   == phi) ]
                            if selection.empty:
                                symbol = ''
                            else:
                                calc = selection.iloc[0]['calc']
                                if (calc is True or
                                    str(calc).lower() == 'true'):
                                    symbol = ('<font color="#198754">'
                                              '<b>●</b>' '</font>')
                                elif (calc is False or
                                      str(calc).lower() == 'false'):
                                    symbol = ('<font color="#DC3545">'
                                              '<b>●</b>' '</font>')
                                else: # calc == 'no'
                                    symbol = ''
                            row.append(Paragraph(symbol, cell_style))
                        data.append(row)
                    page_width = (landscape(A3)[0] - 20 * mm)
                    first_column = 18 * mm
                    remaining_width = ( page_width - first_column)
                    n_columns = len(configurations)
                    if n_columns > 0:
                        column_widths = ([first_column]
                                         + [remaining_width / n_columns]
                                         * n_columns)
                    else:
                        column_widths = [first_column]
                    table = Table(data, colWidths=column_widths, repeatRows=1)
                    table.setStyle(
                    TableStyle([# Header
                                ('BACKGROUND', (0, 0), (-1, 0),
                                 colors.HexColor('#E9ECEF')),
                                ('FONTNAME', (0, 0), (-1, 0),
                                 'Helvetica-Bold'),
                                # Erste Spalte
                                ('BACKGROUND', (0, 1), (0, -1),
                                 colors.HexColor('#F5F5F5')),
                                # Gitter
                                ('GRID', (0, 0), (-1, -1), 0.4,
                                 colors.HexColor('#B0B0B0')),
                                ('BOX', (0, 0), (-1, -1), 0.8,
                                 colors.HexColor('#777777')),
                                # Ausrichtung
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                # Padding
                                ('TOPPADDING', (0, 0), (-1, -1), 5),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),]))
                    story.append(table)
                    story.append(Spacer(1, 6 * mm))
        story.append(PageBreak())
    doc.build(story)
    print('Calculation status PDF saved.')
    
###############################################################################
################################ OPTIMIZATIONS ################################
###############################################################################

if device == 27:
    print('start opt')
    for seedNr in [3]:
        for studyNR in [
                        # 221,
                        # 222,
                        # 223,
                        # 224,
                        # 225,
                        # 226,
                        ]:
            # print(f'opti {seedNr} and {studyNR}')
            fct_opt(study           = studyNR,
                    seed            = seedNr,
                    RUN_DEVICE      = device,
                    RUN_COMPalgo    = True,
                    RUN_COMPplot    = False,
                    RUN_GRID        = False,
                    RUN_GRIDplot    = False,
                    RUN_PLOTcost    = False,
                    RUN_PLOTcons    = False,
                    RUN_PLOTboth    = False,
                    RUN_PLOTpara    = False,
                    RUN_PLOTgrid    = False,
                    RUN_PLOTstat    = False,
                    )
    for seedNr in [2,3]:
        for studyNR in [
                        # 221,
                        # 222,
                        # 223,
                        # 224,
                        # 225,
                        # 226,
                        ]:
            # print(f'opti {seedNr} and {studyNR}')
            fct_opt(study           = studyNR,
                    seed            = seedNr,
                    RUN_DEVICE      = device,
                    RUN_COMPalgo    = True,
                    RUN_COMPplot    = False,
                    RUN_GRID        = False,
                    RUN_GRIDplot    = False,
                    RUN_PLOTcost    = False,
                    RUN_PLOTcons    = False,
                    RUN_PLOTboth    = False,
                    RUN_PLOTpara    = False,
                    RUN_PLOTgrid    = False,
                    RUN_PLOTstat    = False,
                    )

elif device == 891:
    print('start opt')
    for studyNR in [
                    # 1,
                    # 2,
                    # 3,
                    # 5,
                    # 100,
                    # 105,
                    200, # l+r+num
                    # 201, # l+r+num
                    # 210, # l+r+LTP -> gibt alles auch einzeln (211-214)
                    # 220, # num+method
                    # 221, # num+L+method
                    # 230, # num+L
                  ]:
        for seedNr in [1,2,3]:
            fct_opt(study           = studyNR,
                    RUN_DEVICE      = device,
                    seed            = seedNr,
                    ALGORITHMs      = (
                                      'GA',
                                      'DE',
                                      # 'ISRES',
                                      # 'MixedVariableGA',
                                      'PSO',
                                      # 'PatternSearch',
                                      # 'grid',
                                      ),
                    RUN_COMPalgo    = False,
                    RUN_COMPplot    = False,
                    RUN_GRID        = False,
                    RUN_GRIDplot    = True,
                    RUN_PLOTcost    = True,
                    RUN_PLOTcons    = True,
                    RUN_PLOTboth    = True,
                    RUN_PLOTpara    = True,
                    RUN_PLOTgrid    = False,
                    RUN_PLOTstat    = True,
                    )

elif device == 89:
    print('start opt')
    for studyNR in [
                    # 1,
                    # 2,
                    # 3,
                    # 5,
                    # 100,
                    # 105,
                    # 200,
                    201,
                    # 210
                  ]:
        for seedNr in [1,2,3]:
            fct_opt(study           = studyNR,
                    RUN_DEVICE      = device,
                    seed            = seedNr,
                    ALGORITHMs      = (
                                      'GA',
                                      'DE',
                                      # 'ISRES',
                                      # 'MixedVariableGA',
                                      'PSO',
                                      # 'PatternSearch',
                                      # 'grid',
                                      ),
                    RUN_COMPalgo    = True,
                    RUN_COMPplot    = False,
                    RUN_GRID        = False,
                    RUN_GRIDplot    = False,
                    RUN_PLOTcost    = False,
                    RUN_PLOTcons    = False,
                    RUN_PLOTboth    = False,
                    RUN_PLOTpara    = False,
                    RUN_PLOTgrid    = False,
                    RUN_PLOTstat    = False,
                    )

elif device == 0: # reference study
    print('start opt')
    for seedNr in [3]:
        for studyNR in [210]:
            fct_opt(study           = studyNR,
                    seed            = seedNr,
                    RUN_DEVICE      = device,
                    RUN_COMPalgo    = True,
                    RUN_COMPplot    = False,
                    RUN_GRID        = False,
                    RUN_GRIDplot    = False,
                    RUN_PLOTcost    = False,
                    RUN_PLOTcons    = False,
                    RUN_PLOTboth    = False,
                    RUN_PLOTpara    = False,
                    RUN_PLOTgrid    = False,
                    )

###############################################################################
################################# Ref. Studies ################################
###############################################################################

refStudieseval = False
if refStudieseval:
    output = [1,2,4]
    folders = []
    dimension = 'unitcell2'
    pathFolder = r'Optimization - 2026-10-10 - Reference Studies'
    model = 'Model'
    studies = [
               ('Nguyen',         'VR'),
               ('Nguyen',         'RI'),
               ('Yasser',         'VR'),
               ('Stacho',         'Quarry'),
               ('Stacho',         'CrushedPebble'),
               ('Stacho',         'Pebble'),
               ('Bouassida',      'VR'),
               ]
    for soil, meth in studies:
        folder = f'Refstudy_{soil}{meth}_{dimension}'
        folders.append(folder)
        print(f'\nWorking in {folder} ...')
        port_ip, port_pw, path = open_plaxisData(device,
                                                 dimension,
                                                 pathFolder)
        path_data = os.path.join(path, folder)
        # MODEL
        if 1 in output:
            try:
                params = fct_modelParams(soil=soil,
                                         si_method=meth)
                fct_model(path_data, model,
                          dimension,
                          port_ip, port_pw,
                          *params
                          )
            except Exception as e:
                print(f'{folder} not possible')
                print(f'MODEL Error: {e}')
                traceback.print_exc()
                raise ValueError
        # EVALUATION
        if 2 in output:
            try:
                fct_eval(path_data, model, dimension,
                         port_ip, port_pw,
                         testing=False,
                         output = ['settlement']
                         )
            except Exception as e:
                print(f'{folder} not possible')
                print(f'EVAL Error: {e}')
                traceback.print_exc()
                raise ValueError
        # PLOT
        if 3 in output:
            try:
                fct_plot(path, [folder])
            except Exception as e:
                print(f'{folder} not possible')
                print(f'PLOT Error: {e}')
                # traceback.print_exc()
        # Notification
        print('\a')
    # Plot all models together
    _, _, path = open_plaxisData(device, 'spacer', pathFolder)
    if 4 in output:
        try:
            folders_plot = [folder
                            for folder in folders
                            if os.path.isfile(os.path.join(path,
                                              folder,
                                              'input_parameters.txt'))
                            ]
            missing = [folder
                       for folder in folders
                       if folder not in folders_plot
                       ]
            if missing: print(f'Missing models: {missing}')
            fct_plot(path,
                     folders_plot)
        except Exception as e:
            print(f'{folders} not possible')
            print(f'PLOT Error: {e}')
            traceback.print_exc()
            try:
                save_error(path_data, folder, 'mulitPLOT', e)
            except: print('Error within model error plotting')

print('\a') # notification
