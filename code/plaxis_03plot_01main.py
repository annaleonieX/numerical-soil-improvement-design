# -*- coding: utf-8 -*-
'''
Created on Wed Apr  8 15:07:29 2026
@author: annaleoniesalzmann
'''

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import os
import pandas as pd
import time

from plaxis_03plot_02helpers import (fct_setPlottext,
                                     fct_parseDimension,
                                     fct_parseKey,
                                     find_maxPhase,
                                     plot_resistance,
                                     plot_resistanceBar,
                                     plot_resiRatiosBar,
                                     plot_resiRatios,
                                     plot_XoverLoad,
                                     plot_pointsDistribution_comp,
                                     # plot_interfaceSlabDef,
                                     # plot_interfaceSlabDefPhase,
                                     # plot_interfaceShaftSig,
                                     # plot_interfaceShaftDef,
                                     # plot_interfaceShaftDefPhase,
                                     plot_runtime,
                                     plot_approachVariation,
                                     plot_Matrix,
                                     )

from plaxis_00calling_02helpers import (fct_writeRuntime,
                                        open_plaxisData)

###############################################################################
#------------------------------------ Main ------------------------------------
###############################################################################

def fct_plot(path,
             folders,
             foldersComp          = [],
             plotMultiple         = False,
             plotTotal            = False, # puts all files into one file
             plotVariationTornado = False,
             plotVariationForce   = False,
             plotVariationTime    = False,
             plotRuntime          = False,
             plotRuntimeTotal     = False,
             plotMatrix           = True, # useful for pile number/distance var.
             ):
    fct_setPlottext()
    try:
        folders = sorted(folders,
                         key=lambda x:
                             int(os.path.basename(x).split('_')[1]))
    except (ValueError, IndexError): pass
    if len(folders) == 0: pass
    elif len(folders) == 1:
        folder = folders[0]
        t_start = time.time()
        pathFolder = os.path.join(path, folder)
        suffixes, depth = fct_parseDimension(folder.split('_')[-1],
                                             pathFolder)
        for suffix in suffixes:
            pathFile = os.path.join(pathFolder,
                                    f'summary_value{suffix}',
                                    f'Fsu_phases{suffix}.csv')
            df = pd.read_csv(pathFile, sep=';')
            plot_resistance(pathFolder, uw=False, df=df, suffix=suffix);
            plot_resistance(pathFolder, uw=True,  df=df, suffix=suffix);
            plot_resiRatios(pathFolder, uw=False, df=df, suffix=suffix);
            plot_resiRatios(pathFolder, uw=True,  df=df, suffix=suffix);
            phase = find_maxPhase(path, folder)
            print(f'Considered phase is {phase} (only 1 folder).')
            typs = ['elem', 'soil', 'base','shaft']
            for typ in typs:
                plot_pointsDistribution_comp(os.path.join(pathFolder,
                                             f'results_{typ}',
                                             f'results_{typ}_p{phase}'),
                                             typ, phase)
        dfs = fct_buildDfs(path, folders)
        for _, models in dfs.items():
            # to compare different suffixes of only one model
            plot_resistance(path, uw=False, models=models, category=folder)
            plot_resistance(path, uw=True,  models=models, category=folder)
            plot_resiRatios(path, uw=False, models=models, category=folder)
            plot_resiRatios(path, uw=True,  models=models, category=folder)
        fct_writeRuntime(pathFolder, time.time()-t_start, 'plot')
    else:
        phasesList = []
        for folder in folders:
            phase = find_maxPhase(path, folder)
            phasesList.append(phase)
        if phasesList and all(value == phasesList[0]
                              for value in phasesList):
            phase = phasesList[0]
        else:
            print(phasesList)
            phase = -1 # different phases available due to excavation steps
        print(f'Considered phase is {phase}.')
        forces = ['F_elem',    'F_soil',    'F_base',    'F_shaft',
                  'F_elem_uw', 'F_soil_uw', 'F_base_uw', 'F_shaft_uw']
        forces = ['F_elem',    'F_soil',    'F_base',    'F_shaft']
        # forces = []
        settls = ['settl',    'settlDiff', 'tilt',     'tiltDiff',
                  'distortion','relativeDeflection',     'deflectionRatio']
        dfs = fct_buildDfs(path, folders)
        if plotMultiple and len(folders) > 0:
            filenames = plot_caseMultiple(path, dfs, phase)
            if plotTotal:
                plot_caseOverview(path, filenames);
        for category, models in dfs.items():
            foldersC = sorted(sorted(set("_".join(f.split("_")[:3])
                                         for f in list(models.keys()))),
                              key=lambda x:
                                  int(os.path.basename(x).split('_')[1])
                              if os.path.basename(x).split('_')[1].isdigit()
                              else float('inf'))
            if plotVariationTornado:
                for force in forces:
                    # print(f'Dual Force {force} in progress!')
                    fn = plot_approachVariation(path,
                                                category,
                                                foldersC,
                                                force,
                                                phase,
                                                typPlot='dual')
                    # for settl in settls:
                    #     fn = plot_approachVariation(path,
                    #                                 category,
                    #                                 foldersC,
                    #                                 settl,
                    #                                 phase,
                    #                                 typPlot='dual')
                        # print(fn)
                    if plotTotal:
                        plot_totalPic(os.path.join(path,
                                      f'plot_variations_{category}',
                                      f'plot_variations_{force}_p{phase}'),
                                      fn,
                                      f'Plot_Tornado_{force}_p{phase}_all')
                print('plotVariationTornado done.')
            if plotVariationForce:
                for force in forces:
                    print(f'Force per Force {force} in progress!')
                    fn = plot_approachVariation(path,
                                                category,
                                                foldersC,
                                                force,
                                                phase,
                                                typPlot='force')
                for settl in settls:
                    fn = plot_approachVariation(path,
                                                category,
                                                foldersC,
                                                settl,
                                                phase,
                                                typPlot='force')
                    # print(fn)
                print('plotVariationValue done.')
            if plotVariationTime:
                for force in forces:
                    print(f'Force per Time {force} in progress!')
                    fn = plot_approachVariation(path,
                                                category,
                                                foldersC,
                                                force,
                                                phase,
                                                typPlot='time')
                fn = plot_approachVariation(path,
                                            category,
                                            foldersC,
                                            settls[-1],#same for all
                                            phase,
                                            typPlot='time')
                print('plotVariationTime done.')
            if plotMatrix:
                for rows_i in [[1,2,3,4],
                               [1,3], [2,4],
                               [1],[2],[3],[4],
                               ]:
                    for title, values in [
                        ('forces',             ['F_elem', 'F_base']),
                        ('top forces',         ['F_elem', 'F_soil']),
                        ('bottom forces',      ['F_base', 'F_shaft']),
                        ('unit weight forces', ['F_elem_uw', 'F_base_uw']),
                        ('ratios',             ['LSR','RSR','SAR']),
                        ('unit weight ratios', ['LSR_uw','RSR_uw','SAR_uw']),
                        ('deformation',       ['u']),
                        ('deformations',       ['u', 'tilt']),
                        # ('slab tilting',       ['tilt']),
                        # ('slab settlements',   ['u']),
                        ('local moment',       ['M_local']),
                        ('moment',             ['M_local','M_global']),
                        ]:
                        plot_Matrix(path, category, foldersC,
                                    values, title, phase, rows=rows_i)
            if plotRuntime and len(foldersC) >= 2:
                for typPlot in ['parts','total']:
                    plot_runtime(path, foldersC, category,
                                 typPlot=typPlot,
                                 parts=('model','eval'))
        if plotRuntimeTotal and len(dfs) >= 2:
            for typPlot in ['parts','total']:
                plot_runtime(path, folders, 'total',
                             typPlot=typPlot,
                             parts=('model','eval'))
    if plotMatrix and len(foldersComp) > 0:
        foldersComp = folders+foldersComp
        phasesList = []
        for folder in foldersComp:
            phase = find_maxPhase(path, folder)
            phasesList.append(phase)
        if phasesList and all(value == phasesList[0]
                              for value in phasesList):
            phase = phasesList[0]
        else:
            print(phasesList)
            phase = -1 # different phases available due to excavation steps
        print(f'Considered phase is {phase} for foldersComp.')
        for rows_i in [[1,2,3,4],
                       [1,3], [2,4],
                       [1],[2],[3],[4],
                       ]:
            # print(rows_i)
            for title, values in [
                ('forces',             ['F_elem', 'F_base']),
                # ('top forces',         ['F_elem', 'F_soil']),
                # ('bottom forces',      ['F_base', 'F_shaft']),
                # ('unit weight forces', ['F_elem_uw', 'F_base_uw']),
                # ('ratios',             ['LSR','RSR','SAR']),
                # ('unit weight ratios', ['LSR_uw','RSR_uw','SAR_uw']),
                ('deformations',       ['u', 'tilt']),
                ('deformation',       ['u']),
                # ('slab tilting',       ['tilt']),
                # ('slab settlements',   ['u']),
                # ('local moment',       ['M_local']),
                # ('moment',             ['M_local','M_global']),
                ]:
                plot_Matrix(path,
                            'all',
                            foldersComp,
                            values, title, phase, rows=rows_i)
    print(f'Plots {folders+foldersComp} are complited.')

###############################################################################
#------------------------------- Main: MULTI ----------------------------------
###############################################################################

def fct_buildDfs(path, folders):
    dfs = {}
    # print(f'Searched folders in fct_buildDfs are\n{folders}')
    for folder in folders:                  # e.g. siLength_2_unitcell2
        category, _, dimension, _ = fct_parseKey(folder)
        category   = folder.split('_')[0]   # e.g. siLength
        if category not in dfs: dfs[category] = {}
        dimension  = folder.split('_')[2]   # e.g. unitcell2
        pathFolder = os.path.join(path, folder)
        suffixes, depth = fct_parseDimension(dimension, pathFolder)
        for suffix in suffixes:
           key      = f'{folder}{suffix}'
           pathFile = os.path.join(pathFolder,
                                   f'summary_value{suffix}',
                                   f'Fsu_phases{suffix}.csv')
           if not os.path.exists(pathFile): continue
           dfs[category][key] = pd.read_csv(pathFile, header=0, sep=';')
    return dfs
# EXAMPLE
#     folders = ['siLength_2_unitcell3','siLength_2_quarter3',
#                'mesh_fine_unitcell2', 'mesh_medium_unitcell2'
#                'mesh_fine_unitcell3', 'mesh_medium_unitcell3']
#     dfs = {"siLength": {"siLength_2_unitcell3": DataFrame,
#                         "siLength_2_quarter3_EdgeX":  DataFrame,
#                         "siLength_2_quarter3_EdgeY":  DataFrame,
#                         "siLength_2_quarter3_Middle":  DataFrame,
#                         "siLength_2_quarter3_Corner":  DataFrame,
#                         "siLength_3_quarter3_EdgeX":  DataFrame,
#                         "siLength_3_quarter3_EdgeY":  DataFrame,
#                         "siLength_3_quarter3_Middle":  DataFrame,
#                         "siLength_3_quarter3_Corner":  DataFrame},
#            "mesh": {"mesh_fine_unitcell2":      DataFrame,
#                     "mesh_medium_unitcell2":    DataFrame,
#                     "mesh_fine_unitcell3":      DataFrame,
#                     "mesh_medium_unitcell3":    DataFrame}}

def plot_caseMultiple(path, dfs, phase):
    filenames = {"res":     [],
                 "rat":     [],
                 "settl":   []}
    for cat, models in dfs.items():
        filename = plot_resistance(path, uw=False, models=models, category=cat)
        filenames["res"].append(filename)
        filename = plot_resistance(path, uw=True,  models=models, category=cat)
        filenames["res"].append(filename)
        filename = plot_resiRatios(path, uw=False, models=models, category=cat)
        filenames["rat"].append(filename)
        filename = plot_resiRatios(path, uw=True,  models=models, category=cat)
        filenames["rat"].append(filename)
        outputs = [
                   ['Mlocal',    'Bending moment',               'moment [kNm/m]'],
                   ['tilt',      'Tilting',                      'tilt [%]'],
                   ['F_elem',    'Force on the top of the pile', 'R_elem [kN]'],
                   ['F_elem_uw', 'Force on the top of the pile', 'R_elem_uw [kN]'],
                   ['F_soil',    'Force on the top of the soil', 'R_soil [kN]'],
                   ['F_soil_uw', 'Force on the top of the soil', 'R_soil_uw [kN]'],
                   ['F_shaft',   'Force due to shaft resistance','R_shaft [kN]'],
                   ['F_shaft_uw','Force due to shaft resistance','R_shaft_uw [kN]'],
                   ['F_base',    'Force due to end bearing',     'R_base [kN]'],
                   ['F_base_uw', 'Force due to end bearing',     'R_base_uw [kN]'],
                   ['u',         'Settlement over Loading',      'settlement [m]'],
                   ]
        for output in outputs:
            filename = plot_XoverLoad(path, models, cat, output = output)
        filenames["settl"].append(filename)
        plot_resistanceBar(path, models, cat, phase=int(phase), uw=False);
        plot_resiRatiosBar(path, models, cat, phase=int(phase), uw=False);
        plot_resistanceBar(path, models, cat, phase=int(phase), uw=True);
        plot_resiRatiosBar(path, models, cat, phase=int(phase), uw=True);
    print('Plots in plot_caseMultiple are done!')
    return filenames

def plot_caseOverview(path, filenames):
    if not filenames: return
    for pngs, name in [(filenames["res"],     "Overview_Resistance"),
                       (filenames["rat"],     "Overview_Ratios"),
                       (filenames["settl"],   "Overview_Settlements")]:
        pathTotal = os.path.join(path, 'plot_variations_total')
        plot_totalPic(pathTotal, pngs, name)
    return

def plot_totalPic(path, pdfs, name, save=True):
    try:
        from PyPDF2 import PdfMerger # has to be installed previously
        merger = PdfMerger()
        for pdf in pdfs:
            if pdf is None: continue
            if isinstance(pdf, str) and pdf.lower().endswith(".pdf"):
                merger.append(pdf)
        output_path = os.path.join(path, f"{name}.pdf")
        if save:
            merger.write(output_path)
            print(f"Saved merged PDF: {output_path}")
        merger.close()
    except Exception as e:
        print("Error during plot_totalPic-function:")
        print(e)
        output_path = os.path.join(path, f"{name}.pdf")
    return output_path

if __name__ == "__main__":
    device     = 0
    pathFolder = r'Optimization - 2026-08-04 - v1 - Analyse 01ref'
    pathFolder = r'Optimization - 2026-09-14-v1 - 01 - surrounding'
    pathFolder = r'Gruppeneffekte'
    pathFolder = 'Optimization - 2026-09-26 - Sensitivity Study'
    folders = ['model_base_unitcell2',
               'model_base_unitcell3']
    folders = ['eval_new_unitcell2',
               'eval_new_unitcell3',
               'eval_old_unitcell2',
               'eval_old_unitcell3']
    folders = ['eval_new_unitcell2',
               'eval_new_unitcell3',
               'eval_old_unitcell2',
               'eval_old_unitcell3']
    folders = ['mesh_fineMedium_unitcell2']
    folders = ['surrounding_0.25_quarter3',
               'surrounding_0.5_quarter3',
               'surrounding_1_quarter3',
               'surrounding_2_quarter3',
               'surrounding_3_quarter3',
               'surrounding_4_quarter3']
    folders = ['siNum_1_unitcell2',
               'siNum_1_quarter3',
               'siNum_3_quarter3',
               'siNum_5_quarter3',
               'siNum_7_quarter3',
               'siNum_9_quarter3',
               'siNum_11_quarter3',
               'siNum_13_quarter3',
               'siNum_15_quarter3',
               'siNum_17_quarter3',
               'siNum_19_quarter3',
               'siNum_21_quarter3',
               'siNum_23_quarter3',
               'siNum_25_quarter3',
               'siNum_27_quarter3',
               'siNum_29_quarter3',
               ]
    folders = ['meshingFiner_medium_unitcell2',
               'meshingFiner_fineMedium_unitcell2',
               'meshingFiner_fine_unitcell2',
               'meshingFiner_medium_unitcell3',
               'meshingFiner_fineMedium_unitcell3',
               'meshingFiner_fine_unitcell3',
               ]
    folders = ['meshingFinerM_medium_unitcell2',
               'meshingFinerM_fineMedium_unitcell2',
               'meshingFinerM_fine_unitcell2',
               'meshingFinerM_medium_unitcell3',
               'meshingFinerM_fineMedium_unitcell3',
               'meshingFinerM_fine_unitcell3',
               ]
    folders = ['loadRef_450_unitcell2',
               'loadRef_450_unitcell3',
               'loadRef_700_unitcell2',
               'loadRef_700_unitcell3',
               ]
    folders = ['ltp_0_unitcell2',
               ]
    _, _, path = open_plaxisData(device, 'spacer', pathFolder)
    fct_plot(path,
             folders,
             plotMultiple         = True,
             plotTotal            = False, # gather everything in one pdf
             plotVariationTornado = False,
             plotVariationForce   = False, # already within Tornado
             plotVariationTime    = False, # already within Tornado
             plotRuntime          = False,
             plotRuntimeTotal     = False, # gather everything
             plotMatrix           = False)