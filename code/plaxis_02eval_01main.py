# -*- coding: utf-8 -*-
'''
Created on Wed Apr  8 13:26:41 2026
@author: annaleoniesalzmann
'''

###############################################################################
#-------------------------------- USER-INPUT ----------------------------------
###############################################################################

testingDefault          = False
testingDefaultDeep      = False
plottingDefault         = True
accuracyDefault         = 'full'
offsetDefault           = 0.001
port_outDefault         = 10001
counterDefault          = [0,5]
phaseNameDefault        = 'Soil Improvement'
outputDefault           = ['settlement','tilt',
                           'moment','Npl','FoS',
                           'Fsoil','Felem',
                           'Fbase','Fshaft',
                           'ratios']

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import os.path, time, math, ast, traceback, glob, re
import plxscripting.easy as plx
import numpy as np
import pandas as pd
from math import isclose
from copy import deepcopy
from plaxis_00calling_02helpers import (fct_writeRuntime,
                                        fct_getPhases,
                                        open_plaxisData,
                                        close_plaxis)
from plaxis_02eval_02helpers import (fct_classify_interface,
                                     fct_data2D_interface,     # get
                                     fct_data3D_interface,     # get,apply
                                     fct_data2D_lineCS,        # getCS
                                     fct_data3D_horiCS,        # getCS,apply
                                     fct_data2D_InternalForce,
                                     fct_data3D_InternalForce,
                                     fct_calc2D_shaftForce,
                                     fct_calc2D_circleForce,
                                     fct_data3D_gridIP,        # getSingle
                                     fct_calc3D_surfIG,
                                     fct_calc2D_tilt,
                                     fct_calc3D_tilt,
                                     fct_plateMoment,
                                     fct_getForcebyWissmann)   # apply

###############################################################################
#------------------------------- Helpers: Data --------------------------------
###############################################################################

def fct_getInputparams(path, searchedDict):
    pathData = os.path.join(path, 'input_parameters.txt')
    if not os.path.isfile(pathData):
        raise FileNotFoundError(f'File {pathData} not found.')
    with open(pathData, 'r') as f: text = f.read()
    if searchedDict not in text:
        raise ValueError(f'Keyword {searchedDict} not found in {pathData}.')
    start     = text.find(searchedDict)
    start     = text.find('{', start)
    end       = text.find('}', start) + 1
    if (start == -1 or
        end == 0):
        raise ValueError(f'Bracket after {searchedDict} not found.')
    try: params = ast.literal_eval(text[start:end])
    except Exception as e:
        raise ValueError('Error durring transformation '
                         f'of {searchedDict}: {e}.')
    return params

def fct_updateTXTData(path, suffix, line=''):
    print(line)
    subfolder = f'summary_value{suffix}'
    pathFile = os.path.join(path, subfolder, f'{subfolder}.txt')
    with open(pathFile, 'a') as file:
        file.write(line + '\n')

def fct_createCSVphase(path, suffix, phase, descriptions, values):
    subfolder = f'summary_value{suffix}'
    pathFile = os.path.join(path,
                            subfolder,
                            f'{subfolder}_p{phase}.csv')
    new_df = pd.DataFrame({'Forcedescription': descriptions,
                           f'phase_{phase}': values})
    # FALL 1: Datei existiert noch nicht
    if not os.path.exists(pathFile):
        os.makedirs(os.path.dirname(pathFile), exist_ok=True)
        new_df.to_csv(pathFile, sep=';', index=False)
        return
    # FALL 2: Datei existiert → einlesen
    existing_df = pd.read_csv(pathFile, sep=';')
    # bereits vorhandene descriptions
    existing_descr = set(existing_df['Forcedescription'])
    # nur neue Zeilen filtern
    new_rows = new_df[~new_df['Forcedescription'].isin(existing_descr)]
    # wenn es neue gibt → anhängen
    if not new_rows.empty:
        updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
        os.makedirs(os.path.dirname(pathFile), exist_ok=True)
        updated_df.to_csv(pathFile, sep=';', index=False)
    return

def fct_createCSVphases(path, suffix,
                        testing=testingDefault):
    subfolder = f'summary_value{suffix}'
    files = glob.glob(os.path.join(path, subfolder, f'{subfolder}_p*.csv'))
    if not files:
        if testing:
            print('No phase files found.')
        return
    df_list = []
    for file in files:
        df = pd.read_csv(file, sep=';')
        df = df.set_index('Forcedescription')
        df_list.append(df)
    # for i, df in enumerate(df_list): # test
    #     if not df.index.is_unique:
    #         print(f"Duplicate index in df_list[{i}]:")
    #         print(df.index[df.index.duplicated()].tolist())
    # for i, df in enumerate(df_list): # test
    #     if not df.index.is_unique:
    #         print("DUPLIKAT IN FILE", files[i])
    #         print(df.index[df.index.duplicated(keep=False)])
    # print("ALL INDICES:")
    # for i, df in enumerate(df_list): # test
    #     print(i, files[i])
    #     print(df.index.tolist())
    for i, df in enumerate(df_list):
        # print(f"\nDataFrame {i}")
        if not df.index.is_unique:
            print("  -> Index nicht eindeutig")
            print(df.index[df.index.duplicated()])
        if not df.columns.is_unique:
            print("  -> Spalten nicht eindeutig")
            print(df.columns[df.columns.duplicated()])
    df_combined = pd.concat(df_list, axis=1)
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.sort_index(inplace=True)
    out_name = f'{subfolder}_total.csv'
    out_path = os.path.join(path, subfolder, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_combined.reset_index().to_csv(out_path, sep=';', index=False)
    return

def fct_createCSVdataUW(path, suffix):
    # files
    subfolder = f'summary_value{suffix}'
    nameFile = f'{subfolder}_total.csv'
    pathFile = os.path.join(path, subfolder, nameFile)
    out_name = f'{subfolder}_uw.csv'
    out_path = os.path.join(path, subfolder, out_name)
    # data
    key_word = 'phase_'
    df = pd.read_csv(pathFile, sep=';')
    phase_cols = [col for col in df.columns
                  if col.startswith(key_word)]
    if len(phase_cols) < 2: return # at least two different phases
    phase_colsWint = [(int(c.split('_')[1]), c) for c in phase_cols]
    phase_colsWint.sort()
    phase_cols = [c for _, c in phase_colsWint]
    ref_col = phase_cols[0]
    df_out = pd.DataFrame()
    df_out['Forcedescription'] = df['Forcedescription']
    for col in phase_cols[1:]:
        df_out[col] = df[col] - df[ref_col]
    df_out.to_csv(out_path, sep=';', index=False)
    return

def format_value(value):
    if (value != 0 and
        (abs(value) < 1e-1 or
         abs(value) >= 1e5)):
        return f'{value:.3e}'
    return f'{value:.3f}'

def fct_updateData(path, phase,
                   suffix, typ,
                   variant, values,
                   times, descr,
                   results  = None,
                   accuracy = 'full',
                   last     = False
                   ):
    if len(times) == 0: return
    if not last:
        fct_updateTXTData(path, suffix, '-' * 34)
        fct_updateTXTData(path, suffix,
                          f'{"Option":<10} {f"{typ}":<12} {"Time [s]":<10}')
        fct_updateTXTData(path, suffix, '-' * 34)
        descri_list = []
        values_list = []
        for var_i_idx, (var_i, F_i, t_i, des_i) in enumerate(
                zip(variant, values, times, descr)):
            fct_updateTXTData(path,
                              suffix,
                              f'{var_i:<10} {format_value(F_i):<12} '
                              f'{t_i:<10.2f} {des_i}'
                              )
            if results is not None:
                if typ.startswith('F'):
                    typ = typ[2:].split(' ')[0].lower()
                subfolder = f'results_{typ}{suffix}'
                dataname = f'{subfolder}_var{var_i}_p{phase}.csv'
                pathFile = os.path.join(path,
                                        subfolder,
                                        f'{subfolder}_p{phase}',
                                        dataname)
                os.makedirs(os.path.dirname(pathFile),
                            exist_ok=True)
                result = results[var_i_idx]
                # print(result)
                if result is not None:
                    result.to_csv(pathFile,
                                  index=False,
                                  sep=';')
            descri_list.append(f'{typ}_{des_i}')
            values_list.append(F_i)
        fct_createCSVphase(path, suffix, phase,
                           descri_list, values_list)
    if last:
        fct_createCSVphases(path, suffix)
        fct_createCSVdataUW(path, suffix)
    return

###############################################################################
#-------------------------- Helpers: Forces -----------------------------------
###############################################################################

def plot_fosCurve(pathFile):
    import matplotlib.pyplot as plt
    plt.figure()
    df = pd.read_csv(pathFile, sep=';')
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o')
    plt.xlabel("|u| (Total displacement)")
    plt.ylabel("SumMsf (FoS)")
    plt.title("FoS Curve")
    plt.grid(True)
    fig = pathFile.replace(".csv", ".pdf")
    plt.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()
    return

def fct_safeDiv(n, d):
    if n is None or d is None:
        return np.nan
    if np.isnan(n) or np.isnan(d) or d == 0:
        return np.nan
    return n / d
def fct_safeSub(a, b):
    if a is None or b is None: return np.nan
    return a - b

def fct_saveRefvalues(Fsu_phases_df, csv_path,
                      phase,
                      phaseName,
                      load_step=np.nan,
                      s=np.nan,
                      sd=np.nan,
                      t=np.nan,
                      td=np.nan,
                      b=np.nan,
                      Npl=np.nan,
                      M_global=np.nan,
                      M_local=np.nan,
                      F_elem=np.nan,  F_elem_0=np.nan,
                      F_soil=np.nan,  F_soil_0=np.nan,
                      F_base=np.nan,  F_base_0=np.nan,
                      F_shaft=np.nan, F_shaft_0=np.nan,
                      s_elem=np.nan,  s_elem_0=np.nan,
                      s_soil=np.nan,  s_soil_0=np.nan,
                      s_init=np.nan,  s_init_0=np.nan,
                      fos=np.nan):
    # deltas (due to excavation area)
    F_elem_uw  = fct_safeSub(F_elem,  F_elem_0)
    F_soil_uw  = fct_safeSub(F_soil,  F_soil_0)
    F_base_uw  = fct_safeSub(F_base,  F_base_0)
    F_shaft_uw = fct_safeSub(F_shaft, F_shaft_0)
    s_elem_uw  = fct_safeSub(s_elem,  s_elem_0)
    s_soil_uw  = fct_safeSub(s_soil,  s_soil_0)
    s_init_uw  = fct_safeSub(s_init,  s_init_0)
    # ratios
    LSR    = fct_safeDiv(F_elem,     F_soil)
    LSR_uw = fct_safeDiv(F_elem_uw,  F_soil_uw)
    RSR    = fct_safeDiv(F_shaft,    F_base)
    RSR_uw = fct_safeDiv(F_shaft_uw, F_base_uw)
    SAR    = fct_safeDiv(s_soil,     s_init)
    SAR_uw = fct_safeDiv(s_soil_uw,  s_init_uw)
    # Dataset
    new_row = pd.DataFrame([{'Phase':      phase,
                             'Name':       phaseName,
                             'Load':       load_step,
                             'u':          s,
                             'u_diff':     sd,
                             'tilt':       t,
                             'tilt_diff':  td,
                             'distortion': b,
                             'plastic points': Npl,
                             'M_global':   M_global,
                             'M_local':    M_local,
                             'F_shaft':    F_shaft, 'F_shaft_uw': F_shaft_uw,
                             'F_base':     F_base,  'F_base_uw':  F_base_uw,
                             'F_elem':     F_elem,  'F_elem_uw':  F_elem_uw,
                             'F_soil':     F_soil,  'F_soil_uw':  F_soil_uw,
                             's_elem':     s_elem,  's_elem_uw':  s_elem_uw,
                             's_init':     s_init,  's_init_uw':  s_init_uw,
                             's_soil':     s_soil,  's_soil_uw':  s_soil_uw,
                             'LSR':        LSR,     'LSR_uw':     LSR_uw,
                             'RSR':        RSR,     'RSR_uw':     RSR_uw,
                             'SAR':        SAR,     'SAR_uw':     SAR_uw,
                             'FoS':        fos}])
    # Remove existing phase (overwrite behavior)
    if (not Fsu_phases_df.empty and 'Phase' in Fsu_phases_df.columns):
        mask = Fsu_phases_df['Phase'] != phase
        Fsu_phases_df = Fsu_phases_df.loc[mask]
    # Append new result
    Fsu_phases_df = pd.concat([Fsu_phases_df, new_row], ignore_index=True)
    Fsu_phases_df.sort_values(by='Phase', inplace=True)
    Fsu_phases_df.to_csv(csv_path, index=False, sep=';')
    return Fsu_phases_df

###############################################################################
#------------------------------------ Main ------------------------------------
###############################################################################

def fct_eval(path, model, dimension, port_input, plaxispw,
             counter         = counterDefault,
             accuracy        = accuracyDefault,
             testing         = testingDefault,
             testingDeep     = testingDefaultDeep,
             output          = outputDefault,
             phaseStartName  = phaseNameDefault,
             offset          = offsetDefault,
             plotting        = plottingDefault,
             port_out        = port_outDefault):

    t_start = time.time()
    ### Determine loadsteps ---------------------------------------------------
    load_steps = np.loadtxt(os.path.join(path, 'load_steps.csv'),
                            delimiter=';')
    load_steps = np.insert(load_steps, 0, 0) # 0 for phases before
    dimension  = str(dimension) # to be safe

    ### Model opening ---------------------------------------------------------
    close_plaxis(port_out)
    s_i, g_i = plx.new_server('localhost', port_input, password=plaxispw)
    if dimension.endswith('2'):   modelname = model+'.p2dx'
    elif dimension.endswith('3'): modelname = model+'.p3d'
    else: raise ValueError('Dimension {dimension} not defined.')
    s_i.open(os.path.join(path, modelname))

    ### Evaluate model --------------------------------------------------------
    if dimension.endswith('2'): fct_eval2D(path, model, dimension,
                                           s_i, g_i, plaxispw,
                                           load_steps,
                                           counter        = counter,
                                           accuracy       = accuracy,
                                           testing        = testing,
                                           output         = output,
                                           phaseStartName = phaseStartName,
                                           offset         = offset,
                                           plotting       = plotting,
                                           port_out       = port_out
                                           )
    elif dimension.endswith('3'): fct_eval3D(path, model, dimension,
                                             s_i, g_i, plaxispw,
                                             load_steps,
                                             counter        = counter,
                                             accuracy       = accuracy,
                                             testing        = testing,
                                             output         = output,
                                             phaseStartName = phaseStartName,
                                             offset         = offset,
                                             plotting       = plotting
                                             )
    close_plaxis(port_out)
    t_end = time.time()
    runtime = t_end - t_start
    fct_writeRuntime(path, runtime, 'eval', accuracy)
    return # values for optimization necessary !!!

def fct_evalCHECK(model, path, dimension,
                  port_input, plaxispw,
                  port_out = port_outDefault):
    dimension  = str(dimension)
    close_plaxis(port_out)
    s_i, g_i = plx.new_server('localhost', port_input, password=plaxispw)
    if dimension.endswith('2'):   modelname = model+'.p2dx'
    elif dimension.endswith('3'): modelname = model+'.p3d'
    else: raise ValueError('Dimension {dimension} not defined.')
    try:
        s_i.open(os.path.join(path, modelname))
        close_plaxis(port_out)
        s_o, g_o = plx.new_server('localhost',
                                  port_out,
                                  password=plaxispw)
        g_i.gotostages()
        g_i.view(g_i.Phases[-1])
        g_o.close()
        check = True
    except:
        check = False
    try: g_o.close()
    except: pass
    return check

###############################################################################
#---------------------------------- 2D Case -----------------------------------
###############################################################################

def fct_eval2D(path, model, dimension,
               s_i, g_i, plaxispw, load_steps,
               counter        = counterDefault,
               accuracy       = accuracyDefault,
               testing        = testingDefault,
               testingDeep    = testingDefaultDeep,
               output         = outputDefault,
               phaseStartName = phaseNameDefault,
               offset         = offsetDefault,
               plotting       = plottingDefault,
               port_out       = port_outDefault
               ):
    ### 2D - Modify the output ------------------------------------------------
    suffix = ''
    # first value will be considered for the forces in Fsu_phases
    if testing: # first analysis
        variant_Felem  = [0,1,2,3]
        variant_Fsoil  = [0,1,2,3]
        variant_Fbase  = [0,1,2,3]
        variant_Fshaft = [0,1,2,3] # 2 and 3 only for deepTesting
        variant_tilted = [0,1,2]
    else:
        variant_Felem  = [0]
        variant_Fsoil  = [] # diff
        variant_Fbase  = [0]
        variant_Fshaft = [] # diff
        variant_tilted = [0]
    # variables to store temporary datasets
    F_times   = []
    F_results = []
    F_values  = []
    F_descr   = []

    ### 2D - Identify the Material Data via the Plaxis Model ------------------
    g_i.gotostructures()

    # what is the parameters of the slab
    slab_elems_poly = [] # g_i.Polygons
    for polygon in g_i.Polygons:
        if not polygon.Soil.Material: continue
        if 'Slab' in polygon.Soil.Material.Name.value:
            x_coords_global = [point.x.value + polygon.x.value
                               for point in polygon.Points]
            y_coords_global = [point.y.value + polygon.y.value
                               for point in polygon.Points]
            data = {'element': polygon,
                    'x_left':   min(x_coords_global),
                    'x_right':  max(x_coords_global),
                    'y_bottom': min(y_coords_global),
                    'y_top':    max(y_coords_global),
                    'weight':  polygon.Soil.Material.gammaUnsat}
            slab_elems_poly.append(data) # multiple slabs due to excavation
    if slab_elems_poly:
        x_left      = min(s['x_left']   for s in slab_elems_poly)
        x_right     = max(s['x_right']  for s in slab_elems_poly)
        slab_bottom = min(s['y_bottom'] for s in slab_elems_poly)
        slab_top    = max(s['y_top']    for s in slab_elems_poly)
        slab_weight = (sum(s['weight']  for s in slab_elems_poly)
                       / len(slab_elems_poly))
    else:
        print('No slab polygons found.')
        for polygon in g_i.Polygons:
            if not polygon.Soil.Material: continue
            if 'LTP' in polygon.Soil.Material.Name.value:
                x_coords_global = [point.x.value + polygon.x.value
                                   for point in polygon.Points]
                y_coords_global = [point.y.value + polygon.y.value
                                   for point in polygon.Points]
                data = {'element': polygon,
                        'x_left':   min(x_coords_global),
                        'x_right':  max(x_coords_global),
                        'y_bottom': min(y_coords_global),
                        'y_top':    max(y_coords_global),
                        'weight':   0}
                slab_elems_poly.append(data)
        if slab_elems_poly:
            x_left      = min(s['x_left']   for s in slab_elems_poly)
            x_right     = max(s['x_right']  for s in slab_elems_poly)
            slab_top    = max(s['y_top']    for s in slab_elems_poly)
            slab_bottom = slab_top
            slab_weight = 0
        else:
            for point in g_i.Points: # max from evenly distributed lineload
                x_coords_global = [point.x.value]
                y_coords_global = [point.y.value]
                data = {'x_left':   min(x_coords_global),
                        'x_right':  max(x_coords_global),
                        'y_bottom': min(y_coords_global),
                        'y_top':    max(y_coords_global),
                        'weight':   0}
                slab_elems_poly.append(data)
                if slab_elems_poly:
                    x_left      = min(s['x_left']   for s in slab_elems_poly)
                    x_right     = max(s['x_right']  for s in slab_elems_poly)
                    slab_top    = max(s['y_top']    for s in slab_elems_poly)
                    slab_bottom = slab_top
                    slab_weight = 0
                else:
                    raise ValueError
    slab_width  = x_right - x_left

    # what are the soil improvements (number depends on the no. of layers)
    si_elems_pol   = [] # g_i.Polygons
    for polygon in g_i.Polygons:
        if not polygon.Soil.Material: continue
        if 'SI' in  polygon.Soil.Material.Name.value:
            x_coords_global = [point.x.value + polygon.x.value
                               for point in polygon.Points]
            y_coords_global = [point.y.value + polygon.y.value
                               for point in polygon.Points]
            si_top_depth    = max(y_coords_global)
            si_base_depth   = min(y_coords_global)
            si_height       = si_top_depth - si_base_depth
            # what is the radius of the soil improvement
            points_by_y = {}
            for xi, yi in zip(x_coords_global, y_coords_global):
                if yi not in points_by_y:
                    points_by_y[yi] = []
                points_by_y[yi].append(xi)
            si_radii = []
            for x_list in points_by_y.values():
                if len(x_list) > 1:
                    radius = max(x_list) - min(x_list)
                    si_radii.append(radius)
            if si_radii: si_radius = np.mean(si_radii)
            else: raise ValueError('SI radius could not be determined.')
            si_elems_pol.append(polygon)
    if not si_elems_pol:
        raise ValueError('No SI polygons found.')

    # what is the interface above SI and Slab
    si_tops_line = None
    slab_int_top = None
    for line in g_i.Lines: 
        if (hasattr(line, 'NegativeInterface') or 
            hasattr(line, 'PositiveInterface')):
            y_coords = [line.First.y.value, line.Second.y.value]
            if all(abs(y_coord - si_top_depth) < 1e-3
                   for y_coord in y_coords):
                si_tops_line = line
            if all(abs(y_coord - slab_top) < 1e-3
                   for y_coord in y_coords):
                slab_int_top = line
    if si_tops_line is None:
        raise ValueError('No line found.')

    # extract SI unit weight automatically from material
    si_material = si_elems_pol[0].Soil.Material
    if hasattr(si_material, 'gammaUnsat'):
        si_weight = si_material.gammaUnsat
    elif hasattr(si_material, 'gammaSat'):
        si_weight = si_material.gammaSat
    else:
        raise ValueError('No unit weight found for SI material.')

    # what are the interfaces of the soil improvements
    line_list      = []
    si_bases_line  = []
    si_shafts_line = []
    tol = 1e-2
    for line in g_i.Lines:
        if (hasattr(line, 'NegativeInterface') or
            hasattr(line, 'PositiveInterface')):
            start = [line.First.x.value,  line.First.y.value]
            end   = [line.Second.x.value, line.Second.y.value]
            line_list.append((start, end, line))
    for lineData in line_list: # searched base interface
        if (isclose(lineData[0][1], si_base_depth, abs_tol=tol) and
            isclose(lineData[1][1], si_base_depth, abs_tol=tol)):
            si_bases_line.append(lineData[-1])   # line
    if not si_bases_line:
        raise ValueError('Base Interfaces not defined. '
                         f'{line_list} vs. {si_base_depth}')
    for lineData in line_list: # searched shaft interface
        if (isclose(lineData[0][0], si_radius, abs_tol=tol) and
            isclose(lineData[1][0], si_radius, abs_tol=tol)):
            si_shafts_line.append(lineData[-1]) # line
    if not si_shafts_line:
        raise ValueError('Shaft Interfaces not defined. '
                         f'{line_list} vs. {si_radius}')
    slab_height = abs(slab_top - slab_bottom)

    ### 2D - Identify loading and phase of the foundation slab
    g_i.gotostages()
    # check if model is calculated
    phase = g_i.Phases[-1]
    status = int(phase.CalculationResult.value)
    if status != 1:
        raise RuntimeError(f"Model calculation failed in final phase "
                           f"'{phase.Identification.value}': "
                           f"CalculationResult={status}, "
                           f"LogInfo={phase.LogInfo.value}."
                           )
    phaseStart = next(num
                      for num, phase in enumerate(g_i.Phases)
                      if phaseStartName in phase.Identification.value)
    try:
        phaseSlab = next(num
                         for num, phase in enumerate(g_i.Phases)
                         if 'Slab' in phase.Identification.value)
    except:
        phaseSlab = next(num
                         for num, phase in enumerate(g_i.Phases)
                         if 'LTP' in phase.Identification.value)
    phaseSi = next(num
                   for num, phase in enumerate(g_i.Phases)
                   if 'Improvement' in phase.Identification.value)
    phaseFoS = int(any("Failure" in phase.Identification.value
                       for phase in g_i.Phases))
    
    ### 2D - Define Output Log ------------------------------------------------
    # which phases are necessary
    ltp_params = fct_getInputparams(path, 'LTP_PARAMS')
    if ('height' not in ltp_params ):
        raise ValueError('height is missing in LTP_PARAMS.')
    ltp_height = ltp_params.get('height', None)
    if ltp_height > 0:
        ltp_material = fct_getInputparams(path, 'LTP_MATERIAL')
        ltp_weight = ltp_material.get('gammaunsat', None)
        transfer_weight = ltp_height * ltp_weight
    csv_path = os.path.join(path,
                            f'summary_value{suffix}',
                            f'Fsu_phases{suffix}.csv')
    concrete_weight = slab_weight * slab_height
    if phaseStart >= phaseSlab:
        shift = phaseStart - phaseSlab
        loading = [ls - concrete_weight
                   for ls in load_steps[shift:]]
        # if phaseStart == phaseSlab: -[RC, RC+L1, RC+L2, ...]
    elif phaseStart <= phaseSi:
        if ltp_height == 0:
            zeros = [0.0] * (phaseSi - phaseStart + 1)
            loading = zeros + [ls - concrete_weight
                               for ls in load_steps]
            # if phaseStart == phaseSi: -[0, RC, RC+L1, RC+L2, ...]
            # print(loading)
        else:
            zeros = [0.0] * (phaseSi - phaseStart + 1)
            loading1 = [ls - concrete_weight - transfer_weight
                        for ls in load_steps]
            loading = zeros + [-transfer_weight] + loading1
            # if phaseStart == phaseSi: -[0, RC, RC+L1, RC+L2, ...]
    else: raise ValueError('Not defined.')
    loading[:] = [round(x, 3) for x in loading]

    print(f'Relevant Loading for this sequence is \n{loading}.')
    diff = (phaseSlab - phaseStart + 1)
    phasesNeeded = fct_getPhases(phaseStart,
                                 len(load_steps)-1+phaseFoS,
                                 accuracy, diff,
                                 csv_path=csv_path)
    phasesTotal = fct_getPhases(phaseStart,
                                len(load_steps)-1+phaseFoS,
                                'full', diff,
                                csv_path=csv_path)
    phase_to_index = {p: i
                      for i, p in enumerate(phasesTotal)}
    if os.path.exists(csv_path):
        Fsu_phases_df = pd.read_csv(csv_path, sep=';')
    else:
        Fsu_phases_df = pd.DataFrame()
    # write calculated data down
    subfolder = f'summary_value{suffix}'
    pathFile = os.path.join(path, subfolder, f'{subfolder}.txt')
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    if accuracy in ['full', 'bare']:
        open(pathFile, 'w').close() # compute empty data
    print('The geometry was successfully determined of '
          f'{os.path.basename(path)}.')
    
    ### 2D - Get Output Data --------------------------------------------------
    for phase in phasesNeeded:
        variant_Felem_p  = deepcopy(variant_Felem)
        variant_Fsoil_p  = deepcopy(variant_Fsoil)
        variant_Fbase_p  = deepcopy(variant_Fbase)
        variant_Fshaft_p = deepcopy(variant_Fshaft)
        variant_tilted_p = deepcopy(variant_tilted)
        close_plaxis(port_out)
        s_o, g_o = plx.new_server('localhost',
                                  port_out,
                                  password=plaxispw)
        g_i.gotostages()
        phaseGI = g_i.Phases[int(phase)]
        phaseName = phaseGI.Identification.value
        g_i.view(phaseGI)
        fct_updateTXTData(path, suffix,
                          f'\nFor {phaseName}:')
        if 'Failure' in phaseName:
            if 'FoS' in output:
                fos = float(g_o.Phases[int(phase)].Reached.SumMsf.value)
                Fsu_phases_df = fct_saveRefvalues(Fsu_phases_df,
                                                  csv_path,
                                                  phase,
                                                  phaseName,
                                                  fos=fos)
                # curve_data = g_o.getcurves(g_o.Phases[int(phase)],
                #                            g_o.ResultTypes.Soil.Utot,
                #                            g_o.ResultTypes.Phases.SumMsf,
                #                            "node")
                # df_curve = pd.DataFrame(curve_data)
                # fos_path = os.path.join(path, f'summary_value{suffix}',
                #                         f'FoS_evaluation{suffix}.csv')
                # df_curve.to_csv(fos_path, index=False, sep=';')
                # plot_fosCurve(fos_path)
            else: fos = None
            continue

        if phase >= phaseSi:
            if slab_int_top is not None:
                slab_int_name = slab_int_top.NegativeInterface.Name.value
                slab_int_outp = next((i
                                      for i in g_o.Interfaces
                                      if slab_int_name in i.Name.value),
                                     None)
            si_top_int_name   = si_tops_line.NegativeInterface.Name.value
            si_top_int_outp   = next((i
                                      for i in g_o.Interfaces
                                      if si_top_int_name in i.Name.value),
                                     None)
            si_base_int_name  = si_bases_line[0].NegativeInterface.Name.value
            si_base_int_outp  = next((i
                                      for i in g_o.Interfaces
                                      if si_base_int_name in i.Name.value),
                                     None)
            try:
                si_shaft_int_name = si_shafts_line[0].NegativeInterface.Name.value
                si_shaft_int_outp = next((i
                                          for i in g_o.Interfaces
                                          if si_shaft_int_name in i.Name.value),
                                         None)
            except: # for old model
                si_shaft_int_name = si_shafts_line[0].PositiveInterface.Name.value
                si_shaft_int_outp = next((i
                                          for i in g_o.Interfaces
                                          if si_shaft_int_name in i.Name.value),
                                         None)

            try:
                center = 0
                t0 = time.time()
                (si_top_int_outp_elem,
                 si_top_int_outp_soil) = fct_classify_interface(g_o,
                                                                phase,
                                                                si_top_int_outp,
                                                                center,
                                                                si_radius)
                t_top = time.time() - t0
                t0 = time.time()
                (si_base_int_outp_elem,
                 _) = fct_classify_interface(g_o,
                                             phase,
                                             si_base_int_outp,
                                             center,
                                             si_radius)
                t_base = time.time() - t0
            except StopIteration:
                raise ValueError(f"Interfacetransformation not possible:"
                                 f"\n{si_top_int_outp}"
                                 f"\n{si_base_int_outp}")

            ### ELEM FORCE [2D]
            if 'Felem' in output:
                F_times.clear()
                F_results.clear()
                F_values.clear()
                F_descr.clear()
                typ       = 'elem'
                subfolder = f'results_{typ}'
                # F_values = [[F_elem_1, s_elem_1],...[F_i, s_i]]
                var_i = 0
                if (ltp_height == 0 or
                    "ltple" in os.path.basename(path).lower()):
                    variant_Felem_p = [v
                                       for v in variant_Felem_p
                                       if v != var_i]
                    if not variant_Felem_p:
                        variant_Felem_p = [3] # internal force tool
                if var_i in variant_Felem_p: #0
                    if si_top_int_outp_elem is not None:
                        try:
                            t0 = time.time()
                            #elem
                            results = fct_data2D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp_elem)
                            results.sort_values(by='x',
                                                inplace=True,
                                                ignore_index=True)
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathSave = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i = fct_calc2D_circleForce(results,
                                                         pathSave=pathSave)
                            s_i = F_i/(math.pi * si_radius**2)
                            # for history data
                            F_results.append(results)
                            F_values.append([F_i, s_i])
                            F_descr.append('due to the elem interface s_n '
                                           '(trapezoidal)')
                            F_times.append(time.time() - t0 + t_top)
                        except:
                            print('si_top_int_outp_elem not correct defined')
                            variant_Felem_p.remove(var_i)
                    else: # for Phase 1
                        print('si_top_int_outp_elem not defined')
                        variant_Felem_p.remove(var_i)
                        if not variant_Felem_p: variant_Felem_p.append(var_i+1)
                var_i += 1
                if var_i in variant_Felem_p: #1
                    # with crosssection
                    t0 = time.time()
                    results = fct_data2D_lineCS(g_o,
                                                int(phase),
                                                (0,
                                                 si_top_depth-offset),
                                                (slab_width,
                                                 si_top_depth-offset))
                    results.sort_values(by='x',
                                        inplace=True,
                                        ignore_index=True)
                    index_split = (results['x'] - si_radius).abs().idxmin()
                    results = results.iloc[:index_split+1]
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_circleForce(results,
                                                 pathSave=pathSave)
                    s_i = F_i/(math.pi * si_radius**2)
                    F_descr.append('due to crosssection stresses sig_yy '
                                   'under pile head (trapezoidal)')
                    F_results.append(results)
                    F_values.append([F_i, s_i])
                    F_times.append(time.time() - t0)
                var_i += 1
                variant_Felem_p = [v
                                   for v in variant_Felem_p
                                   if v != var_i]
                if not variant_Felem_p:
                    variant_Felem_p = [var_i+1]
                if var_i in variant_Felem_p: #2
                    try:
                        t0 = time.time()
                        results = fct_data2D_interface(g_o,
                                                       int(phase),
                                                       si_top_int_outp)
                        results.sort_values(by='x',
                                            inplace=True,
                                            ignore_index=True)
                        index_split = (results['x'] - si_radius).abs().idxmin()
                        #elem
                        results = results.iloc[:index_split+1]
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathSave = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i = fct_calc2D_circleForce(results,
                                                     pathSave=pathSave)
                        s_i = F_i/(math.pi * si_radius**2)
                        F_results.append(results)
                        F_values.append([F_i, s_i])
                        F_descr.append('due to the top interface s_n '
                                       'with limits '
                                       '(trapezoidal)')
                        F_times.append(time.time() - t0)
                    except: # for Phase 1
                        print('si_top_int_outp not correct defined')
                        variant_Felem_p.remove(var_i)
                        if not variant_Felem_p: variant_Felem_p.append(var_i+1)
                var_i += 1
                if var_i in variant_Felem_p: #3
                    # with 'force in volume' tool
                    t0 = time.time()
                    middle_elem = si_radius/2
                    si_name = si_elems_pol[0].Soil.Material.Name.value
                    si_outp = next(material for material in g_o.Materials
                                   if material.Name.value == si_name)
                    l_elem = [[middle_elem, si_top_depth],
                              [middle_elem, si_top_depth-0.3]]
                    results = fct_data2D_InternalForce(g_o,
                                                       int(phase),
                                                       material=si_outp,
                                                       line=l_elem)
                    F_i = results.loc[results['y'].idxmax(),
                                      'N'] * 2*math.pi * middle_elem
                    s_i = F_i / (math.pi * si_radius**2)
                    middle_soil = si_radius/2 + slab_width/2
                    F_descr.append('due to InternalForceAnalysis '
                                   '(material and line)')
                    F_results.append(results)
                    F_values.append([F_i, s_i])
                    F_times.append(time.time() - t0)
                if plotting:
                    fct_updateData(path, phase, suffix,
                                   f"F_{typ} [kN]", variant_Felem_p,
                                   [row[0] for row in F_values],
                                   F_times, F_descr, F_results,
                                   accuracy=accuracy)
                F_elem = F_values[0][0]
                s_elem = F_values[0][1]
                dataname = f'{subfolder}_line_p{phase}.csv'
                pathFile = os.path.join(path,
                                        subfolder,
                                        f'{subfolder}_p{phase}',
                                        dataname)
                os.makedirs(os.path.dirname(pathFile),
                            exist_ok=True)
                F_results[0].to_csv(pathFile,
                                    index=False,
                                    sep=';')
            else:
                F_elem = None
                s_elem = None

            ### SOIL FORCE [2D]
            if 'Fsoil' in output:
                # print(F_values)
                F_values_elem = deepcopy(F_values)
                F_descr_elem = [re.sub(r'\(([^)]*)\)',
                                       r' - \1 (force difference)',
                                       descr)
                                for descr in F_descr]
                F_times.clear()
                F_results.clear()
                F_values.clear()
                F_descr.clear()
                typ         = 'soil'
                subfolder   = f'results_{typ}'
                # control
                t0 = time.time()
                f_load = loading[phase_to_index[phase]] # s_init
                dis = fct_getInputparams(path,
                                         'SI_PARAMS').get('dis2D',
                                                          None)
                F_load = f_load * (dis*dis)
                tc = time.time() - t0
                variant_Fsoil_p = (list(range(100,
                                              100 + len(F_values_elem)))
                                   + variant_Fsoil_p)
                # print(F_values_elem, F_descr_elem)
                for F_value_elem, descr in zip(F_values_elem,
                                               F_descr_elem):
                    t0 = time.time()
                    F_i = F_load - F_value_elem[0]
                    s_i = F_i/(math.pi * (slab_width**2 - si_radius**2))
                    F_results.append(None)
                    F_values.append([F_i, s_i])
                    F_descr.append(descr)
                    F_times.append(time.time() - t0 + tc)
                var_i = 0
                if var_i in variant_Fsoil_p: #0
                    # with crosssection
                    t0 = time.time()
                    results = fct_data2D_lineCS(g_o, int(phase),
                                                (0,
                                                 si_top_depth-offset),
                                                (slab_width,
                                                 si_top_depth-offset))
                    results.sort_values(by='x',
                                        inplace=True,
                                        ignore_index=True)
                    index_split = (results['x'] - si_radius).abs().idxmin()
                    results = results.iloc[index_split+1:]
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_circleForce(results,
                                                 pathSave=pathSave)
                    s_i = F_i/(math.pi * (slab_width**2
                                          - si_radius**2))
                    F_descr.append('due to crosssection stresses sig_yy '
                                   'under pile head (trapezoidal)')
                    F_results.append(results)
                    F_values.append([F_i, s_i])
                    F_times.append(time.time() - t0)
                var_i += 1
                if var_i in variant_Fsoil_p: #1
                    if si_top_int_outp_soil is not None:
                        try:
                            t0 = time.time()
                            #soil
                            results = fct_data2D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp_soil)
                            results.sort_values(by='x', inplace=True,
                                                ignore_index=True)
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathSave = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i = fct_calc2D_circleForce(results,
                                                         pathSave=pathSave)
                            s_i = F_i/(math.pi * (slab_width**2
                                                  - si_radius**2))
                            # for history data
                            F_results.append(results)
                            F_values.append([F_i, s_i])
                            F_descr.append('due to the soil interface '
                                           's_n (trapezoidal)')
                            F_times.append(time.time() - t0 + t_top)
                            dataname = f'{subfolder}_line{var_i}_p{phase}.csv'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            os.makedirs(os.path.dirname(pathFile),
                                        exist_ok=True)
                            results.to_csv(pathFile, index=False, sep=';')
                        except:
                            print('si_top_int_outp_soil not correct defined')
                            variant_Fsoil_p.remove(var_i)
                            if not variant_Fsoil_p:
                                variant_Fsoil_p.append(var_i+1)
                    else: # for Phase 1
                        print('si_top_int_outp_soil not defined')
                        variant_Fsoil_p.remove(var_i)
                        if not variant_Fsoil_p:
                            variant_Fsoil_p.append(var_i+1)
                var_i += 1
                variant_Fsoil_p = [v
                                   for v in variant_Fsoil_p
                                   if v != var_i]
                if not variant_Fsoil_p:
                    variant_Fsoil_p = [var_i+1]
                if var_i in variant_Fsoil_p: #2
                    try:
                        t0 = time.time()
                        results = fct_data2D_interface(g_o,
                                                       int(phase),
                                                       si_top_int_outp)
                        results.sort_values(by='x', inplace=True,
                                            ignore_index=True)
                        index_split = (results['x']
                                       - si_radius).abs().idxmin()
                        #soil
                        results = results.iloc[index_split:]
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathSave = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i = fct_calc2D_circleForce(results,
                                                     pathSave=pathSave)
                        s_i = F_i/(math.pi * (slab_width**2
                                              - si_radius**2))
                        F_results.append(results)
                        F_values.append([F_i, s_i])
                        F_descr.append('due to the top interface s_n '
                                       'with limits '
                                       '(trapezoidal)')
                        F_times.append(time.time() - t0)
                        dataname = f'{subfolder}_line{var_i}_p{phase}.csv'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        os.makedirs(os.path.dirname(pathFile),
                                    exist_ok=True)
                        results.to_csv(pathFile, index=False, sep=';')
                    except: # for Phase 1
                        print('si_top_int_outp not correct defined')
                        variant_Fsoil_p.remove(var_i)
                        if not variant_Fsoil_p:
                            variant_Fsoil_p.append(var_i+1)
                var_i += 1
                if var_i in variant_Fsoil_p: #3
                    # with 'force in volume' tool
                    F_descr.append('due to InternalForceAnalysis '
                                   '(material and line)')
                    t0 = time.time()
                    middle_soil = si_radius/2 + slab_width/2
                    g_i.gotostructures()
                    soil_name = g_i.Soillayers[0].Soil.Material.Name.value
                    soil_outp = next(material for material in g_o.Materials
                                     if material.Name.value == soil_name)
                    l_soil = [[middle_soil, si_top_depth],
                              [middle_soil, si_top_depth-0.3]]
                    results = fct_data2D_InternalForce(g_o,
                                                       int(phase),
                                                       material=soil_outp,
                                                       line=l_soil)
                    F_i = results.loc[results['y'].idxmax(),
                                      'N'] * 2*math.pi * middle_soil
                    s_i = F_i / (math.pi * (slab_width**2
                                            - si_radius**2))
                    F_results.append(pd.concat([results, results],
                                               ignore_index=True))
                    F_values.append([F_i, s_i])
                    F_times.append(time.time() - t0)
                if plotting:
                    fct_updateData(path, phase, suffix,
                                   f"F_{typ} [kN]", variant_Fsoil_p,
                                   [row[0] for row in F_values],
                                   F_times, F_descr, F_results,
                                   accuracy=accuracy)
                F_soil = F_values[0][0]
                s_soil = F_values[0][1]
            else:
                F_soil = None
                s_soil = None

            ### BASE FORCE [2D]
            if 'Fbase' in output:
                F_times.clear()
                F_results.clear()
                F_values.clear()
                F_descr.clear()
                typ = 'base'
                var_i = 0
                if var_i in variant_Fbase_p: #0
                    # Get interface stresses along the base for each phase
                    t0 = time.time()
                    results = fct_data2D_interface(g_o,
                                                   int(phase),
                                                   si_base_int_outp_elem)
                    results.sort_values(by='x',
                                        inplace=True,
                                        ignore_index=True)
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_circleForce(results, si_radius, pathSave)
                    F_descr.append('due to base interface s_n '
                                   '(trapezoidal)')
                    F_results.append(results)
                    F_values.append(F_i)
                    F_times.append(time.time() - t0 + t_base)
                var_i += 1
                variant_Fbase_p = [v
                                   for v in variant_Fbase_p
                                   if v != var_i]
                if not variant_Fbase_p:
                    variant_Fbase_p = [var_i+1]
                if var_i in variant_Fbase_p: #1
                    # Get interface stresses along the base for each phase
                    t0 = time.time()
                    results = fct_data2D_interface(g_o,
                                                   int(phase),
                                                   si_base_int_outp)
                    results.sort_values(by='x',
                                        inplace=True,
                                        ignore_index=True)
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_circleForce(results, si_radius, pathSave)
                    F_descr.append('due to base interface s_n with limits'
                                   '(trapezoidal)')
                    F_results.append(results)
                    F_values.append(F_i)
                    F_times.append(time.time() - t0)
                var_i += 1
                if var_i in variant_Fbase_p: #2
                    # Get crosssection stresses along the base
                    t0 = time.time()
                    results = fct_data2D_lineCS(g_o,
                                                int(phase),
                                                (x_left,
                                                 si_base_depth-offset),
                                                (x_right,
                                                 si_base_depth-offset))
                    results.sort_values(by='x',
                                        inplace=True,
                                        ignore_index=True)
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_circleForce(results,
                                                 si_radius,
                                                 pathSave)
                    F_descr.append('due to crosssection stresses sig_yy '
                                   'above pile base (trapezoidal)')
                    F_results.append(results)
                    F_values.append(F_i)
                    F_times.append(time.time() - t0)
                var_i += 1
                if var_i in variant_Fbase_p: #3
                    # with 'force in volume' tool
                    t0 = time.time()
                    middle_elem = si_radius/2
                    si_name = si_elems_pol[0].Soil.Material.Name.value
                    si_outp = next(material
                                   for material in g_o.Materials
                                   if material.Name.value == si_name)
                    l_base = [[middle_elem, si_base_depth],
                              [middle_elem, si_base_depth + 0.3]]
                    results = fct_data2D_InternalForce(g_o,
                                                       int(phase),
                                                       material = si_outp,
                                                       line = l_base)
                    F_i = (results.loc[results['y'].idxmin(), 'N']
                           * 2*math.pi * middle_elem)
                    F_descr.append('due to InternalForceAnalysis '
                                   '(material and line)')
                    F_results.append(results)
                    F_values.append(F_i)
                    F_times.append(time.time() - t0)
                if plotting:
                    fct_updateData(path, phase, suffix,
                                   f"F_{typ} [kN]", variant_Fbase_p,
                                   F_values,
                                   F_times, F_descr, F_results,
                                   accuracy=accuracy)
                F_base = F_values[0]
            else: F_base = None

            ### SHAFT FORCE [2D]
            if 'Fshaft' in output:
                F_times.clear()
                F_results.clear()
                F_values.clear()
                F_descr.clear()
                typ = 'shaft'
                subfolder = f'results_{typ}'
                t0 = time.time()
                F_pile = math.pi * si_radius**2 * si_height * si_weight
                F_descr.append('due to stresses from var=0 '
                               'above and below pile (force difference)')
                F_results.append(None)
                F_values.append(F_elem - F_base - F_pile)
                F_times.append(time.time() - t0)
                variant_Fshaft_p = [100] + variant_Fshaft_p
                var_i = 0
                if var_i in variant_Fshaft_p: #0
                    F_descr.append('due to interface stresses s_1 with limits '
                                   '(trapezoidal)')
                    t0 = time.time()
                    results = fct_data2D_interface(g_o,
                                                   int(phase),
                                                   si_shaft_int_outp)
                    results.sort_values(by='y',
                                        inplace=True,
                                        ignore_index=True)
                    dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                    pathSave = os.path.join(path,
                                            subfolder,
                                            f'{subfolder}_p{phase}',
                                            dataname)
                    F_i = fct_calc2D_shaftForce(results,
                                                si_top_depth,
                                                si_base_depth,
                                                pathSave)
                    F_results.append(results)
                    F_values.append(F_i)
                    F_times.append(time.time() - t0)
                if any(v in variant_Fshaft_p for v in [var_i+1,
                                                       var_i+2,
                                                       var_i+3]):
                    t0 = time.time()
                    results_cs_full = None
                    results_cs_full = fct_data2D_lineCS(g_o,
                                                        int(phase),
                                                        (si_radius,
                                                         si_top_depth),
                                                        (si_radius,
                                                         si_base_depth))
                    results_cs_full.sort_values(by='y',
                                                inplace=True,
                                                ignore_index=True)
                    tc = time.time() - t0 # extraction time of the diff values
                    var_i += 1
                    if var_i in variant_Fshaft_p: #1
                        F_descr.append('due to crosssection stresses s_xy '
                                       '(trapezoidal)')
                        t0 = time.time()
                        results = results_cs_full.drop(columns=['sig_xx',
                                                                'sig_yy'])
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathSave = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i = fct_calc2D_shaftForce(results,
                                                    si_top_depth,
                                                    si_base_depth,
                                                    pathSave)
                        F_results.append(results)
                        F_values.append(-F_i)
                        F_times.append(time.time() - t0 + tc)
                    var_i += 1
                    if var_i in variant_Fshaft_p: #2
                        if testingDeep:
                            F_descr.append('due to crosssection stresses s_xx '
                                           '(trapezoidal)')
                            t0 = time.time()
                            results = results_cs_full.drop(columns=['sig_xy',
                                                                    'sig_yy'])
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathSave = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i = fct_calc2D_shaftForce(results,
                                                        si_top_depth,
                                                        si_base_depth,
                                                        pathSave)
                            F_results.append(results)
                            F_values.append(F_i)
                            F_times.append(time.time() - t0 + tc)
                        else: variant_Fshaft_p.remove(var_i)
                    var_i += 1
                    if var_i in variant_Fshaft_p: #3
                        if testingDeep:
                            F_descr.append('due to crosssection stresses s_yy '
                                           '(trapezoidal)')
                            t0 = time.time()
                            results = results_cs_full.drop(columns=['sig_xy',
                                                                    'sig_xx'])
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathSave = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i = fct_calc2D_shaftForce(results,
                                                        si_top_depth,
                                                        si_base_depth)
                            F_results.append(results)
                            F_values.append(F_i)
                            F_times.append(time.time() - t0 + tc)
                        else: variant_Fshaft_p.remove(var_i)
                    del results_cs_full
                else: var_i += 3
                if plotting:
                    fct_updateData(path, phase, suffix,
                                   f"F_{typ} [kN]", variant_Fshaft_p,
                                   F_values,
                                   F_times, F_descr, F_results,
                                   accuracy=accuracy)
                F_shaft = F_values[0]
            else: F_shaft = None
            F_times.clear()
            F_results.clear()
            F_values.clear()
            F_descr.clear()

        else:
            F_elem  = None
            F_soil  = None
            s_elem  = None
            s_soil  = None
            F_base  = None
            F_shaft = None

        ### Get design values for each phase [2D]
        typ       = 'constraints'
        subfolder = f'results_{typ}'
        variants  = []
        if (('tilt' in output or
             'settlement' in output) and
            phase >= phaseSlab):
            dx = 0.01 # 1cm
            results = {}
            methods = []
            var_i = 0
            if var_i in variant_tilted_p: #0
                methods.append(("CS", [(0, slab_top-offset),
                                       (slab_width, slab_top-offset)]))
            var_i += 1
            if var_i in variant_tilted_p: #1
                methods.append(("IP", {'x': list(np.arange(0,
                                                           slab_width + dx,
                                                           dx)),
                                       'y': slab_top-offset}))
            var_i += 1
            if var_i in variant_tilted_p: #2
                if slab_int_top is not None:
                    methods.append(("IF", slab_int_outp))
                else: variant_tilted_p.remove(var_i)
            for countTilt, (method, coords) in enumerate(methods):
                dataname = f'{subfolder}_tilt{method}_p{phase}.pdf'
                pathSave = os.path.join(path,
                                        subfolder,
                                        f'{subfolder}_p{phase}',
                                        dataname)
                (settlement,
                 settlementDiff,
                 tilt,
                 tiltDiffMax,
                 distortion,
                 relativeDeflection,
                 deflectionRatio,
                 time_calc) = fct_calc2D_tilt(g_o,
                                                          int(phase),
                                                          coords,
                                                          pathSave=pathSave)
                if countTilt == 0: # CS
                    s  = settlement
                    sd = settlementDiff
                    t  = tilt
                    td = tiltDiffMax
                    b  = distortion
                results[method] = {
                    "Settlement [m]":               settlement,
                    "Settlement difference [m]":    settlementDiff,
                    "Tilt [%]":                     tilt,
                    "Maximum tilt [%]":             tiltDiffMax,
                    "Distortion [%]":               distortion,
                    "Relative Deflection [m]":      tiltDiffMax,
                    "Deflection Ratio [-]":         distortion,
                    "Time [sec]":                   time_calc
                    }
                variants.append(countTilt)
                F_times.append(time_calc)
                F_values.append([settlement,
                                 settlementDiff,
                                 tilt,
                                 tiltDiffMax,
                                 distortion,
                                 relativeDeflection,
                                 deflectionRatio])
                F_descr.append([f'settlement values due to {method}',
                                f'diff. settl. values due to {method}',
                                f'tilt values due to {method}',
                                f'diff. tilt values due to {method}',
                                f'distortion values due to {method}',
                                f'relativeDefl values due to {method}',
                                f'deflectionRatio values due to {method}'])
            df = pd.DataFrame(results).T
            df.loc["Mean"] = df.mean(axis=0)
            os.makedirs(os.path.join(path, subfolder), exist_ok=True)
            pathSave = os.path.join(path,
                                    subfolder,
                                    f'{subfolder}_p{phase}',
                                    f"{subfolder}_tilt_p{phase}.csv")
            df.to_csv(pathSave, sep=";")
            if plotting:
                for index, name in [[0, 'settl [m]'],
                                    [1, 'settlDiff [m]'],
                                    [2, 'tilt [%]'],
                                    [3, 'tiltDiff [%]'],
                                    [4, 'distortion [%]'],
                                    [5, 'relativeDeflection [m]'],
                                    [6, 'deflectionRatio [-]'],
                                    ]:
                    if ('tilt' not in output and
                        'settlement' in output and
                        index > 0): # for uc2 unnecessary
                        break
                    fct_updateData(path, phase, suffix, name,
                                   variants,
                                   [row[index] for row in F_values],
                                   F_times,
                                   [row[index] for row in F_descr],
                                   accuracy=accuracy)
        else:
            s=None
            sd=None
            t=None
            td=None
            b=None

        F_times.clear()
        F_results.clear()
        F_values.clear()
        F_descr.clear()
        variants.clear()

        if 'settlement' in output:
            u = g_o.getcurveresults(g_o.Curvepoints[0],
                                    g_o.Phases[int(phase)],
                                    g_o.ResultTypes.Soil.Uy)
            print(f'Settlement is {u}.')
        else: u = None
        # what is the dummy plate:
        target_plate = None
        if 'moment' in output:
            for plate in g_i.Plates:
                try:
                    mat_name = plate.Material.value.Identification.value
                    if 'DummySlab' in mat_name:
                        target_plate = plate
                        break
                except Exception as e:
                    if testing: print(f'Plate Error: {e}')
                    target_plate = None
        # print(target_plate)
        if ('moment' in output and
            target_plate is not None):
            try:
                # if testing:
                #     print('target plate for moment found')
                t0 = time.time()
                pathFile = os.path.join(path,
                                        subfolder,
                                        f'{subfolder}_p{phase}')
                moment = fct_plateMoment(g_o,
                                         g_i,
                                         dimension,
                                         phaseSlab,
                                         int(phase),
                                         target_plate,
                                         x_left,
                                         x_left,
                                         x_right*2,
                                         x_right*2,
                                         path_save = pathFile,
                                         si_r      = si_radius)
                F_times.append(time.time() - t0)
                F_descr.append('moment values due to dummy plate')
                if plotting:
                    # if testing:
                    #     print(moment)
                    #     print(variants)
                    #     print(F_values)
                    #     print(F_times)
                    #     print(F_descr)
                    fct_updateData(path, phase, suffix,
                                   'moment_glo', [0],
                                   [moment['Mmax_global']],
                                   F_times, F_descr,
                                   accuracy=accuracy)
                    fct_updateData(path, phase, suffix,
                                   'moment_loc', [0],
                                   [moment['Mmax_local']],
                                   F_times, F_descr,
                                   accuracy=accuracy)
            except Exception as e:
                print(f'EVAL Moment Error: {e}')
                traceback.print_exc()
                moment = {"Mmax_global": None,
                          "Mmax_local": None}
        else: moment = {"Mmax_global": None, "Mmax_local": None}

        F_times.clear()
        F_results.clear()
        F_values.clear()
        F_descr.clear()
        variants.clear()

        if 'Npl' in output:
            t0 = time.time()
            state = g_o.getresults(g_o.Phases[int(phase)],
                                   g_o.ResultTypes.Soil.PlasticPoint,
                                   "Stresspoint")
            # print(state)
            Npl = int(sum(v == 4 for v in state))
            variants.append(0)
            F_times.append(time.time() - t0)
            F_values.append(Npl)
            # print(F_values)
            F_descr.append('plastic points')
            if plotting:
                fct_updateData(path, phase, suffix,
                               'N_plastic', variants, F_values,
                               F_times, F_descr,
                               accuracy=accuracy)
        else: Npl = None

        ### Start values
        fct_updateData(path, phase, suffix,
                       'spacer', [], [], [], [],
                       last=True)
        s_init    = loading[phase_to_index[phase]]
        load_step = s_init + concrete_weight if s_init < 0 else 0

        ### 2D - Summaries and recalculate values
        # Reference Value
        # print(phase, phaseStart)
        if ('Fshaft' in output or
            'Fbase' in output or
            'Felem' in output or
            'Fsoil' in output
            ):
            if phase == phaseStart: # reference phase currently calculated
                [F_shaft_0, F_base_0]           = [F_shaft, F_base]
                [F_elem_0,  F_soil_0]           = [F_elem,  F_soil]
                [s_elem_0,  s_soil_0, s_init_0] = [s_elem,  s_soil, s_init]
                if phaseStart == phaseSi:
                    fct_updateData(path, phase, suffix, "F_pile [kN]", [0,1],
                                   [F_base_0+F_shaft_0, -F_pile], [0,0],
                                   ['due to first F_base+F_shaft value',
                                    'due to F_pile calculation'],
                                   accuracy=accuracy
                                   )
            else: # reference phase was already calculated
                ref_row = Fsu_phases_df[Fsu_phases_df['Phase'] == phaseStart]
                if ref_row.empty:
                    raise ValueError(f'Ref.-Phase {phaseStart} not found.')
                if 'Fshaft' in output:
                    F_shaft_0 = ref_row['F_shaft'].iloc[0]
                if 'Fbase' in output:
                    F_base_0  = ref_row['F_base'].iloc[0]
                if 'Felem' in output:
                    F_elem_0  = ref_row['F_elem'].iloc[0]
                if 'Fsoil' in output:
                    F_soil_0  = ref_row['F_soil'].iloc[0]
                s_elem_0  = ref_row['s_elem'].iloc[0]
                s_soil_0  = ref_row['s_soil'].iloc[0]
                s_init_0  = ref_row['s_init'].iloc[0]
        else:
            F_elem_0 = None
            F_soil_0 = None
            F_base_0 = None
            F_shaft_0 = None
            s_elem_0 = None
            s_soil_0 = None
            s_init_0 = None
        # Join Dataset
        Fsu_phases_df = fct_saveRefvalues(Fsu_phases_df,
                                          csv_path,
                                          phase,
                                          phaseName,
                                          load_step,
                                          s,sd,t,td,b,
                                          Npl,
                                          moment['Mmax_global'],
                                          moment['Mmax_local'],
                                          F_elem,  F_elem_0,
                                          F_soil,  F_soil_0,
                                          F_base,  F_base_0,
                                          F_shaft, F_shaft_0,
                                          s_elem,  s_elem_0,
                                          s_soil,  s_soil_0,
                                          s_init,  s_init_0
                                          )
        g_o.close()

    return g_o

###############################################################################
#---------------------------------- 3D Case -----------------------------------
###############################################################################

def fct_eval3D(path, model, dimension,
               s_i, g_i, plaxispw, load_steps,
               counter        = counterDefault,
               accuracy       = accuracyDefault,
               testing        = testingDefault,
               testingDeep    = testingDefaultDeep,
               output         = outputDefault,
               phaseStartName = phaseNameDefault,
               offset         = offsetDefault,
               plotting       = plottingDefault,
               port_out       = port_outDefault):

    ### 3D - Modify the output ------------------------------------------------
    # first value will be considered for the forces in Fsu_phases
    if testing:
        variant_Felem  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        variant_Fsoil  = [0,1,2,3,4,5,6,7]
        variant_Fbase  = [0,1,2,3,4,5,6,7,8]
        variant_Fshaft = [0,1,2]
        variant_tilted = [0,1,2,3]
    else:
        variant_Felem  = [0]
        variant_Fsoil  = [] # diff + if_tri
        variant_Fbase  = [0]
        variant_Fshaft = [] # diff
        variant_tilted = [0]
    if dimension not in ['unitcell3', 'quarter3', 'full3']:
        raise ValueError('Function invalid for this dimension {dimension}')

    ### 3D - Identify the Material Data via the Plaxis Model ------------------
    g_i.gotostructures()
    # g_i.Phases[-2] # in case there is a phi-c-reduction phase

    # what is the parameters of the slab
    slab_elems_vol = [] # g_i.Volumes
    for volume in g_i.Volumes:
        if not volume.Soil.Material: continue # Material not defined
        name = volume.Soil.Material.Name.value # Materialname
        if 'Slab' in name:
            slab_data = {'element': volume,
                         'baseX':   volume.x.value,
                         'baseY':   volume.y.value,
                         'baseZ':   volume.z.value,
                         'weight':  volume.Soil.Material.gammaUnsat}
            slab_elems_vol.append(slab_data)
    if not slab_elems_vol:
        raise ValueError('No slab volumes found.')
    slab_bottom = min(s['baseZ']  for s in slab_elems_vol)
    slab_weight = (sum(s['weight'] for s in slab_elems_vol) /
                   len(slab_elems_vol))
    # define base point
    countsX = {}
    for s in slab_elems_vol:
        val = s['baseX']
        countsX[val] = countsX.get(val, 0) + 1
    slab_baseX = max(countsX, key=countsX.get)
    countsY = {}
    for s in slab_elems_vol:
        val = s['baseY']
        countsY[val] = countsY.get(val, 0) + 1
    slab_baseY = max(countsY, key=countsY.get)

    # what are the soil improvements (depends on the number of the layers)
    si_elems_all = []
    for vol in g_i.Volumes:
        if (vol.Soil.Material and
            'SI' in vol.Soil.Material.Name.value):
            si_elems_all.append((round(vol.x.value, 3),
                                 round(vol.y.value, 3),
                                 round(vol.z.value, 3), vol))
    if not si_elems_all: raise ValueError('No SI volumes found.')

    # what are the interfaces of the soil improvements
    surf_list = []
    for surf in g_i.Surfaces:
        if (hasattr(surf, 'NegativeInterface') or 
            hasattr(surf, 'PositiveInterface')):
            surf_list.append((surf.x.value,
                              surf.y.value,
                              surf.z.value, surf))

    # what is the distance of the raster
    if dimension in ['unitcell3']:
        dis_x1 = g_i.SoilContour[1].x.value - g_i.SoilContour[0].x.value
        dis_x2 = g_i.SoilContour[2].x.value - g_i.SoilContour[3].x.value
        dis_y1 = g_i.SoilContour[3].y.value - g_i.SoilContour[0].y.value
        dis_y2 = g_i.SoilContour[2].y.value - g_i.SoilContour[1].y.value
        dis_x = (dis_x1 + dis_x2) / 2
        dis_y = (dis_y1 + dis_y2) / 2
        num_si_x = 1
        num_si_y = 1
    elif dimension in ['quarter3', 'full3']:
        points = np.array([(vx, vy, vz)
                           for vx, vy, vz, _ in si_elems_all])
        # how many
        if dimension in ['full3']:
            # all soil improvements are modelled if 'full3'
            unique_x = np.unique(points[:, 0])
            unique_y = np.unique(points[:, 1])
            num_si_x = len(unique_x)
            num_si_y = len(unique_y)
        elif dimension in ['quarter3']:
            # Slab width is unknown -> no explicit geometry info is available.
            # And the model type (full / half / quarter pile representation)
            # cannot be derived directly from geometry.
            # As a workaround, SI_PARAMS (numx, numy) 
            # are read from input_parameters.txt
            # to infer the discretization and indirectly the model extent.
            si_params = fct_getInputparams(path, 'SI_PARAMS')
            if ('numx' not in si_params or 'numy' not in si_params):
                raise ValueError('Keys numx and/or numy are '
                                 'missing in SI_PARAMS.')
            num_si_x  = si_params.get('numx', None)
            num_si_y  = si_params.get('numy', None)
        # raster
        dx_list = []
        dy_list = []
        for i, (x0, y0, _) in enumerate(points):
            same_y = points[np.isclose(points[:, 1], y0)]
            dx = np.abs(same_y[:, 0] - x0)
            dx = dx[dx > 1e-6]
            if len(dx) > 0: dx_list.append(np.min(dx))
            same_x = points[np.isclose(points[:, 0], x0)]
            dy = np.abs(same_x[:, 1] - y0)
            dy = dy[dy > 1e-6]
            if len(dy) > 0: dy_list.append(np.min(dy))
        dis_x = np.median(dx_list) if dx_list else 0
        dis_y = np.median(dy_list) if dy_list else 0
        if (num_si_x <= 1 or num_si_y <= 1):
            if dimension not in ['quarter3']:
                # for full3
                # Slab width is unknown because no explicit info is available
                # si_params already known for quarter3
                si_params = fct_getInputparams(path, 'SI_PARAMS')
            if num_si_x == 1: dis_x = si_params.get('disx', None)
            if num_si_y == 1: dis_y = si_params.get('disy', None)
            # if num_si_x == 1: dis_x = abs(slab_baseX - points[0, 0]) * 2
            # if num_si_y == 1: dis_y = abs(slab_baseY - points[0, 1]) * 2
        if (dis_x == 0 or dis_y == 0):
            raise ValueError('Distance of the raster is ({dis_x}, {dis_y}).')
    else: raise ValueError('Not defined.')
    dis_x = round(dis_x, 3)
    dis_y = round(dis_y, 3)
    print(f'Distance of the raster is ({dis_x}, {dis_y}).')
    print(f'Number of SI points is ({num_si_x}, {num_si_y}).')

    # what are the predefined structures
    g_i.gotostages()
    # check if model is calculated
    phase = g_i.Phases[-1]
    status = int(phase.CalculationResult.value)
    if status != 1:
        raise RuntimeError(f"Model calculation failed in final phase "
                           f"'{phase.Identification.value}': "
                           f"CalculationResult={status}, "
                           f"LogInfo={phase.LogInfo.value}."
                           )
    # see output
    g_i.view(g_i.Phases[-1])
    close_plaxis(port_out)
    _, g_o = plx.new_server('localhost', port_out, password=plaxispw)
    si_searched = [(point.x.value,
                    point.y.value,
                    point.z.value)
                   for point in g_o.Curvepoints]
    print(f'Searched Curvepoints are at {si_searched}.')
    try: g_o.close()
    except: pass
    si_elems_vol   = [] # g_i.Volumes
    si_radii       = []
    si_tops_depth  = []
    si_bases_surf  = [] # g_i.Surfaces [Curvepoint1, Curvepoint2, ...]
    si_shafts_surf = [] # g_i.Surfaces [Curvepoint1, Curvepoint2, ...]
    for sx, sy, sz in si_searched:
        si_tops_depth.append(sz)

        # searched soil improvement volume
        distances_si = [np.sqrt((vx - sx)**2 + (vy - sy)**2)
                        for vx, vy, _, _ in si_elems_all]
        best_index = np.argmin(distances_si)
        si_elems_vol.append(si_elems_all[best_index][-1]) # vol

        # searched base interface
        si_baseValue = si_elems_vol[-1].z.value
        si_bases_all = [(ix, iy, iz, surf)
                         for ix, iy, iz, surf in surf_list
                         if iz == si_baseValue]
        if si_bases_all:
            distances_base = [np.sqrt((ix - sx)**2 + (iy - sy)**2)
                              for ix, iy, _, _ in si_bases_all]
            best_index = np.argmin(distances_base)
            si_bases_surf.append(si_bases_all[best_index][-1]) # surf
            print(f'si_bases_all (last) at [{sx}, {sy}, {sz}]: '
                  f'{si_bases_all[best_index][-1].Name.value}')
            # si_bases_depth.append(si_bases_all[best_index][2]) # iz
        else: raise ValueError('Base Interfaces need to be defined.')

        # searched shaft interface
        candidates_shaft = [(ix, iy, iz, surf)
                            for ix, iy, iz, surf in surf_list
                            if iz < si_baseValue]
        if candidates_shaft:
            distances_shaft = [np.sqrt((ix - sx)**2 + (iy - sy)**2)
                               for ix, iy, _, _ in candidates_shaft]
            best_index = np.argmin(distances_shaft)
            si_shafts_surf.append(candidates_shaft[best_index][-1]) # surf
        else: raise ValueError('Shaft Interfaces need to be defined.')

    si_top_depth  = np.mean(si_tops_depth)
    si_base_depth = si_baseValue
    si_height     = si_top_depth - si_base_depth
    print(f'Boundaries of the SI is (top={si_top_depth}, '
          f'bottom={si_base_depth}).\n')

    # find interface under and above the foundation slab
    g_i.gotostructures()
    si_tops_surf  = None
    slab_top   = None
    for surface in g_i.Surfaces:
        if (abs(surface.x.value - slab_baseX) < 1e-3 and
            abs(surface.y.value - slab_baseY) < 1e-3):
            z_val = surface.z.value
            if (abs(z_val - si_top_depth) < 1e-3 and
                (hasattr(surface, 'NegativeInterface') or
                 hasattr(surface, 'PositiveInterface'))):
                si_tops_surf = surface
            if (hasattr(surface, 'SurfaceLoad') and
                surface.SurfaceLoad):
                slab_top = z_val
    if (si_tops_surf is None or
        slab_top is None):
        raise ValueError('Load surface of SI top interface not found.')
    slab_height = round(abs(slab_top - slab_bottom),3)
    print(f'Slab height is {slab_height}.')
    slab_int_top = None
    for surface in g_i.Surfaces:
        if (abs(surface.x.value - slab_baseX) < 1e-3 and
            abs(surface.y.value - slab_baseY) < 1e-3):
            z_val = surface.z.value
            if (abs(z_val - slab_top) < 1e-3 and
                (hasattr(surface, 'NegativeInterface') or
                 hasattr(surface, 'PositiveInterface'))):
                slab_int_top = surface

    ### 3D - Identify loading and phase of the foundation slab
    g_i.gotostages()
    phaseStart = next(num for num, phase in enumerate(g_i.Phases)
                      if phaseStartName in phase.Identification.value)
    phaseSlab = next(num for num, phase in enumerate(g_i.Phases)
                      if 'Slab' in phase.Identification.value)
    phaseSi = next(num for num, phase in enumerate(g_i.Phases)
                    if 'Improvement' in phase.Identification.value)
    phaseFoS = int(any("Failure" in phase.Identification.value
                       for phase in g_i.Phases))

    ### 3D - Get Output Data --------------------------------------------------
    # Get output interface at every curvepoint [3D]
    x_vals = [p[0] for p in si_searched]
    y_vals = [p[1] for p in si_searched]
    xmin = min(x_vals)
    xmax = max(x_vals)
    ymin = min(y_vals)
    ymax = max(y_vals)
    tol = 1e-6
    mapping = {}
    for si_x, si_y, si_z in si_searched:
        on_xmin = abs(si_x - xmin) < tol
        on_xmax = abs(si_x - xmax) < tol
        on_ymin = abs(si_y - ymin) < tol
        on_ymax = abs(si_y - ymax) < tol
        if on_xmax and on_ymax:   suffix = '_Middle'
        elif on_xmin and on_ymin: suffix = '_Corner'
        elif on_xmin and on_ymax: suffix = '_EdgeY'
        elif on_xmax and on_ymin: suffix = '_EdgeX'
        else: raise ValueError(f'Unknown position for {(si_x, si_y, si_z)}')
        mapping[(si_x, si_y, si_z)] = suffix

    for count, ((si_x, si_y, si_z), suffix) in enumerate(mapping.items()):
        if count <  counter[0]: continue
        if count >= counter[1]: break
        si_radius  = si_radii[count]
        quadLimits = [si_x - dis_x/2,
                      si_x + dis_x/2,
                      si_y - dis_y/2,
                      si_y + dis_y/2]

        # which phases are necessary
        elems_poly = [] # g_i.Polygons
        # for polygon in g_i.Polygons:
        #     if not polygon.Soil.Material: continue
        #     if 'LTP' in polygon.Soil.Material.Name.value:
        #         x_coords_global = [point.x.value + polygon.x.value
        #                            for point in polygon.Points]
        #         y_coords_global = [point.y.value + polygon.y.value
        #                            for point in polygon.Points]
        #         data = {'element': polygon,
        #                 'x_left':   min(x_coords_global),
        #                 'x_right':  max(x_coords_global),
        #                 'y_bottom': min(y_coords_global),
        #                 'y_top':    max(y_coords_global),
        #                 'weight':  polygon.Soil.Material.gammaUnsat}
        #         elems_poly.append(data) # multiple slabs due to excavation
        if elems_poly:
            # x_left      = min(s['x_left']   for s in elems_poly)
            # x_right     = max(s['x_right']  for s in elems_poly)
            ltp_weight = (sum(s['weight']  for s in elems_poly)
                           / len(elems_poly))
            ltp_bottom = min(s['y_bottom'] for s in elems_poly)
            ltp_top    = max(s['y_top']    for s in elems_poly)
            ltp_height = abs(ltp_top - ltp_bottom)
        else:
            ltp_params = fct_getInputparams(path, 'LTP_PARAMS')
            if ('height' not in ltp_params ):
                raise ValueError('height is missing in LTP_PARAMS.')
            ltp_height = ltp_params.get('height', None)
            if ltp_height > 0:
                ltp_material = fct_getInputparams(path, 'LTP_MATERIAL')
                ltp_weight = ltp_material.get('gammaunsat', None)
            else: ltp_weight = 0
        transfer_weight = ltp_height * ltp_weight
        csv_path = os.path.join(path,
                                f'summary_value{suffix}',
                                f'Fsu_phases{suffix}.csv')
        print(f'Loadsteps for this sequence are\n{load_steps.tolist()}.')
        concrete_weight = slab_weight * slab_height
        if phaseStart >= phaseSlab:
            shift = phaseStart - phaseSlab
            loading = [ls - concrete_weight
                       for ls in load_steps[shift:]]
            # if phaseStart == phaseSlab: -[RC, RC+L1, RC+L2, ...]
        elif phaseStart <= phaseSi:
            if ltp_height == 0:
                zeros = [0.0] * (phaseSi - phaseStart + 1)
                loading = zeros + [ls - concrete_weight
                                   for ls in load_steps]
                # if phaseStart == phaseSi: -[0, RC, RC+L1, RC+L2, ...]
            else:
                zeros = [0.0] * (phaseSi - phaseStart + 1)
                loading1 = [ls - concrete_weight - transfer_weight
                            for ls in load_steps]
                loading = zeros + [-transfer_weight] + loading1
                # if phaseStart == phaseSi: -[0, RC, RC+L1, RC+L2, ...]
        else: raise ValueError('Not defined.')
        loading[:] = [round(x, 3) for x in loading]

        print(f'Relevant Loadsteps for this sequence are\n{loading}.')
        diff = (phaseSlab - phaseStart + 1)
        phasesNeeded = fct_getPhases(phaseStart,
                                     len(load_steps)-1+phaseFoS,
                                     accuracy, diff,
                                     csv_path=csv_path)
        phasesTotal = fct_getPhases(phaseStart,
                                    len(load_steps)-1+phaseFoS,
                                    'full', diff,
                                    csv_path=csv_path)
        phase_to_index = {p: i
                          for i, p in enumerate(phasesTotal)}
        if os.path.exists(csv_path):
            Fsu_phases_df = pd.read_csv(csv_path, sep=';')
        else:
            Fsu_phases_df = pd.DataFrame()
        # write calculated data down
        subfolder = f'summary_value{suffix}'
        pathFile = os.path.join(path, 
                                subfolder,
                                f'{subfolder}.txt')
        os.makedirs(os.path.dirname(pathFile), exist_ok=True)
        if accuracy in ['full', 'bare']:
            open(pathFile, 'w').close() # compute empty data

        # extract SI unit weight automatically from material
        # g_i.gotostages()
        si_material = si_elems_vol[count].Soil.Material
        if hasattr(si_material, 'gammaUnsat'):
            si_weight = si_material.gammaUnsat
        elif hasattr(si_material, 'gammaSat'):
            si_weight = si_material.gammaSat
        elif hasattr(si_material, 'gamma'):
            si_weight = si_material.gamma
        else:
            raise ValueError('No unit weight found for SI material.')
        print('The geometry was successfully determined of '
              f'{os.path.basename(path)}.')
        si_top_int_name   = si_tops_surf.NegativeInterface.Name.value
        si_base_int_name  = si_bases_surf[count].NegativeInterface.Name.value
        si_shaft_int_name = si_shafts_surf[count].NegativeInterface.Name.value
        if slab_int_top is not None:
            slab_int_name  = slab_int_top.NegativeInterface.Name.value
        print(f'\nFor Curvepoint {count} at position{suffix} '
              f'({si_x}, {si_y}):\n'
              f'    si_elem_vol   is {si_elems_vol[count].Name.value},\n'
              f'    si_base_surf  is {si_bases_surf[count].Name.value},\n'
              f'    si_base_int   is {si_base_int_name},\n'
              f'    si_shaft_surf is {si_shafts_surf[count].Name.value},\n'
              f'    si_shaft_int  is {si_shaft_int_name},')

        close_plaxis(port_out)
        s_o, g_o = plx.new_server('localhost', port_out, password=plaxispw)
        start_time = time.time()
        while True:
            try:
                g_i.view(g_i.Phases[-1])
                break
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed >= 180:
                    raise ValueError("View could not be opened"
                                     f"within 3 minutes: {e}")
                    break
                print(f"View not yet available: {e}")
                time.sleep(10)
        start_time = time.time()
        while True:
            try:
                si_top_int_outp = next(i
                                       for i in g_o.Interfaces
                                       if si_top_int_name in i.Name.value)
                break
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed >= 360:
                    raise ValueError(f"intf_top = {si_top_int_name} could "
                                     f"not be opened within 6 minutes: {e}")
                    break
                print(f"intf_top not yet available: {e}")
                time.sleep(10)
        try: # in 3D
            si_top_int_outp  = next((i
                                     for i in g_o.Interfaces
                                     if si_top_int_name
                                     in i.Name.value),
                                    None)
            si_base_int_outp  = next((i
                                      for i in g_o.Interfaces
                                      if si_base_int_name in i.Name.value),
                                     None)
            si_shaft_int_outp = next((i
                                      for i in g_o.Interfaces
                                      if si_shaft_int_name in i.Name.value),
                                     None)
            if slab_int_top is not None:
                slab_int_outp  = next((i
                                       for i in g_o.Interfaces
                                       if slab_int_name
                                       in i.Name.value),
                                      None)
            print(f'    si_top_int_outp   is {si_top_int_outp.Name.value},\n'
                  f'    si_base_int_outp  is {si_base_int_outp.Name.value},\n'
                  f'    si_shaft_int_outp is {si_shaft_int_outp.Name.value}.')
        except StopIteration:
            raise ValueError(f"Interface nicht gefunden:"
                             f"\n{si_top_int_outp}"
                             f"\n{si_shaft_int_name}"
                             f"\n{slab_int_name}")

        # elem & base
        si_mateName   = si_elems_vol[count].Soil.Material.Name.value
        si_material   = next(material
                             for material in g_o.Materials
                             if material.Name.value == si_mateName)
        si_soilName   = si_elems_vol[count].Soil.Name.value
        si_soil       = next(soil
                             for soil in g_o.Soils
                             if soil.Name.value == si_soilName)

        # define multipli
        soil = g_o.ResultTypes.Soil
        phas = g_o.Phases[-1]
        u1 = g_o.getsingleresult(phas, soil.Ux,
                                 (si_x+si_radius/2,
                                  si_y+si_radius/2,
                                  si_z-si_radius))
        u2 = g_o.getsingleresult(phas, soil.Ux,
                                 (si_x-si_radius/2,
                                  si_y+si_radius/2,
                                  si_z-si_radius))
        u3 = g_o.getsingleresult(phas, soil.Ux,
                                 (si_x-si_radius/2,
                                  si_y-si_radius/2,
                                  si_z-si_radius))
        u4 = g_o.getsingleresult(phas, soil.Ux,
                                 (si_x+si_radius/2,
                                  si_y-si_radius/2,
                                  si_z-si_radius))
        u = []
        for ui in [u1, u2, u3, u4]:
            try:
                v = float(ui)
                u.append(v if np.isfinite(v) else None)
            except Exception:
                u.append(None)
        valid_count = sum(v is not None for v in u)
        if valid_count == 4:   multipli = 1
        elif valid_count == 2: multipli = 2
        elif valid_count == 1: multipli = 4
        else: raise ValueError(f"Invalid SI data: {u}")
        print(f'Position{suffix} for SI is ({si_x}, {si_y}) '
              f'with multipli {multipli}.\n')

        for phase in phasesNeeded:
            variant_Felem_p  = deepcopy(variant_Felem)
            variant_Fsoil_p  = deepcopy(variant_Fsoil)
            variant_Fbase_p  = deepcopy(variant_Fbase)
            variant_Fshaft_p = deepcopy(variant_Fshaft)
            variant_tilted_p = deepcopy(variant_tilted)
            phaseGI = g_i.Phases[int(phase)]
            phaseName = phaseGI.Identification.value
            fct_updateTXTData(path, suffix,
                              f'\nFor {phaseName}:')
            if 'Failure' in phaseName:
                if 'FoS' in output:
                    fos = float(g_o.Phases[int(phase)].Reached.SumMsf.value)
                    Fsu_phases_df = fct_saveRefvalues(Fsu_phases_df, csv_path,
                                                      phase, phaseName,
                                                      fos=fos)
                    # curve_data = g_o.getcurves(g_o.Phases[int(phase)],
                    #                            g_o.ResultTypes.Soil.Utot,
                    #                            g_o.ResultTypes.Phases.SumMsf,
                    #                            "node")
                    # df_curve = pd.DataFrame(curve_data)
                    # fos_path = os.path.join(path, f'summary_value{suffix}',
                    #                         f'FoS_evaluation{suffix}.csv')
                    # df_curve.to_csv(fos_path, index=False, sep=';')
                    # plot_fosCurve(fos_path)
                else: fos = None
                continue

            if phase >= phaseSi:
                # variables to store temporary datasets
                F_times   = []
                F_results = []
                F_values  = []
                F_descr   = []
                try:
                    t0 = time.time()
                    center = np.array([si_x, si_y])
                    (si_top_int_outp_elem,
                     si_top_int_outp_soil) = fct_classify_interface(g_o,
                                                                    phase,
                                                                    si_top_int_outp,
                                                                    center,
                                                                    si_radius)
                    t_top = time.time() - t0
                except:
                    print(f"Interfacetransformation not possible:"
                          f"\n{si_top_int_outp}")
                    si_top_int_outp_elem = None
                    si_top_int_outp_soil = None
                try:
                    t0 = time.time()
                    (si_base_int_outp_elem,
                     _) = fct_classify_interface(g_o,
                                                 phase,
                                                 si_base_int_outp,
                                                 center,
                                                 si_radius)
                    t_base = time.time() - t0
                except:
                    print(f"Interfacetransformation not possible:"
                          f"\n{si_base_int_outp}")
                    si_base_int_outp_elem = None

                ### ELEM FORCE [3D]
                if 'Felem' in output:
                    F_times.clear()
                    F_results.clear()
                    F_values.clear()
                    F_descr.clear()
                    # print('Calculation for variant_Felem_p')
                    F_descr.clear()
                    typ       = 'elem'
                    subfolder = f'results_{typ}{suffix}'
                    var_i     = -1
                    # print(f'print1.0: {var_i} in {variant_Felem_p}')
                    if any(v in variant_Felem_p for v in [var_i+1,
                                                          var_i+2]):
                        # print(f'print1.1: {variant_Felem_p}')
                        if si_top_int_outp_elem is not None:
                            # print(f'print1.2: {si_top_int_outp_elem}')
                            t0 = time.time()
                            circleLimits = [si_x, si_y, si_radius+offset]
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp_elem)
                            dx = results['x'] - si_x
                            dy = results['y'] - si_y
                            mask = dx**2 + dy**2 <= (si_radius+offset)**2
                            results = results.loc[mask].copy() # just in case
                            area_i = math.pi * si_radius**2 / multipli
                            tc = time.time() - t0
                            # print(f'print1.3: {tc}')
                            var_i += 1
                            if var_i in variant_Felem_p: #0
                                # Get interface stresses
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                F_i, _ = fct_calc3D_surfIG(results,
                                                           's_n',
                                                           method = 'triangularCircle',
                                                           area = area_i,
                                                           boundaries = circleLimits,
                                                           pathSave = pathFile)
                                F_descr.append('due to interface stresses s_n '
                                               '(triangularCircle)')
                                F_results.append(results)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc + t_top)
                                # print(f'print1.4: {F_values}')
                            var_i += 1
                            if var_i in variant_Felem_p: #1
                                # Get interface stresses
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                           method = 'simple',
                                                           area = area_i,
                                                           boundaries = circleLimits,
                                                           pathSave = pathFile)
                                F_descr.append('due to interface stresses s_n '
                                               '(mean)')
                                F_results.append(results)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc + t_top)
                                # print(f'print1.5: {F_values}')
                        else:
                            print('si_top_int_outp_elem defined as None')
                            var_i += 1
                            if var_i in variant_Felem_p:#0
                                F_descr.append('due to interface stresses s_n '
                                               '(triangularCircle)')
                                F_results.append(None)
                                F_values.append(0*multipli)
                                F_times.append(0)
                            var_i += 1
                            if var_i in variant_Felem_p:
                                variant_Felem_p.remove(var_i) #1
                                if not variant_Felem_p:
                                    variant_Felem_p.append(var_i+1)
                    else: var_i += 2
                    var_i += 1
                    # print(f'print0.0: {var_i} in {variant_Felem_p}')
                    if var_i in variant_Felem_p: #2
                        try:
                            # print(f'print0.1: {var_i} in {variant_Felem_p}')
                            # Get interface stresses
                            t0 = time.time()
                            circleLimits = [si_x, si_y, si_radius+offset]
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp)
                            # print(f'print0.2: {si_top_int_outp}')
                            dx = results['x'] - si_x
                            dy = results['y'] - si_y
                            mask = dx**2 + dy**2 <= (si_radius+offset)**2
                            results = results.loc[mask].copy()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            # print(f'print0.3: {pathFile}')
                            F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                       method = 'triangularCircle',
                                                       area = area_i,
                                                       boundaries = circleLimits,
                                                       pathSave = pathFile)
                            F_descr.append('due to interface stresses s_n '
                                           'with limits '
                                           '(triangularCircle)')
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0)
                            # print(f'print0.4: {F_descr}')
                        except: # for Phase 1
                            print('si_top_int_outp not correct defined')
                            variant_Felem_p.remove(var_i)
                            if not variant_Felem_p:
                                variant_Felem_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Felem_p: #3
                        try:
                            F_descr.append('due to grid stresses under pile head '
                                           '(wissmann quadratur)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _, results = fct_getForcebyWissmann(g_o,
                                                                     int(phase),
                                                                     si_x,
                                                                     si_y,
                                                                     si_z-offset,
                                                                     si_radius,
                                                                     pathSave=pathFile)
                            F_results.append(results)
                            F_values.append(F_i)
                            F_times.append(time.time() - t0)
                        except:
                            print('fct_getForcebyWissmann not correct defined')
                            variant_Felem_p.remove(var_i)
                            if not variant_Felem_p:
                                variant_Felem_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Felem_p: #4
                        try:
                            F_descr.append('due to grid stresses under pile head '
                                           '(simpsonCircle21)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            boundaries = [si_x, si_y, si_radius]
                            results = fct_data3D_gridIP(g_o,
                                                        int(phase),
                                                        21, 21, si_z-offset,
                                                        boundaries)
                            F_results.append(results)
                            area_i = math.pi * si_radius**2 / multipli
                            F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                       method = 'simpsonCircle',
                                                       boundaries = boundaries,
                                                       area = area_i,
                                                       pathSave = pathFile)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0)
                        except:
                            print('fct_calc3D_surfIG sim21 not correct defined')
                            variant_Felem_p.remove(var_i)
                            if not variant_Felem_p:
                                variant_Felem_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Felem_p: #5
                        try:
                            F_descr.append('due to grid stresses under pile head '
                                           '(simpsonCircle15)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            boundaries = [si_x, si_y, si_radius]
                            results = fct_data3D_gridIP(g_o,
                                                        int(phase),
                                                        15, 15, si_z-offset,
                                                        boundaries)
                            F_results.append(results)
                            area_i = math.pi * si_radius**2 / multipli
                            F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                       method = 'simpsonCircle',
                                                       boundaries = boundaries,
                                                       area = area_i,
                                                       pathSave = pathFile)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0)
                        except:
                            print('fct_calc3D_surfIG sim15 not correct defined')
                            variant_Felem_p.remove(var_i)
                            if not variant_Felem_p:
                                variant_Felem_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Felem_p: #6
                        try:
                            F_descr.append('due to grid stresses under pile head '
                                           '(simpsonCircle11)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            boundaries = [si_x, si_y, si_radius]
                            results = fct_data3D_gridIP(g_o, int(phase),
                                                        11, 11, si_z-offset,
                                                        boundaries)
                            F_results.append(results)
                            area_i = math.pi * si_radius**2 / multipli
                            F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                       method = 'simpsonCircle',
                                                       boundaries = boundaries,
                                                       area = area_i,
                                                       pathSave = pathFile)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0)
                        except:
                            print('fct_calc3D_surfIG sim11 not correct defined')
                            variant_Felem_p.remove(var_i)
                            if not variant_Felem_p:
                                variant_Felem_p.append(var_i+1)
                    if any(v in variant_Felem_p for v in [var_i+1,
                                                          var_i+2]):
                        t0 = time.time()
                        boundaries = [si_x, si_y, si_radius+offset]
                        results = fct_data3D_horiCS(g_o,
                                                    int(phase),
                                                    si_z-offset,
                                                    boundaries)
                        area_i = math.pi * si_radius**2 / multipli
                        tc = time.time() - t0
                        var_i += 1
                        if var_i in variant_Felem_p: #7
                            try:
                                F_descr.append('due to crosssection stresses '
                                               'under pile head '
                                               '(triangularCircle)')
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                           method = 'triangularCircle',
                                                           area = area_i,
                                                           boundaries = boundaries,
                                                           pathSave = pathFile)
                                F_results.append(results)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc)
                            except:
                                print('fct_calc3D_surfIG triCS not correct defined')
                                variant_Felem_p.remove(var_i)
                                if not variant_Felem_p:
                                    variant_Felem_p.append(var_i+1)
                        var_i += 1
                        if var_i in variant_Felem_p: #8
                            try:
                                t0 = time.time()
                                F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                           method = 'simple',
                                                           area = area_i)
                                F_descr.append('due to crosssection stresses '
                                               'under pile head (mean)')
                                F_results.append(None)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc)
                            except:
                                print('fct_calc3D_surfIG meanIG not correct defined')
                                variant_Felem_p.remove(var_i)
                                if not variant_Felem_p:
                                    variant_Felem_p.append(var_i+1)
                    else:
                        var_i += 2
                    var_i += 1
                    if var_i in variant_Felem_p: #9
                        # only first layer will be considered
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material and first soil)')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           soil=si_soil[0])
                        F_results.append(results)
                        F_values.append(results.loc[results['z'].idxmax(), 'N']
                                        *multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Felem_p: #10
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material and soil)')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           soil=si_soil)
                        F_results.append(results)
                        F_values.append(results.loc[results['z'].idxmax(), 'N']
                                        *multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Felem_p: #11
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material, soil and offset line at '
                                       f'{si_x-offset} and {si_y-offset})')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           soil=si_soil,
                                                           line=[[si_x-offset,
                                                                  si_y-offset,
                                                                  si_z-offset],
                                                                 [si_x-offset,
                                                                  si_y-offset,
                                                                  si_z-0.3]])
                        F_results.append(results)
                        F_values.append(results.loc[results['z'].idxmax(), 'N']
                                        *multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Felem_p: #12
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material, soil and line at '
                                       f'{si_x} and {si_y})')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           soil=si_soil,
                                                           line=[[si_x,
                                                                  si_y,
                                                                  si_z-offset],
                                                                 [si_x,
                                                                  si_y,
                                                                  si_z-0.3]])
                        F_results.append(results)
                        F_values.append(results.loc[results['z'].idxmax(), 'N']
                                        *multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Felem_p: #13
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material and offset line at '
                                       f'{si_x-offset} and {si_y-offset})')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           line=[[si_x-offset,
                                                                  si_y-offset,
                                                                  si_z-offset],
                                                                 [si_x-offset,
                                                                  si_y-offset,
                                                                  si_z-0.3]])
                        F_results.append(results)
                        F_values.append(results.loc[results['z'].idxmax(), 'N']
                                        *multipli)
                        F_times.append(time.time() - t0)
                    if plotting:
                        fct_updateData(path, phase, suffix,
                                       f"F_{typ} [kN]", variant_Felem_p, F_values,
                                       F_times, F_descr, F_results,
                                       accuracy=accuracy)
                    F_elem = F_values[0]
                    s_elem = F_elem / (math.pi * si_radius**2)
                else:
                    F_elem = None
                    s_elem = None
    
                ### SOIL FORCE [3D]
                if 'Fsoil' in output:
                    F_values_elem = deepcopy(F_values)
                    F_descr_elem = [re.sub(r'\(([^)]*)\)',
                                           r' - \1 (force difference)',
                                           descr)
                                    for descr in F_descr
                                    ]
                    F_times.clear()
                    F_results.clear()
                    F_values.clear()
                    F_descr.clear()
                    s_i       = None
                    typ       = 'soil'
                    subfolder = f'results_{typ}{suffix}'
                    # control
                    t0 = time.time()
                    area_i = ((dis_x * dis_y) -
                              math.pi * si_radius**2/multipli)
                    f_load = loading[phase_to_index[phase]] # s_init
                    F_load = f_load * (dis_x * dis_y)
                    tc = time.time() - t0
                    # variants -> [100, 101, 102, 1, 2, 3]
                    variant_Fsoil_p = (list(range(100,
                                                  100 + len(F_values_elem)))
                                       + variant_Fsoil_p)
                    for F_value_elem, descr in zip(F_values_elem,
                                                   F_descr_elem):
                        t0 = time.time()
                        # print(F_load)
                        # print(F_elem)
                        F_i = F_load - F_value_elem
                        F_descr.append(descr)
                        F_results.append(None)
                        F_values.append([F_i, F_i/area_i])
                        F_times.append(time.time() - t0 + tc)
                    var_i     = -1
                    if any(v in variant_Fsoil_p for v in [var_i+1,
                                                          var_i+2]):
                        if si_top_int_outp_soil is not None:
                            t0 = time.time()
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp_soil,
                                                           boundaries=quadLimits)
                            # if plate is smaller than defined area (middle pile)
                            x_min_overlap = results['x'].min()
                            x_max_overlap = results['x'].max()
                            y_min_overlap = results['y'].min()
                            y_max_overlap = results['y'].max()
                            tc = time.time() - t0
                            var_i += 1
                            if var_i in variant_Fsoil_p: #0
                                # normal soil forces in the top interface
                                # (triangulate over area minus circle)
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                method = 'triangularRectangle'
                                F_i, s_i = fct_calc3D_surfIG(results, 's_n',
                                                                  method = method,
                                                                  hole = [si_x,
                                                                          si_y,
                                                                          si_radius],
                                                                  pathSave = pathFile)
                                F_results.append(results)
                                F_values.append([F_i*multipli, s_i])
                                F_descr.append('due to soil interface '
                                               'stresses s_n '
                                               '(triangularRectangle)')
                                F_times.append(time.time() - t0 + tc + t_top)
                            var_i += 1
                            if var_i in variant_Fsoil_p: #1
                                # normal soil forces in the top interface
                                # (overall mean value)
                                t0 = time.time()
                                area_i = (((x_max_overlap-x_min_overlap) *
                                           (y_max_overlap-y_min_overlap)) -
                                          math.pi * si_radius**2/multipli)
                                F_i, s_i = fct_calc3D_surfIG(results, 's_n',
                                                                  method = 'simple',
                                                                  area = area_i)
                                F_results.append(results)
                                F_values.append([F_i*multipli, s_i])
                                F_descr.append('due to soil interface '
                                               'stresses s_n (mean)')
                                F_times.append(time.time() - t0 + tc + t_top)
                        else:
                            var_i += 1
                            if var_i in variant_Fsoil_p: #0
                                F_descr.append('due to interface stresses s_n '
                                               '(triangularRectangle)')
                                F_results.append(None)
                                F_values.append(0*multipli)
                                F_times.append(0)
                            var_i += 1
                            if var_i in variant_Fsoil_p:
                                variant_Fsoil_p.remove(var_i) #1
                    else: var_i += 2
                    var_i += 1
                    if var_i in variant_Fsoil_p: #2
                        try:
                            # normal soil forces in the top interface
                            # (triangulate over area minus circle)
                            t0 = time.time()
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_top_int_outp,
                                                           boundaries=quadLimits)
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            method = 'triangularRectangle'
                            F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                       method = method,
                                                       pathSave = pathFile)
                            F_i = F_i*multipli - F_elem
                            s_i = F_i / ((quadLimits[1]-quadLimits[0])*
                                         (quadLimits[3]-quadLimits[2])-
                                         math.pi * si_radius**2)
                            F_results.append(results)
                            F_values.append([F_i, s_i])
                            F_descr.append('due to top interface stresses s_n '
                                           '(triangularRectangle - Felem)')
                            F_times.append(time.time() - t0)
                        except: # for Phase 1
                            print('si_top_int_outp not correct defined')
                            variant_Fsoil_p.remove(var_i)
                            if not variant_Fsoil_p:
                                # if nothing was considered instead
                                variant_Fsoil_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Fsoil_p: #3
                        # sigmaZZ for the horizontal cut in the base slab
                        # -> simpson with 21 elements
                        t0 = time.time()
                        results = fct_data3D_gridIP(g_o,
                                                    int(phase),
                                                    21, 21, si_top_depth-offset,
                                                    quadLimits)
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        method = 'simpsonRectangle'
                        F_total_i, s_total_i = fct_calc3D_surfIG(results,
                                                                 'sig_zz',
                                                 method = method,
                                                 pathSave = pathFile)
                        # if testing:
                        #    print(f'F_total_i for 21 simpson grid is {F_total_i}')
                        F_i  = F_total_i*multipli - F_elem
                        s_i  = s_total_i - s_elem
                        F_descr.append('due to grid stresses sig_zz '
                                       '(simpsonRectangle21)')
                        F_results.append(results)
                        F_values.append([F_i, s_i])
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Fsoil_p: #4
                        # sigmaZZ for the horizontal cut in the base slab
                        # -> simpson with 11 elements
                        t0 = time.time()
                        results = fct_data3D_gridIP(g_o,
                                                    int(phase),
                                                    11, 11, si_top_depth-offset,
                                                    quadLimits)
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        method = 'simpsonRectangle'
                        F_total_i, s_total_i = fct_calc3D_surfIG(results,
                                                 'sig_zz',
                                                 method = method,
                                                 pathSave = pathFile)
                        F_i  = F_total_i - F_elem/multipli
                        s_i  = s_total_i - s_elem
                        F_descr.append('due to grid stresses sig_zz '
                                       '(simpsonRectangle11)')
                        F_results.append(results)
                        F_values.append([F_i*multipli, s_i])
                        F_times.append(time.time() - t0)
                    if any(v in variant_Fsoil_p for v in [var_i+1,
                                                          var_i+2]):
                        t0 = time.time()
                        results = fct_data3D_horiCS(g_o,
                                                    int(phase),
                                                    si_top_depth-offset,
                                                    boundaries = quadLimits)
                        x_min_overlap = results['x'].min()
                        x_max_overlap = results['x'].max()
                        y_min_overlap = results['y'].min()
                        y_max_overlap = results['y'].max()
                        # if plate is smaller than defined area (corner pile)
                        tc = time.time() - t0
                        var_i += 1
                        if var_i in variant_Fsoil_p: #5
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            method = 'triangularRectangle'
                            F_total_i, s_total_i = fct_calc3D_surfIG(results,
                                                   'sig_zz',
                                                   method = method,
                                                   pathSave = pathFile)
                            F_i  = F_total_i*multipli - F_elem
                            s_i  = s_total_i - s_elem
                            F_descr.append('due to crosssection stresses sig_zz '
                                           '(triangularRectangle)')
                            F_results.append(results)
                            F_values.append([F_i, s_i])
                            F_times.append(time.time() - t0 + tc)
                        var_i += 1
                        if var_i in variant_Fsoil_p: #6
                            # sigmaZZ for the horizontal cut in the base slab
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_total_i, s_total_i = fct_calc3D_surfIG(results,
                                                   'sig_zz',
                                                   method = 'simple',
                                                   pathSave = pathFile)
                            F_i  = F_total_i - F_elem/multipli
                            s_i  = s_total_i - s_elem
                            F_descr.append('due to crosssection stresses sig_zz '
                                           '(mean)')
                            F_results.append(results)
                            F_values.append([F_i*multipli, s_i])
                            F_times.append(time.time() - t0 + tc)
                    else: var_i += 2
                    var_i += 1
                    if var_i in variant_Fsoil_p: #7
                        # soil needs to be clustered around the soil improvement
                        g_i.gotostructures()
                        soil_name = g_i.Soillayers[0].Soil.Material.Name.value
                        soil_outp = next(material
                                         for material in g_o.Materials
                                         if material.Name.value == soil_name)
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material)')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o, int(phase),
                                                           material=soil_outp)
                        F_results.append(results)
                        F_i = results.loc[results['z'].idxmax(), 'N']
                        area_i = ((dis_x * dis_y) -
                                  math.pi * si_radius**2/multipli)
                        F_values.append([F_i, F_i/area_i])
                        F_times.append(time.time() - t0)
                        g_i.gotostages()
                    if plotting:
                        fct_updateData(path, phase, suffix,
                                       f"F_{typ} [kN]", variant_Fsoil_p,
                                       [row[0] for row in F_values],
                                       F_times, F_descr, F_results,
                                       accuracy=accuracy)
                    F_soil = F_values[0][0]
                    s_soil = F_values[0][1]
                else:
                    F_soil = None
                    s_soil = None

                ### BASE FORCE [3D]
                if 'Fbase' in output:
                    F_times.clear()
                    F_results.clear()
                    F_values.clear()
                    F_descr.clear()
                    # print('Calculation for variant_Fbase_p')
                    area_i    = math.pi * si_radius**2 / multipli
                    typ       = 'base'
                    subfolder = f'results_{typ}{suffix}'
                    var_i     = -1
                    if any(v in variant_Fbase_p for v in [var_i+1,
                                                          var_i+2]):
                        if si_base_int_outp_elem is not None:
                            t0 = time.time()
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_base_int_outp_elem)
                            dx = results['x'] - si_x
                            dy = results['y'] - si_y
                            mask = dx**2 + dy**2 <= (si_radius+offset)**2
                            results = results.loc[mask].copy()
                            tc = time.time() - t0
                            var_i += 1
                            if var_i in variant_Fbase_p: #0
                                # Get interface stresses along the base
                                F_descr.append('due to interface stresses s_n '
                                               '(triangularCircle)')
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                           method = 'triangularCircle',
                                                           area = area_i,
                                                           pathSave = pathFile)
                                F_results.append(results)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc + t_base)
                            var_i += 1
                            if var_i in variant_Fbase_p: #1
                                # Get interface stresses along the base
                                F_descr.append('due to interface stresses s_n '
                                               '(mean)')
                                t0 = time.time()
                                dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                                pathFile = os.path.join(path,
                                                        subfolder,
                                                        f'{subfolder}_p{phase}',
                                                        dataname)
                                F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                           method = 'simple',
                                                           area = area_i,
                                                           pathSave = pathFile)
                                F_results.append(results)
                                F_values.append(F_i*multipli)
                                F_times.append(time.time() - t0 + tc + t_base)
                        else:
                            var_i += 1
                            if var_i in variant_Fbase_p: #0
                                F_descr.append('due to interface stresses s_n '
                                               '(triangularCircle)')
                                F_results.append(None)
                                F_values.append(0*multipli)
                                F_times.append(0)
                            var_i += 1
                            if var_i in variant_Fbase_p: #1
                                variant_Fbase_p.remove(var_i)
                    else: var_i += 2
                    var_i += 1
                    if var_i in variant_Fbase_p: #5
                        try:
                            # Get interface stresses
                            t0 = time.time()
                            circleLimits = [si_x, si_y, si_radius+offset]
                            results = fct_data3D_interface(g_o,
                                                           int(phase),
                                                           si_base_int_outp)
                            dx = results['x'] - si_x
                            dy = results['y'] - si_y
                            mask = dx**2 + dy**2 <= (si_radius+offset)**2
                            results = results.loc[mask].copy()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _ = fct_calc3D_surfIG(results, 's_n',
                                                       method = 'triangularCircle',
                                                       area = area_i,
                                                       boundaries = circleLimits,
                                                       pathSave = pathFile)
                            F_descr.append('due to interface stresses s_n '
                                           'with limits '
                                           '(triangularCircle)')
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0)
                        except: # for Phase 1
                            print('si_base_int_outp not correct defined')
                            variant_Fbase_p.remove(var_i)
                            if not variant_Fbase_p:
                                variant_Fbase_p.append(var_i+1)
                    var_i += 1
                    if var_i in variant_Fbase_p: #0
                        F_descr.append('due to grid stresses at pile base '
                                       '(wissmann quadratur)')
                        t0 = time.time()
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i, _, results = fct_getForcebyWissmann(g_o,
                                                                 int(phase),
                                                                 si_x,
                                                                 si_y,
                                                                 si_base_depth,
                                                                 si_radius,
                                                                 pathSave=pathFile)
                        F_results.append(results)
                        F_values.append(F_i)
                        F_times.append(time.time() - t0)
                    circleLimits = [si_x, si_y, si_radius-offset]
                    var_i += 1
                    if var_i in variant_Fbase_p: #1
                        # Simpson innerhalb einen Kreises anwenden
                        F_descr.append('due to grid stresses at pile base '
                                       '(simpsonCircle21)')
                        t0 = time.time()
                        results = fct_data3D_gridIP(g_o, int(phase),
                                                    21, 21,
                                                    si_base_depth,
                                                    circleLimits)
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                   method = 'simpsonCircle',
                                                   boundaries = circleLimits,
                                                   area = area_i,
                                                   pathSave = pathFile)
                        F_results.append(results)
                        F_values.append(F_i*multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Fbase_p: #2
                        # Simpson innerhalb einen Kreises anwenden
                        F_descr.append('due to grid stresses at pile base '
                                       '(simpsonCircle11)')
                        t0 = time.time()
                        results = fct_data3D_gridIP(g_o, int(phase),
                                                    11, 11,
                                                    si_base_depth,
                                                    circleLimits)
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                   method = 'simpsonCircle',
                                                   boundaries = circleLimits,
                                                   area = area_i,
                                                   pathSave = pathFile)
                        F_results.append(results)
                        F_values.append(F_i*multipli)
                        F_times.append(time.time() - t0)
                    var_i += 1
                    if var_i in variant_Fbase_p: #3
                        # Simpson innerhalb eines Kreises anwenden
                        F_descr.append('due to grid stresses at pile base '
                                       '(simpsonCircle5)')
                        t0 = time.time()
                        results = fct_data3D_gridIP(g_o, int(phase),
                                                    5, 5,
                                                    si_base_depth,
                                                    circleLimits)
                        dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                        pathFile = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                   method = 'simpsonCircle',
                                                   boundaries = circleLimits,
                                                   area = area_i,
                                                   pathSave = pathFile)
                        F_results.append(results)
                        F_values.append(F_i*multipli)
                        F_times.append(time.time() - t0)
                    if any(v in variant_Fbase_p for v in [var_i+1,
                                                          var_i+2]):
                        t0 = time.time()
                        circleLimits = [si_x, si_y, si_radius]
                        results = fct_data3D_horiCS(g_o, int(phase),
                                                    si_base_depth,
                                                    circleLimits)
                        tc = time.time() - t0
                        var_i += 1
                        if var_i in variant_Fbase_p: #3
                            # Get crosssection stresses along the base
                            F_descr.append('due to crosssection stresses '
                                           'at pile base (triangularCircle)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                       method = 'triangularCircle',
                                                       area = area_i,
                                                       boundaries = circleLimits,
                                                       pathSave = pathFile)
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0 + tc)
                        var_i += 1
                        if var_i in variant_Fbase_p: #4
                            # Get crosssection stresses along the base
                            F_descr.append('due to crosssection stresses '
                                           'at pile base (mean)')
                            t0 = time.time()
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _ = fct_calc3D_surfIG(results, 'sig_zz',
                                                       method = 'simple',
                                                       area = area_i,
                                                       boundaries = circleLimits,
                                                       pathSave = pathFile)
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0 + tc)
                    else: var_i += 2
                    var_i += 1
                    if var_i in variant_Fbase_p: #8
                        F_descr.append('due to InternalForceAnalysis '
                                       '(material and last soil)')
                        t0 = time.time()
                        results = fct_data3D_InternalForce(g_o,
                                                           int(phase),
                                                           material=si_material,
                                                           soil=si_soil[-1])
                        F_i = results.loc[results['z'].idxmin(), 'N']
                        F_results.append(results)
                        F_values.append(F_i*multipli)
                        F_times.append(time.time() - t0)
                    if plotting:
                        fct_updateData(path, phase, suffix,
                                       f"F_{typ} [kN]", variant_Fbase_p,
                                       F_values,
                                       F_times, F_descr, F_results,
                                       accuracy=accuracy)
                    F_base = F_values[0]
                else:
                    F_base = None

                ### SHAFT FORCE [3D]
                if 'Fshaft' in output:
                    F_times.clear()
                    F_results.clear()
                    F_values.clear()
                    F_descr.clear()
                    # print('Calculation for variant_Fshaft_p')
                    typ       = 'shaft'
                    subfolder = f'results_{typ}{suffix}'
                    # Get stresses along the shaft due to force reduction
                    F_descr.append('due to stresses from var=0 '
                                   'above and below pile (force difference)')
                    F_results.append(None)
                    t0 = time.time()
                    F_pile    = math.pi * si_radius**2 * si_height * si_weight
                    F_values.append(F_elem - F_base - F_pile)
                    F_times.append(time.time() - t0)
                    variant_Fshaft_p = [100] + variant_Fshaft_p
                    var_i     = -1
                    if any(v in variant_Fshaft_p for v in [var_i+1,
                                                           var_i+2]):
                        t0 = time.time()
                        results = fct_data3D_interface(g_o,
                                                       int(phase),
                                                       si_shaft_int_outp)
                        mask = results['z'] >= si_base_depth
                        results = results.loc[mask]
                        mask = results['z'] <= si_top_depth
                        results = results.loc[mask]
                        si_height = (results['z'].max() - results['z'].min())
                        area_shaft = 2*math.pi * si_radius*si_height / multipli
                        tc = time.time() - t0
                        var_i += 1
                        if var_i in variant_Fshaft_p: #0
                            F_descr.append('due to interface stresses s_1 '
                                           '(triangularShaft)')
                            t0 = time.time()
                            shaftLimits = [si_x, si_y,
                                           si_top_depth,
                                           si_base_depth,
                                           si_radius]
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _ = fct_calc3D_surfIG(results, 's_1',
                                                       method = 'triangularShaft',
                                                       area = area_shaft,
                                                       boundaries = shaftLimits,
                                                       pathSave = pathFile)
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0 + tc)
                        var_i += 1
                        if var_i in variant_Fshaft_p: #1
                            F_descr.append('due to interface stresses s_1 '
                                           '(mean)')
                            t0 = time.time()
                            F_i, _ = fct_calc3D_surfIG(results, 's_1',
                                                       method = 'simple',
                                                       area = area_shaft)
                            F_results.append(results)
                            F_values.append(F_i*multipli)
                            F_times.append(time.time() - t0 + tc)
                    else: var_i += 2
                    var_i += 1
                    if var_i in variant_Fshaft_p: #2
                        try: # necessary because failure can occur
                            shaftLimits = [si_x, si_y,
                                           si_top_depth - offset,
                                           si_base_depth + offset,
                                           si_radius]
                            results = fct_data3D_gridIP(g_o,
                                                        int(phase),
                                                        21, 21, None,
                                                        shaftLimits)
                            si_height = abs(si_base_depth - si_top_depth)
                            area_shaft = 2*math.pi * si_radius*si_height / multipli
                            dataname = f'{subfolder}_var{var_i}_p{phase}.pdf'
                            pathFile = os.path.join(path,
                                                    subfolder,
                                                    f'{subfolder}_p{phase}',
                                                    dataname)
                            F_i, _ = fct_calc3D_surfIG(results, 'shear',
                                                       method = 'simpsonShaft',
                                                       boundaries = shaftLimits,
                                                       area = area_shaft,
                                                       pathSave = pathFile)
                            F_results.append(results)
                            F_values.append(-F_i*multipli) # always pos. output
                            F_descr.append('due to grid stresses sig_zz '
                                           '(simpsonShaft21)')
                            F_times.append(time.time() - t0)
                        except Exception as e:
                            print(f'[ERROR] simpsonShaft21 failed: {e}')
                            traceback.print_exc()
                            variant_Fshaft_p.remove(var_i)
                    if plotting:
                        fct_updateData(path, phase, suffix,
                                       f"F_{typ} [kN]", variant_Fshaft_p,
                                       F_values,
                                       F_times, F_descr, F_results,
                                       accuracy=accuracy)
                    F_shaft = F_values[0]
                else: F_shaft=None

                F_times.clear()
                F_results.clear()
                F_values.clear()
                F_descr.clear()

            else:
                F_elem = None
                s_elem = None
                F_soil = None
                s_soil = None
                F_base = None
                F_shaft = None

            ### Get design values for each phase [3D]
            typ = 'constraints'
            variants = []
            if ('tilt' in output and
                phase >= phaseSlab): # also get entire surface at once
                # 'total' for the first suffix since for all suffixes the same
                if (count == 0 and
                    dimension not in ['unitcell3']):
                    quadLimitsMulti = [
                                       ('_total',
                                        [0,xmax, 0,ymax]),
                                       ('', quadLimits)
                                       ]
                else: quadLimitsMulti = [('',quadLimits)]
                dx = 0.01 # 1cm
                for name, quadLimit in quadLimitsMulti:
                    nx_f = (quadLimit[1]-quadLimit[0])/dx
                    ny_f = (quadLimit[3]-quadLimit[2])/dx
                    methods = []
                    results = {}
                    quadLimitOff = [quadLimit[0] - offset,
                                    quadLimit[1] + offset,
                                    quadLimit[2] - offset,
                                    quadLimit[3] + offset]
                    var_i   = 0
                    if var_i in variant_tilted_p: #0
                        methods.append(("CS", (slab_top-offset,
                                               quadLimitOff)))
                    var_i += 1
                    if var_i in variant_tilted_p: #1
                        methods.append(("line", (si_x, si_y,
                                                 slab_top-offset,
                                                 dis_x, dis_y,
                                                 dx, 2)))
                    var_i += 1
                    if var_i in variant_tilted_p: #2
                        methods.append(("grid", (math.ceil((nx_f-1)/2)*2+1,
                                                 math.ceil((ny_f-1)/2)*2+1,
                                                 slab_top-offset,
                                                 quadLimit)))
                    var_i += 1
                    if var_i in variant_tilted_p: #3
                        if slab_int_top is not None:
                            methods.append(("IF", (slab_int_outp,
                                                   quadLimitOff)))
                        else: variant_tilted_p.remove(var_i)
                    subfolder = f'results_tilt{name}{suffix}'
                    # print(methods)
                    for countTilt, (method, inputdata) in enumerate(methods):
                        dataname = f'{subfolder}_tilt{method}_p{phase}.pdf'
                        pathSave = os.path.join(path,
                                                subfolder,
                                                f'{subfolder}_p{phase}',
                                                dataname)
                        (settlement, settlementDiff, tilt, tiltDiffMax,
                         distortion, relativeDeflection, deflectionRatio,
                         time_calc) = fct_calc3D_tilt(g_o,
                                                      int(phase),
                                                      method,
                                                      inputdata,
                                                      pathSave=pathSave)
                        if countTilt == 0: #CS
                            s  = settlement
                            sd = settlementDiff
                            t  = tilt
                            td = tiltDiffMax
                            b  = distortion
                        results[method] = {
                            "Settlement [m]":            settlement,
                            "Settlement difference [m]": settlementDiff,
                            "Tilt [%]":                  tilt,
                            "Maximum tilt [%]":          tiltDiffMax,
                            "Distortion [%]":            distortion,
                            "Relative Deflection [m]":   relativeDeflection,
                            "Deflection Ratio [-]":      deflectionRatio,
                            "Time [sec]":                time_calc
                            }
                        variants.append(countTilt)
                        F_times.append(time_calc)
                        F_values.append([settlement,
                                         settlementDiff,
                                         tilt,
                                         tiltDiffMax,
                                         distortion,
                                         relativeDeflection,
                                         deflectionRatio])
                        F_descr.append([
                            f'settlement values due to {method}{name}',
                            f'diff. settl. values due to {method}{name}',
                            f'tilt values due to {method}{name}',
                            f'diff. tilt values due to {method}{name}',
                            f'distortion values due to {method}{name}',
                            f'relativeDefl values due to {method}{name}',
                            f'deflectionRatio values due to {method}{name}'])
                    df = pd.DataFrame(results).T
                    df.loc["Mean"] = df.mean(axis=0)
                    os.makedirs(os.path.join(path, subfolder),
                                exist_ok=True)
                    df.to_csv(os.path.join(path, subfolder,
                              f"{subfolder}_tilt_p{phase}.csv"),
                              sep=";")
                if plotting:
                    for index, name in [[0, 'settl [m]'],
                                        [1, 'settlDiff [m]'],
                                        [2, 'tilt [%]'],
                                        [3, 'tiltDiff [%]'],
                                        [4, 'distortion [%]'],
                                        [5, 'relativeDeflection [m]'],
                                        [6, 'deflectionRatio [-]'],
                                        ]:
                        fct_updateData(path, phase, suffix, name, variants,
                                       [row[index] for row in F_values],
                                       F_times,
                                       [row[index] for row in F_descr],
                                       accuracy=accuracy)
            else:
                s=None
                sd=None
                t=None
                td=None
                b=None

            F_times.clear()
            F_results.clear()
            F_values.clear()
            F_descr.clear()
            variants.clear()

            # what is the dummy plate
            if 'moment' in output:
                for plate in g_i.Plates:
                    try:
                        mat_name = plate.Material.value.Identification.value
                        if 'DummySlab' in mat_name:
                            if testing:
                                print(f'{plate.Name.value}: {mat_name} '
                                      f'for phase {phase}')
                            target_plate = plate
                            break
                    except Exception as e:
                        if testing: print(f'Plate Error: {e}')
                        target_plate = None
            if ('moment' in output and
                target_plate is not None):
                try:
                    t0 = time.time()
                    subfolder = f'summary_plot{suffix}'
                    pathFile = os.path.join(path, subfolder)
                    moment = fct_plateMoment(g_o, g_i,
                                             dimension,
                                             phaseSlab,
                                             int(phase),
                                             target_plate,
                                             si_x, si_y,
                                             dis_x, dis_y,
                                             path_save=pathFile,
                                             si_r = si_radius)
                    F_times.append(time.time() - t0)
                    F_descr.append('moment values due to dummy plate')
                    if plotting:
                        if testing:
                            print(target_plate)
                            print(moment)
                            print(variants)
                            print(F_values)
                            print(F_times)
                            print(F_descr)
                        fct_updateData(path, phase, suffix,
                                       'M_global', [0],
                                       [moment['Mmax_global']],
                                       F_times, F_descr,
                                       accuracy=accuracy)
                        fct_updateData(path, phase, suffix,
                                       'M_local', [0],
                                       [moment['Mmax_local']],
                                       F_times, F_descr,
                                       accuracy=accuracy)
                except Exception as e:
                    print(f'EVAL Moment Error: {e}')
                    traceback.print_exc()
                    moment = {"Mmax_global": None,
                              "Mmax_local": None}
            else: moment = {"Mmax_global": None,
                            "Mmax_local": None}
            # if 'settlements' in output:
            #     # Get settlement for different positions [3D]
            #     ls_curve = fct_data3D_settlement(g_o, int(phase), count)
            #     subfolder = 'summary_value_Mstage'
            #     dataname = f'results_Mstage{suffix}_p{phase}.csv'
            #     pathFile = os.path.join(path, subfolder,
            #                    f'{subfolder}_p{phase}', dataname)
            #     os.makedirs(os.path.dirname(pathFile), exist_ok=True)
            #     ls_curve.to_csv(pathFile, index=False, sep=';')
            #     # Get settlement above SI for each phase
            #     u = g_o.getcurveresults(g_o.Curvepoints[count],
            #                             g_o.Phases[int(phase)],
            #                             g_o.ResultTypes.Soil.Uz)
            # else: u = None

            F_times.clear()
            F_results.clear()
            F_values.clear()
            F_descr.clear()
            variants.clear()

            if 'Npl' in output:
                t0 = time.time()
                state = g_o.getresults(g_o.Phases[int(phase)],
                                       g_o.ResultTypes.Soil.PlasticPoint,
                                       "Stresspoint")
                Npl = int(sum(v == 4 for v in state))
                variants.append(1) # since reindexing only valid if unique
                F_times.append(time.time() - t0)
                F_values.append(Npl)
                F_descr.append('plastic points')
                if plotting:
                    fct_updateData(path, phase, suffix,
                                   'N_plastic', variants, F_values,
                                   F_times, F_descr,
                                   accuracy=accuracy)
            else: Npl = None

            ### Start values
            fct_updateData(path, phase, suffix,
                           'spacer', [], [], [], [],
                           last=True)
            s_init    = loading[phase_to_index[phase]]
            load_step = s_init + concrete_weight if s_init < 0 else 0

            ### 3D - Summaries and recalculate values
            # Reference Value
            if ('Fshaft' in output or
                'Fbase' in output or
                'Felem' in output or
                'Fsoil' in output
                ):
                if phase == phaseStart: # reference phase currently calculated
                    [F_shaft_0, F_base_0]           = [F_shaft, F_base]
                    [F_elem_0,  F_soil_0]           = [F_elem,  F_soil]
                    [s_elem_0,  s_soil_0, s_init_0] = [s_elem,  s_soil, s_init]
                    if phaseStart == phaseSi:
                        fct_updateData(path, phase, suffix, "F_pile [kN]", [0,1],
                                       [F_base_0+F_shaft_0, -F_pile], [0,0],
                                       ['due to first F_base+F_shaft value',
                                        'due to F_pile calculation'],
                                       accuracy=accuracy)
                else: # reference phase was already calculated
                    ref_row = Fsu_phases_df[Fsu_phases_df['Phase'] == phaseStart]
                    if ref_row.empty:
                        raise ValueError(f'Ref.-Phase {phaseStart} not found.')
                    if 'Fshaft' in output:
                        F_shaft_0 = ref_row['F_shaft'].iloc[0]
                    if 'Fbase' in output:
                        F_base_0  = ref_row['F_base'].iloc[0]
                    if 'Felem' in output:
                        F_elem_0  = ref_row['F_elem'].iloc[0]
                    if 'Fsoil' in output:
                        F_soil_0  = ref_row['F_soil'].iloc[0]
                    s_elem_0  = ref_row['s_elem'].iloc[0]
                    s_soil_0  = ref_row['s_soil'].iloc[0]
                    s_init_0  = ref_row['s_init'].iloc[0]
            else:
                F_elem_0 = None
                F_soil_0 = None
                F_base_0 = None
                F_shaft_0 = None
                s_elem_0 = None
                s_soil_0 = None
                s_init_0 = None
            # Join Dataset
            Fsu_phases_df = fct_saveRefvalues(Fsu_phases_df,
                                              csv_path,
                                              phase,
                                              phaseName,
                                              load_step,
                                              s,sd,t,td,b,
                                              Npl,
                                              moment['Mmax_global'],
                                              moment['Mmax_local'],
                                              F_elem,  F_elem_0,
                                              F_soil,  F_soil_0,
                                              F_base,  F_base_0,
                                              F_shaft, F_shaft_0,
                                              s_elem,  s_elem_0,
                                              s_soil,  s_soil_0,
                                              s_init,  s_init_0)
        close_plaxis(port_out)
    return g_o

if __name__ == "__main__":
    pathFolder   = 'Optimization - 2026-09-26 - Sensitivity Study'
    device       = 27
    name = 'model_base';    val_test = False;
    name = 'model_totalF';  val_test = True;
    name = 'model_simpson'; val_test = True;
    name = 'model_base';    val_test = True;
    name = 'eval_new1';     val_test = True;
    name = 'eval_act';      val_test = True;
    name = 'mesh_medium';       val_test = True;
    name = 'mesh_fineMedium';   val_test = True;
    nameGen = 'meshingFinerM';         val_test = True;
    for dimension in  [
                       # 'unitcell2',
                       'unitcell3',
                       ]:
        for name in [
                     f'{nameGen}_medium',
                     f'{nameGen}_fineMedium',
                     f'{nameGen}_fine',
                     ]:
            ip, pw, path = open_plaxisData(device, dimension, pathFolder)
            path_data    = os.path.join(path, f'{name}_{dimension}')
            model        = r'Model'
            fct_eval(path_data, model, dimension, ip, pw,
                     output = ['moment'],
                     testing=True)