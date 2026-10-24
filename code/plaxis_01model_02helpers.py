# -*- coding: utf-8 -*-
'''
Created on Mon Apr 13 11:44:42 2026
@author: annaleoniesalzmann
'''

differentDeformationDefault = True # if true also deactive surface displacements
thinfirstlayer_Default      = False # 'force in volume' still wrong
excv_stepDefault            = 1.0 # reduce if soil collapses while excav. [m]

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import numpy as np
from plaxis_00calling_02helpers import fct_formatSave
from plaxis_03plot_02helpers import fct_setPlottext
import os
import matplotlib.pyplot as plt
import math
from copy import deepcopy

###############################################################################
#-------------------------- Helper for fct_modelModeller ----------------------
###############################################################################

def fct_positionsGet(min_val, edge, spacing, n):
    return [min_val + edge + i * spacing for i in range(n)]

def fct_positionsBuild(minv, maxv, num, disedge, dis):
    if num % 2 == 1: center = (minv + maxv) / 2
    else:            center = (minv + maxv) / 2 - dis / 2
    if num <= 2:     return [center]
    else:            return [center, minv + disedge]

# Parameter Transformation ----------------------------------------------------
def fct_setProperties(g_i, soil_layer):
    m_models = ['LE', 'MC', 'HS', 'HSS','CM','SS']
    soil = g_i.soilmat()

    # Linear-elastic
    if soil_layer['Model'] == m_models[0]: # LE
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       "linearelastic",
                           'DrainageType',    soil_layer['Drainage'],
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'Eref',            soil_layer['ERef'],
                           'nu',              soil_layer['nu'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter'])
        if soil_layer['Drainage'] != 'Non-porous':
            soil.setproperties('gammasat',    soil_layer['gammasat'])

    # Mohr-Coloumb
    elif soil_layer['Model'] == m_models[1]: # MC
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       "mohrcoulomb",
                           'DrainageType',    soil_layer['Drainage'],
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'gammasat',        soil_layer['gammasat'],
                           'Eref',            soil_layer['ERef'],
                           'nu',              soil_layer['nu'],
                           'cref',            soil_layer['cRef'],
                           'phi',             soil_layer['phi'],
                           'psi',             soil_layer['psi'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter'])

    # Hardening Soil
    elif soil_layer['Model'] == m_models[2]: # HS
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       "hardeningsoil",
                           'DrainageType',    soil_layer['Drainage'],
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'gammasat',        soil_layer['gammasat'],
                           'E50ref',          soil_layer['E50Ref'],
                           'EOedRef',         soil_layer['EoedRef'],
                           'EURRef',          soil_layer['EurRef'],
                           'PowerM',          soil_layer['powerm'],
                           'cref',            soil_layer['cRef'],
                           'phi',             soil_layer['phi'],
                           'psi',             soil_layer['psi'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter'],
                           'K0Determination', 'Manual',
                           'K0Primary',       soil_layer['K0'],
                           'POP',             soil_layer['POP'])

    # Hardening Soil Small
    elif soil_layer['Model'] == m_models[3]: # HSS
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       "hssmall",
                           'DrainageType',    soil_layer['Drainage'],
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'gammaSat',        soil_layer['gammasat'],
                           'E50ref',          soil_layer['E50Ref'],
                           'EOedRef',         soil_layer['EoedRef'],
                           'EURRef',          soil_layer['EurRef'],
                           'PowerM',          soil_layer['powerm'],
                           'cRef',            soil_layer['cRef'],
                           'phi',             soil_layer['phi'],
                           'psi',             soil_layer['psi'],
                           'G0Ref',           soil_layer['G0Ref'],
                           'gamma07',         soil_layer['gamma07'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter'],
                           'K0Determination', 'Manual',
                           'K0Primary',       soil_layer['K0'],
                           'POP',             soil_layer['POP'])

    # Concrete Model
    elif soil_layer['Model'] == m_models[4]:
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       "concrete",
                           'DrainageType',    'Non-porous',
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'E28',             soil_layer['E28'],
                           'nu',              soil_layer['nu'],
                           'fC28',            soil_layer['fC28'],
                           'fC0N',            soil_layer['fC0N'],
                           'fCFN',            soil_layer['fCFN'],
                           'fCUN',            soil_layer['fCUN'],
                           'phiMax',          soil_layer['phiMax'],
                           'epsCPP',          soil_layer['epsCPP'],
                           'aDuct',           soil_layer['aDuct'],
                           'fT28',            soil_layer['fT28'],
                           'GC28',            soil_layer['GC28'],
                           'GT28',            soil_layer['GT28'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter'])

    # Soft Soil Model
    elif soil_layer['Model'] == 'SS':
        lambda_mod = soil_layer['Cc'] / ( 2.3 * (1 + soil_layer['einit']))
        kappa_mod = 2 * soil_layer['Cs'] / (2.3 * (1 + soil_layer['einit']))
        soil.setproperties('Identification',  soil_layer['Name'],
                           'Colour',          soil_layer['Colour'],
                           'SoilModel',       5,
                           'DrainageType',    soil_layer['Drainage'],
                           'gammaUnsat',      soil_layer['gammaunsat'],
                           'gammaSat',        soil_layer['gammasat'],
                           'eInit',           soil_layer['einit'],
                           'lambdaModified',  lambda_mod,
                           'kappaModified',   kappa_mod,
                           'nuUR',            soil_layer['nu'],
                           'cRef',            soil_layer['cRef'],
                           'phi',             soil_layer['phi'],
                           'psi',             soil_layer['psi'],
                           'OCR',             soil_layer['OCR'],
                           'K0Determination', 'Manual',
                           'K0PrimaryIsK0Secondary', True,
                           'K0Primary',       soil_layer['K0'],
                           'InterfaceStrengthDetermination', 'Manual',
                           'Rinter',          soil_layer['Rinter']
                           )

    else: raise ValueError(f"Soilmodel {soil_layer['Model']} not defined.")
    return soil

def create_excvSteps(diff, excv_step=excv_stepDefault):
    """
    Creates excavation steps with maximum step height excv_step.
    Last step is corrected to exactly match total depth.
    """
    if diff <= 0:
        return []
    n = max(1, math.ceil(diff / excv_step))
    dz = round(diff / n, 2)
    steps = [dz] * n
    rest = round(diff - sum(steps[:-1]), 2)
    steps[-1] = rest
    return steps

###############################################################################
#------------------------------ Main for modelMain ----------------------------
###############################################################################

def fct_modelModeller(s_i, g_i, dimension,
                      port_input, plaxispw,
                      soil_params, soil_material,
                      line_load, loadsteps,
                      slab_params, slab_material,
                      si, si_params, si_material,
                      intf, intf_material,
                      ltp_params, ltp_material,
                      thinfirstlayer = thinfirstlayer_Default,
                      # 'force in volume' still wrong
                      excv_step = excv_stepDefault,
                      # reduce if soil collapses while excav.
                      differentDeformation = differentDeformationDefault,
                      ):

    ### Model implementation --------------------------------------------------
    s_i.new() # Open new model and set model properties
    if dimension == 'unitcell2':
        g_i.Project.setproperties('ModelType', 'Axisymmetry')

    ### Create soil materials -------------------------------------------------
    for soil_layer in soil_material:
        fct_setProperties(g_i, soil_layer)

    ### Assign soil materials to layers ---------------------------------------
    # Set model dimensions
    Geometry_1 = g_i.SoilContour
    if dimension.endswith('2'):
        Geometry_1.initializerectangular(0,
                                         soil_params['depth'],
                                         soil_params['length'],
                                         0)
    elif dimension.endswith('3'):
        Geometry_1.initializerectangular(soil_params['minx'],
                                         soil_params['miny'],
                                         soil_params['maxx'],
                                         soil_params['maxy'])
    # Create borehole and assign materials
    if   dimension.endswith('2'): g_i.borehole(0)
    elif dimension.endswith('3'): g_i.borehole(soil_params['minx'],
                                               soil_params['miny'])
    for i, thickness in enumerate(soil_params['layers']):
        material = g_i.Materials[soil_params['layer_seq'][i]]
        if (i==0 and thinfirstlayer and thickness>0.30):
            # first 30 cm
            g_i.soillayer(g_i.Boreholes[-1], 0.30)
            g_i.Soillayers[-1].Soil.Material = material
            # remaining thickness
            g_i.soillayer(g_i.Boreholes[-1], thickness - 0.30)
            g_i.Soillayers[-1].Soil.Material = material
        else:
            g_i.soillayer(g_i.Boreholes[-1], thickness)
            g_i.Soillayers[-1].Soil.Material = material
    g_i.Borehole_1.Head = soil_params['head'] # water table

    g_i.gotostructures()
    ### Create refined soilstructure ------------------------------------------
    if (dimension.endswith('3') and
        soil_params['mesh'] != 1.0):
        overhang0 = si_params['disx']*2 # overhang of the refined area
        overhang1 = si_params['disy']*2 # overhang of the refined area
        overhang2 = slab_params['minx'] # possible overhang
        overhang3 = slab_params['miny'] # possible overhang
        o = min(overhang0,overhang1,overhang2,overhang3)
        if dimension == 'quarter3': # only two sides need to have an overhang
            refine_surf = g_i.surface(
                (slab_params['minx']-o,slab_params['miny']-o,si_params['OK']),
                (slab_params['minx']-o,slab_params['maxy'],  si_params['OK']),
                (slab_params['maxx'],  slab_params['maxy'],  si_params['OK']),
                (slab_params['maxx'],  slab_params['miny']-o,si_params['OK']))
        elif dimension == 'full3':
            refine_surf = g_i.surface(
                (slab_params['minx']-o,slab_params['miny']-o,si_params['OK']),
                (slab_params['minx']-o,slab_params['maxy']+o,si_params['OK']),
                (slab_params['maxx']+o,slab_params['maxy']+o,si_params['OK']),
                (slab_params['maxx']+o,slab_params['miny']-o,si_params['OK']))
        elif dimension == 'unitcell3':
            refine_surf = g_i.surface(
                (slab_params['minx'], slab_params['miny'], si_params['OK']),
                (slab_params['minx'], slab_params['maxy'], si_params['OK']),
                (slab_params['maxx'], slab_params['maxy'], si_params['OK']),
                (slab_params['maxx'], slab_params['miny'], si_params['OK']))
        soil_refine = g_i.extrude(refine_surf, 0, 0,
                                  max(-si_params['height']-o,
                                      soil_params['depth']-si_params['OK']))
        g_i.delete(refine_surf)
    else: soil_refine=[]

    ### Create load transfer platform -----------------------------------------
    if ltp_params['height'] > 0:
        ltp_material_LE = ltp_material.copy()
        ltp_material_LE['Model'] = 'LE'
        ltp_material_LE['Name']  = 'LTP_LE'
        ltp_material_LE['Colour']  = 32768
        print('LTP is considered.')
        ltp_soil   = fct_setProperties(g_i, ltp_material)
        ltp_soilLE = fct_setProperties(g_i, ltp_material_LE)
        print('LE for LTP considered until loading.')
        # print(ltp_soilLE)
        if dimension.endswith('2'): 
            ltp_elem = g_i.polygon((0,                     ltp_params['OK']),
                                   (0,                     ltp_params['UK']),
                                   (soil_params['length'], ltp_params['UK']),
                                   (soil_params['length'], ltp_params['OK']))
            ltp_elem[-1].Material = ltp_soil
        elif dimension.endswith('3'):
            ltp_surf = g_i.surface(
                (ltp_params['minx'], ltp_params['miny'], ltp_params['OK']),
                (ltp_params['maxx'], ltp_params['miny'], ltp_params['OK']),
                (ltp_params['maxx'], ltp_params['maxy'], ltp_params['OK']),
                (ltp_params['minx'], ltp_params['maxy'], ltp_params['OK']))
            ltp_elem = g_i.extrude(ltp_surf, 0, 0, -ltp_params['height'])
            g_i.delete(ltp_surf)
            ltp_elem[-1].Material = ltp_soil
    else:
        ltp_soilLE = 0
        ltp_soil   = 0
        ltp_elem   = []

    ### Model symmetry --------------------------------------------------------
    soil_fixed = []
    if dimension in ['unitcell3','quarter3','full3']:
        minx  = slab_params['minx']
        maxx  = slab_params['maxx']
        miny  = slab_params['miny']
        maxy  = slab_params['maxy']
        z_top = slab_params['OK']
        z_bot = soil_params['depth']
        if not differentDeformation:
            if dimension in ['unitcell3']:
                surf_fixed = g_i.surfdispl(g_i.surface((minx, miny, z_bot),
                                                       (minx, maxy, z_bot),
                                                       (minx, maxy, z_top),
                                                       (minx, miny, z_top)),
                              'Displacement_x', 'Fixed',
                              'Displacement_y', 'Free',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)
                surf_fixed = g_i.surfdispl(g_i.surface((minx, miny, z_bot),
                                                       (maxx, miny, z_bot),
                                                       (maxx, miny, z_top),
                                                       (minx, miny, z_top)),
                              'Displacement_x', 'Free',
                              'Displacement_y', 'Fixed',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)
            if dimension in ['unitcell3', 'quarter3']:
                surf_fixed = g_i.surfdispl(g_i.surface((maxx, miny, z_bot),
                                                       (maxx, maxy, z_bot),
                                                       (maxx, maxy, z_top),
                                                       (maxx, miny, z_top)),
                              'Displacement_x', 'Fixed',
                              'Displacement_y', 'Free',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)
                surf_fixed = g_i.surfdispl(g_i.surface((minx, maxy, z_bot),
                                                       (maxx, maxy, z_bot),
                                                       (maxx, maxy, z_top),
                                                       (minx, maxy, z_top)),
                              'Displacement_x', 'Free',
                              'Displacement_y', 'Fixed',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)
            if dimension in [ 'quarter3']:
                surf_fixed = g_i.surfdispl(g_i.surface(
                                          (maxx, soil_params['miny'], z_bot),
                                          (maxx, slab_params['miny'], z_bot),
                                          (maxx, slab_params['miny'], 0),
                                          (maxx, soil_params['miny'], 0)),
                              'Displacement_x', 'Fixed',
                              'Displacement_y', 'Free',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)
                surf_fixed = g_i.surfdispl(g_i.surface(
                                          (soil_params['minx'], maxy, z_bot),
                                          (slab_params['minx'], maxy, z_bot),
                                          (slab_params['minx'], maxy, 0),
                                          (soil_params['minx'], maxy, 0)),
                              'Displacement_x', 'Free',
                              'Displacement_y', 'Fixed',
                              'Displacement_z', 'Free')
                soil_fixed.append(surf_fixed)

    ### Model excavation geometry ---------------------------------------------
    excv_depths = {'soil': [],
                   'slab': [],
                   'ltp':  []
                   }
    excv_elems = deepcopy(excv_depths)
    excv_fixed = deepcopy(excv_depths)
    if si_params['OK'] != 0:
        ### Define the excavation area
        # define excavation partition for numerical stabel redistribution
        z = 0
        if round(slab_params['OK'],3) < 0:
            excv_depths['soil'].append(z)
            diff = abs(slab_params['OK'])
            for dz in create_excvSteps(diff, excv_step=excv_step):
                z -= dz
                excv_depths['soil'].append(z)
        if slab_params['height'] > 0:
            excv_depths['slab'].append(z)
            diff = slab_params['height']
            for dz in create_excvSteps(diff, excv_step=excv_step):
                z -= dz
                excv_depths['slab'].append(z)
        if ltp_params['height'] > 0:
            excv_depths['ltp'].append(z)
            diff = ltp_params['height']
            for dz in create_excvSteps(diff, excv_step=excv_step):
                z -= dz
                excv_depths['ltp'].append(z)
        print(f'Modelled excavation steps are\n{excv_depths}.')
        # 2d
        if dimension.endswith('2'):
            for excv_typ, excv_depths_list in excv_depths.items():
                for top, bottom in zip(excv_depths_list[:-1],
                                       excv_depths_list[1:]):
                    excv_elem = g_i.polygon((0,                     top),
                                            (0,                     bottom),
                                            (soil_params['length'], bottom),
                                            (soil_params['length'], top))
                    excv_elems[excv_typ].append(excv_elem)
        elif dimension.endswith('3'):
            ### Define the excavation volume
            for excv_typ, excv_depths_list in excv_depths.items():
                for top, bottom in zip(excv_depths_list[:-1],
                                       excv_depths_list[1:]):
                    excv_surf = g_i.surface(
                        (minx,                slab_params['miny'], top),
                        (slab_params['maxx'], slab_params['miny'], top),
                        (slab_params['maxx'], slab_params['maxy'], top),
                        (minx,                slab_params['maxy'], top))
                    excv_elem = g_i.extrude(excv_surf, 0, 0, (bottom-top))
                    g_i.delete(excv_surf)
                    excv_elems[excv_typ].append(excv_elem)
            ### Stabilize the excavation pit
            if dimension in ['full3','quarter3']:
                for excv_typ, excv_depths_list in excv_depths.items():
                    for top, bottom in zip(excv_depths_list[:-1],
                                           excv_depths_list[1:]):
                        excv_fixed_z = []
                        # xMin
                        surf_fixed = g_i.surface((minx, miny, top),
                                                 (minx, miny, bottom),
                                                 (minx, maxy, bottom),
                                                 (minx, maxy, top))
                        excv_fixed_z.append(surf_fixed)
                        g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                                 'Displacement_y', 'Free',
                                                 'Displacement_z', 'Free')
                        # yMin
                        surf_fixed = g_i.surface((minx, miny, top),
                                                 (minx, miny, bottom),
                                                 (maxx, miny, bottom),
                                                 (maxx, miny, top))
                        excv_fixed_z.append(surf_fixed)
                        g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                                 'Displacement_y', 'Fixed',
                                                 'Displacement_z', 'Free')
                        if dimension in ['full3']:
                            # xMax
                            surf_fixed = g_i.surface((maxx, miny, top),
                                                     (maxx, miny, bottom),
                                                     (maxx, maxy, bottom),
                                                     (maxx, maxy, top))
                            excv_fixed_z.append(surf_fixed)
                            g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                                     'Displacement_y', 'Free',
                                                     'Displacement_z', 'Free')
                            # yMax
                            surf_fixed = g_i.surface((minx, maxy, top),
                                                     (minx, maxy, bottom),
                                                     (maxx, maxy, bottom),
                                                     (maxx, maxy, top))
                            excv_fixed_z.append(surf_fixed)
                            g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                                     'Displacement_y', 'Fixed',
                                                     'Displacement_z', 'Free')
                        excv_fixed[excv_typ].append(excv_fixed_z)

    ### Model building geometry -----------------------------------------------
    if dimension in ['full3','quarter3']:
        if ltp_params['OK'] > 0:
            minx   = ltp_params['minx']
            maxx   = ltp_params['maxx']
            miny   = ltp_params['miny']
            maxy   = ltp_params['maxy']
            top    = ltp_params['OK']
            bottom = ltp_params['UK']
            excv_fixed_z = []
            # xMin
            surf_fixed = g_i.surface((minx, miny, top),
                                     (minx, miny, bottom),
                                     (minx, maxy, bottom),
                                     (minx, maxy, top))
            excv_fixed_z.append(surf_fixed)
            g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                     'Displacement_y', 'Free',
                                     'Displacement_z', 'Free')
            # yMin
            surf_fixed = g_i.surface((minx, miny, top),
                                     (minx, miny, bottom),
                                     (maxx, miny, bottom),
                                     (maxx, miny, top))
            excv_fixed_z.append(surf_fixed)
            g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                     'Displacement_y', 'Fixed',
                                     'Displacement_z', 'Free')
            if dimension in ['full3']:
                # xMax
                surf_fixed = g_i.surface((maxx, miny, top),
                                         (maxx, miny, bottom),
                                         (maxx, maxy, bottom),
                                         (maxx, maxy, top))
                excv_fixed_z.append(surf_fixed)
                g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                         'Displacement_y', 'Free',
                                         'Displacement_z', 'Free')
                # yMax
                surf_fixed = g_i.surface((minx, maxy, top),
                                         (minx, maxy, bottom),
                                         (maxx, maxy, bottom),
                                         (maxx, maxy, top))
                excv_fixed_z.append(surf_fixed)
                g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                         'Displacement_y', 'Fixed',
                                         'Displacement_z', 'Free')
            excv_fixed['ltp'].append(excv_fixed_z)
        if slab_params['OK'] > 0:
            minx   = slab_params['minx']
            maxx   = slab_params['maxx']
            miny   = slab_params['miny']
            maxy   = slab_params['maxy']
            top    = slab_params['OK']
            bottom = slab_params['UK']
            excv_fixed_z = []
            # xMin
            surf_fixed = g_i.surface((minx, miny, top),
                                     (minx, miny, bottom),
                                     (minx, maxy, bottom),
                                     (minx, maxy, top))
            excv_fixed_z.append(surf_fixed)
            g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                     'Displacement_y', 'Free',
                                     'Displacement_z', 'Free')
            # yMin
            surf_fixed = g_i.surface((minx, miny, top),
                                     (minx, miny, bottom),
                                     (maxx, miny, bottom),
                                     (maxx, miny, top))
            excv_fixed_z.append(surf_fixed)
            g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                     'Displacement_y', 'Fixed',
                                     'Displacement_z', 'Free')
            if dimension in ['full3']:
                # xMax
                surf_fixed = g_i.surface((maxx, miny, top),
                                         (maxx, miny, bottom),
                                         (maxx, maxy, bottom),
                                         (maxx, maxy, top))
                excv_fixed_z.append(surf_fixed)
                g_i.surfdispl(surf_fixed,'Displacement_x', 'Fixed',
                                         'Displacement_y', 'Free',
                                         'Displacement_z', 'Free')
                # yMax
                surf_fixed = g_i.surface((minx, maxy, top),
                                         (minx, maxy, bottom),
                                         (maxx, maxy, bottom),
                                         (maxx, maxy, top))
                excv_fixed_z.append(surf_fixed)
                g_i.surfdispl(surf_fixed,'Displacement_x', 'Free',
                                         'Displacement_y', 'Fixed',
                                         'Displacement_z', 'Free')
            excv_fixed['slab'].append(excv_fixed_z)

    ### Create slab material --------------------------------------------------
    if slab_params['height'] > 0:
        # dummy plate for bending moments
        bendingMoment = True
        if bendingMoment:
            h = slab_params['height']
            # print(r'https://bentleysystems.service-now.com/community?'
            #       r'id=community_question&sys_id=d4095ef247558a909091861f536d4340')
            # soil = g_i.platemat()
            # soil.setproperties('Identification',  soil_layer['Name'],
            #                    'Colour',          soil_layer['Colour'],
            #                    'SoilModel',       "linearelastic",
            #                    'DrainageType',    soil_layer['Drainage'],
            #                    'gammaUnsat',      soil_layer['gamma'],
            #                    'Eref',            soil_layer['ERef'],
            #                    'nu',              soil_layer['nu'],
            #                    'InterfaceStrengthDetermination', 'Manual',
            #                    'Rinter',          soil_layer['Rinter'])
            # https://bentleysystems.service-now.com/community
            # id=kb_article&sysparm_article=KB0108381
            plate_height = 0.5 * (slab_params['OK'] + slab_params['UK'])
            soil_plate = g_i.platemat()
            Edummy = slab_material['ERef']*1e-5
            # 2D case
            if dimension.endswith('2'):
                soil_plate.setproperties('Identification', "DummySlab",
                                         'MaterialType', "Elastic",
                                         'EA1', Edummy * h,
                                         'EI',  Edummy * h**3/12,
                                         'Structnu',  slab_material['nu'],
                                         'w',  0.0)
                plate_ref = g_i.line((0,                     plate_height),
                                     (soil_params['length'], plate_height))
                slab_plate = g_i.plate(plate_ref[-1]) # point, point, line
            # 3D case
            elif dimension.endswith('3'):
                soil_plate.setproperties('Identification', "DummySlab",
                                         'MaterialType', "Elastic",
                                         'E1', Edummy,
                                         'Structnu12',  slab_material['nu'],
                                         'd',  h,
                                         'w',  0.0)
                plate_ref = g_i.surface((minx, miny, plate_height),
                                        (maxx, miny, plate_height),
                                        (maxx, maxy, plate_height),
                                        (minx, maxy, plate_height))
                slab_plate = g_i.plate(plate_ref) # Polygon
            slab_plate.Material = soil_plate
        else:
            soil_plate = 0
            slab_plate = []
        # Geometry of the slab + material
        slab_soil = fct_setProperties(g_i, slab_material)
        if dimension.endswith('2'):
            slab_elem = g_i.polygon(
                (0,                     slab_params['OK']),
                (0,                     slab_params['UK']),
                (soil_params['length'], slab_params['UK']),
                (soil_params['length'], slab_params['OK']))
            slab_elem[-1].Material = slab_soil
        elif dimension.endswith('3'):
            slab_surf = g_i.surface((minx, miny, slab_params['OK']),
                                    (maxx, miny, slab_params['OK']),
                                    (maxx, maxy, slab_params['OK']),
                                    (minx, maxy, slab_params['OK']))
            slab_elem = g_i.extrude(slab_surf, 0, 0, -slab_params['height'])
            slab_elem[-1].Material = slab_soil
            g_i.delete(slab_surf)
    else:
        slab_soil = 0
        slab_elem = []
        soil_plate = 0
        slab_plate = []

    # Custom interface properties
    pos_z = slab_params['UK']
    if intf: # (forces between slab and ltp)
        slab_int = fct_setProperties(g_i, intf_material)
        slab_int.setproperties('InterfaceStiffnessDetermination',
                               intf_material['InterfaceStiffnessDet'],
                               'knInter',    intf_material['knInter'],
                               'ksInter',    intf_material['ksInter'])
        if dimension.endswith('2'):
            slab_int = g_i.neginterface((0,                     pos_z),
                                        (soil_params['length'], pos_z),
                                        'MaterialMode', 'Custom',
                                        'Material', slab_int)
        elif dimension.endswith('3'):
            slab_int = g_i.neginterface((minx, miny, pos_z),
                                        (maxx, miny, pos_z),
                                        (maxx, maxy, pos_z),
                                        (minx, maxy, pos_z),
                                        'MaterialMode', 'Custom',
                                        'Material', slab_int)
        else: slab_int = []
    else:
        slab_bottom_intf = True
        if slab_bottom_intf and ltp_params['height'] > 0:
            # Rinter=1 due to rough ground
            if dimension.endswith('2'):
                slab_int = g_i.neginterface((0,                     pos_z),
                                            (soil_params['length'], pos_z))
            elif dimension.endswith('3'):
                slab_int = g_i.neginterface((minx, miny, pos_z),
                                            (maxx, miny, pos_z),
                                            (maxx, maxy, pos_z),
                                            (minx, maxy, pos_z))
            else: slab_int = []
        else: slab_int = []

    slab_tops_intf = False
    if slab_tops_intf:
        # General interface to extract deformation
        # not used to reduce model -> used is CS-offset
        pos_z = slab_params['OK']
        if dimension.endswith('2'):
            slab_tops = g_i.neginterface((0,                     pos_z),
                                         (soil_params['length'], pos_z))
        elif dimension.endswith('3'):
            slab_tops = g_i.neginterface((minx, miny, pos_z),
                                         (maxx, miny, pos_z),
                                         (maxx, maxy, pos_z),
                                         (minx, maxy, pos_z))
    else: slab_tops = []

    si_top_int = True
    if si_top_int:
        # General interface to extract forces
        pos_z = si_params['OK']
        if dimension.endswith('2'):
            si_tops = g_i.neginterface((0,                     pos_z),
                                       (soil_params['length'], pos_z))
        elif dimension.endswith('3'):
            si_tops = g_i.neginterface((minx, miny, pos_z),
                                       (maxx, miny, pos_z),
                                       (maxx, maxy, pos_z),
                                       (minx, maxy, pos_z))
        else: si_tops = []
    else: si_tops = []

    ### Create loading --------------------------------------------------------
    Load = True
    uniform = True
    if Load:
        if dimension.endswith('3'):
            load_surf = g_i.surface((minx, miny, slab_params['OK']),
                                    (maxx, miny, slab_params['OK']),
                                    (maxx, maxy, slab_params['OK']),
                                    (minx, maxy, slab_params['OK']))
        if uniform:
            if dimension.endswith('2'):
                loading = g_i.lineload((0,
                                        slab_params['OK']),
                                       (soil_params['length'],
                                        slab_params['OK']),
                                       'qy_start', -line_load)
            elif dimension.endswith('3'):
                loading = g_i.surfload(load_surf, 'sigz', -line_load)
                # g_i.delete(load_surf) ### ATTENTION: should NOT be deleted!
            else: loading = []
        else: # to evaluate influnce in the 3d model (unitcell not useful!)
            if dimension.endswith('2'):
                loading = g_i.lineload((0,                     slab_params['OK']),
                                       (soil_params['length'], slab_params['OK']),
                                       'Distribution', 'Linear',
                                       'qy_start', -line_load,
                                       'qy_end',   -line_load)
            elif dimension.endswith('3'):
                loading = g_i.surfload(load_surf,
                                       'Distribution', 'x-aligned increment',
                                       'sigz',       -line_load,
                                       'sigz_inc_x', -line_load/(maxx-minx))
                # g_i.delete(load_surf) ### ATTENTION: should NOT be deleted!
            else: loading = []
    else: loading = []

    ### Model soil improvement element ----------------------------------------
    if si:
        si_soil  = fct_setProperties(g_i, si_material)
        si_shaft_UK = si_params['UK'] - 1.0
        if si_shaft_UK < soil_params['depth']:
            si_shaft_UK = soil_params['depth']
        topextension = False # otherwise tension cut off failure within LTP
        if (topextension and ltp_params['height'] > 0):
            si_shaft_OK = si_params['OK'] + ltp_params['height']/2
        else: si_shaft_OK = si_params['OK']
        si_shaft_height = si_shaft_UK - si_shaft_OK
        si_base_radius_ext = min(0.5,
                                 (min(si_params['disx'],
                                      si_params['disy'])/2
                                  - si_params['radius'] - 0.05)
                                 )
        if dimension.endswith('2'):
            # Soil improvement
            si_elems = g_i.polygon((0,                   si_params['OK']),
                                   (si_params['radius'], si_params['OK']),
                                   (si_params['radius'], si_params['UK']),
                                   (0,                   si_params['UK']))
            si_elems[-1].Material = si_soil
            # Soil improvement interface - SHAFT
            # si_shafts = g_i.neginterface((si_params['radius'], si_shaft_OK),
            #                              (si_params['radius'], si_shaft_UK))
            si_shafts = g_i.posinterface((si_params['radius'], si_shaft_OK),
                                         (si_params['radius'], si_shaft_UK))
            # Soil improvement interface - BASE
            si_base_radius = si_params['radius'] + si_base_radius_ext
            if si_base_radius > soil_params['length']: # stays inside the model
                si_base_radius = soil_params['length'] - 0.005
                # to be on the safe side: 0.005
            si_bases  = g_i.neginterface((0,              si_params['UK']),
                                         (si_base_radius, si_params['UK']))
            
        elif dimension.endswith('3'):
            si_elems = []
            si_shafts = []
            si_bases = []
            # Define soil improvement raster
            if dimension == 'unitcell3':
                si_pos = np.array([[(minx + maxx) / 2,
                                    (miny + maxy) / 2]])
                # si_pos_all = si_pos.copy()
            elif dimension in ['full3','quarter3']:
                if dimension == 'quarter3':
                    nx      =  si_params['numx'] // 2
                    ny      =  si_params['numy'] // 2
                    # nx_all  = (si_params['numx'] + 1) // 2
                    # ny_all  = (si_params['numy'] + 1) // 2
                else: # full3
                    nx = si_params['numx']
                    ny = si_params['numy']
                    # nx_all = nx
                    # ny_all = ny
                si_pos_x     = fct_positionsGet(minx,
                                                si_params['dis_edgex'],
                                                si_params['disx'],
                                                nx)
                si_pos_y     = fct_positionsGet(miny,
                                                si_params['dis_edgey'],
                                                si_params['disy'],
                                                ny)
                # si_pos_x_all = fct_positionsGet(minx, si_params['dis_edgex'],
                #                                 si_params['disx'], nx_all)
                # si_pos_y_all = fct_positionsGet(miny, si_params['dis_edgey'],
                #                                 si_params['disy'], ny_all)
                # Meshgrid
                X, Y         = np.meshgrid(si_pos_x, si_pos_y)
                si_pos       = np.column_stack([X.ravel(), Y.ravel()])
                # X_all, Y_all = np.meshgrid(si_pos_x_all, si_pos_y_all)
                # si_pos_all   = np.column_stack([X_all.ravel(),
                #                                 Y_all.ravel()])
            else: raise ValueError('Position for SI not defined.')
            # Soil improvement
            for i, (si_pos_xi, si_pos_yi) in enumerate(si_pos, start=1):
                si_poly = g_i.polycurve((si_pos_xi,
                                         si_pos_yi,
                                         si_params['OK']),
                                        (1,0,0), (0,1,0),
                                        'Arc', 0, 180, si_params['radius'],
                                        'Arc', 0, 180, si_params['radius'])
                si_poly[0].Offset2 = -si_params['radius']
                si_surf   = g_i.surface(si_poly[0])
                g_i.delete(si_poly[0])
                si_volume = g_i.extrude(si_surf, 0, 0,
                                        si_params['UK'] - si_params['OK'])
                si_volume[-1].Material = si_soil
                g_i.delete(si_surf)
                # Shaft interface
                si_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                         si_shaft_OK),
                                        (1,0,0), (0,1,0),
                                        'Arc', 0, 180, si_params['radius'],
                                        'Arc', 0, 180, si_params['radius'])
                si_poly[0].Offset2 = -si_params['radius']
                si_shaft_surf = g_i.extrude(si_poly[0], 0, 0, si_shaft_height)
                g_i.delete(si_poly[0])
                g_i.neginterface(si_shaft_surf)
                # Base interface
                si_base_radius = si_params['radius'] + si_base_radius_ext
                checks = [("maxx", si_pos_xi, ">"),
                          ("minx", si_pos_xi, "<"),
                          ("maxy", si_pos_yi, ">"),
                          ("miny", si_pos_yi, "<"),
                          ]
                for ref, pos, comp in checks:
                    if (comp == ">" and
                        pos + si_base_radius > soil_params[ref]):
                        si_base_radius = soil_params[ref] - pos - 0.01
                    if (comp == "<"
                        and pos - si_base_radius < soil_params[ref]):
                        si_base_radius = pos + 0.01 - soil_params[ref]
                si_base_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                              si_params['UK']),
                                             (1,0,0), (0,1,0),
                                             'Arc', 0, 180, si_base_radius,
                                             'Arc', 0, 180, si_base_radius)
                si_base_poly[0].Offset2 = -si_base_radius
                si_base_surf = g_i.surface(si_base_poly[0])
                g_i.neginterface(si_base_surf)
                g_i.delete(si_base_poly[0])
                # Group
                si_elems.append(si_volume[0])
                si_shafts.append(si_shaft_surf)
                si_bases.append(si_base_surf)
            if dimension == 'quarter3':
                # Half soil improvement on the y axis
                if si_params['numx'] % 2:
                    n_si = si_params['numx']
                    for i in range(int(n_si/2)):
                        si_pos_xi = slab_params['maxx']
                        si_pos_yi = (slab_params['miny']
                                     + si_params['dis_edgey']
                                     + i * si_params['disy'])
                        # Shaft interface
                        si_half = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                 si_shaft_OK),
                                                (1,0,0), (0,1,0),
                                                'Arc', 0, 180,
                                                si_params['radius'])
                        si_half[1].ArcProperties.RelativeStartAngle1 = 180
                        si_half[0].Offset2 = si_params['radius']
                        si_shaft_surf = g_i.extrude(si_half[0],
                                                    0, 0, si_shaft_height)
                        g_i.delete(si_half[0])
                        g_i.neginterface(si_shaft_surf)
                        # Soil improvement
                        si_half = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                 si_params['OK']),
                                                (1,0,0), (0,1,0),
                                                'Arc', 0, 180,
                                                si_params['radius'])
                        si_half[1].ArcProperties.RelativeStartAngle1 = 180
                        si_half[0].Offset2 = si_params['radius']
                        si_half[0].close('line')
                        si_surf = g_i.surface(si_half[0])
                        g_i.delete(si_half[0])
                        si_volume = g_i.extrude(si_surf,0,0,
                                                si_params['UK']
                                                - si_params['OK'])
                        si_volume[-1].Material = si_soil
                        g_i.delete(si_surf)
                        # Base interface
                        si_base_radius = si_params['radius'] + si_base_radius_ext
                        checks = [("minx", si_pos_xi, "<"),
                                  ("maxy", si_pos_yi, ">"),
                                  ("miny", si_pos_yi, "<"),
                                  ]
                        for ref, pos, comp in checks:
                            if (comp == ">" and
                                pos + si_base_radius > soil_params[ref]):
                                si_base_radius = soil_params[ref] - pos - 0.01
                            if (comp == "<"
                                and pos - si_base_radius < soil_params[ref]):
                                si_base_radius = pos + 0.01 - soil_params[ref]
                        si_base_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                      si_params['UK']),
                                                     (1,0,0), (0,1,0),
                                                     'Arc', 0, 180,
                                                     si_base_radius)
                        si_base_poly[1].ArcProperties.RelativeStartAngle1 = 180
                        si_base_poly[0].Offset2 = si_base_radius
                        si_base_poly[0].close('line')
                        si_base_surf = g_i.surface(si_base_poly[0])
                        g_i.neginterface(si_base_surf)
                        g_i.delete(si_base_poly[0])
                        # Group
                        si_elems.append(si_volume[0])
                        si_shafts.append(si_shaft_surf)
                        si_bases.append(si_base_surf)
                # half soil improvement on the x axis
                if si_params['numy'] % 2:
                    n_si = si_params['numy']
                    for i in range(int(n_si/2)):
                        si_pos_xi = (slab_params['minx'] +
                                     si_params['dis_edgex'] +
                                     i * si_params['disx'])
                        si_pos_yi = slab_params['maxy']
                        # Shaft interface
                        si_half = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                 si_shaft_OK),
                                                (1,0,0), (0,1,0),
                                                'Arc', 0, 180,
                                                si_params['radius'])
                        si_half[1].ArcProperties.RelativeStartAngle1 = -90
                        si_half[0].Offset1 = -si_params['radius']
                        si_shaft_surf = g_i.extrude(si_half[0],
                                                    0, 0, si_shaft_height)
                        g_i.neginterface(si_shaft_surf)
                        # Soil improvement interface
                        si_half = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                 si_params['OK']),
                                                (1,0,0), (0,1,0),
                                                'Arc', 0, 180,
                                                si_params['radius'])
                        si_half[1].ArcProperties.RelativeStartAngle1 = -90
                        si_half[0].Offset1 = -si_params['radius']
                        si_half[0].close('line')
                        si_surf = g_i.surface(si_half[0])
                        si_volume = g_i.extrude(si_surf,0,0,
                                                si_params['UK']-si_params['OK'])
                        si_volume[-1].Material = si_soil
                        g_i.delete(si_surf)
                        g_i.delete(si_half[0])
                        # Base interface
                        si_base_radius = si_params['radius'] + si_base_radius_ext
                        checks = [("maxx", si_pos_xi, ">"),
                                  ("minx", si_pos_xi, "<"),
                                  ("miny", si_pos_yi, "<"),
                                  ]
                        for ref, pos, comp in checks:
                            if (comp == ">" and
                                pos + si_base_radius > soil_params[ref]):
                                si_base_radius = soil_params[ref] - pos - 0.01
                            if (comp == "<"
                                and pos - si_base_radius < soil_params[ref]):
                                si_base_radius = pos + 0.01 - soil_params[ref]
                        si_base_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                      si_params['UK']),
                                                     (1,0,0), (0,1,0),
                                                     'Arc', 0, 180,
                                                     si_base_radius)
                        si_base_poly[1].ArcProperties.RelativeStartAngle1 = -90
                        si_base_poly[0].Offset1 = -si_base_radius
                        si_base_poly[0].close('line')
                        si_base_surf = g_i.surface(si_base_poly[0])
                        g_i.neginterface(si_base_surf)
                        g_i.delete(si_base_poly[0])
                        # Group
                        si_elems.append(si_volume[0])
                        si_shafts.append(si_shaft_surf)
                        si_bases.append(si_base_surf)
                # quarter soil improvement at xmax and ymax
                if (si_params['numx'] % 2 and
                    si_params['numy'] % 2):
                    # Soil improvement
                    si_pos_xi = slab_params['maxx']
                    si_pos_yi = slab_params['maxy']
                    si_quarter = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                si_params['OK']),
                                               (1,0,0), (0,1,0),
                                               'Arc',0,90, si_params['radius'],
                                               'Line',90,  si_params['radius'],
                                               'Line',90,  si_params['radius'])
                    si_quarter[1].ArcProperties.RelativeStartAngle1 = -90
                    si_quarter[0].Offset1 = -si_params['radius']
                    si_surf = g_i.surface(si_quarter[0])
                    si_volume = g_i.extrude(si_surf,0,0,
                                            si_params['UK']-si_params['OK'])
                    si_volume[-1].Material = si_soil
                    g_i.delete(si_surf)
                    g_i.delete(si_quarter[0]) # 0=Polycurve, 1,2,3=Segment
                    # Soil improvement interface
                    # Shaft interface (1.0 to include the next node)
                    si_shaft_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                   si_shaft_OK),
                                                  (1,0,0), (0,1,0),
                                                  'Arc',0,90,
                                                  si_params['radius'])
                    si_shaft_poly[1].ArcProperties.RelativeStartAngle1 = -90
                    si_shaft_poly[0].Offset1 = -si_params['radius']
                    si_shaft_surf = g_i.extrude(si_shaft_poly[0], 
                                                0, 0, si_shaft_height)
                    g_i.neginterface(si_shaft_surf)
                    g_i.delete(si_shaft_poly[0])
                    # Base interface
                    si_base_radius = si_params['radius'] + si_base_radius_ext
                    checks = [("minx", si_pos_xi, "<"),
                              ("miny", si_pos_yi, "<"),
                              ]
                    for ref, pos, comp in checks:
                        if (comp == ">" and
                            pos + si_base_radius > soil_params[ref]):
                            si_base_radius = soil_params[ref] - pos - 0.01
                        if (comp == "<"
                            and pos - si_base_radius < soil_params[ref]):
                            si_base_radius = pos + 0.01 - soil_params[ref]
                    si_base_poly = g_i.polycurve((si_pos_xi, si_pos_yi,
                                                  si_params['UK']),
                                                 (1,0,0), (0,1,0),
                                                 'Arc', 0, 90, si_base_radius,
                                                 'Line', 90,   si_base_radius,
                                                 'Line', 90,   si_base_radius)
                    si_base_poly[1].ArcProperties.RelativeStartAngle1 = -90
                    si_base_poly[0].Offset1 = -si_base_radius
                    si_base_surf = g_i.surface(si_base_poly[0])
                    g_i.neginterface(si_base_surf)
                    g_i.delete(si_base_poly[0])
                    # Group
                    si_elems.append(si_volume[0])
                    si_shafts.append(si_shaft_surf)
                    si_bases.append(si_base_surf)

            si_elems  = g_i.group(si_elems)
            si_shafts = g_i.group(si_shafts)
            si_bases  = g_i.group(si_bases)
            g_i.point(minx, miny, slab_params['OK'])
            g_i.point(maxx, maxy, slab_params['OK'])

        else: raise ValueError('Dimension not defined')

        si_position = []
        if dimension.endswith('2'):
            if dimension == 'unitcell2':
                si_position.append((0, si_params['OK']))
            else:
                raise ValueError('Curvepoints not defined')
        elif dimension.endswith('3'):
            if dimension == 'unitcell3':
                si_position.append(((minx + maxx) / 2,
                               (miny + maxy) / 2,
                               si_params['OK']))
            elif dimension in ['full3', 'quarter3']:
                if dimension == 'quarter3':
                    xmax_eff = 2 * maxx - minx
                    ymax_eff = 2 * maxy - miny
                else:  # full3
                    xmax_eff = maxx
                    ymax_eff = maxy
                x_positions = fct_positionsBuild(minx, xmax_eff,
                                              si_params['numx'],
                                              si_params['dis_edgex'],
                                              si_params['disx'])
                y_positions = fct_positionsBuild(miny, ymax_eff,
                                              si_params['numy'],
                                              si_params['dis_edgey'],
                                              si_params['disy'])
                for x in x_positions:
                    for y in y_positions:
                        si_position.append((x, y, si_params['OK']))
                si_position = list(set(si_position))
                # print(si_position)
            else:
                raise ValueError('Curvepoints not defined')
            # inport si_position for curvepoints - not necessary for 2D
            for point_si in si_position:
                g_i.point(*point_si) # single coordinates not tuple
        else:
            raise ValueError('Dimension not defined')

        # for polycurves in g_i.Polycurves:
        #     g_i.delete(polycurves)

        ### Evaluate uplift force ---------------------------------------------
        soil_layer  = 0
        soil_depth  = 0
        force_unsat = 0
        force_sat   = 0
        for soil in g_i.Materials:
            if 'Soil' not in soil.Name.value: continue
            # Soil improvement unit weights
            si_gamma_unsat = si_soil.gammaUnsat
            si_gamma_sat   = si_soil.gammaSat
            si_bottom       = si_params['UK']
            # Original soil unit weights
            soil_gamma_unsat = soil.gammaUnsat
            soil_gamma_sat   = soil.gammaSat
            soil_depth_layer = soil_params['layers'][soil_layer]
            soil_depth  -= soil_depth_layer
            rel_height   = min(soil_depth, si_bottom)
            force_unsat += (si_gamma_unsat - soil_gamma_unsat) * rel_height
            force_sat   += (si_gamma_sat   - soil_gamma_sat  ) * rel_height

            warning = False
            # Check unsaturated unit weight
            if warning:
                if force_unsat < 0:
                    print(f'SI gamma_unsat ({si_gamma_unsat}) is smaller than '
                          f'soil gamma_unsat ({soil_gamma_unsat}) '
                          f'for soil layer {soil.Name.value}. '
                          f'Risk of uplift.')
                # Check saturated unit weight
                if force_sat < 0:
                    print(f'SI gamma_sat ({si_gamma_sat}) is smaller than '
                          f'soil gamma_sat ({soil_gamma_sat}) '
                          f'for soil layer {soil.Name.value}. '
                      f'Risk of uplift.')
    else: # not SI
        si_soil     = 0
        si_bases    = 0
        si_shafts   = 0
        si_tops     = 0
        si_elems    = []
        si_position = []

    return (s_i, g_i,
            loading, soil_plate, soil_refine, soil_fixed,
            si_soil, si_position, si_elems, si_bases, si_shafts, si_tops,
            slab_soil, slab_elem, slab_int, slab_plate, slab_tops,
            ltp_soilLE, ltp_soil, ltp_elem,
            excv_elems, excv_fixed)

def fct_modelPhaser(g_i, dimension, path,
                    loading, loadsteps, line_load,
                    soil_refine, soil_params,
                    si_params, si_soil, si, si_elems,
                    si_bases, si_shafts, si_tops,
                    slab_params, slab_soil, slab_elem,
                    soil_plate, slab_int, slab_tops,
                    ltp_params, ltp_soilLE, ltp_soil, ltp_elem,
                    excv_elems, excv_fixed,
                    SRF,
                    differentDeformation=differentDeformationDefault):
    g_i.gotostages()
    phaseCounter = 0
    # Initial phase
    II = g_i.InitialPhase
    g_i.setcurrentphase(II)
    if differentDeformation:
        if dimension == 'unitcell2':
            g_i.set(g_i.Deformations.BoundaryXMin, II, "Horizontally fixed") # left
            g_i.set(g_i.Deformations.BoundaryXMax, II, "Horizontally fixed") # right
            g_i.set(g_i.Deformations.BoundaryYMin, II, "Fully fixed") # bottom
            g_i.set(g_i.Deformations.BoundaryYMax, II, "Free") # top
        elif dimension == 'unitcell3':
            g_i.set(g_i.Deformations.BoundaryXMin, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryXMax, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryYMin, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryYMax, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryZMin, II, "Fully fixed") # bottom
            g_i.set(g_i.Deformations.BoundaryZMax, II, "Free") # to the top
        elif dimension == 'quarter3':
            g_i.set(g_i.Deformations.BoundaryXMin, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryXMax, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryYMin, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryYMax, II, "Horizontally fixed")
            g_i.set(g_i.Deformations.BoundaryZMin, II, "Fully fixed") # bottom
            g_i.set(g_i.Deformations.BoundaryZMax, II, "Free") # top
        elif dimension == 'full3':
            g_i.set(g_i.Deformations.BoundaryXMin, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryXMax, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryYMin, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryYMax, II, "Normally fixed")
            g_i.set(g_i.Deformations.BoundaryZMin, II, "Fully fixed") # bottom
            g_i.set(g_i.Deformations.BoundaryZMax, II, "Free") # top
    g_i.deactivate(g_i.Interfaces,           II)
    g_i.deactivate(g_i.Plates,               II)
    if dimension.endswith('2'):
        g_i.deactivate(g_i.Lines,            II)
        g_i.deactivate(g_i.LineLoads,        II)
        g_i.deactivate(g_i.Polygons,         II)
        for poly in g_i.Polygons:
            if 'Borehole' in poly.Name.value:
                g_i.activate(poly,           II)
    elif dimension.endswith('3'):
        g_i.deactivate(g_i.Surfaces,         II)
        g_i.deactivate(g_i.SurfaceLoads,     II)
        g_i.deactivate(g_i.Volumes,          II)
        if differentDeformation:
            g_i.deactivate(g_i.SurfaceDisplacements,     II)
        else:
            excv_s = [surf
                      for excv_fix in excv_fixed.values()
                      for surfs in excv_fix
                      for surf in surfs]
            g_i.deactivate(excv_s,           II)
        if soil_params['mesh'] != 1.0:
            g_i.activate(soil_refine[0],   II) # to reduce meshpoints overall
        for poly in g_i.Volumes:
            if 'Borehole' in poly.Name.value:
                g_i.activate(poly,           II)
    II.DeformCalcType.set('K0 procedure')
    II.Identification.set(f'Phase {phaseCounter}: Initial State')
    phaseCounter += 1

    # Excavation
    if si_params['OK'] != 0:
        # print(excv_elems)
        for excv_typ, excv_vols in excv_elems.items():
            num = 1
            # print(excv_vols)
            for excv_vol in excv_vols:
                # print(excv_vol)
                EX = g_i.phase(g_i.Phases[-1])
                g_i.setcurrentphase(EX)
                g_i.setwaterdry(excv_vol[0],                  EX)
                g_i.deactivate(excv_vol[0],                   EX)
                if dimension in ['full3', 'quarter3']:
                    g_i.activate(excv_fixed[excv_typ][num-1], EX)
                name = f'Phase {phaseCounter}: Excavation {excv_typ} {num}'
                EX.Identification.set(name)
                phaseCounter += 1
                num          += 1

    # Activate soil improvement
    if si:
        SI = g_i.phase(g_i.Phases[-1])
        g_i.setcurrentphase(SI)
        if dimension.endswith('2'):
            # print(si_elems,si_shafts,si_bases,si_tops)
            g_i.setmaterial(si_elems[0],     SI, si_soil)
            g_i.activate(si_shafts[2],       SI) # [point,point,line,interface]
            g_i.activate(si_bases[2],        SI) # [point,point,line,interface]
            g_i.activate(si_tops[2],         SI) # [point,point,line,interface]
        elif dimension.endswith('3'):
            # print(si_elems,si_shafts,si_bases,si_tops)
            g_i.setmaterial(si_elems,        SI, si_soil) # [Volume,Soil]
            g_i.activate(si_shafts,          SI) # list
            g_i.activate(si_bases,           SI) # list
            g_i.activate(si_tops[0],         SI) # [Polygon,NegativeInterface]
        SI.Identification.set(f'Phase {phaseCounter}: Soil Improvement')
        phaseCounter += 1

    # Activate ltp
    if ltp_params['height'] > 0:
        LTP = g_i.phase(g_i.Phases[-1])
        g_i.setcurrentphase(LTP)
        if ltp_params['OK'] > 0:
            for excv_surfs in excv_fixed['ltp']:
                g_i.activate(excv_surfs,      LTP)
        else: # excavation
            for excv_surfs in excv_fixed['ltp']:
                g_i.deactivate(excv_surfs,    LTP)
        g_i.activate(ltp_elem[0],             LTP)
        g_i.setmaterial(ltp_elem[0],          LTP, ltp_soilLE)
        LTP.Deform.ResetDisplacementsToZero = True
        LTP.Deform.ResetSmallStrain         = False
        LTP.Identification.set(f'Phase {phaseCounter}: LTP')
        phaseCounter += 1

    # Activate slab
    if slab_params['height'] > 0:
        SB = g_i.phase(g_i.Phases[-1])
        g_i.setcurrentphase(SB)
        if slab_params['OK'] > 0:
            for excv_surfs in excv_fixed['slab']:
                g_i.activate(excv_surfs,         SB)
        else: # excavation
            for excv_surfs in excv_fixed['slab']:
                g_i.deactivate(excv_surfs,       SB)
        g_i.activate(slab_elem[0],               SB)
        g_i.setmaterial(slab_elem[0],            SB, slab_soil)
        if soil_plate != 0:
            g_i.activate(g_i.Plates,             SB) # slab_plate
            g_i.setmaterial(g_i.Plates,          SB, soil_plate)
        if dimension.endswith('2'):
            if len(slab_tops)>0:
                g_i.activate(slab_tops[2],       SB)
            g_i.deactivate(g_i.LineLoads,        SB)
        elif dimension.endswith('3'):
            if len(slab_tops)>0:
                g_i.activate(slab_tops[0],       SB)
            g_i.deactivate(g_i.SurfaceLoads,     SB)
        if len(slab_int) > 0:
            if ltp_params['height'] > 0:
                g_i.activate(slab_int[0],        SB)
                # slab_int shpuld not be on top of the si_tops
        # g_i.setmaterial(ltp_elem[0],             SB, ltp_soil) # stabilisation
        SB.Deform.ResetDisplacementsToZero = True
        SB.Deform.ResetSmallStrain         = False
        SB.Identification.set(f'Phase {phaseCounter}: Foundation Slab')
        phaseCounter += 1

    ### Activate loading ------------------------------------------------------
    StepwiseLoading = True
    # Iterative loading procedure (to get better accuracy in the start-stages)
    if StepwiseLoading:
        if loadsteps < 1:
            raise ValueError(f'Loadstep Number {loadsteps} not possible.')
        elif loadsteps == 1:
            load_steps = np.array([-line_load])
        else:
            log_space  = np.logspace(0, 1, loadsteps)
            scaled     = ((log_space - log_space.min())
                          / (log_space.max() - log_space.min()))
            startvalue = 10
            load_steps = -1 * np.round(startvalue
                                       + scaled * (line_load
                                                   - startvalue), 3)
        np.savetxt(os.path.join(path,'load_steps.csv'),
                   load_steps, delimiter=';')
        # plot (x-axis: labels and grid only at integer loadsteps)
        fct_setPlottext()
        fig, ax = plt.subplots(figsize=(12/2.54, 6/2.54))
        x = np.arange(1, len(load_steps) + 1)
        ax.plot(x, load_steps, marker='o')
        ax.set_xlabel('Loadstep')
        ax.set_ylabel(r'Applied load $F_{\mathrm{load}}$ [kN]')
        ax.set_title('Loadstep distribution')
        ax.set_xticks(x)
        ax.invert_yaxis()
        ax.grid(True)
        fig.subplots_adjust(bottom=0.22)
        fct_formatSave(True, fig, path, 'load_steps')
        plt.close(fig)
        # activate loading
        for count, load in enumerate(load_steps):
            phase_name = f'Phase {count+phaseCounter}: Load step {count+1}'
            if load == load_steps[0]:
                LS = g_i.phase(g_i.Phases[-1])
                g_i.setcurrentphase(LS)
                LS.Deform.ResetDisplacementsToZero = True
                LS.Deform.ResetSmallStrain         = False
                if dimension.endswith('2'):
                    g_i.activate(g_i.LineLoads,        LS)
                    g_i.set(g_i.LineLoads[0].qy_start, LS, load)
                if dimension.endswith('3'):
                    g_i.activate(g_i.SurfaceLoads,     LS)
                    g_i.set(g_i.SurfaceLoads[0].sigz,  LS, load)
                if ltp_params['height'] > 0:
                    g_i.setmaterial(ltp_elem[0],       LS, ltp_soil)
            else:
                LS = g_i.phase(g_i.Phases[-1])
                g_i.setcurrentphase(LS)
                if dimension.endswith('2'):
                    g_i.set(g_i.LineLoads[0].qy_start, LS, load)
                if dimension.endswith('3'):
                    g_i.set(g_i.SurfaceLoads[0].sigz,  LS, load)
            LS.Identification.set(phase_name)
    # Activate total loading in one step (same result)
    else:
        count = 0
        phase_name = f'Phase {count+phaseCounter}: Load step total'
        LS = g_i.phase(g_i.Phases[-1])
        g_i.setcurrentphase(LS)
        LS.Deform.ResetDisplacementsToZero  = True
        LS.Deform.ResetSmallStrain          = False
        g_i.activate(loading, LS)
        LS.Identification.set(phase_name)

    if SRF: # additional phase to evaluate Factor of Safety # SafetyFactor
        SF = g_i.phase(g_i.Phases[-1])
        g_i.setcurrentphase(SF)
        SF.DeformCalcType                   = "Safety"
        SF.Deform.UseDefaultIterationParams = False
        SF.Deform.MaxSteps                  = 200
        SF.Deform.MaxIterations             = 80
        SF.Deform.DesiredMinIterations      = 8
        SF.Deform.DesiredMaxIterations      = 20
        SF.Deform.UseSubspaceAccelerator    = True
        SF.Identification.set(f'Phase {count+phaseCounter+1}: Failure')

    return g_i
