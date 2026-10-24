# -*- coding: utf-8 -*-
'''
Created on Wed Apr  8 13:04:40 2026
@author: annaleoniesalzmann
'''

port_outDefault         = 10001
thinfirstlayer_Default  = False
SRF_Default             = False
excv_stepDefault        = 1.0 # reduce if soil collapses while excav. [m]

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import plxscripting.easy as plx
import os.path
import time
import copy

from plaxis_00calling_02helpers import (fct_writeRuntime,
                                        open_plaxisData,
                                        close_plaxis)
from plaxis_01model_02helpers   import (fct_modelModeller,
                                        fct_modelPhaser)
from plaxis_01model_03refstudy import fct_modelParamsPaper

###############################################################################
#------------------------------------ Main ------------------------------------
###############################################################################

def fct_modelParams(meshRef        = None,
                    ref_height     = None,        # slab OK (excavation)
                    raster         = None,        # quadratic or triangular
                    disx           = None,        # distance of SI elements
                    disy           = None,        # distance of SI elements
                    si_length      = None,        # length or height of the SI
                    si_radius      = None,        # radius of the SI
                    si_numberx     = None,        # number of SIs in in x dir
                    si_numbery     = None,        # number of SIs in in y dir
                    si_method      = 'ST',        # for better model
                    si_stiffness   = None,        # kPa
                    si_nu          = 0.2,
                    si_gamma       = None,        # before 25 (E>12e6) else 20
                    si_Rinter      = 1.0,         # for concrete more 0.9
                    soil           = 5,           # how many layers (only 1 or 2)
                    ltp_mat        = {},          # diff. material properties
                    ltp_height     = None,        # [meters]
                    slab_height    = None,        # [meters]
                    intf           = None,        # if interface considered
                    line_load      = None,        # [kPa]
                    loadsteps      = None,        # step num [0 to line_load]
                    water_depth    = None,
                    toweroverFac   = None,        #  more/less sorrounding
                    si_top         = None,
                    ):

    if (isinstance(soil, str) and any(name in soil.lower()
                                     for name in ('nguyen',
                                                  'yasser',
                                                  'stacho',
                                                  'bouassida'
                                                  ))):
        if   'nguyen' in soil.lower():      raster = 'triangular'
        elif 'yasser' in soil.lower():      raster = 'turnedquad'
        elif 'stacho' in soil.lower():      raster = 'quadratic'
        elif 'bouassida' in soil.lower():   raster = 'quadratic'
        # print(f'fct_modelParams is {slab_height}')
        params = fct_modelParamsPaper(soil,
                                      meshRef        = meshRef,
                                      ref_height     = ref_height,
                                      raster         = raster,
                                      disx           = disx,
                                      disy           = disy,
                                      si_length      = si_length,
                                      si_radius      = si_radius,
                                      si_numberx     = si_numberx,
                                      si_numbery     = si_numbery,
                                      si_method      = si_method,
                                      si_stiffness   = si_stiffness,
                                      ltp_height     = ltp_height,
                                      slab_height    = slab_height,
                                      line_load      = line_load,
                                      loadsteps      = loadsteps,
                                      water_depth    = water_depth,
                                      # *,
                                      # case           = None,
                                      toweroverFac   = toweroverFac,
                                      # paramsX        = None,
                                      # paramsY        = None,
                                      intf           = intf,
                                      # layers         = None,
                                      # soil_mat       = None,
                                      # si_mat         = None,
                                      ltp_mat        = ltp_mat,
                                      # slab_mat       = None,
                                      # intf_mat       = None
                                      )
        return params

    testmodel = soil == 'Refmodel'
    # Only use defaults when no value was supplied
    if raster is None:      raster = 'quadratic'
    if meshRef is None:     meshRef = (0.2,
                                       0.2,
                                       1.0,
                                       1.0,
                                       1.0) if testmodel else 'fineMedium'
    if ref_height is None:      ref_height  = 2.5   if testmodel else 0.0
    if si_numberx is None:      si_numberx  = 7
    if si_numbery is None:      si_numbery  = 7
    if disx is None:            disx        = 3.0   if testmodel else 1.5
    if disy is None:            disy        = 3.0   if testmodel else 1.5
    if si_stiffness is None:    si_stiffness   = 100e3
    if si_length is None:       si_length   = 12    if testmodel else 9
    if si_radius is None:       si_radius   = 0.35  if testmodel else 0.30
    if si_gamma is None:        si_gamma    = 19    if testmodel else 24
    if ltp_height is None:      ltp_height  = 0.0   if testmodel else 0.5
    if slab_height is None:     slab_height = 2.5   if testmodel else 0.3
    if intf is None:            intf        = False if testmodel else True
    if line_load is None:       line_load   = 450   if testmodel else 100
    if loadsteps is None:       loadsteps   = 9     if testmodel else 5
    if water_depth is None:     water_depth = -24   if testmodel else -20
    if toweroverFac is None:    toweroverFac = 3    if testmodel else 2.5
    
    if si_top is None:
        if testmodel: si_topboundary = 0 # no excavation
        else:         si_topboundary = ref_height - ltp_height - slab_height
    else: si_topboundary = si_top

    ### Mesh ------------------------------------------------------------------
    if isinstance(meshRef, str):
        mesh_map  = {#general
                     'coarse':          (0.25, 0.25, 1.0, 1.0, 1.0),
                     'mediumCoarse':    (0.20, 0.20, 0.9, 0.9, 0.9),
                     'medium':          (0.15, 0.15, 0.8, 0.8, 0.8),
                     'fineMedium':      (0.10, 0.10, 0.7, 0.7, 0.7),
                     'fine':            (0.05, 0.05, 0.6, 0.6, 0.6),
                     #specific
                     'fineMeshing':     (0.05, 0.20, 1.0, 1.0, 1.0),
                     'fineImprovement': (0.20, 0.05, 1.0, 1.0, 1.0),
                     'fineSoillayer':   (0.20, 0.20, 0.6, 1.0, 1.0),
                     'fineFoundation':  (0.20, 0.20, 1.0, 0.6, 1.0),
                     'fineLtp':         (0.20, 0.20, 1.0, 1.0, 0.6),
                     }
        mesh_distri, mesh_si, mesh_soil, mesh_slab, mesh_ltp = mesh_map[meshRef]
    elif len(meshRef) == 5:
        mesh_distri, mesh_si, mesh_soil, mesh_slab, mesh_ltp = meshRef
    else:
        raise ValueError("meshRef has to be STRING or TUPLE with (mesh_distri"
                         ", mesh_si, mesh_soil, mesh_slab, mesh_ltp)")
    dis_edgex   = disx/2 # distance of SI to the edge in 3D [meter]
    dis_edgey   = disy/2 # distance of SI to the edge in 3D [meter]
    paramsX     = (si_numberx-1)*disx + 2*dis_edgex
    paramsY     = (si_numbery-1)*disy + 2*dis_edgey
    toweroverX  = paramsX * toweroverFac
    toweroverY  = paramsY * toweroverFac

    ### Model the sourroundings -----------------------------------------------
    if testmodel:
        calc_layers   = [24]
        calc_seq      = [0]
        soil_material = [{'Model':      'HSS',
                          'Name':       'Soft',
                          'Colour':      5886249,
                          'Drainage':   'Drained',
                          'gamma':       19,
                          'gammaunsat':  19,
                          'gammasat':    19,
                          'ERef':        8000,
                          'E50Ref':      8000,
                          'EoedRef':     8000,
                          'EurRef':      24000,
                          'G0Ref':       72000,
                          'gamma07':     0.0002,
                          'powerm':      0.8,
                          'cRef':        20,
                          'phi':         25,
                          'psi':         0,
                          'Rinter':      1,
                          'K0':          0.577,
                          'POP':         0,
                          'nu':          0.2,
                          }
                         ]
    elif isinstance(soil, (int, tuple)):
        # soil_material_refs = [# due to Florian Spirkl
        #                       {'Model':         'HS', 
        #                        'Name':          'Soil_Soft',
        #                        'Colour':        5886249,
        #                        'Drainage':      'Drained',
        #                        'gamma':         17,
        #                        'gammaunsat':    17,
        #                        'gammasat':      17,
        #                        'ERef':          0.92e3,
        #                        'E50Ref':        3.125e3,
        #                       ' EoedRef':       2.5e3,
        #                        'EurRef':        16.25e3,
        #                        'G0Ref':         720e3,
        #                        'gamma07':       0.0002,
        #                        'powerm':        0.85,
        #                        'cRef':          5,
        #                        'phi':           15,
        #                        'psi':           0,
        #                        'Rinter':        1,
        #                        'K0':            0.741,
        #                        'POP':           0,
        #                        'nu':            0.426,
        #                        },
        #                       {'Model':         'HS',
        #                        'Name':          'Soil_Medium',
        #                        'Colour':        22016,
        #                        'Drainage':      'Drained',
        #                        'gamma':         18,
        #                        'gammaunsat':    18,
        #                        'gammasat':      18,
        #                        'ERef':          3.0e3,
        #                        'E50Ref':        10.0e3,
        #                        'EoedRef':       8.0e3,
        #                        'EurRef':        50.0e3,
        #                        'G0Ref':         250e3,
        #                        'gamma07':       0.0002,
        #                        'powerm':        0.70,
        #                        'cRef':          2,
        #                        'phi':           15,
        #                        'psi':           0,
        #                        'Rinter':        1,
        #                        'K0':            0.550,
        #                        'POP':           0,
        #                        'nu':            0.350,
        #                        },
        #                       {'Model':         'HS',
        #                        'Name':          'Soil_Stiff',
        #                        'Colour':        13607400,
        #                        'Drainage':      'Drained',
        #                        'gammaunsat':    19,
        #                        'gammasat':      19,
        #                        'gamma':         19,
        #                        'ERef':          62.4e3,
        #                        'E50Ref':        100e3,
        #                        'EoedRef':       80e3,
        #                        'EurRef':        240e3,
        #                        'G0Ref':         260e3,
        #                        'gamma07':       0.0001,
        #                        'powerm':        0.55,
        #                        'cRef':          1,
        #                        'phi':           37.5,
        #                        'psi':           0,
        #                        'nu':            0.281,
        #                        'Rinter':        1,
        #                        'K0':            0.391,
        #                        'POP':           0,
        #                        'thickness':     si_length-0.5,
        #                        }
        #                       ]
        # soil_material_refs = [# due to Klemens Phüringer TOTAL
        #                                              # layer 1
        #                       {'Model':         'HS',
        #                       'Name':           'Soil_Mutterboden',
        #                       'Colour':         5886249,
        #                       'Drainage':       'Drained',
        #                       'gamma':          18.5,
        #                       'gammaunsat':     18.5,
        #                       'gammasat':       18.5,
        #                       'ERef':           1,
        #                       'E50Ref':         1,
        #                       'EoedRef':        3,
        #                       'EurRef':         9,
        #                       'G0Ref':          720e3,
        #                       'gamma07':        0.0002,
        #                       'powerm':         0.5,
        #                       'cRef':           0.1,
        #                       'phi':            1,
        #                       'psi':            0,
        #                       'Rinter':         1,
        #                       'K0':             0.966,
        #                       'POP':            0,
        #                       'nu':             0.2,
        #                       'thickness':      0.2,
        #                       },
        #                                             # layer 2
        #                       {'Model':         'HS',
        #                       'Name':           'Soil_Soft1',
        #                       'Colour':         5886249,
        #                       'Drainage':       'Drained',
        #                       'gamma':          18.5,
        #                       'gammaunsat':     18.5,
        #                       'gammasat':       18.5,
        #                       'ERef':           3.5e3,
        #                       'E50Ref':         3.5e3,
        #                       'EoedRef':        3.5e3,
        #                       'EurRef':         10.5e3,
        #                       'G0Ref':          720e3,
        #                       'gamma07':        0.0002,
        #                       'powerm':         0.80,
        #                       'cRef':           0.1,
        #                       'phi':            22.5,
        #                       'psi':            0,
        #                       'Rinter':         1,
        #                       'K0':             0.617,
        #                       'POP':            0,
        #                       'nu':             0.2,
        #                       'thickness':      4-0.7,
        #                       },
        #                                             # layer 3
        #                       {'Model':         'HS',
        #                        'Name':          'Soil_Soft2',
        #                        'Colour':        5886249,
        #                        'Drainage':      'Drained',
        #                        'gamma':         19.5,
        #                        'gammaunsat':    19.5,
        #                        'gammasat':      19.5,
        #                        'ERef':          3500,
        #                        'E50Ref':        3500,
        #                        'EoedRef':       3500,
        #                        'EurRef':        10.5e3,
        #                        'G0Ref':         720e3,
        #                        'gamma07':       0.0002,
        #                        'powerm':        0.80,
        #                        'cRef':          0.1,
        #                        'phi':           22.5,
        #                        'psi':           0,
        #                        'Rinter':        1,
        #                        'K0':            0.617,
        #                        'POP':           0,
        #                        'nu':            0.2,
        #                        'thickness':     4,
        #                        },
        #                                              # layer 4
        #                       {'Model':         'HS',
        #                       'Name':           'Soil_Stiff1',
        #                       'Colour':         22016,
        #                       'Drainage':       'Drained',
        #                       'gammaunsat':     20,
        #                       'gammasat':       21, 
        #                       'ERef':           15e3,
        #                       'E50Ref':         15e3,
        #                       'EoedRef':        15e3,
        #                       'EurRef':         45e3,
        #                       'G0Ref':          260e3,
        #                       'gamma07':        0.0001,
        #                       'powerm':         0.70,
        #                       'cRef':           10,
        #                       'phi':            27.5,
        #                       'psi':            0,
        #                       'nu':             0.2,
        #                       'Rinter':         1,
        #                       'K0':             0.538,
        #                       'POP':            0,
        #                       'thickness':      0.5+0.5,
        #                       },
        #                                             # layer 5
        #                       {'Model':         'HS',
        #                       'Name':           'Soil_Stiff2',
        #                       'Colour':         13607400,
        #                       'Drainage':       'Drained',
        #                       'gammaunsat':     20,
        #                       'gammasat':       21,
        #                       'ERef':           30e3,
        #                       'E50Ref':         30e3,
        #                       'EoedRef':        30e3,
        #                       'EurRef':         90e3,
        #                       'G0Ref':          260e3,
        #                       'gamma07':        0.0001,
        #                       'powerm':         0.70,
        #                       'cRef':           20,
        #                       'phi':            27.5,
        #                       'psi':            0,
        #                       'nu':             0.2,
        #                       'Rinter':         1,
        #                       'K0':             0.538,
        #                       'POP':            0,
        #                       'thickness':      4,
        #                       },
        #                                             # layer 6
        #                       {'Model':         'HS',
        #                       'Name':           'Soil_Stiff3',
        #                       'Colour':         13607400,
        #                       'Drainage':       'Drained',
        #                       'gammaunsat':     20,
        #                       'gammasat':       21, 
        #                       'ERef':           70e3,
        #                       'E50Ref':         70e3,
        #                       'EoedRef':        70e3,
        #                       'EurRef':         210e3,
        #                       'G0Ref':          260e3,
        #                       'gamma07':        0.0001,
        #                       'powerm':         0.70,
        #                       'cRef':           20,
        #                       'phi':            27.5,
        #                       'psi':            0,
        #                       'nu':             0.2,
        #                       'Rinter':         1,
        #                       'K0':             0.538,
        #                       'POP':            0,
        #                       'thickness':      0.5,
        #                       } # so base always is working
        #                       ]
        soil_material_refs = [# due to Klemens Phüringer
                              {'Model':         'HS',
                              'Name':           'Soil_Soft1',
                              'Colour':         5886249,
                              'Drainage':       'Drained',
                              'gamma':          18.5,
                              'gammaunsat':     18.5,
                              'gammasat':       18.5,
                              'ERef':           3.5e3,
                              'E50Ref':         3.5e3,
                              'EoedRef':        3.5e3,
                              'EurRef':         10.5e3,
                              'G0Ref':          72e3,
                              'gamma07':        0.0002,
                              'powerm':         0.80,
                              'cRef':           0.1,
                              'phi':            22.5,
                              'psi':            0,
                              'Rinter':         1,
                              'K0':             0.617,
                              'POP':            0,
                              'nu':             0.2,
                              'thickness':      4 + 1,
                              }, # 4-si_topboundary
                              {'Model':         'HS',
                               'Name':          'Soil_Soft2',
                               'Colour':        5886249,
                               'Drainage':      'Drained',
                               'gamma':         18.5,
                               'gammaunsat':    18.5,
                               'gammasat':      18.5,
                               'ERef':          7e3,
                               'E50Ref':        7e3,
                               'EoedRef':       7e3,
                               'EurRef':        21e3,
                               'G0Ref':         72e3,
                               'gamma07':       0.0002,
                               'powerm':        0.80,
                               'cRef':          0.1,
                               'phi':           22.5,
                               'psi':           0,
                               'Rinter':        1,
                               'K0':            0.617,
                               'POP':           0,
                               'nu':            0.2,
                               'thickness':     9-0.5-4
                               },
                              {'Model':         'HS',
                              'Name':           'Soil_Stiff1',
                              'Colour':         22016,
                              'Drainage':       'Drained',
                              'gammaunsat':     20,
                              'gammasat':       21,
                              'ERef':           15e3,
                              'E50Ref':         15e3,
                              'EoedRef':        15e3,
                              'EurRef':         45e3,
                              'G0Ref':          260e3,
                              'gamma07':        0.0001,
                              'powerm':         0.70,
                              'cRef':           10,
                              'phi':            27.5,
                              'psi':            0,
                              'nu':             0.2,
                              'Rinter':         1,
                              'K0':             0.538,
                              'POP':            0,
                              'thickness':      0.5+0.5,
                              },
                              {'Model':         'HS',
                              'Name':           'Soil_Stiff2',
                              'Colour':         13607400,
                              'Drainage':       'Drained',
                              'gammaunsat':     20,
                              'gammasat':       21,
                              'ERef':           30e3,
                              'E50Ref':         30e3,
                              'EoedRef':        30e3,
                              'EurRef':         90e3,
                              'G0Ref':          260e3,
                              'gamma07':        0.0001,
                              'powerm':         0.70,
                              'cRef':           20,
                              'phi':            27.5,
                              'psi':            0,
                              'nu':             0.2,
                              'Rinter':         1,
                              'K0':             0.538,
                              'POP':            0,
                              'thickness':      3
                              },
                              {'Model':         'HS',
                              'Name':           'Soil_Stiff3',
                              'Colour':         13607401,
                              'Drainage':       'Drained',
                              'gammaunsat':     20,
                              'gammasat':       21, 
                              'ERef':           70e3,
                              'E50Ref':         70e3,
                              'EoedRef':        70e3,
                              'EurRef':         210e3,
                              'G0Ref':          260e3,
                              'gamma07':        0.0001,
                              'powerm':         0.70,
                              'cRef':           20,
                              'phi':            27.5,
                              'psi':            0,
                              'nu':             0.2,
                              'Rinter':         1,
                              'K0':             0.538,
                              'POP':            0,
                              'thickness':      0.5,
                              } # so base always is working
                              ]

        soil_material = []
        if isinstance(soil, int):
            layerCount = soil
            layers = []
            for i in range(layerCount):
                ref_mat = soil_material_refs[i].copy()
                layers.append(ref_mat['thickness'])
                soil_material.append(ref_mat)
        elif isinstance(soil, tuple):
            layerCount = len(soil)
            layers = []
            for i in range(layerCount):
                ref_mat = soil_material_refs[i].copy()
                ref_mat['thickness'] = abs(soil[i])
                layers.append(ref_mat['thickness'])
                soil_material.append(ref_mat)
        calc_layers = layers
        calc_seq = list(range(layerCount))
    elif isinstance(soil, list):
        calc_layers = soil[0]
        soil_material = soil[1]
        calc_seq = list(range(len(calc_layers)))
    else:
        raise ValueError('Not defined!')

    soil_depth = -sum(calc_layers)
    if water_depth < soil_depth:
        water_depth = soil_depth
    if (si_topboundary - si_length) < soil_depth - 0.5:
        # to take the rock into consideration
        # 0.5 = base interface is possible to extract
        si_length = round(si_topboundary - soil_depth - 0.5,3)
    soil_params = {'length':     0,
                   'depth':      soil_depth,
                   'head':       water_depth,
                   'minx':       0,
                   'maxx':       toweroverX+paramsX+toweroverX,
                   'miny':       0,
                   'maxy':       toweroverY+paramsY+toweroverY,
                   'mesh':       mesh_soil,
                   'meshdistri': mesh_distri,
                   'layers':     calc_layers,
                   'layer_seq':  calc_seq}

    ### Structural element: Soil improvement ----------------------------------
    si_params   = {'UK':         round(si_topboundary-si_length,3),
                   'OK':         round(si_topboundary,3),
                   'height':     si_length,
                   'radius':     si_radius,
                   'mesh':       mesh_si,
                   'raster':     raster,
                   'dis2D':      disx,
                   'disx':       disx, 'dis_edgex': dis_edgex, 'numx': si_numberx,
                   'disy':       disy, 'dis_edgey': dis_edgey, 'numy': si_numbery}
    si_material = {'Model':      'LE',
                   'Name':       f'SI_{si_method}',
                   'Method':     si_method,
                   'Drainage':   'Drained',
                   'Colour':     15272433,
                   'ERef':       si_stiffness,  # 2.8e7 for RSS
                   'nu':         si_nu,         # 0.2 for RSS
                   'Rinter':     si_Rinter,     # 0.9 for RSS
                   'gammaunsat': si_gamma,      # 24 for RSS
                   'gammasat':   si_gamma,
                   'gamma':      si_gamma,
                   }
    if si_length>0 and si_radius>0: si=True
    else: si=False

    ### Soil materials (Load Transfer Platform LTP) ---------------------------
    ltp_params    = {'height': ltp_height,
                     'mesh':   mesh_ltp,
                     'UK':     round(si_topboundary,3),
                     'OK':     round(si_topboundary+ltp_height,3),
                     'minx':   toweroverX,
                     'maxx':   toweroverX+paramsX,
                     'miny':   toweroverY,
                     'maxy':   toweroverY+paramsY,
                     }
    if isinstance(ltp_mat, int) and ltp_mat == 1:
        ltp_material  = {'Model':       'MC',
                         'Name':        'ltp',
                         'Colour':      9498246,
                         'Drainage':    'Drained',
                         'gammaunsat':  19,
                         'gammasat':    19,
                         'ERef':        8000,
                         'E50Ref':      8000,
                         'EoedRef':     8000,
                         'EurRef':      24000,
                         'G0Ref':       72000,
                         'gamma07':     0.0002,
                         'powerm':      0.8,
                         'cRef':        20,
                         'phi':         25,
                         'psi':         0,
                         'nu':          0.2,
                         'Rinter':      1,
                         'K0':          0.577,
                         'POP':         0,
                         }
    elif isinstance(ltp_mat, int) and ltp_mat == 2:
        ltp_material = {'Model':        'HSS',
                        'Name':         'ltp',
                        'Colour':       9498246,
                        'Drainage':     'Drained',
                        'gammaunsat':   21,
                        'gammasat':     21,
                        'gamma':        21,
                        'ERef':         8000,
                        'E50Ref':       70000,
                        'EoedRef':      70000,
                        'EurRef':       210000,
                        'G0Ref':        260000,
                        'gamma07':      0.0001,
                        'powerm':       0.7,
                        'cRef':         20,
                        'phi':          27.5,
                        'psi':          0,
                        'nu':           0.2,
                        'Rinter':       1,
                        'K0':           0.538,
                        'POP':          0,
                        }
    elif isinstance(ltp_mat, int) and ltp_mat == 3: #due to Florian Spirkl
        ltp_material = {'Model':        'HS',
                        'Name':         'ltp',
                        'Colour':       9498246,
                        'Drainage':     'Drained',
                        'gammaunsat':   19,
                        'gammasat':     19,
                        'ERef':         62.4e3,
                        'E50Ref':       100e3,
                        'EoedRef':      80e3,
                        'EurRef':       240e3,
                        'G0Ref':        260000,
                        'gamma07':      0.0001,
                        'powerm':       0.55,
                        'cRef':         1,
                        'phi':          37.5,
                        'psi':          0,
                        'nu':           0.281,
                        'Rinter':       1,
                        'K0':           0.391,
                        'POP':          0,
                        }
    elif isinstance(ltp_mat, int) and ltp_mat == 4: #due to Florian Spirkl
        ltp_material = {'Model':        'HS', # worked but not with MC
                       'Name':          'ltp',
                       'Colour':        9498246,
                       'Drainage':      'Drained',
                        'gammaunsat':   20,
                        'gammasat':     20,
                       'ERef':          75e3,
                       'E50Ref':        75e3,
                       'EoedRef':       75e3,
                       'EurRef':        225e3,
                       'G0Ref':         720e3,
                       'gamma07':       0.0002,
                       'powerm':        0.5,
                       'cRef':          0.1,
                       'phi':           40,
                       'psi':           5,
                       'Rinter':        1,
                       'K0':            0.357,
                       'POP':           0,
                       'nu':            0.2,
                       'thickness':     ltp_height,
                       }
    elif isinstance(ltp_mat, int) and ltp_mat == 5: #due to Florian Spirkl
        ltp_material = {'Model':        'MC',#adaption
                        'Name':         'LTP',
                        'Colour':       9498246,
                        'Drainage':     'Drained',
                        'gammaunsat':   20,
                        'gammasat':     20,
                        'ERef':         75e3,
                        'E50Ref':       75e3,
                        'EoedRef':      75e3,
                        'EurRef':       225e3,
                        'G0Ref':        720e3,
                        'gamma07':      0.0002,
                        'powerm':       0.5,
                        'cRef':         0.1,
                        'phi':          40,
                        'psi':          5,
                        'Rinter':       1,
                        'K0':           0.357,
                        'POP':          0,
                        'nu':           0.2,
                        'thickness':    ltp_height}
    else:
        if testmodel:
            ltp_material = {'Model':      'MC',
                            'Name':         'ltp',
                            'Colour':       9498246,
                            'Drainage':     'Drained',
                            'gammaunsat':   19,
                            'gammasat':     19,
                            'gamma':        19,
                            'ERef':         8000,
                            'E50Ref':       8000,
                            'EoedRef':      8000,
                            'EurRef':       24000,
                            'G0Ref':        72000,
                            'gamma07':      0.0002,
                            'powerm':       0.8,
                            'cRef':         20,
                            'phi':          25,
                            'psi':          0,
                            'nu':           0.2,
                            'Rinter':       1,
                            'K0':           0.577,
                            'POP':          0,
                            }
        else: ltp_material = {'Model':          'MC',#adaptions
                              'Name':           'LTP',
                              'Colour':         9498246,
                              'Drainage':       'Drained',
                              'gammaunsat':     20,
                              'gammasat':       20,
                              'ERef':           75e3,
                              'E50Ref':         75e3,
                              'EoedRef':        75e3,
                              'EurRef':         225e3,
                              'G0Ref':          720e3,
                              'gamma07':        0.0002,
                              'powerm':         0.5,
                              'cRef':           5,
                              'phi':            38,
                              'psi':            5,
                              'Rinter':         1,
                              'K0':             0.357,
                              'POP':            0,
                              'nu':             0.2,
                              'thickness':      ltp_height} # final params!
        ltp_material.update(ltp_mat)

    ### Structural element: Slab ----------------------------------------------
    slab_params   = {'height':      slab_height,
                     'mesh':        mesh_slab,
                     'UK':          round(si_topboundary
                                          +ltp_height,
                                          3),
                     'OK':          round(si_topboundary
                                          +ltp_height
                                          +slab_height,
                                          3),
                     'minx':        toweroverX,
                     'maxx':        toweroverX+paramsX,
                     'miny':        toweroverY,
                     'maxy':        toweroverY+paramsY}
    slab_material = {'Model':       'LE',
                     'Name':        'Slab',
                     'Colour':      6772842,
                     'Drainage':    'Non-porous',
                     'gammaunsat':  25, # LE
                     'gammasat':    25,
                     'gamma':       25,
                     'ERef':        32e6,
                     'nu':          0.2,
                     'Rinter':      (1.0 if testmodel else 0.7),
                     'cRef':        1000000, # MC
                     'phi':         0,
                     'E28':         30000000, # CM
                     'phiMax':      15,
                     'epsCPP':      -0.0025,
                     'fC28':        5000,
                     'fC0N':        0.15,
                     'fCFN':        0.75,
                     'fCUN':        0.1,
                     'aDuct':       10,
                     'fT28':        2600,
                     'GC28':        0.5,
                     'GT28':        0.1,
                     }

    ### Special Interface under Slab ------------------------------------------
    if testmodel:
        intf_material = {'Model':      'LE',
                         'Name':       'SlabInt',
                         'Drainage':   'Drained',
                         'Colour':      14452458,
                         'gamma':       0,
                         'mesh':        mesh_slab,
                         'ERef':        1,
                         'nu':          0.2,
                         'Rinter':      1.0,
                         'InterfaceStiffnessDet': 'Direct',
                         'knInter':      1.0e10,
                         'ksInter':      1.0e10,
                         }
    else:
        intf_material = {'Model':       'LE',
                         'Name':        'SlabInt',
                         'Drainage':    'Drained',
                         'Colour':      14452458,
                         'gammaunsat':  0,
                         'gammasat':    0,
                         'mesh':        mesh_slab,
                         'ERef':        1,
                         'nu':          0.2,
                         # 'Rinter':      1,
                         'Rinter':      0.7,
                         'InterfaceStiffnessDet': 'Direct',
                         'knInter':     1e6,
                         'ksInter':     0.2*1e6}
    ### Loading ---------------------------------------------------------------
    load_params = [line_load, loadsteps]

    return (soil_params,   soil_material,
            slab_params,   slab_material,
            si, si_params, si_material,
            intf,          intf_material,
            ltp_params,    ltp_material,
            load_params)

def fct_model(path, model, dimension,
              port_input, plaxispw,
              soil_params,   soil_material,
              slab_params,   slab_material,
              si, si_params, si_material,
              intf,          intf_material,
              ltp_params,    ltp_material,
              load_params,
              excv_step      = excv_stepDefault,
              thinfirstlayer = thinfirstlayer_Default,
              SRF            = SRF_Default,
              port_out       = port_outDefault):

    ### Set up the sourroundings ----------------------------------------------
    s_i, g_i = plx.new_server('localhost', port_input, password=plaxispw)
    s_o, g_o = plx.new_server('localhost', port_out,   password=plaxispw)
    os.makedirs(path, exist_ok=True)
    dimension = str(dimension)
    if dimension not in ['unitcell2', 'unitcell3', 'quarter3', 'full3']:
        raise ValueError (f'Dimension {dimension} not tested.')

    ### Materialinput as txt --------------------------------------------------
    line_load = load_params[0]
    loadsteps = load_params[1]
    materialdata = True
    if materialdata:
        t_start = time.time()
        file_path = os.path.join(path, 'input_parameters.txt')
        with open(file_path, 'w') as file:
            file.write('MODEL - INPUT PARAMETERS\n')
            file.write('='*60 + '\n\n')
            file.write('GENERAL\n')
            file.write('-'*60 + '\n')
            file.write('dimension    = ' + str(dimension) + '\n')
            file.write('line_load    = ' + str(line_load) + '\n')
            file.write('loadsteps    = ' + str(loadsteps) + '\n')
            file.write('si           = ' + str(si) + '\n')
            file.write('interface    = ' + str(intf) + '\n\n')
            file.write('SOIL_PARAMS\n')
            file.write('-'*60 + '\n')
            file.write(str(soil_params) + '\n\n')
            file.write('SOIL_MATERIAL\n')
            file.write('-'*60 + '\n')
            file.write(str(soil_material) + '\n\n')
            file.write('SLAB_PARAMS\n')
            file.write('-'*60 + '\n')
            file.write(str(slab_params) + '\n\n')
            file.write('SLAB_MATERIAL\n')
            file.write('-'*60 + '\n')
            file.write(str(slab_material) + '\n\n')
            file.write('SOIL IMPROVEMENT (SI)\n')
            file.write('-'*60 + '\n')
            file.write('SI_PARAMS:\n')
            file.write(str(si_params) + '\n\n')
            file.write('SI_MATERIAL:\n')
            file.write(str(si_material) + '\n\n')
            file.write('INTERFACE\n')
            file.write('-'*60 + '\n')
            file.write(str(intf_material) + '\n\n')
            file.write('LOAD TRANSFER PLATFORM (LTP)\n')
            file.write('-'*60 + '\n')
            file.write('LTP_PARAMS:\n')
            file.write(str(ltp_params) + '\n\n')
            file.write('LTP_MATERIAL:\n')
            file.write(str(ltp_material) + '\n\n')

    if dimension == 'quarter3':
        soil_params = copy.deepcopy(soil_params)
        slab_params = copy.deepcopy(slab_params)
        ltp_params  = copy.deepcopy(ltp_params)
        soil_params['maxx'] = (soil_params['minx'] + soil_params['maxx'])/2
        soil_params['maxy'] = (soil_params['miny'] + soil_params['maxy'])/2
        slab_params['maxx'] = (slab_params['minx'] + slab_params['maxx'])/2
        slab_params['maxy'] = (slab_params['miny'] + slab_params['maxy'])/2
        ltp_params['maxx']  = (ltp_params['minx']  + ltp_params['maxx'] )/2
        ltp_params['maxy']  = (ltp_params['miny']  + ltp_params['maxy'] )/2
    elif dimension == 'unitcell3':
        soil_params = copy.deepcopy(soil_params)
        slab_params = copy.deepcopy(slab_params)
        ltp_params  = copy.deepcopy(ltp_params)
        soil_params['minx'] = 0
        soil_params['miny'] = 0
        soil_params['maxx'] = si_params['disx']
        soil_params['maxy'] = si_params['disy']
        slab_params['minx'] = 0
        slab_params['miny'] = 0
        slab_params['maxx'] = si_params['disx']
        slab_params['maxy'] = si_params['disy']
        ltp_params['minx']  = 0
        ltp_params['miny']  = 0
        ltp_params['maxx']  = si_params['disx']
        ltp_params['maxy']  = si_params['disy']
    if si_params['UK'] < soil_params['depth']:
        si_params['UK'] = soil_params['depth']

    ### Create soil body and assign model structure ---------------------------
    # Check element raster and adjust model width
    if dimension.endswith('2'): #for 3D always quadratic 
        if si == True:
            dis    = si_params['dis2D']
            raster = si_params['raster']
            if   raster == 'quadratic':  si_equiR = 1.13
            elif raster == 'turnedquad':
                import math
                si_equiR = 1.13 / math.sqrt(2) # see Yasser Fayrouz
            elif raster == 'triangular': si_equiR = 1.05
            elif raster == 'hexagonal':  si_equiR = 1.28
            else: print('Choose a valid soil improvement layout.')
            soil_params['length'] = round(si_equiR*dis/2,3)
        else:
            soil_params['length'] = 1.0
            print('No improvement is considered in this model.')
    # Modeller
    (s_i, g_i,
     loading, soil_plate, soil_refine, soil_fixed,
     si_soil, si_position, si_elems, si_bases, si_shafts, si_tops,
     slab_soil, slab_elem, slab_int, slab_plate, slab_tops,
     ltp_soilLE, ltp_soil, ltp_elem,
     excv_elems, excv_fixed) = fct_modelModeller(s_i, g_i, dimension,
                                          port_input, plaxispw,
                                          soil_params, soil_material,
                                          line_load, loadsteps,
                                          slab_params, slab_material,
                                          si, si_params, si_material,
                                          intf, intf_material,
                                          ltp_params, ltp_material,
                                          excv_step = excv_step,
                                          thinfirstlayer=thinfirstlayer)
    print('Modeller: structures complited')

    ### Meshing ---------------------------------------------------------------
    g_i.gotomesh()
    if dimension.endswith('2'):
        for soilpoly in g_i.SoilPolygons:                                # soil
            soilpoly.setproperties('CoarsenessFactor', soil_params['mesh'])
        for poly in g_i.Polygons:
            if (si and
                (si_elems[0].Name.value + '_1') in poly.Name.value and
                 si_params['mesh'] != 1):
                poly.setproperties('CoarsenessFactor', si_params['mesh'])
            if (slab_params['height'] > 0 and
                (slab_elem[0].Name.value + '_1') in poly.Name.value and
                 slab_params['mesh'] != 1):
                poly.setproperties('CoarsenessFactor', slab_params['mesh'])
            if (ltp_params['height'] > 0 and
                (ltp_elem[0].Name.value + '_1') in poly.Name.value and
                ltp_params['mesh'] != 1):
                poly.setproperties('CoarsenessFactor', ltp_params['mesh'])
        for line in g_i.Lines:
            if (si and
                intf and
                si_tops[2].Name.value in line.Name.value and
                intf_material['mesh'] != 1):
                line.setproperties('CoarsenessFactor', intf_material['mesh'])
    elif dimension.endswith('3'):
        if (slab_params['height'] > 0 and
            slab_params['mesh'] != 1):
            g_i.apply(slab_elem[0], 'sps',
                      'CoarsenessFactor', slab_params['mesh'])
        if (si and
            intf and
            intf_material['mesh'] != 1):
            g_i.apply(si_tops[0], 'sps',
                      'CoarsenessFactor', intf_material['mesh'])
        if (ltp_params['height'] > 0 and
            ltp_params['mesh'] != 1):
            g_i.apply(ltp_elem[0], 'sps',
                      'CoarsenessFactor', ltp_params['mesh'])
        if soil_params['mesh'] != 1.0:
            g_i.apply(soil_refine[0], 'sps',
                      'CoarsenessFactor', soil_params['mesh'])
        if (si and
            si_params['mesh'] != 1):
            g_i.apply(si_elems, 'apply', 'sps',
                      'CoarsenessFactor', si_params['mesh'])
    print('Modeller: mesh properties complited')

    #https://bentleysystems.service-now.com/community?
    #id=kb_article&sysparm_article=KB0109481
    #DEFAULT CLASSES        Coarseness:     Rel.ElementSize (min 0.4)
        # very coarse:      0.100           2.0
        # coarse:           0.075           1.5
        # medium:           0.050           1.0
        # fine:             0.035           0.7
        # very fine:        0.025           0.5
    # OPT=0
    # unitcell 3 (mediumFine) = Generated 14393 elements, 25583 nodes
    # OPT=1
    # Coarseness 0.035
    # unitcell 3 = Generated 163103 elements, 255868 nodes
    # Coarseness 0.1 (mediumfine)
    # unitcell 3 = Generated 14393 elements, 25583 nodes
    if dimension.endswith('2'):
        g_i.mesh(soil_params['meshdistri'])
    else:
        if dimension.startswith('unitcell'):
            g_i.mesh(soil_params['meshdistri']) # = Coarseness
        else:
            meshingOpt = 1
            if meshingOpt == 0:
                g_i.mesh(soil_params['meshdistri']) # = Coarseness
            if meshingOpt == 1:
                g_i.mesh('Coarseness', soil_params['meshdistri'],
                         'UseEnhancedRefinements', True,
                         'EMRMinElementSize', 1E-5,
                         'UseSweptMeshing', False) # element distribution
            if meshingOpt == 2:
                g_i.mesh('RelativeElementSize', 1,#min 0.4
                         'UseEnhancedRefinements', True,
                         'EMRMinElementSize', 1E-5,
                         'EMRGlobalScale', 1.2,
                         'UseSweptMeshing', False) # element distribution

    start_time = time.time()
    while True:
        try:
            g_i.viewmesh()
            break
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed >= 180:
                raise ValueError("View could not be opened"
                                 f"within 3 minutes: {e}")
                break
            print(f"View not yet available: {e}")
            g_i.sleep(10)
    print('Modeller: meshing complited')

    ### Curvepoints for Analysis ----------------------------------------------
    for point in si_position if si else []:
        g_o.addcurvepoint('Node', point)
    g_o.update
    g_o.close()
    close_plaxis(port_out)
    print('Modeller: curvepoints complited')

    ### Staged construction ---------------------------------------------------
    g_i = fct_modelPhaser(g_i, dimension, path,
                          loading, loadsteps, line_load,
                          soil_refine, soil_params,
                          si_params, si_soil, si,
                          si_elems, si_bases, si_shafts, si_tops,
                          slab_params, slab_soil, slab_elem,
                          soil_plate, slab_int, slab_tops,
                          ltp_params, ltp_soilLE, ltp_soil, ltp_elem,
                          excv_elems, excv_fixed,
                          SRF)
    print('Modeller: phases complited')
    if dimension.endswith('2'):   modelname = model+'.p2dx'
    elif dimension.endswith('3'): modelname = model+'.p3d'
    else:                         modelname = model
    g_i.save(os.path.join(path, modelname)) # new - safer
    print(f'Modeller: {model} has been successfully created and saved.')

    ### Calculation and timer -------------------------------------------------
    calculation = True
    if calculation:
        g_i.calculate()
        g_i.save(os.path.join(path, modelname))
        # check if model is calculated
        phase = g_i.Phases[-1]
        status = int(phase.CalculationResult.value)
        if status != 1:
            try: g_o.close()
            except: pass
            raise RuntimeError(f"Model calculation failed in final phase "
                               f"'{phase.Identification.value}': "
                               f"CalculationResult={status}, "
                               f"LogInfo={phase.LogInfo.value}."
                               )
        else:
            print(f'Modeller: {os.path.basename(os.path.normpath(path))} '
                  'has been successfully calculated and saved.')
    t_end = time.time()
    runtime = t_end - t_start
    fct_writeRuntime(path, runtime, 'model')

    try: g_o.close()
    except: pass
    close_plaxis(port_out)

if __name__ == "__main__":
    name = 'model_base'
    pathFolder = 'Optimization - PLOTTING'
    dimension  = 'unitcell2'
    device     = 0
    model      = r'Model'
    port_ip, port_pw, path = open_plaxisData(device, dimension, pathFolder)
    fct_model(os.path.join(path,
                           f'{name}_{dimension}'),
              model,
              dimension,
              port_ip,
              port_pw,
              *fct_modelParams()) # with default values