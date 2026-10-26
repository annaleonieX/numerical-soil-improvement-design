"""
Reference-Studies:
 - Nguyen: doi:10.1061/(ASCE)GM.1943-5622.0002623, Tab. 1/2/4.
 - Yasser: doi:10.1186/s40703-022-00177-4, Tab. 2/3,
 - Stacho: doi:10.3311/PPci.13727, Tab. 2/4, Abschnitt 3.
 - Bouassida et al. (2024): doi:10.1061/IJGNAI.GMENG-9259, Tab. 1-4.
"""

from copy import deepcopy
from math import isfinite

def fct_modelParamsPaper(soil,
                         meshRef        = None,
                         ref_height     = None,
                         raster         = None,
                         disx           = None,
                         disy           = None,
                         si_length      = None,
                         si_radius      = None,
                         si_numberx     = None,
                         si_numbery     = None,
                         si_method      = None,
                         si_stiffness   = None,
                         ltp_height     = None,
                         slab_height    = None,
                         line_load      = None,
                         loadsteps      = None,
                         water_depth    = None,
                         *,
                         case           = None,
                         toweroverFac   = None,
                         paramsX        = None,
                         paramsY        = None,
                         intf           = None,
                         layers         = None,
                         soil_mat       = None,
                         si_mat         = None,
                         ltp_mat        = None,
                         slab_mat       = None,
                         intf_mat       = None):
    # print(f'fct_modelParamsPaper is {slab_height}')

    # ======================================================================
    # NGUYEN
    # ======================================================================
    if 'nguyen' in soil.lower():
        is_sc = (soil == 'Nguyen_SC' or
                 soil == 'Nguyen_VR' or
                 'sc' in si_method.lower() or
                 'vr' in si_method.lower())
        # expected_method = 'VR' if is_sc else 'RI'
        # if si_method is not None and si_method != expected_method:
        #     raise ValueError(f'Nguyen study requires {expected_method}')
        si_method = 'VR' if is_sc else 'RI'
        cases = ({'reference': (0.90, 1.60, 16.00), # 152mm page 6
                  'si1':       (0.80, 1.62, 14.95),
                  'si5':       (0.80, 2.00, 14.27),
                  'si17':      (0.80, 2.35, 14.32)}
                 if is_sc
                 else
                 {'reference': (0.44, 2.00, 15.60), # 30mm page 7
                  'ri3':       (0.44, 2.25, 16.38),
                  'ri10':      (0.62, 1.86, 16.96),
                  'ri14':      (0.44, 1.88, 16.77)})
        if case is None:
            case = 'reference'
        if case not in cases:
            raise ValueError('Nguyen case: ' + ', '.join(cases))
        diameter, spacing, length = cases[case]
        if si_length    is None: si_length = length
        if si_radius    is None: si_radius = diameter / 2
        if raster       is None: raster = 'triangular'
        if ltp_height   is None: ltp_height = 0.0 if is_sc else 0.4
        if slab_height  is None: slab_height = 0.5
        if ref_height   is None: ref_height = ltp_height + slab_height  # SI head z=0
        if line_load    is None: line_load = 120.0
        if water_depth  is None: water_depth = 0.0  # ground level WATER
        if paramsX is None:
            paramsX = (disx * (si_numberx + 1)
                       if disx is not None and si_numberx is not None
                       else 43.5)  # assumption: fallback for standalone calls
        if paramsY is None:
            paramsY = (disy * (si_numbery + 1)
                       if disy is not None and si_numbery is not None
                       else paramsX)
        if meshRef    is None: meshRef = 'fineMedium'  # assumption
        if si_numberx is None: si_numberx = max(1, round(paramsX/spacing) - 1)
        if si_numbery is None: si_numbery = si_numberx  # assumption: square grid
        if si_numberx < 1 or si_numbery < 1:
            raise ValueError('Nguyen: si_numberx/si_numbery must be positive')
        if disx is None: disx = paramsX / (si_numberx + 1)
        if disy is None: disy = paramsY / (si_numbery + 1)
        if (abs(disx * (si_numberx + 1) - paramsX) > 1e-8 or
            abs(disy * (si_numbery + 1) - paramsY) > 1e-8):
            raise ValueError('Nguyen: spacing must equal slab width / (si_num + 1)')
        calc_layers = [7.0, 4.0, 4.0, 15.0]
        soil_material = [{'Name':       'Soil_Nguyen_Layer_1',
                          'Model':      'MC',  # assumption
                          'Colour':     5886249,
                          'gammaunsat': 15,
                          'gammasat':   16.5,
                          'nu':         0.3,
                          'ERef':       2000.0 * 1.3 * 0.4 / 0.7,
                          'Drainage':   'Drained',  # assumption
                          'cRef':       5,  # assumption
                          'phi':        25,  # assumption
                          'psi':        0.0,  # assumption
                          'Rinter':     1.0,  # assumption
                          'K0':         0.5774,  # assumption
                          },
                         {'Name':       'Soil_Nguyen_Layer_2',
                          'Model':      'MC',  # assumption
                          'Colour':     5886249,
                          'gammaunsat': 16,
                          'gammasat':   18,
                          'nu':         0.3,
                          'ERef':       9000.0 * 1.3 * 0.4 / 0.7,
                          'Drainage':   'Drained',  # assumption
                          'cRef':       1,  # assumption
                          'phi':        30,  # assumption
                          'psi':        0.0,  # assumption
                          'Rinter':     1.0,  # assumption
                          'K0':         0.5,  # assumption
                          },
                         {'Name':       'Soil_Nguyen_Layer_3',
                          'Model':      'MC',  # assumption
                          'Colour':     5886249,
                          'gammaunsat': 15,
                          'gammasat':   16,
                          'nu':         0.3,
                          'ERef':       5000.0 * 1.3 * 0.4 / 0.7,
                          'Drainage':   'Drained',  # assumption
                          'cRef':       5,  # assumption
                          'phi':        25,  # assumption
                          'psi':        0.0,  # assumption
                          'Rinter':     1.0,  # assumption
                          'K0':         0.5774,  # assumption
                          },
                         {'Name':       'Soil_Nguyen_Layer_4',
                          'Model':      'MC',  # assumption
                          'Colour':     5886249,
                          'gammaunsat': 17,
                          'gammasat':   19,
                          'nu':         0.3,
                          'ERef':       35000.0 * 1.3 * 0.4 / 0.7,
                          'Drainage':   'Drained',  # assumption
                          'cRef':       1,  # assumption
                          'phi':        35,  # assumption
                          'psi':        0.0,  # assumption
                          'Rinter':     1.0,  # assumption
                          'K0':         0.4264,  # assumption
                          },
                         ]
        if is_sc: # vr
            if si_stiffness is None:
                Eref = 100000.0 * 1.3 * 0.4 / 0.7
            else: Eref = si_stiffness
            si_material = {'Name':       'SI_NguyenSC',
                           'Model':      'LE',  # assumption
                           'Colour':     15272433,
                           'Method':     'VR',
                           'ERef':       Eref,
                           'nu':         0.3,  # assumption
                           'Drainage':   'Drained',  # assumption
                           'gammaunsat': 19.0,  # assumption
                           'gammasat':   20.0,  # assumption
                           'cRef':       0.0,  # assumption
                           'phi':        42.0,
                           'psi':        0.0,  # assumption
                           'Rinter':     1.0,  # assumption
                           'K0':         0.3309,  # assumption
                           }
        else: # ri
            if si_stiffness is None:
                Eref = 4700.0 * (10e3 / 1000.0)**0.5 * 1000.0
            else: Eref = si_stiffness
            si_material = {'Name':       'SI_NguyenRI',
                           'Model':      'LE',  # assumption
                           'Colour':     15272433,
                           'Method':     'RI', # concrete strength is 10MPa
                           'ERef':       Eref, #ACI
                           'nu':         0.2,  # assumption
                           'Drainage':   'Non-porous',  # assumption
                           'gammaunsat': 24.0,  # assumption
                           'gammasat':   24.0,  # assumption
                           'Rinter':     1.0,  # assumption
                           }
        ltp_material = {'Name':       'LTP',
                        'Model':      'MC',  # assumption
                        'Colour':     5886249,
                        'ERef':       75000.0,
                        'phi':        38.0,
                        'cRef':       5,  # assumption
                        'psi':        5,  # assumption
                        'nu':         0.2,  # assumption
                        'gammaunsat': 19.0,  # assumption
                        'gammasat':   20.0,  # assumption
                        'Drainage':   'Drained',  # assumption
                        'Rinter':     1.0,  # assumption
                        'K0':         0.3843,  # assumption
                        }
        slab_material = {'Name':       'Slab',
                         'Model':      'LE',  # assumption
                         'Colour':     6772842,
                         'Drainage':   'Non-porous',  # assumption
                         'ERef':       32e6,  # assumption
                         'nu':         0.2,  # assumption
                         'gammaunsat': 25,  # assumption
                         'gammasat':   25,  # assumption
                         'Rinter':     1.0,  # assumption
                         }

    # ======================================================================
    # YASSER: Referenzfall ohne Lastverteilungsschicht
    # ======================================================================
    elif 'yasser' in soil.lower():
        if si_length    is None: si_length          = 15.0
        if si_radius    is None: si_radius          = 1/2
        if si_method    is None: si_method          = 'VR'
        if ltp_height   is None: ltp_height         = 0.0
        if slab_height  is None: slab_height        = 0.6 # page 4
        if raster       is None: raster             = 'turnedquad' # see pic
        if line_load    is None: line_load          = 34.0 # page 3
        if water_depth  is None: water_depth        = -20 # assumption
        if ref_height   is None: ref_height = ltp_height + slab_height #page15
        if disx is None:
            disx = (paramsX / (si_numberx + 1)
                    if (paramsX is not None and
                        si_numberx is not None)
                    else 2.2)
        if disy is None:
            disy = (paramsY / (si_numbery + 1)
                    if (paramsY is not None and
                        si_numbery is not None)
                    else 2.2)
        if paramsX is None:
            paramsX = ((si_numberx + 1) * disx
                       if si_numberx is not None
                       else 35)
        if paramsY is None:
            paramsY = ((si_numbery + 1) * disy
                       if si_numbery is not None
                       else 15)
        if si_numberx   is None: si_numberx = max(1, round(paramsX/disx)-1)
        if si_numbery   is None: si_numbery = max(1, round(paramsY/disy)-1)
        calc_layers = [40, 4, 8, 6, 11]
        soil_material = [{'Name':       'Soft_clay',
                          'Model':      'SS',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 17.0,
                          'gammasat':   17.0,
                          'cRef':       5.0,
                          'phi':        25.0,
                          'Cc':         0.46,
                          'Cs':         0.046,
                          'einit':      1.0,
                          'OCR':        1.3,
                          'K0':         1.5,
                          'nu':         0.2, #assumption
                          'psi':        5, #assumption
                          'Rinter':     1, # full contact
                          },
                         {'Name':       'Medium_stiff_clay',
                          'Model':      'SS',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 18.0,
                          'gammasat':   18.0,
                          'cRef':       5.0,
                          'phi':        25.0,
                          'Cc':         0.46,
                          'Cs':         0.046,
                          'einit':      1.1,
                          'OCR':        1.3,
                          'K0':         0.57,
                          'nu':         0.2, #assumption
                          'psi':        5,  #assumption
                          'Rinter':     1, # full contact
                          },
                         {'Name':       'Dense_sand1',
                          'Model':      'MC',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 18.0,
                          'gammasat':   18.0,
                          'cRef':       0.1,
                          'phi':        40.0,
                          'ERef':       50000.0,
                          'K0':         0.35,
                          'nu':         0.2, #assumption
                          'psi':        5, #assumption
                          'Rinter':     1, # full contact
                          },
                         {'Name':       'Very_stiff_clay', #all assumption
                          'Model':      'MC',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 18.0,
                          'gammasat':   18.0,
                          'cRef':       5.0,
                          'phi':        25.0,
                          'K0':         0.57,
                          'ERef':       30000.0,
                          'nu':         0.3,
                          'psi':        0,
                          'Rinter':     1,
                          },
                         {'Name':       'Dense_sand2',
                          'Model':      'MC',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 18.0,
                          'gammasat':   18.0,
                          'cRef':       0.1,
                          'phi':        40.0,
                          'ERef':       50000.0,
                          'K0':         0.35,
                          'nu':         0.2, #assumption
                          'psi':        5, #assumption
                          'Rinter':     1, # full contact
                          },
                         ]
        if si_stiffness is None: Eref = 100000.0
        else: Eref = si_stiffness
        si_material = {'Name':       'SI_Yasser',
                       'Model':      'LE',#MC
                       'Colour':     15272433,
                       'Method':     'VR',
                       'Drainage':   'Drained',
                       'gammaunsat': 19.0,
                       'gammasat':   19.0,
                       'cRef':       0.1,
                       'phi':        45.0,
                       'ERef':       Eref,
                       'K0':         0.29,
                       'nu':         0.2,
                       'psi':        2,
                       'Rinter':     1,
                       }
        ltp_material = {'Name':       'LTP',
                        'Model':      'MC',
                        'Colour':     9498246,
                        'Drainage':   'Drained',
                        'gammaunsat': 19.0,
                        'gammasat':   19.0,
                        'ERef':       100e3,  # kPa
                        'cRef':       0.1,    # kPa
                        'phi':        45.0,
                        'K0':         0.29,
                        'nu':         0.2, # assumptions
                        'psi':        5.0, # assumptions
                        'Rinter':     1.0, # assumptions
                        'POP':        0.0, # assumptions
                        }
        slab_material = {'Name':       'Slab',
                         'Model':      'LE',
                         'Colour':     6772842,
                         'ERef':       21e6, # page 4
                         'Drainage':   'Non-porous',
                         'gammaunsat': 25,
                         'gammasat':   25,
                         'nu':         0.2,
                         'Rinter':     1,
                         }

    # ======================================================================
    # STACHO
    # ======================================================================
    elif 'stacho' in soil.lower():
        if si_method    is None: si_method      = 'VR'
        if ('quarry' in soil.lower() or
            'quarry' in si_method.lower()):
            case = 'quarry'
        elif ('crushedpebble' in soil.lower() or
              'crushedpebble' in si_method.lower()
              ):
            case = 'crushedpebble'
        elif ('pebble' in soil.lower() or
              'pebble' in si_method.lower()
              ):
            case = 'pebble'
        else:
            print("reference case 'quarry' used")
            case = 'quarry'
        if raster       is None: raster         = 'quadratic'
        if disx         is None: disx           = 1.5 # and 1.8 and 2
        if disy         is None: disy           = 1.5 # and 1.8 and 2
        if si_length    is None: si_length      = 5.0
        if si_radius    is None: si_radius      = 0.3
        if ltp_height   is None: ltp_height     = 1.0
        if line_load    is None: line_load      = 80.0
        if paramsX      is None: paramsX        = 80
        if paramsY      is None: paramsY        = 80
        if slab_height  is None: slab_height    = 0 # page 799
        if ref_height   is None: ref_height     = ltp_height + slab_height
        if water_depth  is None: water_depth    = -5 # page 796
        if si_numberx   is None: si_numberx     = max(1,round(paramsX/disx)-1)
        if si_numbery   is None: si_numbery     = max(1,round(paramsY/disy)-1)
        ds = float(disx) - 1.8
        K_installation = 1.306 - (3103 / 3000) * ds + (269 / 300) * ds**2 #tab4
        Eoed_factor    = 1.47 - 0.57 * ds + 0.1 * ds**2 #tab4
        calc_layers = [5, 75]
        soil_material = [{'Name':       'Stacho_Clay',
                          'Model':      'HS',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 19.0,
                          'gammasat':   20.45,
                          'E50Ref':     5170.0,
                          'EoedRef':    5170.0 * Eoed_factor,
                          'EurRef':     25850.0,
                          'powerm':     0.85,
                          'phi':        26.6,
                          'psi':        0.0,
                          'cRef':       10.8,
                          'K0':         K_installation,
                          'nu':         0.2,#assumption
                          'Rinter':     1,#assumption
                          'POP':        0,#assumption
                          },
                         {'Name':       'Stacho_Gravel',
                          'Model':      'HS',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 21.0,
                          'gammasat':   21.55,
                          'E50Ref':     101e3,
                          'EoedRef':    101e3,
                          'EurRef':     303e3,
                          'powerm':     0.5,
                          'phi':        36.1,
                          'psi':        6.1,
                          'cRef':       1.0,
                          'K0':         0.4108,
                          'nu':         0.2,#assumption
                          'Rinter':     1,#assumption
                          'POP':        0,#assumption
                          },
                         ]
        slab_material = {'Model':       'LE', # taken from general studies
                         'Name':        'Slab',
                         'Colour':      6772842,
                         'Drainage':    'Non-porous',
                         'gammaunsat':  25,
                         'gammasat':    25,
                         'gamma':       25,
                         'ERef':        32e6,
                         'nu':          0.2,
                         'Rinter':      1,
                         }
        ltp_material = {'Name':       'LTP_Stacho',
                        'Model':      'HS',
                        'Colour':     5886249,
                        'Drainage':   'Drained',
                        'gammaunsat': 21.0,
                        'gammasat':   21.55,
                        'ERef':       108e3,
                        'E50Ref':     108e3,
                        'EoedRef':    108e3,
                        'EurRef':     324e3,
                        'kx':         864,
                        'ky':         864,
                        'powerm':     0.5,
                        'phi':        36.1,
                        'psi':        6.1,
                        'cRef':       1.0,
                        'K0':         0.4108,
                        'nu':         0.2,#assumption
                        'Rinter':     1,#assumption
                        'POP':        0,#assumption
                        }
        si_material = {'Name':       'SI_Stacho_quarry', # tab 2
                       'Model':      'HS',
                       'Colour':     15272433,
                       'Drainage':   'Drained',
                       'gammaunsat': 16.81,
                       'gammasat':   19.25,
                       'E50Ref':     100000, # 82500
                       'EoedRef':    82500,
                       'EurRef':     400000, # 247500
                       'powerm':     0.5,
                       'phi':        48.0,
                       'kx':         864,
                       'ky':         864,
                       'psi':        18.0,
                       'cRef':       57.0,
                       'K0':         0.3,
                       'nu':         0.2,#assumption
                       'Rinter':     1,#assumption
                       'POP':        0,#assumption
                       'Method':     'VR',
                       }
        if case == 'crushedpebble':
            si_material.update(Name       = 'SI_Stacho_crushedpebble',
                               gammaunsat = 17.36,
                               gammasat   = 19.55,
                               E50Ref     = 100000, # 65800
                               EoedRef    = 65800,
                               EurRef     = 210000, # 197400
                               phi        = 51.8,
                               psi        = 21.8,
                               cRef       = 1.0)
        elif case == 'pebble':
            si_material.update(Name       = 'SI_Stacho_pebble',
                               gammaunsat = 18.0,
                               gammasat   = 19.9,
                               E50Ref     = 60000, #45300,
                               EoedRef    = 45300,
                               EurRef     = 180000, #135900,
                               phi        = 49.9,
                               psi        = 19.9,
                               cRef       = 1.0)
        # Originalmodell: Last direkt auf Tragschicht, keine belegte Platte.
        # 60 m Lastbreite != 80 m Verbesserungsbreite: Builder anpassen.

    # ======================================================================
    # BOUASSIDA (2024)
    # ======================================================================
    elif 'bouassida' in soil.lower():
        if case is None: case = 'hc16'
        cases = {'hc8':  (8.0,  1.29, 34.59, 154),  #Tab.4
                 'hc10': (10.0, 1.62, 21.89, 97),
                 'hc12': (12.0, 1.90, 16.05, 71),
                 'hc14': (14.0, 2.13, 12.75, 57),
                 'hc16': (16.0, 2.33, 10.65, 47),
                 'hc18': (18.0, 2.51, 9.21, 41)}
        if case not in cases:
            raise ValueError('Available Case: hc8/10/12/14/16/18')
        length, spacing, _, _ = cases[case]
        if raster       is None: raster         = 'triangular'
        if disx         is None: disx           = spacing
        if disy         is None: disy           = spacing
        if si_length    is None: si_length      = length
        if si_radius    is None: si_radius      = 0.4
        if si_method    is None: si_method      = 'VR'
        if ltp_height   is None: ltp_height     = 0.5 # fig 2
        if line_load    is None: line_load      = 80.0
        if paramsX      is None: paramsX        = 15.0 # page 2
        if paramsY      is None: paramsY        = 15.0
        if slab_height  is None: slab_height    = 0.5 # fig 2
        if water_depth  is None: water_depth    = 0.0 # guess
        if ref_height   is None: ref_height     = ltp_height + slab_height
        if si_numberx   is None: si_numberx = max(1,round(paramsX/disx)-1)
        if si_numbery   is None: si_numbery = max(1,round(paramsY/disy)-1)

        calc_layers = [10, 10]
        soil_material = [{'Name':       'Bouassida_Clay_1',
                          'Model':      'MC', # page 6
                          'Colour':     5886249,
                          'Drainage':   'Drained', # maybe undrained A
                          'gammaunsat': 17.0,
                          'gammasat':   17.0,
                          'einit':      0.9,
                          'Cc':         0.35,
                          'Cs':         0.035, # assumption
                          'ERef':       3e3,
                          'nu':         0.4,
                          'cRef':       20.0,
                          'phi':        0.0,
                          'psi':        0, # assumption
                          'Rinter':     1, # assumption
                          },
                         {'Name':       'Bouassida_Clay_2',
                          'Model':      'MC',
                          'Colour':     5886249,
                          'Drainage':   'Drained',
                          'gammaunsat': 18.0,
                          'gammasat':   18.0,
                          'einit':      0.9,
                          'Cc':         0.35,
                          'Cs':         0.035, # assumption
                          'ERef':       3.5e3,
                          'nu':         0.35,
                          'cRef':       25.0,
                          'phi':        0.0,
                          'psi':        0, # assumption
                          'Rinter':     1, # assumption
                          },
                         ]
        si_material = {'Name':       'SI_Bouassida', # page 3
                       'Model':      'LE',
                       'Colour':     15272433,
                       'Method':     'VR',
                       'Drainage':   'Drained',
                       'gammaunsat': 19,
                       'gammasat':   19,
                       'gamma':      19,
                       'ERef':       30e3,
                       'nu':         0.25,
                       'Rinter':     1,
                       }
        slab_material = {'Model':       'LE', # taken from general studies
                         'Name':        'Slab',
                         'Colour':      6772842,
                         'Drainage':    'Non-porous',
                         'gammaunsat':  25,
                         'gammasat':    25,
                         'gamma':       25,
                         'ERef':        32e6,
                         'nu':          0.2,
                         'Rinter':      1,
                         }
        ltp_material = {'Model':          'MC',#taken from general study
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
                        'nu':             0.2,}
    else:
        raise ValueError('Unbekannte Studie: ' + str(soil))

    # Uebergebene Werte haben Vorrang ---------------------------------------
    intf_material = {'Model':       'LE',
                     'Name':        'SlabInt',
                     'Drainage':    'Drained',
                     'Colour':      14452458,
                     'gammaunsat':  0,
                     'gammasat':    0,
                     'mesh':        0.7,
                     'ERef':        1,
                     'nu':          0.2,
                     'Rinter':      0.7,
                     'InterfaceStiffnessDet': 'Direct',
                     'knInter':     1e6,
                     'ksInter':     0.2*1e6,
                     }
    if toweroverFac is None: toweroverFac = 1.0  # ignored by proper UC2 builder
    if intf         is None: intf = False
    if layers       is not None: calc_layers = list(layers)
    if soil_mat     is not None: soil_material = deepcopy(soil_mat)
    si_material['Method'] = si_method
    if si_mat       is not None: si_material.update(deepcopy(si_mat))
    if ltp_mat      is not None: ltp_material.update(deepcopy(ltp_mat))
    if slab_mat     is not None: slab_material.update(deepcopy(slab_mat))
    if intf_mat     is not None: intf_material.update(deepcopy(intf_mat))

    # Numerische Arbeitsvorgaben, keine Literaturwerte ----------------------
    if meshRef      is None: meshRef = 'fineMedium'
    if loadsteps    is None: loadsteps = 3 # since loads are small
    mesh_map = {'coarse':          (0.05, 0.25, 1.0, 1.0, 1.0),
                'mediumCoarse':    (0.04, 0.20, 0.9, 0.9, 0.9),
                'medium':          (0.03, 0.15, 0.8, 0.8, 0.8),
                'fineMedium':      (0.02, 0.10, 0.7, 0.7, 0.7),
                'fine':            (0.01, 0.05, 0.6, 0.6, 0.6),
                'fineMeshing':     (0.05, 0.20, 1.0, 1.0, 1.0),
                'fineImprovement': (0.20, 0.05, 1.0, 1.0, 1.0),
                'fineSoillayer':   (0.20, 0.20, 0.6, 1.0, 1.0),
                'fineFoundation':  (0.20, 0.20, 1.0, 0.6, 1.0),
                'fineLtp':         (0.20, 0.20, 1.0, 1.0, 0.6),
                }
    mesh = mesh_map[meshRef] if isinstance(meshRef, str) else meshRef
    if len(mesh) != 5 or any(not isfinite(x) or x <= 0 for x in mesh):
        raise ValueError('meshRef: Name oder fuenf positive Zahlen.')
    mesh_distri, mesh_si, mesh_soil, mesh_slab, mesh_ltp = mesh

    # Noch offene Eingaben gemeinsam nennen --------------------------------
    geometry = dict(ref_height   = ref_height,
                    raster       = raster,
                    disx         = disx,
                    disy         = disy,
                    si_length    = si_length,
                    si_radius    = si_radius,
                    si_numberx   = si_numberx,
                    si_numbery   = si_numbery,
                    ltp_height   = ltp_height,
                    slab_height  = slab_height,
                    line_load    = line_load,
                    water_depth  = water_depth,
                    toweroverFac = toweroverFac,
                    paramsX      = paramsX,
                    paramsY      = paramsY,
                    intf         = intf)
    missing = [key
               for key, value in geometry.items()
               if value is None]
    missing += ['calc_layers[%d]' % i
                for i, h in enumerate(calc_layers)
                if h is None]
    materials = [('soil_material[%d]' % i, mat)
                 for i, mat in enumerate(soil_material)]
    materials.append(('si_material', si_material))
    if slab_height is not None and slab_height > 0:
        materials.append(('slab_material', slab_material))
    if ltp_height is not None and ltp_height > 0:
        materials.append(('ltp_material', ltp_material))
    if intf:
        materials.append(('intf_material', intf_material))
    for name, mat in materials:
        if not mat: missing.append(name)
        missing += [name + '.' + key
                    for key, value in mat.items()
                    if value is None]
    if missing:
        raise ValueError('Noch zu ergaenzen: ' + ', '.join(missing))

    # Positive integer inputs
    for name, value in (('si_numberx', geometry['si_numberx']),
                        ('si_numbery', geometry['si_numbery']),
                        ('loadsteps',  loadsteps),
                        ):
        if (isinstance(value, bool) or
            not isinstance(value, int) or
            value < 1):
            raise ValueError(f'{name} must be a positive integer ({value}).')
    # Interface and grid configuration
    if not isinstance(intf, bool):
        raise ValueError('intf must be True or False.')
    # Finite, strictly positive dimensions
    for name in ('disx','disy','si_radius','si_length','paramsX','paramsY',):
        value = geometry[name]
        if not isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and greater than zero.')
    # Finite, non-negative inputs
    for name in ('slab_height','ltp_height','toweroverFac','line_load'):
        value = geometry[name]
        if not isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative.')
    # Reference and groundwater elevations
    for name, value in (('ref_height',  ref_height),
                        ('water_depth', water_depth),
                        ):
        if not isfinite(value):
            raise ValueError(f'{name} must be finite.')
    # Soil layers and material assignments
    if not calc_layers:
        raise ValueError('At least one soil layer must be defined.')
    if len(calc_layers) != len(soil_material):
        raise ValueError('Each soil layer must have exactly '
                         'one soil material.')
    if any(not isfinite(height) or height <= 0 for height in calc_layers):
        raise ValueError('All soil layer thicknesses must '
                         'be finite and positive.')
    # Vertical position of the soil improvement elements
    soil_depth          = -sum(calc_layers)
    si_topboundary      = round(ref_height - ltp_height - slab_height, 4)
    si_bottomboundary   = si_topboundary - si_length
    if si_topboundary > 0:
        raise ValueError(f'The column heads {si_topboundary} must '
                         'not extend above the soil surface.')
    if si_bottomboundary <= soil_depth:
        raise ValueError('The column toes must remain above the model bottom.')

    # Clearances between column centres and foundation edges
    dis_edgex = (paramsX - (si_numberx - 1) * disx) / 2
    dis_edgey = (paramsY - (si_numbery - 1) * disy) / 2
    if min(dis_edgex, dis_edgey) < si_radius:
        raise ValueError(
            f"All columns must fit entirely within the foundation footprint. "
            f"paramsX={paramsX}, paramsY={paramsY}, "
            f"si_numberx={si_numberx}, si_numbery={si_numbery}, "
            f"disx={disx}, disy={disy}, si_radius={si_radius}, "
            f"dis_edgex={dis_edgex}, dis_edgey={dis_edgey}, "
            f"clearance_x={dis_edgex - si_radius}, "
            f"clearance_y={dis_edgey - si_radius} (must be >= 0)."
            )

    # Model extensions beyond the foundation
    toweroverX = paramsX * toweroverFac
    toweroverY = paramsY * toweroverFac
    soil_params = {'length':     0,
                   'depth':      soil_depth,
                   'head':       water_depth,
                   'minx':       0,
                   'maxx':       2 * toweroverX + paramsX,
                   'miny':       0,
                   'maxy':       2 * toweroverY + paramsY,
                   'mesh':       mesh_soil,
                   'meshdistri': mesh_distri,
                   'layers':     list(calc_layers),
                   'layer_seq':  list(range(len(calc_layers))),
                   }
    si_params = {'UK':          si_topboundary - si_length,
                 'OK':          si_topboundary,
                 'height':      si_length,
                 'radius':      si_radius,
                 'mesh':        mesh_si,
                 'raster':      raster,
                 'dis2D':       disx,
                 'disx':        disx,
                 'dis_edgex':   dis_edgex,
                 'numx':        si_numberx,
                 'disy':        disy,
                 'dis_edgey':   dis_edgey,
                 'numy':        si_numbery,
                 }
    ltp_params = {'height':     ltp_height,
                  'mesh':       mesh_ltp,
                  'UK':         si_topboundary,
                  'OK':         si_topboundary + ltp_height,
                  'minx':       toweroverX,
                  'maxx':       toweroverX + paramsX,
                  'miny':       toweroverY,
                  'maxy':       toweroverY + paramsY,
                  }
    slab_params = {'height':    slab_height,
                   'mesh':      mesh_slab,
                   'UK':        si_topboundary + ltp_height,
                   'OK':        ref_height,
                   'minx':      toweroverX,
                   'maxx':      toweroverX + paramsX,
                   'miny':      toweroverY,
                   'maxy':      toweroverY + paramsY,
                   }
    for mat in soil_material + [si_material,
                                ltp_material,
                                slab_material]:
        if mat.get('gammaunsat') is not None:
            mat['gamma'] = mat['gammaunsat']
    si = si_length > 0 and si_radius > 0
    load_params = [line_load, loadsteps]

    return (soil_params,        soil_material,
            slab_params,        slab_material,
            si, si_params,      si_material,
            intf,               intf_material,
            ltp_params,         ltp_material,
            load_params)
