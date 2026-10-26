# -*- coding: utf-8 -*-
'''
Created on Wed Apr  8 15:07:29 2026
@author: annaleoniesalzmann
'''

###############################################################################
#-------------------------------- USER-INPUT ----------------------------------
###############################################################################

uwDefault   = False
saveDefault = True
dpiDefault  = 600
phase_siDef = 1
phaseDef    = 10
CM_PER_INCH = 2.54
figWidthDEF_cm = 15
exclude_methodsDefault = ('ifL',
                          'fa_LoM',
                          'fa_SLM',
                          'fa_SLoM',
                          'fa_firstSM',
                          'fa_difLM',
                          'fa_difLoM',
                          'fa_difSLM',
                          'fa_difSLoM',
                          'fa_diffirstSM',
                          )

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import glob, re, os, ast, math
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict

from plaxis_00calling_02helpers import (fct_setPlotstyle,
                                        fct_formatSave,
                                        fct_sortKey,
                                        fct_mathLabel)

# Publication settings: input data remain F_...; display labels use R_....
DEFAULT_MODEL_SETTINGS = {
    'siNum': 7, 'qd': 100.0, 'tslab': 0.3,
    'tLTP': 0.5, 's': 1.5, 'ESI': 100000.0,  # ESI in kPa
}

def fct_displayForce(value):
    return re.sub(r'(?<![A-Za-z])F_?(elem|soil|shaft|base)(?=\b|_)',
                  lambda m: 'R_' + m.group(1), str(value), flags=re.I)

def fct_categoryInfo(category):
    """Human-readable category label and heading, independently of CSV keys."""
    c = str(category).lower()
    if 'sinum' in c: return 'Piles per row [-]', 'piles per row'
    if 'ltp' in c: return 'LTP thickness [m]', 'LTP thickness'
    if c.startswith('dis') or 'spacing' in c: return r'Spacing $s$ [m]', 'spacing'
    if 'stiff' in c or c in ('e', 'esi') or c.startswith('e_'):
        return r'SI stiffness $E_{SI}$ [kPa]', 'SI stiffness'
    if c.startswith('load') or c in ('qd', 'q'):
        return 'Surface load [kPa]', 'surface load'
    if c.startswith('slab') or 'tslab' in c:
        return 'Slab thickness [m]', 'slab thickness'
    return str(category).replace('_', ' ').capitalize(), str(category).replace('_', ' ')

def fct_categoryValue(category, number, defaults=DEFAULT_MODEL_SETTINGS):
    if str(number).lower() not in ('default', 'base', 'ref'):
        return str(number)
    c = str(category).lower()
    if 'sinum' in c: return str(defaults['siNum'])
    if 'ltp' in c: return str(defaults['tLTP'])
    if c.startswith('dis'): return str(defaults['s'])
    if c.startswith('load'): return str(defaults['qd'])
    if c.startswith('slab'): return str(defaults['tslab'])
    if c.startswith('e') or 'stiff' in c: return str(defaults['ESI'])
    return str(number)

def fct_positionLabel(suffix, useEdges=True):
    value = str(suffix or '').lstrip('_') or 'Unitcell'
    if not useEdges and value.lower() in ('edgex', 'edgey'):
        return 'Edge'
    return value

def fct_adjustCm(fig, *, left=1.2, right=1.0, bottom=1.2, top=0.9,
                 wspace=None, hspace=None):
    """All margins are physical centimetres; gaps may remain subplot ratios."""
    width, height = (v * CM_PER_INCH for v in fig.get_size_inches())
    kw = dict(left=left/width, right=1-right/width,
              bottom=bottom/height, top=1-top/height)
    if wspace is not None: kw['wspace'] = wspace
    if hspace is not None: kw['hspace'] = hspace
    fig.subplots_adjust(**kw)

def fct_wrapTitle(title, width=66):
    import textwrap
    return '\n'.join(textwrap.wrap(str(title), width=width,
                                   break_long_words=False))

###############################################################################
#--------------------------- Helpers: Plotlayout ------------------------------
###############################################################################

def get_digit(number):
    try:
        float(number)
        return True
    except (ValueError, TypeError):
        return False
    
def find_maxPhase(path,
                  folder,
                  phaseDef=phaseDef):
    files = glob.glob(os.path.join(path,
                                   folder,
                                   'summary_value*',
                                   'Fsu_phases*.csv'))
    if not files:
        print(f'{path} -> {folder} not found')
        phase = phaseDef
    else:
        csv_file = files[0]
        df = pd.read_csv(csv_file, sep=';')
        phases = pd.to_numeric(df['Phase'], errors='coerce')
        if 'Name' in df.columns:
            load_steps = pd.to_numeric(
                df['Name'].astype('string')
                  .str.extract(r'Load step\s+(\d+)')[0],
                errors='coerce'
            )
            load_steps = load_steps.where(phases.notna())
        else:
            load_steps = pd.Series(dtype=float)
        
        if load_steps.notna().any():
            phase = int(phases.loc[load_steps.idxmax()])
        elif phases.notna().any():
            phase = int(phases.max())
        else:
            phase = phaseDef
    return phase

def fct_parseKey(key):
    parts     = key.split('_')
    typ       = parts[0].lower()
    number    = parts[1].lower()
    dimension = parts[2].lower() if len(parts) > 2 else ''
    suffix = '_' + parts[-1] if len(parts) > 3 else ''
    return typ, number, dimension, suffix

def fct_sortModels(models):
    return dict(sorted(models.items(),
                       key=lambda item:
                           (
                           fct_sortKey(fct_parseKey(item[0])[1]),
                           fct_sortKey(item[0])
                           )
                       )
                )

def fct_setPlottext(S=6, M=8, L=10):
    # Apply consistent plot style.
    plt.rc('font',   size=S)
    plt.rc('axes',   titlesize=M,
                     labelsize=M)
    plt.rc('xtick',  labelsize=S)
    plt.rc('ytick',  labelsize=S)
    plt.rc('legend', fontsize=S)
    plt.rc('figure', titlesize=L)

def fct_setPlotline(key):
    # Return linestyle based on dimension/position.
    _, _, dimension, suffix = fct_parseKey(key)
    if dimension == 'all':
        linestyle = {'unitcell2': '-',
                     'unitcell3': (0, (3, 1, 1, 1, 1, 1)),
                     'quarter3': '--',
                     'full3': ':',
                     }
        suffix = 'var'
    elif dimension.endswith('2'): # unitcell2
        linestyle = 'solid'
        suffix = ''
    elif dimension.endswith('3'):
        if len(key.split('_')) > 3: # quarter3 or full3
            if   suffix == '_Middle':      linestyle = 'dashed'  #--
            elif suffix == '_Corner':      linestyle = 'dotted'  #:
            elif suffix == '_EdgeY':       linestyle = 'dashdot' #-.
            elif suffix == '_EdgeX':       linestyle = (0, (3, 1, 1, 1, 1, 1))
            elif suffix == '_MiddleEdgeY': linestyle = 'dashed'  #--
            elif suffix == '_CornerEdgeX': linestyle = 'dotted'  #:
            elif suffix == '_MiddleEdgeX': linestyle = 'dashed'  #--
            elif suffix == '_CornerEdgeY': linestyle = 'dotted'  #:
            else: raise ValueError(f'Position {suffix} is not defined!')
        else: # unitcell3 or all have the same position
            linestyle = (0, (3, 1, 1, 1, 1, 1)) # densely dashdotdotted
            suffix = ''
    else: raise ValueError(f'Dimension {dimension} is not defined!')
    return linestyle, suffix

def fct_formatLegend(fig_or_ax,
                     typ        = 'outside',
                     handles    = None,
                     xLegend    = None,
                     yLegend    = None,
                     title      = None):
    if typ == 'below':
        # call with fct_formatLegend(ax, 'below')
        # if isinstance(fig_or_ax, plt.Figure):
        #     fontsize = max(t.get_fontsize()
        #                    for ax in fig_or_ax.axes
        #                    for t in ax.get_yticklabels())
        # elif isinstance(fig_or_ax, plt.Axes):
        #     fontsize = max(t.get_fontsize()
        #                    for t in fig_or_ax.get_yticklabels())
        # else:
        #     raise TypeError('Input must be a matplotlib Figure or Axes object')
        # print(fontsize)
        fig_or_ax.legend(loc            = 'lower center',
                         bbox_to_anchor = (0.5, -0.14),
                         ncol           = 3,
                         framealpha     = 1,
                         facecolor      = 'white',
                         edgecolor      = 'white')
    elif typ == 'outside':
        # call with fct_formatLegend(ax, 'outside')
        gap=0.04
        fig_or_ax.legend(loc            = 'upper left', # reference point
                         bbox_to_anchor = (1+gap, 1),  # position outside
                         borderaxespad  = 0, # deletes excess distance
                         framealpha     = 1, # background transparent
                         facecolor      = 'white',
                         edgecolor      = 'black',
                         )
    elif typ == 'handles':
        legend = fig_or_ax.legend(handles       = handles,
                                  loc           = 'upper left',
                                  bbox_to_anchor= (xLegend, yLegend),
                                  title         = title,
                                  ncol          = (2
                                                   if len(handles) > 6
                                                   else 1),
                                  edgecolor     = 'black',
                                  )
        return legend
    else: raise ValueError(f'Legendposition {typ} is not defined!')

def fct_getColor(colors, number, seen_numbers, counter, n_models):
    if number not in seen_numbers:
        seen_numbers[number] = counter
        counter += 1
    n_base_colors = len(colors)
    n_levels = max(1, math.ceil(n_models / n_base_colors)) # 8/5 -> 2
    num_idx = seen_numbers[number]
    group = (num_idx // n_levels) % n_base_colors
    level = num_idx % n_levels
    color = colors[group][level]
    return color, seen_numbers, counter

###############################################################################
#----------------------------- Helpers: Plotaxis ------------------------------
###############################################################################

def fct_niceNumber(x, rounding=True): # within fct_niceAxis
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    if rounding:
        if   f < 1.5: nf = 1
        elif f < 3:   nf = 2
        elif f < 7:   nf = 5
        else:         nf = 10
    else:
        if   f <= 1:  nf = 1
        elif f <= 2:  nf = 2
        elif f <= 5:  nf = 5
        else:         nf = 10
    return nf * 10**exp

def fct_niceAxis(data, n_ticks=6): # within fct_formatAxis
    data = np.array(data)
    data = data[~np.isnan(data)] # eliminate NaN (relevant for ratios: SAR,...)
    if len(data) == 0: return 0, 1, [0, 1] # if all Nan or totally empty
    dmin = np.min(data)
    dmax = np.max(data)
    if dmin == dmax: return dmin, dmax, [dmin, dmax] # if no deviation
    data_range = fct_niceNumber(dmax - dmin, rounding=False)
    step = fct_niceNumber(data_range / (n_ticks - 1), rounding=True)
    nice_min = math.floor(dmin / step) * step
    nice_max = math.ceil(dmax / step) * step
    if nice_min == nice_max:
        nice_max += step
    ticks = np.arange(nice_min, nice_max + 0.5 * step, step)
    return nice_min, nice_max, ticks

def fct_formatAxis(axis,
                   xall, yall,
                   title=None,
                   xlabel=None, xinv=False,
                   ylabel=None, yinv=False):
    # X axis
    xmin, xmax, xticks = fct_niceAxis(xall)
    axis.set_xlim(xmin, xmax)
    axis.set_xticks(xticks)
    if xinv:   axis.invert_xaxis()
    if xlabel: axis.set_xlabel(xlabel)
    # Y axis
    ymin, ymax, yticks = fct_niceAxis(yall)
    axis.set_ylim(ymin, ymax)
    axis.set_yticks(yticks)
    if yinv:   axis.invert_yaxis()
    if ylabel: axis.set_ylabel(ylabel)
    # General
    if title: axis.set_title(title)
    axis.set_axisbelow(True)
    axis.grid(color='0.85', lw=0.4, alpha=0.4)

###############################################################################
#--------------------------- Helpers: Information -----------------------------
###############################################################################

def fct_abbrevLabel(descr: str) -> str: # now fct_abbrevFull used = interactive
    mapping = {
        #2D
        'due to the slab interface':                                            'if_tra',
        'due to the top interface':                                             'if_tra',
        'due to base interface':                                                'if_tra',
        'due to interface stresses s_1':                                        'if_tra',
        'due to crosssection stresses along the base':                          'cs_tra',
        'due to crosssection stresses under surface':                           'cs_tra',
        'due to crosssection stresses above SI':                                'cs_tra',
        'due to crosssection stresses s_xy':                                    'cs_tra',
        'due to crosssection stresses s_xx':                                    'cs_tra',
        'due to crosssection stresses s_yy':                                    'cs_tra',
        'due to InternalForceAnalysis with material and line':                  'fa_oneMl',
        'due to the force difference':                                          'if_dif',
        'due to the force difference due to crosssection':                      'cs_dif',
        #2D - shaft
        'due to stresses from var=0 above and below pile (force difference)':   'if_dif',
        'due to interface stresses s_1 with limits (trapezoidal)':              'ifL_tra',
        'due to crosssection stresses s_xy (trapezoidal)':                      'cs_tra',
        'due to crosssection stresses s_xx (trapezoidal)':                      'cs_traX',
        'due to crosssection stresses s_yy (trapezoidal)':                      'cs_traY',

        #3D - elem
        'due to grid under pile head (simpsonCircle21)':                        'gr_sim21',
        'due to grid under pile head (simpsonCircle15)':                        'gr_sim15',
        'due to grid under pile head (simpsonCircle11)':                        'gr_sim11',
        'due to crosssection under pile head (triangularCircle)':               'cs_tri',
        'due to crosssection under pile head (mean)':                           'cs_med',
        'due to InternalForceAnalysis in the first soillayer':                  'fa_oneFs',
        'due to InternalForceAnalysis with material and soil':                  'fa_oneMs',
        'due to InternalForceAnalysis with material, soil and line at {si_x-offset} and {si_y-offset}': 'fa_oneMslOff',
        'due to InternalForceAnalysis with material, soil and line at {si_x} and {si_y}': 'fa_oneMsl',
        'due to InternalForceAnalysis with material and line at {si_x-offset} and {si_y-offset}': 'fa_oneMl',
        'due to Wissmann Quadratur Rule':                                       'gr_wis',

        #3D - soil
        'due to interface stresses s_n (triangularRectangle)':                  'if_tri',
        'due to interface stresses s_n (mean)':                                 'if_med',
        'due to crosssection stresses sig_zz (triangularRectangle)':            'cs_tri',
        'due to crosssection stresses sig_zz (mean)':                           'cs_med',
        'due to grid stresses sig_zz (simpsonRectangle11)':                     'gr_sim11',
        'due to grid stresses sig_zz (simpsonRectangle21)':                     'gr_sim21',
        'due to grid stresses sig_zz (simpsonRectangle5)':                      'gr_sim5',
        'due to InternalForceAnalysis with material':                           'fa_one',

        #3D - base
        'due to crosssection stresses (triangularCircle)':                      'cs_tri',
        'due to crosssection stresses (mean)':                                  'cs_med',
        'due to interface stresses (triangularCircle)':                         'if_tri',
        'due to interface stresses (mean)':                                     'if_med',
        'due to grid stresses (simpsonCircle21)':                               'gr_sim21',
        'due to grid stresses (simpsonCircle11)':                               'gr_sim11',
        'due to InternalForceAnalysis with material and last soil':             'fa_oneMs',
        'due to Wissmann Quadratur Rule':                                       'gr_wis',

        #3D - shaft
        'due to interface stresses (triangularShaft)':                          'if_tri',
        'due to interface stresses (mean)':                                     'if_med',
        'due to grid stresses sig_zz for simpsonShaft21':                       'gr_sim21',
        'due to stresses from var=0 above and below pile (force difference)':   'if_dif',
        'due to interface stresses s_1 (triangularShaft)':                      'if_tri',
        'due to interface stresses s_1 (mean)':                                 'if_mea',
        'due to grid stresses sig_zz (simpsonShaft21)':                         'gr_sim21',
        }
    return mapping.get(descr, None)

def fct_variantLabel_2d(typ, variant):
    mapping = {'elem': {
                        # 'var0': 'if_tra',
                        # 'var1': 'cs_tra',
                        'var1': 'if_tra',
                        'var0': 'cs_tra',
                        'var2': 'ifL_tra',
                        'var3': 'fa_oneMl',
                        },
               'soil': {'var0': 'cs_tra',
                        'var1': 'if_tra',
                        'var2': 'ifL_tra',
                        'var3': 'fa_oneMl',
                        'var100': 'if_dif',
                        'var101': 'cs_dif',
                        'var102': 'if_dif',
                        'var103': 'fa_dif',
                        },
               'base': {'var0': 'if_tra',
                        'var1': 'ifL_tra',
                        'var2': 'cs_tra',
                        'var3': 'fa_oneMl',
                        },
               'shaft': {'var0': 'if_tra',
                         'var1': 'cs_tra',
                         'var2': 'cs_tra',
                         'var3': 'cs_tra',
                         'var100': 'if_dif',
                         },
               }
    return mapping.get(typ, {}).get(variant, variant)

def group_modelsSuf(models):
    groups = defaultdict(dict)
    baseline = {} # to have these values as a reference
    for key, df in models.items():
        _, number, dimension, suffix = fct_parseKey(key)
        # group key = variant (edgeX, middleY, corner, ...)
        if suffix == '': baseline[key] = df
        else: groups[suffix][key] = df
    for suffix in groups: groups[suffix].update(baseline)
    return groups

def fct_getPositions(dimension, num_si_x=None, num_si_y=None):
    if dimension in ['unitcell2']:
        s = ['']
        m = [1]
    elif dimension in ['unitcell3']:
        s = ['_Middle']
        m = [1]
    elif dimension in ['quarter3', 'full3']:
        if num_si_x==None: raise ValueError('{num_si_x} should be defined.')
        if num_si_y==None: raise ValueError('{num_si_y} should be defined.')
        if (num_si_x <= 2 and num_si_y <= 2):
            s = ['']
            if   num_si_x == 1 and num_si_y == 1: m = [4]; s = ['_Middle' ]
            elif num_si_x == 1 and num_si_y == 2: m = [2]; s = ['_EdgeX'  ]
            elif num_si_x == 2 and num_si_y == 1: m = [2]; s = ['_EdgeY'  ]
            elif num_si_x == 2 and num_si_y == 2: m = [1]; s = ['_Corner' ]
            else: raise ValueError('Value Error.')
        elif ((num_si_x == 2 and num_si_y >= 3) or
              (num_si_x >= 3 and num_si_y == 2)):
            if   max(num_si_x, num_si_y) % 2 == 0:
                m = [2, 2]
                s = ['_EdgeY', '_EdgeX']
            elif max(num_si_x, num_si_y) % 2 == 1:
                if   num_si_x == 2:m = [4, 2]; s = ['_Middle', '_EdgeX']
                elif num_si_y == 2: m = [2, 4]; s = ['_EdgeY', '_Middle']
            else: raise ValueError('Value Error.')
            # MiddleY ... Ymax
            # CornerX ... Ymin
            # CornerY ... Xmin
            # MiddleX ... Xmax
        else:
            s = ['_EdgeY', '_EdgeX', '_Middle', '_Corner']
            # EdgeY  ... Xmin, Ymax
            # EdgeX  ... Xmax, Ymin
            # Middle ... Xmax, Ymax
            # Corner ... Xmin, Ymin
            if   (num_si_x % 2 == 1 and num_si_y % 2 == 1): m = [2, 2, 4, 1]
            elif (num_si_x % 2 == 1 and num_si_y % 2 == 0): m = [1, 2, 2, 1]
            elif (num_si_x % 2 == 0 and num_si_y % 2 == 1): m = [2, 1, 2, 1]
            elif (num_si_x % 2 == 0 and num_si_y % 2 == 0): m = [1, 1, 1, 1]
            else: raise ValueError('Value Error.')
        if dimension == 'full3':
            m = [1 for _ in m] # only full SIs
    elif dimension in ['all']: # to see if data is available
        s = ['_Middle', '_EdgeY', '_EdgeX', '_Corner', '']
        m = [1 for _ in s]
    else: raise ValueError('Value Error.')
    return s, m

def fct_parseDimension(dimension, path=None):
    # Determine suffixes and depth variable for 2D/3D cases.
    if dimension in ['unitcell2', '_unitcell2']:
        depth = 'y'
        suffixes, _ = fct_getPositions(dimension)
    elif dimension in ['unitcell3', 'quarter3', 'full3',
                       '_unitcell3','_quarter3','_full3']:
        depth = 'z'
        if dimension in ['unitcell3', '_unitcell3']:
            suffixes, _ = fct_getPositions(dimension)
        else: # 'quarter3','full3'
            if path is None:
                suffixes, _ = fct_getPositions('all')
            else:
                data_path = os.path.join(path, 'input_parameters.txt')
                with open(data_path, 'r') as f: text = f.read()
                start     = text.find('SI_PARAMS')
                start     = text.find('{', start)
                end       = text.find('}', start) + 1
                si_params = ast.literal_eval(text[start:end])
                num_si_x    = si_params.get('numx', None)
                num_si_y    = si_params.get('numy', None)
                suffixes, _ = fct_getPositions(dimension, num_si_x, num_si_y)
    else: raise ValueError(f'Dimension {dimension} is not defined!')
    return suffixes, depth

def fct_parseTextOld(pathFile):
    data = []
    current_phase = None
    current_type = None
    if not os.path.exists(pathFile):
        return None
    with open(pathFile, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('For Phase'):
                current_phase = int(re.findall(r'\d+', line)[0])
            elif 'F_elem' in line: current_type = 'F_elem'
            elif 'F_soil' in line: current_type = 'F_soil'
            elif 'F_base' in line: current_type = 'F_base'
            elif 'F_shaft' in line: current_type = 'F_shaft'
            elif re.match(r'^\d+', line):
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 3:
                    data.append({'phase':       current_phase,
                                 'type':        current_type,
                                 'option':      int(parts[0]),
                                 'force':       float(parts[1]),
                                 'time':        float(parts[2]),
                                 'description': str(parts[3])})
    return data

def fct_parseText(pathFile):
    if not os.path.exists(pathFile):
        return None
    data = []
    current_phase = None
    current_type = None
    with open(pathFile, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Phase
            m = re.match(r"For Phase\s+(\d+):", line)
            if m:
                current_phase = int(m.group(1))
                continue
            # Tabellenüberschrift
            # m = re.match(r"Option\s+(F_\w+)", line)
            # m = re.match(r"Option\s+(\w+)\s+Time\s*\[s\]", line)
            m = re.match(r"Option\s+(\w+)\s+(?:\[[^\]]+\]\s+)?Time\s*\[s\]",
                         line
                         )
            if m:
                current_type = m.group(1)
                continue
            # Datenzeile
            # m = re.match(
            #     r"(\d+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+"
            #     r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(.*)",
            #     line)
            number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            m = re.match(rf"(\d+)\s+({number})\s+({number})\s+(.*)",
                         line
                         )
            if m:
                data.append({"phase":       current_phase,
                             "type":        current_type,
                             "option":      int(m.group(1)),
                             "force":       float(m.group(2)),
                             "time":        float(m.group(3)),
                             "description": m.group(4),
                             })
    return data

def fct_parsePhase(pathFolder, namePattern):
    files = glob.glob(os.path.join(pathFolder, namePattern))
    phases = []
    for f in files:
        match = re.search(r'_p(\d+)\.csv$', f)
        if match: phases.append(int(match.group(1)))
    phasesSorted = sorted(phases)
    return phasesSorted

def getUW(uw):
    if not uw:  uw, UW = '', ''
    else:       uw, UW = '_uw', 'Unit Weight '
    return uw, UW

def build_modelGroupsNum(models):
    counts = defaultdict(set)
    for key in models:
        category, number, _, _ = fct_parseKey(key)
        counts[category].add(number)
    return {k: len(v)
            for k, v in counts.items()}

def build_modelGroups(models):
    number_counts = build_modelGroupsNum(models)
    groups = defaultdict(dict)
    # categories = {fct_parseKey(key)[0] for key in models}
    # multi_category = len(categories) > 1
    models = fct_sortModels(models)
    for key, df in models.items():
        category, number, dimension, suffix = fct_parseKey(key)
        multi_number = number_counts[category] > 1
        is_unitcell = dimension in ["unitcell2",
                                    "unitcell3"]
        is_complex = dimension in ["quarter3",
                                   "full3"]
        # 1) direct groups
        groups[f"{category}_{number}_allDim"][key] = df
        groups[f"{category}_{number}_{dimension}"][key] = df
        if multi_number:
            groups[f"{category}_var_allDim"][key] = df
            groups[f"{category}_var_{dimension}"][key] = df
        # 2) unitcell explicit
        if is_unitcell:
            groups[f"{category}_{number}_allDim_middle"][key] = df
            if multi_number:
                groups[f"{category}_var_allDim_middle"][key] = df
        # 3) quarter3 structure
        if is_complex:
            if suffix: s = suffix.lower()
            else:      s = '_unknown'
            groups[f"{category}_{number}_{dimension}{s}"][key] = df
            if multi_number:
                groups[f"{category}_var_{dimension}{s}"][key] = df
            if s == "_middle":
                groups[f"{category}_{number}_allDim{s}"][key] = df
                if multi_number:
                    groups[f"{category}_var_allDim{s}"][key] = df
    return dict(groups)

# output for build_modelGroups with rad_i = [1,3,var]
    # rad_i_allDim
    # rad_i_unitcell2
    # rad_i_unitcell3
    # rad_i_quarter
    # rad_i_allDim_middle
    # rad_i_quarter_middle
    # rad_i_quarter_middle
    # rad_i_quarter_edgex
    # rad_i_quarter_edgey

###############################################################################
#----------------------------- Helpers: Plotting ------------------------------
###############################################################################

def plot_lineSiWaterSoil(ax, path, all_x, all_y): # for ref_line=True
    _, colors = fct_setPlotstyle(flat=True)
    data_path = os.path.join(path, 'input_parameters.txt')
    with open(data_path, 'r') as f: text = f.read()
    start     = text.find('SOIL_PARAMS')
    start     = text.find('{', start)
    end       = text.find('}', start) + 1
    soil_dict = ast.literal_eval(text[start:end])
    head      = soil_dict['head']
    layers    = soil_dict['layers']
    start     = text.find('SI_PARAMS')
    start     = text.find('{', start)
    end       = text.find('}', start) + 1
    si_dict   = ast.literal_eval(text[start:end])
    si_UK     = si_dict['UK']
    w = 1
    # soil improvement UK
    ax.axhline(si_UK, label = 'SI - bottom',
               c=colors[-1], ls='dashed', lw=w)
    all_y.append(si_UK)
    # water level
    ax.axhline(head, label = 'Waterlevel',
               c=colors[-2], ls='dashed', lw=w)
    all_y.append(head)
    # soil layers
    if len(layers):
        layertot = 0
        for i, layer in enumerate(layers):
            layertot -= layer
            ax.axhline(layertot, label=f'Layer {i+1}',
                       c=colors[-3-i], ls='dashed', lw=w)
            all_y.append(layertot)
    return (all_x, all_y)

###############################################################################
#----------------------------------- Testing ----------------------------------
###############################################################################

def plot_testingMarkers():
    markers, _ = fct_setPlotstyle()
    x = np.arange(10)
    y = np.ones_like(x)
    fig, ax = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 8/CM_PER_INCH), dpi=dpiDefault) ###80%
    for i, m in enumerate(markers):
        ax.plot(x + i,
                y + i,
                label=m,
                marker=m[0],
                markersize=m[1],
                linestyle='-',
                c='r')
    ax.legend(title='Marker')
    ax.set_title('Matplotlib Marker Test')
    ax.grid(True)
    plt.show()
# plot_testingMarkers()

def plot_testingColors():
    for case in range(0, 15):
        _, colors = fct_setPlotstyle(case=case, flat=True)
        x = np.arange(10)
        y = np.ones_like(x)
        fig, ax = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 8/CM_PER_INCH), dpi=dpiDefault) ###80%
        for i, c in enumerate(colors):
            ax.plot(x + i,
                    y + i,
                    label=f'{i}',
                    linestyle='-',
                    c=c)
        ax.legend(title='Color index')
        ax.set_title(f'Matplotlib Color Test (case {case})')
        ax.grid(True)
        plt.show()
# plot_testingColors()

###############################################################################
#------------------- Main: Distribution Comparison - ref from eval ------------
###############################################################################

def _get_model_labels(pathFolders): # for compariso
    model_folders = [os.path.normpath(path).split(os.sep)[-3]
                     for path in pathFolders
                     ]
    split_names = [name.split("_")
                   for name in model_folders
                   ]
    if len({len(parts) for parts in split_names}) == 1:
        n_parts = len(split_names[0])
        varying_indices = [i
                           for i in range(n_parts)
                           if len({parts[i]
                                   for parts in split_names}) > 1
                           ]
        if varying_indices:
            labels = ["_".join(parts[i]
                               for i in varying_indices)
                      for parts in split_names
                      ]
        else:
            labels = model_folders
    else:
        labels = model_folders
    return labels

def _get_base_colors(colors):
    from matplotlib.colors import is_color_like
    base_colors = []
    for color in colors:
        if is_color_like(color):
            base_colors.append(color)
        # e.g. list of shades for one base colour
        elif (isinstance(color, (list, tuple, np.ndarray))
              and len(color) > 0
              and is_color_like(color[0])):
            base_colors.append(color[0])
    if not base_colors:
        raise ValueError("No valid colours found in fct_setPlotstyle().")
    return base_colors

def fct_shortNumber(value):
    def replace(match):
        text = match.group()
        number = float(text)
        if number != 0 and (abs(number) >= 1e3 or
                            abs(number) <= 1e-3):
            mantissa, exponent = f'{number:.6e}'.split('e')
            return f'{mantissa.rstrip("0").rstrip(".")}e{int(exponent)}'
        return text
    return re.sub(r'(?<![\d.])(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?',
                  replace, str(value))

def _get_model_colors(model_labels):
    from matplotlib.colors import to_rgba
    _, colors = fct_setPlotstyle()
    base_colors = _get_base_colors(colors)
    unique_models = list(dict.fromkeys(model_labels))
    n_models = len(unique_models)
    n_base = len(base_colors)
    n_levels = max(1,
                   math.ceil(n_models / n_base))
    if n_levels == 1:  alpha_levels = [1.0]
    else: alpha_levels = np.linspace(1.0, 0.45, n_levels)
    model_colors = {}
    for i, model in enumerate(unique_models):
        base_index = i % n_base
        alpha_index = i // n_base
        model_colors[model] = to_rgba(base_colors[base_index],
                                      alpha=float(alpha_levels[alpha_index])
                                      )
    return model_colors

def _get_variant_linestyles(variants):
    linestyles = ["-", "--", "-.", ":",
                  (0, (5, 1)),
                  (0, (3, 1, 1, 1)),
                  (0, (5, 2, 1, 2)),
                  (0, (1, 2)),
                  ]
    unique_variants = list(dict.fromkeys(variants))
    return {variant: linestyles[i % len(linestyles)]
            for i, variant in enumerate(unique_variants)
            }

def plot_pointsDistribution_comp(pathFolder, typ, phase):
    # always convert to list
    if isinstance(pathFolder, (str, os.PathLike)):
        pathFolders = [str(pathFolder)]
    else:
        pathFolders = [str(path) for path in pathFolder]
    pathLower = [path.lower() for path in pathFolders]
    fct_setPlottext()
    if all("unitcell2" in path
           for path in pathLower):
        return plot_pointsDistribution_comp_2d(pathFolders,
                                              typ,
                                              phase
                                              )
    elif all(("unitcell3" in path
              or "quarter3" in path
              or "full3" in path)
             for path in pathLower):
        return plot_pointsDistribution_comp_3d(pathFolders,
                                              typ,
                                              phase,
                                              quantity=["s_n",
                                                        "sig_yy",
                                                        "s_1"
                                                        ],
                                              direction="x"
                                              )
    else:
        raise ValueError( "Mixed or unknown model types:\n"
                         + "\n".join(pathFolders)
                         )

def plot_pointsDistribution_comp_2d(pathFolders,
                                   typ,
                                   phase,
                                   exclude_methods = exclude_methodsDefault):
    # paths
    if isinstance(pathFolders, (str, os.PathLike)):
        pathFolders = [str(pathFolders)]
    else:
        pathFolders = [str(path) for path in pathFolders]
    if phase == -1:
        path = pathFolders[0]
        phase = find_maxPhase(path.rsplit("\\", 3)[0],
                              path.rstrip("\\").split("\\")[-3]
                              )
    # model labels
    model_labels = _get_model_labels(pathFolders)
    model_colors = _get_model_colors(model_labels)
    # collect files first
    plot_data = []
    for pathFolder, model in zip(pathFolders,
                                 model_labels):
        files = glob.glob(os.path.join(pathFolder,
                                       f"results_{typ}_var*_p{phase}.csv"
                                       )
                          )
        print(f"plot_pointsDistribution_comp_2d ->\n{model} -> {files}")
        for file in files:
            basename = os.path.basename(file)
            variant = (basename
                       .replace(f"results_{typ}_", "")
                       .replace(f"_p{phase}.csv", "")
                       )
            method_label = fct_variantLabel_2d(typ, variant)
            if method_label.startswith(tuple(exclude_methods)):
                continue
            plot_data.append({"file": file,
                              "path": pathFolder,
                              "model": model,
                              "variant": variant
                              })
    if not plot_data:
        print("No files found.")
        return None
    # variant styles
    variants = [data["variant"] for data in plot_data]
    variant_styles = _get_variant_linestyles(variants)
    unique_variants = list(dict.fromkeys(variants))
    plot_groups = [("all", plot_data)]
    plot_groups.extend((variant,
                        [data for data in plot_data
                         if data["variant"] == variant]
                        )
                       for variant in unique_variants
                       )
    file_paths = []
    for group_name, group_data in plot_groups:
        # figure
        fig_height = 4
        fig_width = 15
        fig, ax = plt.subplots(figsize=(fig_width/CM_PER_INCH,
                                        fig_height/CM_PER_INCH)) # 80%
        coord = None
        fixed = None
        # plot
        plotted_data = []
        for data in group_data:
            file = data["file"]
            model = data["model"]
            variant = data["variant"]
            df = pd.read_csv(file, sep=";")
            df = df.drop_duplicates()
            tol = 1e-3
            x_range = df["x"].max() - df["x"].min()
            y_range = df["y"].max() - df["y"].min()
            
            if x_range > tol and y_range <= tol:
                current_coord = "x"
                current_fixed = df["y"].mean()
            
            elif y_range > tol and x_range <= tol:
                current_coord = "y"
                current_fixed = df["x"].mean()
            
            else:
                print("plot_pointsDistribution_comp_2d: "
                      "Cannot identify direction:\n"
                      f"{file}\n"
                      f"x_range = {x_range:.8e}, "
                      f"y_range = {y_range:.8e}")
                continue
            if coord is None:
                coord = current_coord
                fixed = current_fixed
            elif (coord != current_coord
                  or not np.isclose(fixed,
                                    current_fixed,
                                    rtol = 0,
                                    atol = 1e-2,
                                    )
                  ):
                print("Different cross-sections within one plot group: "
                      f"{file} has direction {current_coord!r} "
                      f"and fixed coordinate {current_fixed:.6f}.")
                continue
            # quantity
            if "sig_yy" in df.columns: quantity = "sig_yy"
            elif "s_n" in df.columns:  quantity = "s_n"
            elif "s_1" in df.columns:  quantity = "s_1"
            elif "N" in df.columns:    quantity = "N"
            else:
                print(f"No valid data column:\n{file}")
                continue
            # sort + plot
            df = df.sort_values(coord)
            ax.plot(df[coord],
                    df[quantity],
                    color       = model_colors[model],
                    linestyle   = variant_styles[variant],
                    marker      = ".",
                    markersize  = 5,
                    )
            plotted_data.append(data)
        # labels
        if not ax.lines:
            print(f"No valid data for {group_name}.")
            plt.close(fig)
            continue
        if coord == "x":
            xlabel = (f"x [m] along horizontal cross-section "
                      f"at y = {fixed:.3f} m")
            ylabel = r"$\sigma_y$ [kN/m²]" # sigmaz vs. sigman
        else:
            xlabel = (f"y [m] along vertical cross-section "
                      f"at x = {fixed:.3f} m")
            ylabel = r"$\sigma_x$ [kN/m²]"
        ax.set_xlabel(xlabel)
        if quantity == "N":
            ax.set_ylabel(r"$N$ [kN/m]", labelpad=5)
            ax.set_title(f"Normal force distribution for "
                         f"{typ} at phase {phase}")
        else:
            ax.set_ylabel(ylabel, labelpad=5)
            typname = f'R_{typ}'
            ax.set_title(f"Stress distribution for "
                         f"{fct_mathLabel(typname)} at phase {phase}")
        ax.grid(True, alpha=0.3)
        # Actual figure dimensions [cm]
        fig_width, fig_height = fig.get_size_inches() * CM_PER_INCH
        # Margins and legend spacing [cm]
        left_cm   = 1.7
        right_cm  = 3.5 # Reserved space for legends
        bottom_cm = 1.2
        top_cm    = 0.8
        gap_cm    = 0.2
        # Available axes dimensions [cm]
        ax_width_cm  = fig_width - left_cm - right_cm
        ax_height_cm = fig_height - bottom_cm - top_cm
        if ax_width_cm <= 0 or ax_height_cm <= 0:
            raise ValueError("Figure dimensions are too small "
                             "for the chosen margins.")
        ax.set_position([left_cm        / fig_width,
                         bottom_cm      / fig_height,
                         ax_width_cm    / fig_width,
                         ax_height_cm   / fig_height
                         ])
        # Legend spacing in axes coordinates
        gap_x = gap_cm / ax_width_cm
        gap_y = gap_cm / ax_height_cm
        xLegend = 1 + gap_x
        yLegend = 1
        # legend 1: model
        unique_models = list(dict.fromkeys(data["model"]
                                           for data in plotted_data
                                           ))
        handles_model = [Line2D([0],
                                [0],
                                color       = model_colors[model],
                                linestyle   = "-",
                                linewidth   = 1.5,
                                label = str(fct_shortNumber(model.split('_', 1)[1]
                                                            if '_' in model else model))
                                )
                         for model in unique_models
                         ]
        legend_model = ax.legend(handles        = handles_model,
                                 title          = "Model",
                                 loc            = "upper left",
                                 bbox_to_anchor = (xLegend, yLegend),
                                 borderaxespad  = 0
                                 )
        ax.add_artist(legend_model)
        fig.canvas.draw()
        yLegend = (legend_model.get_window_extent(renderer = fig.canvas
                                                 .get_renderer())
                   .transformed(ax.transAxes.inverted()).y0
                   - gap_y)
        # legend 2: variant
        unique_variants = list(dict.fromkeys(data["variant"]
                                             for data in plotted_data
                                             ))
        variant_handles = [Line2D([0], [0],
                                  color     = "black",
                                  linestyle = variant_styles[variant],
                                  linewidth = 1.5,
                                  label     = fct_variantLabel_2d(typ,
                                                                  variant)
                                  )
                           for variant in unique_variants
                           if not fct_variantLabel_2d(typ,
                                                      variant
                                                      ).startswith(
                                                          exclude_methods)
                           ]
        ax.legend(handles       = variant_handles,
                  title         = "Eval. method",
                  loc           = "upper left",
                  bbox_to_anchor= (xLegend, yLegend),
                  borderaxespad = 0
                  )
        from matplotlib.ticker import FuncFormatter, MaxNLocator        
        # Determine exponent from the actual y-axis data
        y_values = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.lines
        ])
        
        y_values = y_values[np.isfinite(y_values)]
        y_max = np.max(np.abs(y_values))
        
        if y_max > 0:
            exponent = int(math.floor(math.log10(y_max)))
        else:
            exponent = 0
        
        # Scientific notation only for large/small values
        if -2 < exponent < 3:
            exponent = 0
        
        scale = 10.0 ** exponent
        
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: f'{value / scale:g}')
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # Hide Matplotlib's automatic offset text
        ax.yaxis.get_offset_text().set_visible(False)
        
        # Explicit exponent, positioned above y-axis
        if exponent != 0:
            ax.text(
                0, 1.02,
                rf'$\times 10^{{{exponent}}}$',
                transform=ax.transAxes,
                ha='left',
                va='bottom',
                fontsize=5
            )
        # save
        if len(pathFolders) == 1: save_path = pathFolders[0]
        else: save_path = os.path.commonpath(pathFolders)
        print(f'stress distribution plot saved within {save_path}')
        # Save: combined plot + separate plot for each variant
        if group_name == "all": filename = f"results_{typ}_p{phase}_comp"
        else: filename = f"results_{typ}_{group_name}_p{phase}_comp"
        file_path = fct_formatSave(save     = True,
                                   fig      = fig,
                                   path     = save_path,
                                   name     = filename
                                   )
        print(f'saved in {file_path}')
        file_paths.append(file_path)
        plt.show()
        plt.close(fig)
    return file_paths

def fct_findQuantity(df, quantity):
    if isinstance(quantity, str):
        quantity = [quantity]
    for q in quantity:
        if q in df.columns:
            return q
    return None

def plot_pointsDistribution_comp_3d(pathFolder,
                                    typ,
                                    phase,
                                    quantity      = "s_n",
                                    direction     = "x",
                                    fixed_value   = None,
                                    n_line        = 300,
                                    n_grid        = 200,
                                    show_points   = False,
                                    cmap          = "viridis"):
    if phase == -1:
        phase = find_maxPhase(pathFolder.rsplit("\\", 3)[0],
                              pathFolder.rstrip("\\").split("\\")[-3])
    files = glob.glob(os.path.join(pathFolder,
                                   f"results_{typ}_var*_p{phase}.csv"))
    if len(files) == 0: return
    dfs = []
    labels = []
    for file in files:
        dfs.append(pd.read_csv(file))
        label = os.path.basename(file).replace(f"results_{typ}_",
                                               "").replace(f"_p{phase}.csv",
                                                       "")
        labels.append(label)
    # prepare
    # print(dfs)
    prepared = []
    xmin = -np.inf; xmax = +np.inf
    ymin = -np.inf; ymax = +np.inf
    vmin = +np.inf; vmax = -np.inf
    usedQuantities = []
    usedLabels = []
    for df, label in zip(dfs, labels):
        q = fct_findQuantity(df, quantity)
        if q is None:
            print(f"No matching quantity found in {label}")
            continue
        df = (df.groupby(["x", "y"])[q].mean().reset_index())
        prepared.append(df)
        usedQuantities.append(q)
        usedLabels.append(label)
        xmin = max(xmin, df.x.min()) if xmin != -np.inf else df.x.min()
        xmax = min(xmax, df.x.max())
        ymin = max(ymin, df.y.min()) if ymin != -np.inf else df.y.min()
        ymax = min(ymax, df.y.max())
        vmin = min(vmin, df[q].min())
        vmax = max(vmax, df[q].max())
    # section line
    if fixed_value is None:
        if direction == "x": fixed_value = 0.5 * (ymin + ymax)
        else:                fixed_value = 0.5 * (xmin + xmax)
    if direction == "x":
        coord = np.linspace(xmin, xmax, n_line)
        section = np.column_stack((coord,
                                   np.full_like(coord, fixed_value)))
        xlabel = f"x [m] (y={fixed_value:.3f} m)"
    else:
        coord = np.linspace(ymin, ymax, n_line)
        section = np.column_stack((np.full_like(coord, fixed_value),
                                   coord))
        xlabel = f"y [m] (x={fixed_value:.3f} m)"
    # interpolation grid
    gx = np.linspace(xmin, xmax, n_grid)
    gy = np.linspace(ymin, ymax, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    surfaces = []
    sections = []
    for df in prepared:
        from scipy.interpolate import griddata
        points = np.column_stack((df.x, df.y))
        surf = griddata(points, df[quantity], (GX, GY),
                        method="linear")
        line = griddata(points, df[quantity], section,
                        method="linear")
        surfaces.append(surf)
        sections.append(line)
    # Line plots: all variants together + each variant separately
    plot_groups = [(list(range(len(sections))),
                    f"compare_line_{typ}_p{phase}")]
    plot_groups.extend(([i], f"compare_line_{label}_{typ}_p{phase}")
                       for i, label in enumerate(usedLabels)
                       )
    for indices, filename in plot_groups:
        fig, ax = plt.subplots(figsize=(12 / CM_PER_INCH,
                                        7.5 / CM_PER_INCH))
        for i in indices:
            ax.plot(coord, sections[i],
                    label=usedLabels[i],
                    color=f"C{i % 10}"
                    )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(usedQuantities[indices[0]])
        ax.grid(True)
        ax.legend()
        file_path = fct_formatSave(save=True,
                                   fig=fig,
                                   path=pathFolder,
                                   name=filename
                                   )
        plt.close(fig)
    # surface plots
    for df, surf, label, q in zip(prepared, surfaces, usedLabels,
                                  usedQuantities):
        fig, ax = plt.subplots(figsize=(12/CM_PER_INCH, 10/CM_PER_INCH)) ####80%
        im = plt.imshow(surf,
                        origin="lower",
                        extent=[xmin, xmax, ymin, ymax],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="equal")
        if show_points: plt.scatter(df.x, df.y, c="k", s=5)
        plt.title(label)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.colorbar(im, ax=ax, label=q)
        pathName = f"compare_surf{label}_{typ}_p{phase}"
        file_path = fct_formatSave(save=True,
                                   fig=fig,
                                   path=pathFolder,
                                   name=pathName)
    # difference plots
    ref = surfaces[0]
    vmax_diff = max(np.nanmax(np.abs(s-ref))
                    for s in surfaces[1:])
    for surf, label, q in zip(surfaces[1:],
                              usedLabels[1:],
                              usedQuantities[1:]):
        fig, ax = plt.subplots(figsize=(12/CM_PER_INCH, 10/CM_PER_INCH)) ####80%
        diff = surf - ref
        im = plt.imshow(diff,
                        origin="lower",
                        extent=[xmin, xmax,
                                ymin, ymax],
                        cmap="RdBu_r",
                        vmin=-vmax_diff,
                        vmax=vmax_diff,
                        aspect="equal")
        plt.title(f"{label} - {labels[0]}")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.colorbar(im, label=f"Δ {quantity}")
        pathName = f"compare_diff{label}_{typ}_p{phase}"
        file_path = fct_formatSave(save=True,
                                   fig=fig,
                                   path=pathFolder,
                                   name=pathName)
    return file_path

###############################################################################
#------------------------------- Main: Resistance -----------------------------
###############################################################################

def plot_resistance(path,
                    uw=uwDefault, save=saveDefault,
                    df=None, suffix=None,
                    models=None, category=None):
    markers, colors = fct_setPlotstyle()
    if df is not None:
        if (models is not None or
            category is not None): raise ValueError('Typ not defined!')
        filename = plot_resistanceSingle(path, suffix, df,
                                         colors, markers, uw, save)
        return filename
    elif models is not None and category is not None:
        filename = plot_resistanceMultiple(path, models, category,
                                           colors, markers, uw, save)
        return filename
    else: raise ValueError('Typ not defined!')

def plot_resistanceSingle(path, suffix, df,
                          colors, markers, uw=uwDefault,
                          save=saveDefault, dpi=dpiDefault):
    uw, UW = getUW(uw)
    fig,ax = plt.subplots(1, 1, dpi=dpi, figsize=(12/CM_PER_INCH, 9/CM_PER_INCH)) ####80%
    all_u       = df['u'].values
    all_F_base  = df[f'F_base{uw}'].values
    all_F_shaft = df[f'F_shaft{uw}'].values
    all_F_sum   = all_F_base + all_F_shaft
    all_F = np.concatenate([all_F_base, all_F_shaft, all_F_sum])
    ax.plot(all_F_base,  all_u, label='$R_b$',
            c=colors[0][0], lw=1.0,
            marker=markers[1][0], markersize=markers[1][1])
    ax.plot(all_F_shaft, all_u, label='$R_s$',
            c=colors[1][0], lw=1.0,
            marker=markers[2][0], markersize=markers[2][1])
    ax.plot(all_F_sum,   all_u, label='$R_{sum}$',
            c=colors[2][0], lw=1.0,
            marker=markers[3][0], markersize=markers[3][1])
    fct_formatAxis(ax, all_F, all_u, xinv=True,
                   xlabel=f'{UW}Resistance R [kN]',
                   ylabel='Settlement u [m]')
    fct_formatLegend(fig, 'below')
    subfolder = f'summary_plot{suffix}'
    pathFolder = os.path.join(path, subfolder)
    filename = fct_formatSave(save, fig, pathFolder, f'Resistance{suffix}{uw}')
    return filename

def plot_resistanceMultiple(path, models, category,
                            colors, markers, uw=uwDefault,
                            save=saveDefault, dpi=dpiDefault):
    groupedModels = build_modelGroups(models)
    filenames = []
    for key, modelsDict in groupedModels.items():
        filename = plot_resistanceMultiplePlot(path,
                                               modelsDict,
                                               f'{category}_{key}',
                                               colors,
                                               markers,
                                               uw=uwDefault,
                                               save=saveDefault,
                                               dpi=dpiDefault)
        filenames.append(filename)
    return filenames

def plot_resistanceMultiplePlot(path, models, category,
                                colors, markers, uw=uwDefault,
                                save=saveDefault, dpi=dpiDefault):
    from matplotlib.lines import Line2D
    uw, UW = getUW(uw)
    
    # General
    seen_numbers = {}
    style_group = {}
    style_dim = {}
    style_pos = {}
    
    # Plot - General
    fig, axes = plt.subplots(1, 3, figsize=(15/CM_PER_INCH, 6/CM_PER_INCH), dpi=dpi)
    ax_rbase, ax_rshaft, ax_rsum = axes
    all_u_y      = []
    all_F_base   = []
    all_F_shaft  = []
    all_F_sum    = []
    counter = 0
    n_models = len({fct_parseKey(key)[1] for key in models})
    # Track which legend labels were already added
    for idx, (key, df) in enumerate(models.items()):
        cat, number, dimension, suffix = fct_parseKey(key)
        color, seen_numbers, counter = fct_getColor(colors,
                                                    number,
                                                    seen_numbers,
                                                    counter,
                                                    n_models)
        # marker and linestyle for dimension variation
        marker, markersize = markers[int(dimension[-1])]
        linestyle, position = fct_setPlotline(key)
        # values for plotting
        # num  = (f"{cat}{' = ' if get_digit(str(number)) else ' with '} {number}")
        num  = str(number)
        style_group[num] = (cat, color)
        style_dim[dimension] = (marker, markersize)
        position_label = position.lstrip('_') if position else dimension
        style_pos[position_label] = linestyle
        array_u       = df['u'].values
        array_F_base  = df[f'F_base{uw}'].values
        array_F_shaft = df[f'F_shaft{uw}'].values
        array_F_sum   = array_F_base + array_F_shaft
        # plots
        plot_style = dict(color=color,
                          linewidth=1.0,
                          linestyle=linestyle,
                          marker=marker,
                          markersize=markersize,
                          )
        ax_rbase.plot(array_F_base,   array_u, **plot_style)
        ax_rshaft.plot(array_F_shaft, array_u, **plot_style)
        ax_rsum.plot(array_F_sum,     array_u, **plot_style)
        # collect values for axis limits
        all_F_base.extend(array_F_base.tolist())
        all_F_shaft.extend(array_F_shaft.tolist())
        all_F_sum.extend(array_F_sum.tolist())
        all_u_y.extend(array_u.tolist())

    # Axis limits and ticks
    sameaxis = False
    if sameaxis:
        all_F = np.concatenate([all_F_base, all_F_shaft, all_F_sum])
        fct_formatAxis(ax_rbase, all_F, all_u_y,
                       xlabel='$R_{base}$ [kN]', xinv=True,
                       ylabel='Settlement u [m]')
        fct_formatAxis(ax_rshaft, all_F, all_u_y,
                       xlabel='$R_{shaft}$ [kN]', xinv=True)
        fct_formatAxis(ax_rsum, all_F, all_u_y,
                       xlabel='$R_{sum}$ [kN]', xinv=True)
    else:
        fct_formatAxis(ax_rbase, all_F_base, all_u_y,
                       xlabel='$R_{base}$ [kN]', xinv=True,
                       ylabel='Settlement u [m]')
        fct_formatAxis(ax_rshaft, all_F_shaft, all_u_y,
                       xlabel='$R_{shaft}$ [kN]', xinv=True)
        fct_formatAxis(ax_rsum, all_F_sum, all_u_y,
                       xlabel='$R_{sum}$ [kN]', xinv=True)
    ax_rshaft.tick_params(axis='y', labelleft=False)
    ax_rsum.tick_params(axis='y', labelleft=False)

    # legends
    handles_dimensions = [Line2D([0], [0],
                                 color='black',
                                 linestyle='None',
                                 marker=marker,
                                 markersize=markersize,
                                 label=str(dimension),
                                 )
                          for dimension, (marker, markersize) in style_dim.items()
                          ]
    handles_groups = [Line2D([0], [0],
                             color=color,
                             linewidth=1.5,
                             label=str(number),
                             )
                      for number, (category2, color) in style_group.items()
                      ]
    handles_positions = [Line2D([0], [0],
                                color='black',
                                linestyle=linestyle,
                                linewidth=1.5,
                                label=position_label)
                         for position_label, linestyle in style_pos.items()]
    # settings
    fig.suptitle(f'{UW}Resistance curves for the {cat} variation')

    # legends
    xLegend = 0.80
    yLegend = 0.90 # only main title
    gap     = 0.04
    fct_adjustCm(fig, left=1.2, right=3.0, bottom=1.5, top=0.9)
    for title, handles in [(cat,           handles_groups),
                           ('Dimension',       handles_dimensions),
                           ('Position',        handles_positions)]:
        if len(handles) > 0:
            legend = fct_formatLegend(fig,
                                      typ        = 'handles',
                                      handles    = handles,
                                      xLegend    = xLegend,
                                      yLegend    = yLegend,
                                      title      = title)
            fig.canvas.draw()
            yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                       .transformed(fig.transFigure.inverted()).y0
                       - gap)
    prefix = f'{category.split('_')[1]}_{category.split('_')[2]}'
    pathFolder = os.path.join(path,
                              f'plot_variations_{cat}',
                              f'plot_variations_resistance_{prefix}')
    filename = fct_formatSave(save, fig, pathFolder,
                              f'Resistance_{'_'.join(category.split('_')[1:])}{uw}')
    return filename

def plot_resistanceBar(path, models, category, phase, uw=uwDefault,
                       save=saveDefault, dpi=dpiDefault):
    groupedModels = build_modelGroups(models)
    filenames = []
    for key, modelsDict in groupedModels.items():
        filename = plot_resistanceBarPlot(path,
                                          modelsDict,
                                          key,
                                          phase,
                                          uw=uwDefault,
                                          save=saveDefault,
                                          dpi=dpiDefault)
        filenames.append(filename)
    return filenames

def plot_resistanceBarPlot(path, models, category, phase, uw=uwDefault,
                           save=saveDefault, dpi=dpiDefault):
    # print(path)
    uw, UW = getUW(uw)
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _, colors = fct_setPlotstyle()
    fig, axes = plt.subplots(1, 3, figsize=(15/CM_PER_INCH,
                                            4/CM_PER_INCH),
                             dpi=dpi)
    ax_rbase, ax_rshaft, ax_rsum = axes
    labels = []
    values_base = []
    values_shaft = []
    values_sum = []
    bar_colors = []
    seen_numbers = {}
    counter = 0
    bar_hatches = []
    legend_entries_color = {}
    legend_entries_hatch = {}
    hatch_list = ['', '/', '.', 'x', '\\\\', '+', '-']
    dim_to_hatch = {}
    models = fct_sortModels(models)
    # print(models)
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    for key, df in models.items():
        cat, number, dimension, suffix = fct_parseKey(key)
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        if number not in legend_entries_color:
            legend_entries_color[number] = color
        if suffix not in dim_to_hatch:
            dim_to_hatch[suffix] = hatch_list[len(dim_to_hatch) % 
                                                 len(hatch_list)]
        hatch = dim_to_hatch[suffix]
        if suffix not in legend_entries_hatch:
            legend_entries_hatch[suffix] = hatch
        if phase == -1:
            phase_use = find_maxPhase(path,
                                      f'{cat}_{number}_{dimension}')
        else:
            phase_use = phase
        df_phase = df[df['Phase'] == phase_use]
        if df_phase.empty: continue
        val_base  = df_phase[f'F_base{uw}'].values[0]
        val_shaft = df_phase[f'F_shaft{uw}'].values[0]
        val_sum   = val_base + val_shaft
        values_base.append(val_base)
        values_shaft.append(val_shaft)
        values_sum.append(val_sum)
        bar_colors.append(color)
        bar_hatches.append(hatch)
        label = f'{number}_{dimension}{suffix}'
        labels.append(label)
    if not labels:
        print(f'No data {models.keys()} for phase {phase_use} '
              'for plot_resistanceBarPlot')
        return

    ax_rbase.set_title('Base Resistance')
    ax_rshaft.set_title('Shaft Resistance')
    ax_rsum.set_title('Total Resistance')
    x = np.arange(len(labels))
    ax_rbase.bar(x,   values_base,    color=bar_colors,   hatch=bar_hatches)
    ax_rshaft.bar(x,  values_shaft,   color=bar_colors,   hatch=bar_hatches)
    ax_rsum.bar(x,    values_sum,     color=bar_colors,   hatch=bar_hatches)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    handles_color = [Patch(facecolor=color,
                           label=number)
                    for number, color in sorted(
                        legend_entries_color.items(),
                        key=lambda item: fct_sortKey(str(item[0]))
                        )
                    ]
    handles_hatch = [Patch(facecolor='white', edgecolor='black',
                           hatch=hatch,
                           label=suf.lstrip('_') if suf else 'Unitcell')
                 for suf, hatch in legend_entries_hatch.items()]
    xLegend = 0.84
    yLegend = 0.82 # with subtitle (0.82), only main title (0.90)
    gap     = 0.04
    fct_adjustCm(fig, left=1.2, right=3.0, bottom=1.5, top=0.9)
    for title, handles in [(cat,              handles_color),
                           ('Position',       handles_hatch),]:
        if len(handles) > 0:
            legend = fct_formatLegend(fig,
                                      typ        = 'handles',
                                      handles    = handles,
                                      xLegend    = xLegend,
                                      yLegend    = yLegend,
                                      title      = title)
            fig.add_artist(legend)
            fig.canvas.draw()
            yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                       .transformed(fig.transFigure.inverted()).y0
                       - gap)
    # fig.suptitle(f'{UW}Resistance Comparison for {category}')
    pathFolder = os.path.join(
                 path,
                 f'plot_variations_{category.split('_')[0]}',
                 f'plot_variations_resistance_bar_p{phase}'
                 f'_{category.split('_')[1]}',
                 )
    filename = fct_formatSave(save, fig, pathFolder,
                              f'Resistance_bar_{category}{uw}_p{phase}')
    return filename

###############################################################################
#---------------------------------- Main: Ratios ------------------------------
###############################################################################

def plot_resiRatios(path, uw=uwDefault,
                    save=saveDefault,
                    df=None, suffix=None,
                    models=None, category=None):
    markers, colors = fct_setPlotstyle()
    if df is not None:
        if (models is not None or
            category is not None): raise ValueError('Typ not defined!')
        filename = plot_resiRatiosSingle(path, suffix, df,
                                         colors, markers, uw, save)
        return filename
    elif models is not None and category is not None:
        filename = plot_resiRatiosMultiple(path, models, category,
                                           colors, markers, uw, save)
        return filename
    else: raise ValueError('Typ not defined!')

def plot_resiRatiosSingle(path, suffix, df,
                          colors, markers, uw=uwDefault,
                          save=saveDefault, dpi=dpiDefault):
    uw, UW     = getUW(uw)
    fig, ax    = plt.subplots(1, 1, figsize=(15/CM_PER_INCH, 4/CM_PER_INCH), dpi=dpi)
    all_Loads  = df['Load'].values
    all_LSRs   = df[f'LSR{uw}'].values
    all_RSRs   = df[f'RSR{uw}'].values
    all_SARs   = df[f'SAR{uw}'].values
    all_Ratios = np.concatenate([all_LSRs, all_RSRs, all_SARs])
    # plotting
    ax.plot(all_Loads, all_LSRs, label='LSR',
            c=colors[0][0], lw=1.0,
            marker=markers[1][0], markersize=markers[1][1])
    ax.plot(all_Loads, all_RSRs, label='RSR',
            c=colors[1][0], lw=1.0,
            marker=markers[2][0], markersize=markers[2][1])
    ax.plot(all_Loads, all_SARs, label='SAR',
            c=colors[2][0], lw=1.0,
            marker=markers[3][0], markersize=markers[3][1])
    # settings
    fct_formatAxis(ax, all_Loads, all_Ratios,
                   xlabel='$Loading$ [kPa]', xinv=True,
                   ylabel=f'{UW}Ratios')
    fct_formatLegend(fig, 'below')
    subfolder = f'summary_plot{suffix}'
    pathFolder = os.path.join(path, subfolder)
    filename = fct_formatSave(save, fig, pathFolder, f'Ratios{suffix}{uw}')
    return filename

def plot_resiRatiosMultiple(path, models, category,
                            colors, markers, uw=uwDefault,
                            save=saveDefault, dpi=dpiDefault):
    groupedModels = build_modelGroups(models)
    filenames = []
    for key, modelsDict in groupedModels.items():
        filename = plot_resiRatiosMultiplePlot(path,
                                               modelsDict,
                                               f'{category}_{key}',
                                               colors,
                                               markers,
                                               uw=uwDefault,
                                               save=saveDefault,
                                               dpi=dpiDefault)
        filenames.append(filename)
    return filenames

def plot_resiRatiosMultiplePlot(path, models, category,
                                colors, markers, uw=uwDefault,
                                save=saveDefault, dpi=dpiDefault):
    uw, UW       = getUW(uw)
    seen_numbers = {}
    counter      = 0
    style_group = {}
    style_dim = {}
    style_pos = {}
    # Plot - General
    fig_width_cm  = 15.0
    fig_height_cm = 6.0
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_cm/CM_PER_INCH,
                                            fig_height_cm/CM_PER_INCH),
                             dpi=dpi)
    ax_lsr, ax_rsr, ax_sar = axes
    all_loads    = []
    all_lsr      = []
    all_rsr      = []
    all_sar      = []
    categoryShort = category.split('_')[0]
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    # Track which legend labels were already added
    for idx, (key, df) in enumerate(models.items()):
        cat, number, dimension, _ = fct_parseKey(key)
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        marker, markersize = markers[int(dimension[-1])]
        linestyle, position = fct_setPlotline(key)
        # group_label = (f"{categoryShort} = {number}"
        #                if get_digit(number)
        #                else f"{categoryShort} with {number}")
        group_label = fct_categoryValue(cat, number)
        style_group[group_label] = color
        style_dim[dimension] = (marker, markersize)
        # Auch die unterschiedlichen Unitcell-Linienstile erfassen.
        position_label = position.lstrip('_') if position else dimension
        style_pos[position_label] = linestyle
        all_Loads = df['Load'].values
        all_LSRs  = df[f'LSR{uw}'].values
        all_RSRs  = df[f'RSR{uw}'].values
        all_SARs  = df[f'SAR{uw}'].values
        ax_lsr.plot(all_Loads, all_LSRs,
                    c=color, lw=1.0, linestyle=linestyle,
                    marker=marker, markersize=markersize)
        ax_rsr.plot(all_Loads, all_RSRs,
                    c=color, lw=1.0, linestyle=linestyle,
                    marker=marker, markersize=markersize)
        ax_sar.plot(all_Loads, all_SARs,
                    c=color, lw=1.0, linestyle=linestyle,
                    marker=marker, markersize=markersize)
        # Collect values for axis limits
        all_loads.extend(all_Loads)
        all_lsr.extend(all_LSRs)
        all_rsr.extend(all_RSRs)
        all_sar.extend(all_SARs)
    # Axis limits and ticks
    sameaxis = False
    if sameaxis:
        all_ratios = np.concatenate([all_lsr, all_rsr, all_sar])
        fct_formatAxis(ax_lsr, all_Loads, all_ratios,
                       title='LSR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
        fct_formatAxis(ax_rsr, all_Loads, all_ratios,
                       title='RSR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
        fct_formatAxis(ax_sar, all_Loads, all_ratios,
                       title='SAR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
    else:
        fct_formatAxis(ax_lsr, all_Loads, all_lsr,
                       title='LSR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
        fct_formatAxis(ax_rsr, all_Loads, all_rsr,
                       title='RSR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
        fct_formatAxis(ax_sar, all_Loads, all_sar,
                       title='SAR',
                       xlabel='$Loading$ [kPa]', xinv=True,
                       )
    # settings
    fig.suptitle(f'{UW}Ratio curves for the {categoryShort} variation')
    handles_groups = [Line2D([0], [0],
                             color=color,
                             linewidth=1.5,
                             label=label)
                      for label, color in style_group.items()]
    handles_dimensions = [Line2D([0], [0],
                                 color='black',
                                 linestyle='None',
                                 marker=marker,
                                 markersize=markersize,
                                 label=dimension)
                          for dimension, (marker, markersize)
                          in style_dim.items()]
    handles_positions = [Line2D([0], [0],
                                color='black',
                                linestyle=linestyle,
                                linewidth=1.5,
                                label=position)
                         for position, linestyle in style_pos.items()]
    # Layout distances [cm]
    left_cm        = 1.2
    right_cm       = 3.4    # Reserved for legends
    top_cm         = 1.0
    bottom_cm      = 1.2
    wgap_cm        = 1.0    # Gap between graphs
    legend_gap_cm  = 0.4    # Gap between graphs and legends
    legend_top_cm  = 0.6
    # Available area for subplots [cm]
    plot_width_cm = fig_width_cm - left_cm - right_cm
    # Convert physical gaps to Matplotlib spacing
    ncols = len(axes)
    wspace = (ncols * wgap_cm /
              (plot_width_cm - (ncols - 1) * wgap_cm)
              if ncols > 1
              else 0)
    fig.subplots_adjust( left   = left_cm / fig_width_cm,
                        right  = 1 - right_cm / fig_width_cm,
                        bottom = bottom_cm / fig_height_cm,
                        top    = 1 - top_cm / fig_height_cm,
                        wspace = wspace
                        )
    # Legend positions
    xLegend = (fig_width_cm - right_cm + legend_gap_cm) / fig_width_cm
    yLegend = 1 - legend_top_cm / fig_height_cm
    for title, handles in ((cat,            handles_groups),
                           ('Dimension',    handles_dimensions),
                           ('Position',     handles_positions),):
        if not handles: continue
        legend = fct_formatLegend(fig,
                                  typ        = 'handles',
                                  handles    = handles,
                                  xLegend    = xLegend,
                                  yLegend    = yLegend,
                                  title      = title)
        fig.add_artist(legend)
        fig.canvas.draw()
        yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                   .transformed(fig.transFigure.inverted()).y0
                   - legend_gap_cm/fig_height_cm)
    prefix = f'{category.split('_')[1]}_{category.split('_')[2]}'
    pathFolder = os.path.join(path,
                              f'plot_variations_{category.split('_')[0]}',
                              f'plot_variations_ratios_{prefix}')
    filename = fct_formatSave(save, fig, pathFolder, f'Ratios_{category}{uw}')
    return filename

def plot_resiRatiosBar(path, models, category, phase,
                        uw=uwDefault, save=saveDefault, dpi=dpiDefault):
    groupedModels = build_modelGroups(models)
    filenames = []
    for key, modelsDict in groupedModels.items():
        filename = plot_resiRatiosBarPlot(path,
                                          modelsDict,
                                          key,
                                          phase,
                                          uw=uw,
                                          save=save,
                                          dpi=dpi)
        filenames.append(filename)
    return filenames


def plot_resiRatiosBarPlot(path, models, category, phase,
                           uw=uwDefault, save=saveDefault, dpi=dpiDefault):
    uw, UW = getUW(uw)
    _, colors = fct_setPlotstyle()
    fig, axes = plt.subplots(1, 3, figsize=(15/CM_PER_INCH, 6/CM_PER_INCH), dpi=dpi)
    ax_lsr, ax_rsr, ax_sar = axes
    labels      = []
    values_lsr  = []
    values_rsr  = []
    values_sar  = []
    bar_colors  = []
    bar_hatches = []
    seen_numbers = {}
    categoryShort = category.split('_')[0]
    counter = 0
    hatch_list = ['', '/', '.', 'x', '\\\\', '+', '-']
    suffix_to_hatch = {'': ''}
    legend_entries_color = {}
    legend_entries_hatch = {}
    models = fct_sortModels(models)
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    for key, df in models.items():
        cat, number, dimension, suffix = fct_parseKey(key)
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        if number not in legend_entries_color:
            legend_entries_color[number] = color
        # Schraffur nach Position/Suffix, unabhängig von der Dimension
        if suffix not in suffix_to_hatch:
            suffix_to_hatch[suffix] = hatch_list[
                1 + (len(suffix_to_hatch) - 1) % (len(hatch_list) - 1)]
        hatch = suffix_to_hatch[suffix]
        # filter
        if phase == -1:
            phase_use = find_maxPhase(path,
                                      f'{cat}_{number}_{dimension}')
        else: phase_use = phase
        df_phase = df[df['Phase'] == phase_use]
        if df_phase.empty: continue
        legend_entries_hatch[suffix] = hatch
        val_lsr = df_phase[f'LSR{uw}'].values[0]
        val_rsr = df_phase[f'RSR{uw}'].values[0]
        val_sar = df_phase[f'SAR{uw}'].values[0]
        values_lsr.append(val_lsr)
        values_rsr.append(val_rsr)
        values_sar.append(val_sar)
        bar_colors.append(color)
        bar_hatches.append(hatch)
        label = f'{number}_{dimension}{suffix}'
        labels.append(label)
    if not labels:
        print(f'No data {models.keys()} for phase {phase} '
              'for plot_resiRatiosBarPlot')
        return

    # --- plots ---
    x = np.arange(len(labels))
    ax_lsr.bar(x, values_lsr, color=bar_colors, hatch=bar_hatches)
    ax_rsr.bar(x, values_rsr, color=bar_colors, hatch=bar_hatches)
    ax_sar.bar(x, values_sar, color=bar_colors, hatch=bar_hatches)
    ax_lsr.set_title('LSR')
    ax_rsr.set_title('RSR')
    ax_sar.set_title('SAR')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    # --- legends ---
    handles_color = [Patch(facecolor=color,
                           label=f"{number}")
                     for number, color in sorted(
                             legend_entries_color.items(),
                             key=lambda item: fct_sortKey(str(item[0])))]
    handles_hatch = [Patch(facecolor='white',
                           edgecolor='black',
                           hatch=hatch,
                           label=suffix.lstrip('_')
                           if suffix else 'Unitcell')
                     for suffix, hatch in legend_entries_hatch.items()]
    xLegend = 0.80
    yLegend = 0.82 # with subtitle (0.82), only main title (0.90)
    gap     = 0.02
    fct_adjustCm(fig, left=1.2, right=3.0, bottom=1.5, top=0.9)
    for title, handles in [(cat,           handles_color),
                           ('Position',    handles_hatch),]:
        if len(handles) > 0:
            legend = fct_formatLegend(fig,
                                      typ        = 'handles',
                                      handles    = handles,
                                      xLegend    = xLegend,
                                      yLegend    = yLegend,
                                      title      = title)
            fig.add_artist(legend)
            fig.canvas.draw()
            yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                       .transformed(fig.transFigure.inverted()).y0
                       - gap)
    fig.suptitle(f'{UW}Ratio Comparison for {categoryShort}', y=0.98)
    pathFolder = os.path.join(
                 path,
                 f'plot_variations_{category.split('_')[0]}',
                 f'plot_variations_ratios_bar_p{phase}_{category.split('_')[1]}',
                 )
    filename = fct_formatSave(save, fig, pathFolder,
                              f'Ratios_bar_{category}{uw}_p{phase}')
    return filename

###############################################################################
#------------------------ Main: Settlements at Curvepoint ---------------------
###############################################################################

def plot_XoverLoad(path, models, category,
                    output = ['u','Settlement u [m]','settlement'],
                    save=saveDefault, dpi=dpiDefault):
    groupedModels = build_modelGroups(models)
    filenames = []
    for key, modelsDict in groupedModels.items():
        try:
            filename = plot_XoverLoadPlot(path,
                                           modelsDict,
                                           key,
                                           f'{category}_{key}',
                                           output=output,
                                           save=saveDefault,
                                           dpi=dpiDefault)
            filenames.append(filename)
        except Exception as e:
            print("[XoverLoad ERROR]", e)
            traceback.print_exc()
    return filenames

def plot_XoverLoadPlot(path, models, key, category,
                       output=['u', 'Settlement u [m]', 'settlement'],
                       save=saveDefault, dpi=dpiDefault):
    from matplotlib.lines import Line2D

    # General
    markers, colors = fct_setPlotstyle()
    seen_numbers = {}
    counter = 0
    dim_seen = {}
    style_group = {}
    style_dim = {}
    style_pos = {}
    # Plot
    fig, ax = plt.subplots(1, 1,
                           figsize=(12/CM_PER_INCH, 8/CM_PER_INCH),
                           dpi=dpi) ###80%
    all_x = []
    all_y = []
    models = fct_sortModels(models)
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    for model_key, df in models.items():
        # requested result not available
        if output[0] not in df.columns: continue
        df_plot = df.dropna(subset=[output[0], 'Load'])
        if df_plot.empty: continue
        cat, number, dimension, _ = fct_parseKey(model_key)
        # Farbe anhand von number
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        # Symbolform anhand der Dimension
        if dimension not in dim_seen:
            dim_seen[dimension] = len(dim_seen)

        dim_idx = dim_seen[dimension] % (len(markers) - 1)
        marker, markersize = markers[dim_idx + 1]
        # Strichtyp anhand der Position
        linestyle, position = fct_setPlotline(model_key)
        # Eigenschaften für die Legenden sammeln
        # group_label = (f"{cat}{' = ' if get_digit(str(number)) else ' with '}"
        #                f"{number}")
        group_label = fct_categoryValue(cat, number)
        style_group[group_label] = color
        style_dim[dimension] = (marker, markersize)
        position_label = position.lstrip('_') if position else 'Unitcell'
        style_pos[position_label] = linestyle
        # Werte und Kurve
        data_xLoad = df_plot['Load'].values
        data_yDef = np.abs(df_plot[output[0]].values)
        ax.plot(data_xLoad,
                data_yDef,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize)
        all_x.extend(data_xLoad.tolist())
        all_y.extend(data_yDef.tolist())
    # nothing was plotted
    if len(all_x) == 0 or len(all_y) == 0:
        plt.close(fig)
        return None
    # Achsen
    name, unit = output[2].rsplit(' ', 1)  # F_elem, [kN]
    fct_formatAxis(ax, all_x, all_y,
                   title=fct_displayForce(output[1]),
                   xlabel='Surface load [kPa]',
                   xinv=True,
                   ylabel= f'{fct_mathLabel(fct_displayForce(name))} {unit}'
                   )
    # Legendeneinträge
    handles_groups = [Line2D([0], [0],
                             color=color,
                             linewidth=1.5,
                             label=group_label)
                      for group_label, color in style_group.items()]
    handles_dimensions = [Line2D([0], [0],
                                 color='black',
                                 linestyle='None',
                                 marker=marker,
                                 markersize=markersize,
                                 label=str(dimension))
                          for dimension, (marker,
                                          markersize) in style_dim.items()]
    handles_positions = [Line2D([0], [0],
                                color='black',
                                linestyle=linestyle,
                                linewidth=1.5,
                                label=str(position))
                         for position, linestyle in style_pos.items()]
    # Platz für die Legenden rechts
    fct_adjustCm(fig, left=1.3, right=3.5, bottom=1.4, top=1.0)
    xLegend = 1.03 # since Loadvalue has 3 digits = long
    yLegend = 1.00
    gap     = 0.04
    if len(handles_dimensions) < 2 or len(handles_positions) < 2:
        combined = [Line2D([0], [0], color='black',
                           linestyle=line, marker=mark, markersize=size,
                           linewidth=1.3, label=f'{dim} / {pos}')
                    for dim, (mark, size) in style_dim.items()
                    for pos, line in style_pos.items()]
        legend_sets = ((fct_categoryInfo(cat)[0], handles_groups),
                       ('Dimension / position', combined))
    else:
        legend_sets = ((fct_categoryInfo(cat)[0], handles_groups),
                       ('Dimension', handles_dimensions),
                       ('Position', handles_positions))
    for title, handles in legend_sets:
        if not handles: continue
        legend = fct_formatLegend(ax,
                                  typ        = 'handles',
                                  handles    = handles,
                                  xLegend    = xLegend,
                                  yLegend    = yLegend,
                                  title      = title)
        ax.add_artist(legend)
        fig.canvas.draw()
        yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                   .transformed(ax.transAxes.inverted()).y0
                   - gap)
    # Save
    pathFolder = os.path.join(
        path,
        f"plot_variations_{category.split('_')[0]}",
        f"plot_variations_{output[0]}_overLoad_{key.split('_')[1]}")
    filename = fct_formatSave(save, fig, pathFolder,
                              f'{output[0]}_{key}',)
    return filename

###############################################################################
#---------------------------- Main: Interface analysis ------------------------
###############################################################################

def plot_interfaceShaftDefPhase(path, models, category,
                                phase=None,
                                save=saveDefault,
                                dpi=dpiDefault):
    # general
    markers, colors = fct_setPlotstyle()
    seen_numbers = {}
    counter = 0
    fig, ax = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 8/CM_PER_INCH), dpi=dpi) ###80%
    all_x = []
    all_y = []
    dim_seen = {}
    dim_counter = 0
    models = fct_sortModels(models)
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    for key, _ in models.items():
        typ, number, dimension, suffix = fct_parseKey(key)
        # depth
        _, depth = fct_parseDimension(dimension)
        # suffix + path
        pathData = os.path.join(path, '_'.join(key.split('_')[:3]))
        typ = 'shaft'
        subfolder = f'results_{typ}{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        # phase selection
        if phase is None:
            phases = fct_parsePhase(pathFolder, f'{subfolder}_line_p*.csv')
            if not phases: continue
            phase_use = phases[-1]
        elif phase == -1:
            phase_use = find_maxPhase(path,
                                      f'{category}_{number}_{dimension}')
        else: phase_use = phase
        # file load
        filename = f'{subfolder}_line_p{phase_use}.csv'
        pathFile = os.path.join(pathFolder, filename)
        if not os.path.exists(pathFile): continue
        df_shaft = pd.read_csv(pathFile, header=0, sep=';')
        data_xDef = df_shaft['ux'].values * 1000
        data_yDep = df_shaft[depth].values
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        # dimension grouping
        if dimension not in dim_seen:
            dim_seen[dimension] = dim_counter
            dim_counter += 1
        dim_idx = dim_seen[dimension] % len(markers)
        marker, markersize = markers[dim_idx + 1]
        linestyle, position = fct_setPlotline(key)
        # plot
        if number.isdigit(): partition = ' = '
        else: partition = ' = '
        label  = f'{category}{partition}{number} for {dimension}{position}'
        ax.plot(data_xDef, data_yDep, label=label,
                c=color, lw=1, linestyle=linestyle,
                marker=marker, markersize=markersize)
        all_x.extend(data_xDef)
        all_y.extend(data_yDep)
    # reference line
    if len(models)>0:
        plot_lineSiWaterSoil(ax, pathData, all_x, all_y)
    # axis formatting
    fct_formatAxis(ax, all_x, all_y,
                   title=f'Interface Shaft Deformation for Phase {phase_use}',
                   ylabel='depth [m]')
    fct_formatLegend(ax, 'outside')
    # save
    categoryShort = category.split('_')[0]
    pathFolder = os.path.join(path,
                              f'plot_variations_{categoryShort}',
                              'plot_variations_interface')
    filename = fct_formatSave(save, fig, pathFolder,
                              f'Interface_Shaft_Def_{categoryShort}_p{phase_use}')
    print(f'plot_interfaceShaftDefPhase plots {filename}')
    return filename

def plot_interfaceShaftDef(path, models,
                           save=saveDefault, dpi=dpiDefault):
    all_pngs_shaft_d = []
    models = fct_sortModels(models)
    for key, _ in models.items():
        _, _, dimension, suffix = fct_parseKey(key)
        linestyle, _ = fct_setPlotline(key)
        pathData = os.path.join(path, '_'.join(key.split('_')[:3]))
        fig, ax = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 9/CM_PER_INCH), dpi=dpi) ###80%
        all_x   = []
        all_y   = []
        _, depth = fct_parseDimension(dimension)
        # plots
        _, colors = fct_setPlotstyle(flat=True)
        # define used folders/phases
        typ        = 'shaft'
        subfolder  = f'results_{typ}{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        phases     = fct_parsePhase(pathFolder, f'{subfolder}_line_p*.csv')
        for num, phase in enumerate(phases):
            dataname = f'{subfolder}_line_p{phase}.csv'
            pathFile = os.path.join(pathFolder, dataname)
            if not os.path.exists(pathFile): continue
            df_shaft = pd.read_csv(pathFile, header=0, sep=';')
            data_xDef = df_shaft['ux'].values * 1000
            data_yDep = df_shaft[depth].values
            ax.plot(data_xDef, data_yDep,
                    label=f'Phase {phase}',
                    c=colors[num], lw=1, linestyle=linestyle)
            all_x.extend(data_xDef) # for axis
            all_y.extend(data_yDep) # for axis
        # reference line
        ref_line = True
        if ref_line: plot_lineSiWaterSoil(ax, pathData, all_x, all_y)
        # settings
        fct_formatAxis(ax, all_x, all_y,
                       title=f'Interface Shaft Deformation for {key}',
                       xlabel=r'$u_x$ [mm]',
                       ylabel=f'depth {depth} [m]')
        fct_formatLegend(ax, 'outside')
        subfolder = f'summary_plot{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        filename = fct_formatSave(save, fig, pathFolder,
                                  f'Interface_Shaft_Def{suffix}')
        all_pngs_shaft_d.append(filename)
    return all_pngs_shaft_d

def plot_interfaceShaftSig(path, models,
                           save=saveDefault, dpi=dpiDefault):
    # Plots
    all_pngs_shaft_s = []
    models = fct_sortModels(models)
    for key, _ in models.items():
        linestyle, _ = fct_setPlotline(key)
        _, colors    = fct_setPlotstyle(flat=True)
        _, _, dimension, suffix = fct_parseKey(key)
        pathData = os.path.join(path, '_'.join(key.split('_')[:3]))
        all_x      = []
        all_y      = []
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 9/CM_PER_INCH), dpi=dpi) ###80%
        if   dimension[-1] == '2': depth = 'y'; sigma = 's_1'
        elif dimension[-1] == '3': depth = 'z'; sigma = 'sig_zz'
        else: raise ValueError(f'Dimension {dimension[-1]} not defined!')
        # define used folders/phases
        typ        = 'shaft'
        subfolder  = f'results_{typ}{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        phases     = fct_parsePhase(pathFolder, f'{subfolder}_line_p*.csv')
        if not phases: continue
        for num, phase in enumerate(phases):
            dataname = f'{subfolder}_line_p{phase}.csv'
            pathFile = os.path.join(pathFolder, dataname)
            if not os.path.exists(pathFile): continue
            df_shaft = pd.read_csv(pathFile, header=0, sep=';')
            data_xSig = df_shaft[sigma].values
            data_yDep = df_shaft[depth].values
            ax.plot(data_xSig, data_yDep, label = f'Phase {phase}',
                    c=colors[num], lw=1, linestyle=linestyle)
            all_x.extend(data_xSig) # for axis
            all_y.extend(data_yDep) # for axis
        # reference line
        ref_line = True
        if ref_line: plot_lineSiWaterSoil(ax, pathData, all_x, all_y)
        # settings
        fct_formatAxis(ax, all_x, all_y,
                       title=f'Interface Shaft Stress for {key}',
                       xlabel=fr'$\sigma_{{{depth}}}$',
                       ylabel=f'depth {depth} [m]')
        fct_formatLegend(ax, 'outside')
        subfolder = f'summary_plot{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        filename = fct_formatSave(save, fig, pathFolder,
                                  f'Interface_Shaft_Sig{suffix}')
        all_pngs_shaft_s.append(filename)
    return all_pngs_shaft_s

def plot_interfaceSlabDef(path, models,
                          save=saveDefault,
                          dpi=dpiDefault,
                          phase_si=phase_siDef):
    all_pngs_slab = []
    models = fct_sortModels(models)
    for key, _ in models.items():
        _, _, dimension, suffix = fct_parseKey(key)
        _, depth = fct_parseDimension(dimension)
        # plots
        linestyle, position = fct_setPlotline(key)
        _, colors           = fct_setPlotstyle(flat=True)
        pathData = os.path.join(path, '_'.join(key.split('_')[:3]))
        fig, ax    = plt.subplots(1, 1, figsize=(12/CM_PER_INCH, 9/CM_PER_INCH), dpi=dpi) ###80%
        all_x      = []
        all_y      = []
        # define used folders/phases
        typ        = 'soil'
        subfolder  = f'results_{typ}{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        phases     = fct_parsePhase(pathFolder, f'{subfolder}_line_p*.csv')
        if not phases: continue
        # get data
        for num, phase in enumerate(phases):
            if phase <= phase_si: continue
            dataname = f'{subfolder}_line_p{phase}.csv'
            pathFile = os.path.join(pathFolder, dataname)
            if not os.path.exists(pathFile): continue
            df_slab = pd.read_csv(pathFile, header=0, sep=';')
            if   dimension[-1] == '3': center_x = df_slab['x'].max()
            elif dimension[-1] == '2': center_x = df_slab['x'].min()
            data_xrel = abs(df_slab['x'] - center_x).values
            data_ydef = df_slab[f'u{depth}'].values * 1000
            ax.plot(data_xrel, data_ydef, label = f'Phase {phase}',
                    c=colors[num], lw=1, linestyle=linestyle)
            all_x.extend(data_xrel)
            all_y.extend(data_ydef)
        # reference line
        ref_line = True
        if ref_line:
            data_path = os.path.join(pathData, 'input_parameters.txt')
            with open(data_path, 'r') as f: text = f.read()
            start     = text.find('SI_PARAMS')
            start     = text.find('{', start)
            end       = text.find('}', start) + 1
            si_params = ast.literal_eval(text[start:end])
            radius    = si_params.get('radius', None)
            if radius is not None:
                ax.axvline(radius, color='blue', 
                           linestyle='dashdot', lw=1, 
                           label = 'SI radius')
                all_x.append(radius)
        # settings
        fct_formatAxis(ax, all_x, all_y,
                       title=f'Interface Slab Deformation for {key}',
                       xlabel='$x$ [m]',
                       ylabel=f'$u_{{{depth}}}$ [mm]')
        fct_formatLegend(ax, 'outside')
        subfolder = f'summary_plot{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        filename = fct_formatSave(save, fig, pathFolder,
                                  f'Interface_Slab_Def{suffix}')
        all_pngs_slab.append(filename)
    return all_pngs_slab

def plot_interfaceSlabDefPhase(path, models, category, phase=None,
                             save=saveDefault, dpi=dpiDefault):
    _, colors = fct_setPlotstyle()
    fig, ax = plt.subplots(1, 1, figsize=(15/CM_PER_INCH, 8/CM_PER_INCH), dpi=dpi)
    labels = []
    values = []
    bar_colors = []
    bar_hatches = []
    seen_numbers = {}
    counter = 0
    hatch_list = ['', '/', '.', 'x', '\\\\', '+', '-']
    dim_to_hatch = {}
    legend_entries_color = {}
    legend_entries_hatch = {}
    models = fct_sortModels(models)
    n_models = len({fct_parseKey(key)[1]
                    for key in models})
    for key, _ in models.items():
        cat, number, dimension, suffix = fct_parseKey(key)
        _, depth = fct_parseDimension(dimension)
        color, seen_numbers, counter = fct_getColor(colors, number,
                                                    seen_numbers,
                                                    counter, n_models)
        if number not in legend_entries_color:
            legend_entries_color[number] = color
        if dimension not in dim_to_hatch:
            dim_to_hatch[dimension] = hatch_list[len(dim_to_hatch) %
                                                    len(hatch_list)]
        hatch = dim_to_hatch[dimension]
        if dimension not in legend_entries_hatch:
            legend_entries_hatch[dimension] = hatch
        pathData = os.path.join(path, '_'.join(key.split('_')[:3]))

        # phase selection
        typ        = 'soil'
        subfolder  = f'results_{typ}{suffix}'
        pathFolder = os.path.join(pathData, subfolder)
        if phase is None:
            phases = fct_parsePhase(pathFolder, f'{subfolder}_line_p*.csv')
            if not phases:
                filename = f"{subfolder}_line_pX.csv"
                print(f"Path not found: {os.path.join(pathFolder, filename)}")
                continue
            phase_use = phases[-1]
        elif phase == -1:
            phase_use = find_maxPhase(path,
                                      f'{category}_{number}_{dimension}')
        else: phase_use = phase

        dataname  = f'{subfolder}_line_p{phase_use}.csv'
        pathFile = os.path.join(pathFolder, dataname)
        if not os.path.exists(pathFile): continue
        df_slab   = pd.read_csv(pathFile, header=0, sep=';')
        data_ydef = df_slab[f'u{depth}'].values * 1000
        
        max_def   = np.max(np.abs(data_ydef))
        label     = f'{number}_{dimension}'
        labels.append(label)
        values.append(max_def)
        bar_colors.append(color)
        bar_hatches.append(hatch)
    if not labels:
        print('No data found for plot_interfaceSlabDefPhase!')
        return
    print(f'plot_interfaceSlabDefMax {labels}\n{values}')
    x = np.arange(len(labels))
    ax.bar(x, values, color=bar_colors, hatch=bar_hatches)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Maximum Slab Deformation for Phase {phase_use}')
    ax.set_title(f'Interface Slab Deformation for Phase {phase_use}')
    ax.set_ylabel('$u_{max}$ [mm]')
    ax.set_xlabel('Model')
    handles_color = [Patch(facecolor=color,
                           label=str(number))
                    for number, color in sorted(
                        legend_entries_color.items(),
                        key=lambda item: fct_sortKey(str(item[0]))
                        )
                    ]
    handles_hatch = [Patch(facecolor='white', edgecolor='black',
                           hatch=hatch, label=dimension)
                     for dimension, hatch in legend_entries_hatch.items()]
    xLegend = 0.80
    yLegend = 0.82 # with subtitle (0.82), only main title (0.90)
    gap     = 0.04
    fct_adjustCm(fig, left=1.2, right=3.0, bottom=1.5, top=0.9)
    for title, handles in [(cat,            handles_color),
                           ('Dimension',    handles_hatch),]:
        if len(handles) > 0:
            legend = fct_formatLegend(fig,
                                      typ        = 'handles',
                                      handles    = handles,
                                      xLegend    = xLegend,
                                      yLegend    = yLegend,
                                      title      = title)
            fig.add_artist(legend)
            fig.canvas.draw()
            yLegend = (legend.get_window_extent(fig.canvas.get_renderer())
                       .transformed(fig.transFigure.inverted()).y0
                       - gap)
    categoryShort = category.split('_')[0]
    pathFolder = os.path.join(path,
                              f'plot_variations_{categoryShort}',
                              'plot_variations_interface')
    filename = fct_formatSave(save, fig, pathFolder,
                              f'Interface_Slab_Def_{categoryShort}_p{phase_use}')
    print(f'plot_interfaceSlabDefPhase plots {filename}')
    return filename

###############################################################################
#-------------- Main: Force Variation due to model and Position ---------------
###############################################################################

def fct_getValue(pathFile, typ, phase,
                 inputTyp='csv'):
    if inputTyp == 'txt':
        data = fct_parseText(pathFile)
        if not data:
            return None
        for entry in data:
            if ((phase is not None and entry['phase'] != phase)
                or entry['type'] != typ):
                continue
            return entry['force']   # <-- FIRST ONLY
        return None
    elif inputTyp == 'csv':
        df = pd.read_csv(pathFile, sep=';')
        df = df[df['Phase'] == phase]
        if (df.empty or
            typ not in df.columns):
            print(f'value not found {pathFile} - typ: {typ} - typ: {phase}')
            return None
        return df[typ].values[0]
    else: raise ValueError("inputTyp not supported")

def x_label(name, show_dim):
    number = fct_parseKey(name)[1]
    dimension = fct_parseKey(name)[2]
    return (f'{number}\n{dimension}'
            if show_dim
            else str(number))

def position_name(suffix):
    return fct_positionLabel(suffix, useEdges=False)


def add_ratio_guides(ax):
    ax.axhline(0.00,
               color='black',
               linewidth=1,
               zorder=0)
    for limit in (-0.05, 0.05):
        ax.axhline(limit,
                   color='black',
                   linewidth=1,
                   linestyle='--',
                   zorder=0)

def get_category_labels(categories): # for multi catgeories
    if not categories:
        return {}, ''
    categories = list(categories)
    # Longest common substring of all categories.
    common = categories[0]
    for category in categories[1:]:
        best = ''
        for i in range(len(common)):
            for j in range(i + 1, len(common) + 1):
                candidate = common[i:j]
                if len(candidate) > len(best) and candidate in category:
                    best = candidate
        common = best
    # Only use the common part if it is meaningful.
    if not common:
        return {cat: cat for cat in categories}, ''
    labels = {}
    for category in categories:
        if category == common:
            labels[category] = 'Default'
        else:
            label = category.replace(common, '', 1)
            labels[category] = label if label else 'Default'
    return labels, common

def get_position_color(suffix):
    _, colors = fct_setPlotstyle()
    POSITION_COLORS = {
                       "unitcell":  colors[4][0],  # Red
                       "middle":    colors[3][0],  # Yellow
                       "edgey":     colors[2][0],  # Green
                       "edgex":     colors[1][0],  # Turquoise
                       "corner":    colors[0][0],  # Purple
                       }
    position = str(suffix or "").strip().lstrip("_").lower()
    if position == "": position = "unitcell"
    return POSITION_COLORS[position]

def plot_Matrix(path, category, names,
                colomnTyps, nameTyps,
                phase,
                rows=[1, 2, 3, 4],
                save=saveDefault,
                dpi=dpiDefault,
                useEdges=False,
                defaults=DEFAULT_MODEL_SETTINGS):
    _, base_colors = fct_setPlotstyle()
    multi_category = category == 'all'
    if not colomnTyps:
        raise ValueError("colomnTyps darf nicht leer sein.")
    if multi_category:
        c_max = len(base_colors)
        categories = list(dict.fromkeys(fct_parseKey(name)[0]
                                        for name in names))[:c_max]
        formatted = {cat: fct_shortNumber(cat)
                     for cat in categories}
        labels, common = get_category_labels(list(formatted.values()))
        category_labels = {cat: fct_categoryInfo(cat)[1]
                           for cat in categories}
        names = [name
                 for name in names
                 if fct_parseKey(name)[0] in categories]
    if not names:
        raise ValueError(f'No models {category} {names}')
    # x-Beschriftung: Dimension nur bei gemeinsamem Vergleich von q3 und f3
    dimensions = {fct_parseKey(name)[2]
                  for name in names}
    show_dim = False
    # show_dim = {'quarter3', 'full3'} <= dimensions
    # print("DEBUG x_labels:")
    # for name in names:
    #     value = x_label(name, show_dim)
    #     print(f"{name} -> {value!r} ({type(value).__name__})")
    # Dimension in Category aufnehmen, wenn full3 und quarter3
    # gemeinsam im Plot vorkommen
    if multi_category and {'full3', 'quarter3'} <= dimensions:
        for cat in categories:
            cat_dimensions = {fct_parseKey(name)[2]
                                  for name in names
                                  if fct_parseKey(name)[0] == cat
                                  }
            dims = [dim
                    for dim in ('full3', 'quarter3')
                    if dim in cat_dimensions]
            if dims:
                category_labels[cat] += f" ({', '.join(dims)})"
    x_labels = sorted({x_label(name, show_dim)
                       for name in names}, key=fct_sortKey)
    x_pos = {label: i for i, label in enumerate(x_labels)}
    force_count = len(colomnTyps)
    row_count   = len(rows)
    max_name_len = max((len(part)
                        for label in x_labels
                        for part in str(label).split('\n')),
                       default=0)
    extra_cm = max(0, max_name_len - 5) * 0.1
    fig_width_cm  = 12 / 2 * force_count + 3.5
    fig_height_cm = (3.0 + 8 / 2 * row_count
                     + row_count * extra_cm)
    fig, axes = plt.subplots(row_count,
                         force_count,
                         figsize=(fig_width_cm / CM_PER_INCH,
                                  fig_height_cm / CM_PER_INCH),
                         dpi=dpi,
                         squeeze=False
                         )
    repetitions = max(1, math.ceil(len({fct_parseKey(name)[1]
                                        for name in names})
                                   / len(base_colors)))
    # Layout distances [cm]
    left_cm   = 1.4
    # in cm
    right_cm  = 3.5 # in cm
    top_cm    = 1.9 if row_count > 1 else 1.0 # title space
    bottom_cm = 0.9 + extra_cm
    wgap_cm   = 1.4
    hgap_cm   = 1.0 + extra_cm
    # Available area for subplots [cm]
    plot_width_cm  = fig_width_cm - left_cm - right_cm
    plot_height_cm = fig_height_cm - top_cm - bottom_cm
    # Convert physical gaps to Matplotlib spacing
    wspace = (force_count * wgap_cm /
              (plot_width_cm - (force_count - 1) * wgap_cm)
              if force_count > 1 else 0)
    hspace = (row_count * hgap_cm /
              (plot_height_cm - (row_count - 1) * hgap_cm)
              if row_count > 1 else 0)
    fig.subplots_adjust(left   = left_cm / fig_width_cm,
                        right  = 1 - right_cm / fig_width_cm,
                        top    = 1 - top_cm / fig_height_cm,
                        bottom = bottom_cm / fig_height_cm,
                        wspace = wspace,
                        hspace = hspace
                        )
    markers,  colors  = fct_setPlotstyle(flat=True, r=repetitions)
    markersS, colorsS = fct_setPlotstyle(flat=True, r=1)
    model_style  = {}
    suffix_style = {}
    if multi_category:
        from matplotlib.colors import to_rgba
        category_colors = {cat: base_colors[i][0]
                           for i, cat in enumerate(categories)}
        # print("DEBUG fct_parseKey vs. cat:")
        # for name in names:
        #     value = fct_parseKey(name)[1]
        #     print(f'Plotter: {name}: {value!r} ({type(value).__name__})')
        category_numbers = {
            cat: sorted({fct_parseKey(name)[1]
                         for name in names
                         if fct_parseKey(name)[0] == cat},
                        key=fct_sortKey)
            for cat in categories}
    linestyles_dim, _ = fct_setPlotline('pos_0_all')
    def get_model_style(name):
        cat, number, _, _ = fct_parseKey(name)
        key = (cat, number) if multi_category else number
        if key not in model_style:
            index = len(model_style)
            if multi_category:
                numbers = category_numbers[cat]
                # Erst bei mehr als einer Modellnummer wird Alpha abgestuft.
                alpha = (1.0
                         if len(numbers) == 1
                         else (1.0 - 0.6 * numbers.index(number)
                               / (len(numbers) - 1))
                         )
                model_style[key] = (to_rgba(category_colors[cat], alpha),
                                    ('.', 4))
            else:
                model_style[key] = (colors[index % len(colors)],
                                    markers[index % len(markers)])
        return model_style[key]
    def get_line_style(suffix):
        if not multi_category:
            return '-'
        dimension = ('unitcell3'
                     if suffix == ''
                     else 'quarter3')
        linestyle, _ = fct_setPlotline(f'position_0_{dimension}{suffix}')
        return linestyle
    def get_suffix_styleOld(suffix):
        if suffix not in suffix_style:
            index = len(suffix_style)
            suffix_style[suffix] = (colorsS[index % len(colorsS)],
                                    markersS[index % len(markersS)])
        return suffix_style[suffix]
    def get_suffix_style(suffix): # hardcoded
        if suffix not in suffix_style:
            index = len(suffix_style)
            suffix_style[suffix] = (get_position_color(suffix),
                                    markersS[index % len(markersS)])
        return suffix_style[suffix]
    x_category = (fct_categoryInfo(categories[0])[0]
                  if multi_category and len(categories) == 1
                  else 'Piles per row [-]'
                  if any('sinum' in fct_parseKey(n)[0].lower() for n in names)
                  else fct_categoryInfo(category)[0]
                  if not multi_category else 'Model variation')
    def format_number_axes(ax):
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels([fct_categoryValue(category if not multi_category
                                              else categories[0],
                                              fct_shortNumber(label), defaults)
                            for label in x_labels],
                           rotation=45,
                           ha='right')
        ax.set_xlabel(x_category)
        ax.grid(True,
                linestyle='-',
                alpha=0.2)
    for col, colomnTyp in enumerate(colomnTyps):
        values = {}
        for name in names:
            phase_use = (find_maxPhase(path, name)
                         if phase == -1
                         else phase)
            model_values = {}
            pattern = os.path.join(path,
                                   name,
                                   'summary_value*',
                                   'Fsu_phases*.csv')
            for file_path in sorted(glob.glob(pattern)):
                folder = os.path.basename(os.path.dirname(file_path))
                suffix = folder[len('summary_value'):]
                # Nur das zum Ordnersuffix passende Ergebnis verwenden.
                if os.path.basename(file_path) != f'Fsu_phases{suffix}.csv':
                    continue
                value = fct_getValue(file_path, colomnTyp, phase_use)
                if value is None or not np.isfinite(value):
                    continue
                model_values[suffix] = abs(float(value))
            if model_values:
                values[name] = model_values
        if 'F' in colomnTyp:      unit = 'kN'
        elif colomnTyp == 'u':    unit = 'm'
        elif 'M' in colomnTyp:    unit = 'kNm/m'
        elif 'tilt' in colomnTyp: unit = '°'
        else:                     unit = '-'

        if 1 in rows: ax_pos  = axes[rows.index(1), col]
        if 2 in rows: ax_rat  = axes[rows.index(2), col]
        if 3 in rows: ax_var  = axes[rows.index(3), col]
        if 4 in rows: ax_dev  = axes[rows.index(4), col]

        if 1 in rows:
            # Row 1: Middle – Edge – Corner.
            for name, model_values in values.items():
                color, (marker,
                        marker_size) = get_model_style(name)
                dimension = fct_parseKey(name)[2]
                linestyle = linestyles_dim.get(dimension, '-')
                line_points = []
                for x, suffix in ((0, '_Middle'),
                                  (2, '_Corner')):
                    if suffix in model_values:
                        line_points.append((x, model_values[suffix]))
                edge_values = [model_values[suffix]
                               for suffix in (('_EdgeY',) if not useEdges
                                              else ('_EdgeY', '_EdgeX'))
                               if suffix in model_values]
                if edge_values:
                    line_points.append((1, float(np.mean(edge_values))))
                if line_points:
                    line_points.sort(key=lambda point: point[0])
                    ax_pos.plot([point[0] for point in line_points],
                                [point[1] for point in line_points],
                                color=color,
                                linestyle=linestyle,
                                linewidth=1
                                )
                    for x, y in line_points:
                        if x != 1: ax_pos.scatter(x, y,
                                                  color=color,
                                                  marker=marker,
                                                  s=marker_size**2,
                                                  zorder=3)
                for edge_value in edge_values:
                    ax_pos.scatter(1, edge_value,
                                   color=color,
                                   marker=marker,
                                   s=marker_size**2,
                                   zorder=4)
                if not line_points and '' in model_values:
                    ax_pos.axhline(model_values[''],
                                   color=color,
                                   linestyle=linestyle,
                                   linewidth=1
                                   )
            ax_pos.set_xticks((0, 1, 2))
            ax_pos.set_xticklabels(('Middle', 'Edge', 'Corner'))
            ax_pos.set_xlabel('Position')
            ax_pos.set_ylabel(f'{fct_mathLabel(fct_displayForce(colomnTyp))} [{unit}]')
            ax_pos.set_title(fct_mathLabel(fct_displayForce(colomnTyp)))
            ax_pos.grid(True, linestyle='-', alpha=0.2)

        if 2 in rows:
            # Row 2: EdgeY, EdgeX und Corner relativ zu Middle.
            add_ratio_guides(ax_rat)
            for suffix in (('_EdgeY', '_Corner') if not useEdges
                           else ('_EdgeY', '_EdgeX', '_Corner')):
                grouped_points = defaultdict(list)
                for name, model_values in values.items():
                    middle = model_values.get('_Middle')
                    other = model_values.get(suffix)
                    if middle is None or other is None or middle == 0:
                        continue
                    cat, _, dim, _ = fct_parseKey(name)
                    group = (cat, dim) if multi_category else ('all', '')
                    grouped_points[group].append((x_pos[x_label(name,
                                                                show_dim)],
                                                  (other - middle)
                                                   / middle))
                for (cat, dim), points in grouped_points.items():
                    points.sort(key=lambda point: point[0])
                    color, (marker,
                            marker_size) = ((category_colors[cat],
                                             get_suffix_style(suffix)[1])
                                             if multi_category
                                             else get_suffix_style(suffix))
                    ax_rat.plot([point[0] for point in points],
                                [point[1] for point in points],
                                color=color,
                                linewidth=1,
                                linestyle=get_line_style(suffix),
                                marker=marker,
                                markersize=marker_size,
                                label=(f'{cat}: {position_name(suffix)}'
                                        ' vs. Middle'
                                       if multi_category
                                       else f'{position_name(suffix)}'
                                             ' vs. Middle'))
            ax_rat.set_ylabel(r'$\delta_{\mathrm{M}}$ [-]')
            ax_rat.set_title(fct_mathLabel(fct_displayForce(colomnTyp)))
            format_number_axes(ax_rat)

        all_suffixes = sorted({suffix
                               for model_values in values.values()
                               for suffix in model_values
                               if useEdges or suffix != '_EdgeX'}) # row 3 and 4
        if 3 in rows:
            for suffix in all_suffixes:
                grouped_points = defaultdict(list)
                for name, model_values in values.items():
                    if suffix not in model_values: continue
                    cat, _, dim, _ = fct_parseKey(name)
                    group = (cat, dim) if multi_category else ('all', '')
                    grouped_points[group].append((x_pos[x_label(name, show_dim)],
                                                   model_values[suffix]))
                for (cat, dim), points in grouped_points.items():
                    points.sort(key=lambda point: point[0])
                    color, (marker,
                            marker_size) = ((category_colors[cat],
                                             get_suffix_style(suffix)[1])
                                             if multi_category
                                             else get_suffix_style(suffix))
                    label = (f'{cat}: {position_name(suffix)}'
                             if multi_category
                             else position_name(suffix))
                    if suffix == '': # unitcell2
                        if multi_category or len(points) == 1:
                            ax_var.scatter([point[0]
                                            for point in points],
                                           [point[1]
                                            for point in points],
                                           color=color,
                                           marker=marker,
                                           s=marker_size**2,
                                           zorder=3,
                                           label=label)
                            for _, value in points:
                                ax_var.axhline(value,
                                               color=color,
                                               linewidth=1,
                                               linestyle=get_line_style(suffix),
                                               alpha=0.7,
                                               zorder=1)
                        else:
                            ax_var.plot([point[0]
                                         for point in points],
                                        [point[1]
                                         for point in points],
                                        color=color,
                                        linewidth=1,
                                        linestyle=get_line_style(suffix),
                                        marker=marker,
                                        markersize=marker_size,
                                        label=label)
                    else:
                        ax_var.plot([point[0]
                                     for point in points],
                                    [point[1]
                                     for point in points],
                                    color=color,
                                    linewidth=1,
                                    linestyle=get_line_style(suffix),
                                    marker=marker,
                                    markersize=marker_size,
                                    label=label)
            ax_var.set_ylabel(f'{fct_mathLabel(fct_displayForce(colomnTyp))} [{unit}]')
            ax_var.set_title(fct_mathLabel(fct_displayForce(colomnTyp)))
            format_number_axes(ax_var)

        unitcellPlot = 'single' # single or mean
        if (4 in rows and unitcellPlot == 'mean'):
            # Zeile 4: Vergleich mit dem Mittelwert aller vorhandenen
            # Unitcell-Referenzen; dieselbe Definition wie bisher.
            unitcell_values = [model_values['']
                               for name, model_values in values.items()
                               if (fct_parseKey(name)[2] == 'unitcell2'
                                   and '' in model_values)]
            unitcell_ref = (float(np.mean(unitcell_values))
                            if unitcell_values
                            else None)
            if multi_category:
                unitcell_refs = {}
                for cat in categories:
                    refs = [model_values['']
                            for name, model_values in values.items()
                            if (fct_parseKey(name)[0] == cat
                                and fct_parseKey(name)[2] == 'unitcell2'
                                and '' in model_values)]
                    if refs:
                        unitcell_refs[cat] = float(np.mean(refs))
            add_ratio_guides(ax_dev)
            if (multi_category or
                (unitcell_ref is not None and
                 unitcell_ref != 0)):
                for suffix in all_suffixes:
                    if suffix == '': continue
                    grouped_points = defaultdict(list)
                    for name, model_values in values.items():
                        cat, _, dim, _ = fct_parseKey(name)
                        ref = (unitcell_refs.get(cat)
                               if multi_category
                               else unitcell_ref)
                        if (suffix not in model_values or
                            ref is None or
                            ref == 0):
                            continue
                        # group = ((cat, dim)
                        #          if multi_category
                        #          else ('all', ''))
                        group = (cat, dim)
                        grouped_points[group].append((
                            x_pos[x_label(name, show_dim)],
                            (model_values[suffix] - ref) / ref))
                    for (cat, dim), points in grouped_points.items():
                        points.sort(key=lambda point: point[0])
                        # points.sort(key=lambda point: point[0],
                        #             reverse=True)
                        color, (marker,
                                marker_size) = (
                            (category_colors[cat], get_suffix_style(suffix)[1]) if multi_category
                            else get_suffix_style(suffix))
                        ax_dev.plot([point[0] for point in points],
                                    [point[1] for point in points],
                                    color=color,
                                    linewidth=1,
                                    linestyle=get_line_style(suffix),
                                    marker=marker,
                                    markersize=marker_size,
                                    label=(f'{cat}: {position_name(suffix)}'
                                            ' vs. Unitcell'
                                       if multi_category
                                       else f'{position_name(suffix)}'
                                             ' vs. Unitcell'))
                ax_dev.set_ylabel(r'$\delta_{\mathrm{UC}}$ [-]')
                ax_dev.set_title(fct_mathLabel(fct_displayForce(colomnTyp)))
                format_number_axes(ax_dev)

        elif (4 in rows and unitcellPlot == 'single'):
            # Zeile 4: Vergleich mit der passenden Unitcell-Referenz.
            # Bei nur einem vorhandenen Unitcell2 wird dieses für alle Modelle
            # verwendet. Bei mehreren Unitcell2 wird anhand von Typ und Nummer
            # das passende Modell herangezogen.
            unitcell_refs = {}
            for name in values:
                cat, number, dimension, _ = fct_parseKey(name)
                unitcell_names = [ref_name
                                  for ref_name in values
                                  if (fct_parseKey(ref_name)[0] == cat
                                      and fct_parseKey(ref_name)[2] == 'unitcell2'
                                      and '' in values[ref_name])
                                  ]
                if len(unitcell_names) == 1:
                    ref_name = unitcell_names[0]
                else:
                    matching = [ref_name
                                for ref_name in unitcell_names
                                if fct_parseKey(ref_name)[1] == number
                                ]
                    ref_name = matching[0] if matching else None
                unitcell_refs[name] = (values[ref_name]['']
                                       if ref_name is not None
                                       else None
                                       )
            add_ratio_guides(ax_dev)
            for suffix in all_suffixes:
                if suffix == '': continue
                grouped_points = defaultdict(list)
                for name, model_values in values.items():
                    cat, _, dim, _ = fct_parseKey(name)
                    ref = unitcell_refs.get(name)
                    if (suffix not in model_values
                        or ref is None
                        or ref == 0):
                        continue
                    # group = ((cat, dim)
                    #          if multi_category
                    #          else ('all', ''))
                    group = (cat, dim)
                    grouped_points[group].append((x_pos[x_label(name,
                                                                show_dim)],
                                                  (model_values[suffix]
                                                   - ref)
                                                  / ref))
                for (cat, dim), points in grouped_points.items():
                    points.sort(key=lambda point: point[0])
                    # points.sort(key=lambda point: point[0],
                    #             reverse=True)
                    color, (marker,
                            marker_size) = ((category_colors[cat],
                                             get_suffix_style(suffix)[1])
                                             if multi_category
                                             else get_suffix_style(suffix)
                                             )
                    ax_dev.plot([point[0] for point in points],
                                [point[1] for point in points],
                                color=color,
                                linewidth=1,
                                linestyle=get_line_style(suffix),
                                marker=marker,
                                markersize=marker_size,
                                label=(f'{cat}: {position_name(suffix)}'
                                        ' vs. Unitcell'
                                        if multi_category
                                        else f'{position_name(suffix)}'
                                              ' vs. Unitcell'))
            ax_dev.set_ylabel(r'$\delta_{\mathrm{UC}}$ [-]')
            ax_dev.set_title(fct_mathLabel(fct_displayForce(colomnTyp)))
            format_number_axes(ax_dev)

    # Legend spacing [cm]
    legend_gap_cm = 0.15
    ax_ref = axes[0, -1]
    gap_x = legend_gap_cm / (fig_width_cm *
                             ax_ref.get_position().width)
    gap_y = legend_gap_cm / (fig_height_cm *
                             ax_ref.get_position().height)
    if 1 in rows:
        ax_i = axes[rows.index(1), -1]
        if multi_category:
            number_alpha = {number: color[3]
                            for (_,number), (color,_) in model_style.items()}
            handles_models = [Line2D([0], [0],
                                     color='0.45',
                                     alpha=number_alpha[number],
                                     marker='.',
                                     markersize=4,
                                     linestyle='None',
                                     label=fct_categoryValue(category if not multi_category else categories[0], fct_shortNumber(number), defaults))
                              for number in sorted({key[1]
                                                    for key in model_style},
                                       key=fct_sortKey)
                              ]
        else:
            handles_models = [Line2D([0], [0],
                                     color=color,
                                     marker=marker,
                                     markersize=marker_size,
                                     linestyle='None',
                                     label=fct_categoryValue(category if not multi_category else categories[0], fct_shortNumber(number), defaults))
                              for number, (color, (marker,
                                                   marker_size))
                              in sorted(model_style.items(),
                                        key=lambda item:
                                            fct_sortKey(str(item[0])))
                              ]
        legend_models = fct_formatLegend(ax_i,
                                         typ        = 'handles',
                                         handles    = handles_models,
                                         xLegend    = 1+gap_x,
                                         yLegend    = 1,
                                         title      = (fct_categoryInfo(category)[0] if not multi_category else 'Model variation'))
        ax_i.add_artist(legend_models)
        fig.canvas.draw()
        yLegend = (legend_models.get_window_extent(fig.canvas.get_renderer())
                   .transformed(ax_i.transAxes.inverted()).y0
                   - gap_y)
        handles_dimensions = [Line2D([0], [0],
                                     color='black',
                                     linestyle=linestyles_dim.get(dimension, '-'),
                                     linewidth=1.5,
                                     label=dimension)
                              for dimension in linestyles_dim
                              if dimension in dimensions
                              ]
        legend_dimensions = fct_formatLegend(ax_i,
                                             typ        = 'handles',
                                             handles    = handles_dimensions,
                                             xLegend    = 1+gap_x,
                                             yLegend    = yLegend,
                                             title      = 'Dimension')
        if multi_category:
            ax_i.add_artist(legend_dimensions)
            fig.canvas.draw()
            if not any(r in rows for r in (2, 3, 4)):
                yLegend = (legend_dimensions.get_window_extent(fig.canvas
                                                               .get_renderer())
                           .transformed(ax_i.transAxes.inverted()).y0
                           - gap_y)
                handles_categories = [Line2D([0], [0],
                                             color=category_colors[cat],
                                             marker='.',
                                             linestyle='-',
                                             label=category_labels[cat])
                                      for cat in categories
                                      ]
                legend_categories = fct_formatLegend(ax_i,
                                                     typ        = 'handles',
                                                     handles    = handles_categories,
                                                     xLegend    = 1+gap_x,
                                                     yLegend    = yLegend,
                                                     title      = 'Category')
                ax_i.add_artist(legend_categories)
    if 2 in rows or 4 in rows:
        handles_reference = [Line2D([0], [0],
                                    color='black',
                                    linestyle='-',
                                    linewidth=1,
                                    label='Zero deviation (0%)'),
                             Line2D([0], [0],
                                    color='black',
                                    linestyle='--',
                                    linewidth=1,
                                    label='±5% deviation')
                             ]
    titles = []
    if 2 in rows: titles.append(((axes[rows.index(2), -1],
                                  'Ratios due to Middle')))
    if 3 in rows: titles.append(((axes[rows.index(3), -1],
                                  'Position variation')))
    if 4 in rows: titles.append(((axes[rows.index(4), -1],
                                  'Ratios due to Unitcell')))
    for ax_i, title in titles:
        handles, labels = ax_i.get_legend_handles_labels()
        is_ratio_axis = 'ratio' in title.lower()
        if handles:
            if multi_category:
                present_cats = [cat
                                for cat in categories
                                if any(label.startswith(f'{cat}: ')
                                       for label in labels)]
                handles_cat = [Line2D([0], [0],
                                      color=category_colors[cat],
                                      linewidth=1.5,
                                      label=category_labels[cat])
                               for cat in present_cats]
                legend_cat = fct_formatLegend(ax_i,
                                              typ        = 'handles',
                                              handles    = handles_cat,
                                              xLegend    = 1+gap_x,
                                              yLegend    = 1,
                                              title      = 'Category')
                ax_i.add_artist(legend_cat)
                fig_current = ax_i.figure
                fig_current.canvas.draw()
                yLegend = (legend_cat.get_window_extent(fig_current.canvas
                                                               .get_renderer())
                           .transformed(ax_i.transAxes.inverted()).y0
                           - gap_y)
                present_suffixes = [suffix
                                    for suffix in ('_Middle',
                                                   '_EdgeY',
                                                   *(() if not useEdges else ('_EdgeX',)),
                                                   '_Corner',
                                                   '_MiddleEdgeX',
                                                   '_MiddleEdgeY',
                                                   '_CornerEdgeX',
                                                   '_CornerEdgeY',
                                                   ''
                                                   )
                                    if any(label.split(': ',
                                                       1)[-1].split(' vs.',
                                                                    1)[0]
                                           == position_name(suffix)
                                           for label in labels)]
                handles_pos = [Line2D([0], [0],
                                      color='black',
                                      linestyle=get_line_style(suffix),
                                      marker=get_suffix_style(suffix)[1][0],
                                      markersize=get_suffix_style(suffix)[1][1],
                                      label=position_name(suffix))
                               for suffix in present_suffixes]
                fct_formatLegend(ax_i,
                                 typ        = 'handles',
                                 handles    = handles_pos,
                                 xLegend    = 1+gap_x,
                                 yLegend    = yLegend,
                                 title      = title)
            else:
                unique = dict(zip(labels, handles))
                ax_i.legend(unique.values(),
                            unique.keys(),
                            title=title,
                            loc='upper left',
                            bbox_to_anchor=(1+gap_x, 1),
                            edgecolor='black')
            if is_ratio_axis:
                content_legend = ax_i.get_legend()
                ax_i.add_artist(content_legend)
                ax_i.figure.canvas.draw()
                yLegend = (content_legend.get_window_extent(ax_i.figure
                                                            .canvas
                                                            .get_renderer())
                           .transformed(ax_i.transAxes.inverted()).y0
                           - gap_y)
                fct_formatLegend(ax_i,
                                 typ        = 'handles',
                                 handles    = handles_reference,
                                 xLegend    = 1+gap_x,
                                 yLegend    = yLegend,
                                 title      = 'Reference line')
    if phase == -1:    phaseSuf = 'in the last phase'
    else:              phaseSuf = f'in phase {phase}'
    if multi_category:
        folderSuf = common + '-'.join(category_labels[cat]
                                      for cat in categories)
        variaSuf  = common
    else:
        variaSuf  = category
        folderSuf = category
    if row_count > 1:
        variation_title = (' and '.join(dict.fromkeys(
            fct_categoryInfo(cat)[1] for cat in categories))
            if multi_category else fct_categoryInfo(category)[1])
        title = (f'Impact on {nameTyps} due to {variation_title} variation {phaseSuf}')
        fig.suptitle(fct_wrapTitle(title, 55), y=1 - 0.3/fig_height_cm)
    # Scientific notation for large/small y-axis values
    from matplotlib.ticker import ScalarFormatter
    for ax in axes.flat:
        formatter = ScalarFormatter(useMathText=True,
                                    useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis='y', pad=2)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontsize(label.get_fontsize() + 0.5)
    pathSave = os.path.join(path,
                            f'plot_variations_{folderSuf}',
                             'plot_variations_00_comparisons')
    fct_formatSave(save, fig,
                   pathSave,
                   f"Matrix_p{phase}_{nameTyps.replace(' ', '')}"
                   f"_{''.join(map(str, rows))}")
    return pathSave

###############################################################################
#---------------------- Helpers: Force Variation Tornado ----------------------
###############################################################################

def fct_abbrevFallback(descr: str) -> str:
    if not descr: return ""
    first = descr.strip().split()[0]
    return first[:2].upper() if len(first) > 0 else descr

def fct_abbrevFullOld(descr: str) -> str:
    m = fct_abbrevLabel(descr)
    if m is not None: return m
    return fct_abbrevFallback(descr)

def fct_abbrevFull(descr: str) -> str:
    if not descr: return ""
    descr = descr.strip().lower()
    # special cases
    if "settlement" in descr:
        if "cs" in descr: return "settl_cs"
        if "ip" in descr: return "settl_ip"
        if "if" in descr: return "settl_if"
    if "diff. settl." in descr:
        if "cs" in descr: return "Dsettl_cs"
        if "ip" in descr: return "Dsettl_ip"
        if "if" in descr: return "Dsettl_if"
    if "diff. tilt" in descr:
        if "cs" in descr: return "tiltD_cs"
        if "ip" in descr: return "tiltD_ip"
        if "if" in descr: return "tiltD_if"
    if "tilt" in descr:
        if "cs" in descr: return "tilt_cs"
        if "ip" in descr: return "tilt_ip"
        if "if" in descr: return "tilt_if"
    if "distortion" in descr:
        if "cs" in descr: return "dist_cs"
        if "ip" in descr: return "dist_ip"
        if "if" in descr: return "dist_if"
    if "defl" in descr:
        if "cs" in descr: return "deflR_cs"
        if "ip" in descr: return "deflR_ip"
        if "if" in descr: return "deflR_if"
    if "values due to" in descr:
        if "due to cs" in descr:   return "CS"
        if "due to line" in descr: return "line"
        if "due to grid" in descr: return "grid"

    prefix = None
    if "var=0" in descr:           return "if_dif"
    if   "interface"             in descr:
        if "limits"              in descr: prefix = "ifL"
        else:                              prefix = "if"
    elif "crosssection"          in descr: prefix = "cs"
    elif "grid"                  in descr: prefix = "gr"
    elif "internalforceanalysis" in descr: prefix = "fa"
    elif "var=0"                 in descr: prefix = "var0"
    if prefix is None:
        print(f'{descr} not known')
        return fct_abbrevFallback(descr)

    suffix = None
    match = re.search(r'\((.*?)\)', descr)
    bracket = (match.group(1)
               if match else descr)
    if   "mean"                  in bracket: suffix = "mea"
    elif "triangular"            in bracket: suffix = "tri"
    elif "trapezoidal"           in bracket: suffix = "tra"
    elif "difference"            in bracket: suffix = "dif"
    elif "wissmann"              in bracket: suffix = "wis"
    elif "simpson"               in bracket:
        suffix = f"sim{re.search(r'\d+', bracket).group()}"

    if prefix == "cs" and suffix == "tra":
        if "s_xx" in descr and prefix == "cs":
            suffix = "csX"
        elif "s_xx" in descr and prefix == "tra":
            suffix = "traX"
        elif ("s_yy" in descr and
              "under pile head" not in descr and
              prefix == "tra"):
            suffix = "traY"
        elif ("s_yy" in descr and
              "under pile head" not in descr and
              prefix == "cs"):
            suffix = "csY"

    if prefix == "fa": #InternalForceAnalysis
        suffixFA = []
        if "difference"         in bracket: suffixFA.append("dif")
        if   "last soil"        in descr:   suffixFA.append("lastS")
        elif "first soil"       in descr:   suffixFA.append("firstS")
        elif "soil"             in descr:   suffixFA.append("S")
        if   "offset line"      in descr:   suffixFA.append("Lo")
        elif   "offset_line"    in descr:   suffixFA.append("Lo")
        elif "line"             in descr:   suffixFA.append("L")
        if   "material"         in descr:   suffixFA.append("M")
        suffix = "".join(suffixFA)

    if suffix:
        return f"{prefix}_{suffix}"
    return prefix

def fct_abbrevSuffix(suffix: str) -> str:
    pos_map = {'Middle': 'M',
               'EdgeY': 'EY',
               'EdgeX': 'EX',
               'Corner': 'C'}
    if suffix in pos_map:  return pos_map[suffix]
    if suffix: return suffix[0].upper()
    return suffix

def fct_abbrevNumber(number):
    try:
        num = float(number)
        return num
    except:
        if isinstance(number, str):
            if number.islower(): return number[0] # medium->m
            upper_chars = ''.join(c for c in number if c.isupper())
            if upper_chars: return number[0] + upper_chars # mediumFine->mF
            return number[0]
        raise ValueError(f"Unknown number label: {number}")

###############################################################################
#--------------------------- Main: Time-Force Tornado -------------------------
###############################################################################

def plot_approachVariation(path, category, folders, typForce, phase,
                           typPlot='dual', # dual|force|time
                           save=saveDefault, useEdges=False):
    folders = sorted(folders,
                     key=lambda folder:
                         (
                         fct_sortKey(fct_parseKey(os.path.basename(folder))[1]),
                         fct_sortKey(os.path.basename(folder))
                         )
                     )
    models, legendColor, legendHatch, modelsRef = fct_buildDictvalues(path,
                                                                      folders,
                                                                      typForce,
                                                                      phase,
                                                                      useEdges=useEdges)
    groupedModels = build_modelGroups(models)
    # print(groupedModels)
    filenames = [groupedModels]
    for key, modelsDict in groupedModels.items():
        legendColorShort = {}
        for model in modelsDict:
            category, number, _, _ = fct_parseKey(model)
            # keyColor = (category
            #             + (' = ' if get_digit(str(number)) else ' with ')
            #             + str(number))
            keyColor = str(number)
            legendColorShort[keyColor] = legendColor[keyColor]
        legendHatchShort = {}
        for model in modelsDict:
            _, _, dimension, suffix = fct_parseKey(model)
            keyHatch = f"{dimension}{suffix}"
            legendHatchShort[keyHatch] = legendHatch[keyHatch]
        modelsRefShort = {}
        numbers = {fct_parseKey(model)[1]
                   for model in modelsDict}
        for refModel, refData in modelsRef.items():
            if number == 'var':
                modelsRefShort[refModel] = refData
            else:
                _, number, _, _ = fct_parseKey(refModel)
                if number in numbers:
                    modelsRefShort[refModel] = refData
        # print(path)
        # print(modelsDict)
        filename = plot_approachVariationPlot(path,
                                              category,
                                              typForce,
                                              phase,
                                              key,
                                              modelsDict,
                                              modelsRefShort,
                                              legendColorShort,
                                              legendHatchShort,
                                              typPlot)
        filenames.append(filename)
    return filenames

def fct_buildDictvalues(path, folders, typForce, phaseRef,
                        useEdges=False,
                        refModels = ['unitcell2'],
                        alpha_list = [1.0, 0.85, 0.7, 0.55, 0.4]):
    #refModels = ['unitcell2', 'unitcell3']
    hatch_list   = ['', '//', '..', 'xx', '\\\\', '++', '--']
    _,color_dict = fct_setPlotstyle()
    n_models = len({fct_parseKey(folder)[1] for folder in folders})
    color_shade = min(len(color_dict[0]),
                      max(1,
                          math.ceil(n_models / len(color_dict))
                          )
                      )
    models       = {}
    modelsRef    = {}
    legendColor  = {}
    legendHatch  = {}
    legendAlpha  = {}
    counterColor = 0
    counterAlpha = 0
    for folder in folders:
        category, number, dimension, _ = fct_parseKey(folder)
        suffixes, _ = fct_parseDimension(dimension)
        for suffix in suffixes:
            if not useEdges and suffix == '_EdgeX':
                continue
            subfolder = f'summary_value{suffix}'
            file_name = f'{subfolder}.txt'
            pathFile = os.path.join(path, folder, subfolder, file_name)
            # print(pathFile)
            if not os.path.exists(pathFile): continue
            data = fct_parseText(pathFile)
            if data is None: continue
            # keyColor = (category
            #             + (' = ' if get_digit(str(number)) else ' with ')
            #             + str(number))
            keyColor = str(number)
            if keyColor not in legendColor:
                # base  = counterColor % color_base
                # shade = (counterColor // color_base) % color_shade
                base  = counterColor // color_shade
                shade = counterColor % color_shade
                legendColor[keyColor] = color_dict[base][shade]
                counterColor += 1
            color = legendColor[keyColor]
            keyHatch = f"{dimension}{suffix}"
            if keyHatch not in legendHatch:
                legendHatch[keyHatch] = hatch_list[len(legendHatch)
                                                   % len(hatch_list)]
            hatch = legendHatch[keyHatch]
            keyModel = f"{folder}{suffix}"
            if keyModel not in models:
                models[keyModel] = {"color": color,
                                    "hatch": hatch,
                                    "methods": {},
                                    }
            if typForce.endswith("_uw"):
                ref_force = {}
                for entry in data:
                    if entry['type'] != typForce.replace("_uw", ""):
                        continue
                    descr = fct_abbrevFull(entry["description"])
                    phase = entry["phase"]
                    force = entry["force"]
                    key   = (typForce, descr)
                    if (key not in ref_force or
                        phase < ref_force[key][0]):
                        ref_force[key] = (phase, force)
            for entry in data:
                # print(entry["type"],
                #       entry["description"])
                if phaseRef == -1:
                    phaseRef_use = find_maxPhase(path, folder)
                else: phaseRef_use = phaseRef
                if ((phaseRef_use is not None
                     and entry['phase'] != phaseRef_use) or
                    entry['type'] != typForce.replace("_uw", "")):
                    continue
                force = entry["force"]
                time = entry["time"]
                # print(pathFile)
                descr = fct_abbrevFull(entry['description'])
                if descr not in legendAlpha:
                    legendAlpha[descr] = alpha_list[counterAlpha
                                                    % len(alpha_list)]
                    counterAlpha += 1
                alpha = legendAlpha[descr]
                if typForce.endswith("_uw"):
                    ref = ref_force.get((typForce, descr))
                    if ref is not None:
                        force = force - ref[1]
                if descr not in models[keyModel]["methods"]:
                    models[keyModel]["methods"][descr] = {}
                models[keyModel]["methods"][descr]["force"] = abs(force)
                models[keyModel]["methods"][descr]["time"] = abs(time)
                models[keyModel]["methods"][descr]["alpha"] = alpha
                if dimension in refModels:
                    if keyModel not in modelsRef: # takes the first method
                        modelsRef[keyModel] = {"force": force,
                                               "color": color}
    # print(models)
    return models, legendColor, legendHatch, modelsRef

# models
#  └── rad_1_quarter_middle
#       ├── color zB #123
#       ├── hatch zB '//'
#       └── methods
#            └── description1
#                 ├── force
#                 └── time
#            └── description2
#                 ├── force
#                 └── time
#  └── rad_2_quarter_middle
#       ├── color zB #123
#       ├── hatch zB '//'
#       └── methods
#            └── description1
#                 ├── force
#                 └── time
#            └── description2
#                 ├── force
#                 └── time

def fct_niceScale(value): # regarding time BUT not used anymore
    nice_steps = [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 1,
                  5, 10, 15, 20, 25, 50, 100, 200, 500,
                  1000, 2000, 5000, 10000]
    abs_value = abs(value)
    for s in nice_steps:
        if abs_value <= s:
            return s
    print(f'fct_niceScale makes {abs_value} // {nice_steps[-1]}')
    return nice_steps[-1] * (abs_value // nice_steps[-1] + 1)

def fct_timeUnit(values):
    max_time = np.max(np.abs(values))
    if   max_time <= 120:  return 1,    'sec'
    elif max_time <= 7200: return 60,   'min'
    else:                  return 3600, 'h'

def fct_methodGroup(method):
    prefix = str(method).split('_')[0].lower()
    group_aliases = { 'fa': 'fa',
                     'da': 'fa',  # da_difSLoM belongs to fa
                     'cs': 'cs',
                     'if': 'if',
                     'sr': 'sr',
                     'gr': 'sr',
                     }
    return group_aliases.get(prefix, prefix)

def fct_methodShades(methods):
    groups = sorted({ fct_methodGroup(method)
                     for method in methods
                     })
    alphas = np.linspace(1.0, 0.45, len(groups))
    return dict(zip(groups, alphas))

def fct_finestMeshReference(modelsRef, path=None,
                            phase=None, typForce=None):

    mesh_rank = {
        'fine':         0,
        'finemedium':   1,
        'mediumfine':   1,
        'mfine':        1,
        'medium':       2,
        'mediumcoarse': 3,
        'coarsemedium': 3,
        'mcoarse':      3,
        'coarse':       4,
    }

    def get_mesh_info(name):
        name = os.path.basename(os.path.normpath(str(name)))
        match = re.match(
            r'^(.*?)_([A-Za-z]+)_unitcell[23]$',
            name, re.IGNORECASE
        )

        if not match:
            return None

        family = match.group(1)
        mesh = match.group(2).lower()

        if mesh not in mesh_rank:
            return None

        return family, mesh, mesh_rank[mesh]

    if not modelsRef:
        return {}

    # First select the finest supplied reference
    ranked = [
        (get_mesh_info(key)[2], key)
        for key in modelsRef
        if get_mesh_info(key) is not None
    ]

    if not ranked:
        return dict(modelsRef)

    _, ref_key = min(ranked, key=lambda item: item[0])
    selected = {ref_key: dict(modelsRef[ref_key])}

    if path is None or phase is None or typForce is None:
        return selected

    family, _, current_rank = get_mesh_info(ref_key)

    # Identify study directory
    search_path = os.path.abspath(path)

    for folder in [search_path, *list(
            os.path.dirname(search_path) for _ in [])]:
        pass

    ancestors = []
    folder = search_path
    while True:
        ancestors.append(folder)
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent

    study_dir = None
    candidates = []

    for folder in ancestors:
        possible = glob.glob(os.path.join(
            folder, f'{family}_*_unitcell2'
        ))

        possible = [
            p for p in possible
            if os.path.isdir(p)
            and get_mesh_info(p) is not None
        ]

        if possible:
            study_dir = folder
            candidates = possible
            break

    if study_dir is None:
        return selected

    candidates.sort(
        key=lambda p: get_mesh_info(p)[2]
    )

    # Read a reference value from Fsu_phases.csv
    force_columns = {
        'Felem': 'F_elem',
        'Fsoil': 'F_soil',
        'Fbase': 'F_base',
        'Fshaft': 'F_shaft',
        'Relem': 'F_elem',
        'Rsoil': 'F_soil',
        'Rbase': 'F_base',
        'Rshaft': 'F_shaft',
        'R_elem': 'F_elem',
        'R_soil': 'F_soil',
        'R_base': 'F_base',
        'R_shaft': 'F_shaft',
    }

    column = force_columns.get(typForce, typForce)

    for candidate in candidates:
        info = get_mesh_info(candidate)
        rank = info[2]

        if rank >= current_rank:
            continue

        csv_file = os.path.join(
            candidate, 'summary_value', 'Fsu_phases.csv'
        )

        if not os.path.isfile(csv_file):
            continue

        df = pd.read_csv(csv_file, sep=';')

        if 'Phase' not in df.columns or column not in df.columns:
            continue

        values = df.loc[
            pd.to_numeric(df['Phase'], errors='coerce')
            == int(phase),
            column
        ]

        if values.empty:
            continue

        force = pd.to_numeric(values, errors='coerce').iloc[0]

        if not np.isfinite(force):
            continue

        new_key = os.path.basename(candidate)
        new_value = dict(modelsRef[ref_key])
        new_value['force'] = float(force)

        print(f'Reference: {ref_key} -> {new_key}')

        return {new_key: new_value}

    return selected

def plot_approachVariationPlot(path, category, typForce,
                               phase, key,
                               modelsDict, modelsRef,
                               legendColor, legendHatch,
                               typPlot = 'dual',#force; time
                               perc_plot = False,
                               use_shades = True,
                               exclude_methods = exclude_methodsDefault,
                               save = saveDefault,
                               dpi = dpiDefault):
    from matplotlib.colors import to_rgba
    if typPlot in ('force', 'dual'):
        modelsRef = fct_finestMeshReference( modelsRef,
                                            path=path,
                                            phase=phase,
                                            typForce=typForce
                                            )
    multiple_dim = any(
        'unitcell2' not in model_key.lower()
        for model_key in modelsDict
        )
    method_groups = [
        descr
        for model in modelsDict.values()
        for descr in model["methods"]
        if not descr.startswith(tuple(exclude_methods))
    ]
    group_shades = fct_methodShades(method_groups)
    # Always use method-group shading when enabled
    group_shading = use_shades
    
    vals_label  = []
    vals_force  = []
    vals_time   = []
    bar_colors  = []
    bar_hatches = []
    method_groups = [
        descr
        for model in modelsDict.values()
        for descr in model["methods"]
        if not descr.startswith(tuple(exclude_methods))
    ]
    group_shades = fct_methodShades(method_groups)
    # Group shading if multiple model categories exist
    group_shading = use_shades and len(legendColor) > 1
    for model_key, model in modelsDict.items():
        color = model["color"]
        hatch = model["hatch"]
        for descr, method in model["methods"].items():
            if descr.startswith(tuple(exclude_methods)):
                continue
            vals_label.append(descr)
            vals_force.append(method["force"])
            vals_time.append(method["time"])
            bar_hatches.append(hatch)
            if use_shades:
                alpha = group_shades[fct_methodGroup(descr)]
            else:
                alpha = 1.0
            # Apply transparency to fill only
            rgba = to_rgba(color)
            bar_colors.append(
                (rgba[0], rgba[1], rgba[2], rgba[3] * alpha)
            )
    methods_by_dim = defaultdict(set)
    for model_key, model in modelsDict.items():
        dim = fct_parseKey(model_key)[2]
        methods_by_dim[dim].update(descr for descr in model['methods']
                                   if not descr.startswith(tuple(exclude_methods)))
    dimension_axis = bool(methods_by_dim) and all(
        len(methods) <= 1 for methods in methods_by_dim.values())
    if dimension_axis:
        vals_label = [fct_parseKey(k)[2]
                      for k, model in modelsDict.items()
                      for descr in model['methods']
                      if not descr.startswith(tuple(exclude_methods))]
    if (len(vals_force)==0 or
        len(vals_time)==0):
        print(f'Values not defined for {typForce} in phase {phase} '
              'within plot_approachVariationPlot:\n'
              f'{list(modelsDict.keys())}')
        return None
    if (len(vals_force)==1 or
        len(vals_time)==1):
        # print(f'Only one value defined for {vals_force} and {vals_time}')
        return None
    # print('plot_approachVariationPlot\n'
    #       f'vals_force: {vals_force}\n'
    #       f'vals_time: {vals_time}')
    timeFactor, unitT = fct_timeUnit(vals_time)
    forcesCount = len(vals_force)
    figHeight = max(8, forcesCount * 0.4) / CM_PER_INCH
    if typPlot == 'dual':
        fig, (ax_time, ax_force) = plt.subplots(1, 2,
                                                sharey=True,
                                                figsize=(figWidthDEF_cm/
                                                         CM_PER_INCH,
                                                         figHeight),
                                                gridspec_kw={'width_ratios':
                                                             [1, 1],
                                                             'wspace':
                                                                 0.05},
                                                dpi=dpi
                                                )
        ax = ax_force
    else:
        fig, ax = plt.subplots(1, 1,
                               figsize=(15/CM_PER_INCH, figHeight),
                               dpi=dpi
                               )
        ax_time = ax
        ax_force = ax
    y = np.arange(forcesCount)
    if typPlot in ['dual', 'time']:
        if typPlot == 'dual':   colorTime = 'lightgrey'
        else:                   colorTime = bar_colors
        bars_time = ax_time.barh(y,
                                 [v / timeFactor
                                  for v in vals_time],
                                 color=colorTime,
                                 label=f'Time [{unitT}]'
                                 )
        for bar, h in zip(bars_time, bar_hatches):
            bar.set_hatch(h)
            bar.set_edgecolor('black')
            bar.set_linewidth(0.6)
        if typPlot == 'dual': ax_time.invert_xaxis()
    if typPlot in ['dual', 'force']:
        if typForce == 'settl':
            quantity = 'Total settlement'
            symbol = r'$u_{max}$'
            unit = 'm'
        elif typForce == 'settlDiff':
            quantity = 'Differential settlement'
            symbol = r'$\delta_u$'
            unit = 'm'
        elif typForce == 'tilt':
            quantity = 'Overall tilt'
            symbol = r'$\omega_u$'
            unit = '%'
        elif typForce == 'tiltDiff':
            quantity = 'Angular strain'
            symbol = r'$\alpha_u$'
            unit = '%'
        elif typForce == 'distortion':
            quantity = 'Angular distortion'
            symbol = r'$\beta_u$'
            unit = '%'
        elif typForce == 'relativeDeflection':
            quantity = 'Relative deflection'
            symbol = r'$\Delta u$'
            unit = 'm'
        elif typForce == 'deflectionRatio':
            quantity = 'Deflection ratio'
            symbol = r'$\Delta u/l_{\mathrm{span}}$'
            unit = '-'
        elif typForce.startswith(('F', 'R')):
            quantity = 'Force'
            symbol = fct_mathLabel('R' + typForce[1:]
                                   if typForce.startswith('F')
                                   else typForce
                                   )
            unit = 'kN'
        bars_force = ax_force.barh(y,
                                   vals_force,
                                   color=bar_colors,
                                   label=f'{symbol} [{unit}]'
                                   )
        for bar, h in zip(bars_force, bar_hatches):
            bar.set_hatch(h)
            bar.set_edgecolor('black')
            bar.set_linewidth(0.6)
    # ref plots
    if (multiple_dim and not dimension_axis and
        typPlot in ['dual', 'force']):
        for keyModel, valueDict in modelsRef.items():
            color = valueDict['color']
            ax_force.axvline(abs(valueDict["force"]),
                             color          = color,
                             linestyle      = '--',
                             linewidth      = 1.2)
            if perc_plot:
                ax_force.axvline(-valueDict["force"] * (1 - 0.05),
                                 color      = color,
                                 linestyle  = ':',
                                 linewidth  = 1.0,
                                 alpha      = 0.5)
                ax_force.axvline(-valueDict["force"] * (1 + 0.05),
                                 color      = color,
                                 linestyle  = ':',
                                 linewidth  = 1.0,
                                 alpha      = 0.5)
   # look
    ax_time.set_yticks(y)
    ax_time.set_yticklabels(vals_label)
    if typPlot == 'dual':
        ax_time.set_ylabel('Dimension' if dimension_axis else 'Force evaluation method')
        ax_time.set_xlabel(f'Time [{unitT}]')
        ax_force.set_xlabel(f'{symbol} [{unit}]')
        ax_force.tick_params(axis       = 'y',
                             left       = False,
                             labelleft  = False)
        fig.suptitle(f'{quantity}-Time-Comparison for phase {phase}')
    elif typPlot == 'force':
        ax_force.set_ylabel('Dimension' if dimension_axis else 'Force evaluation method')
        ax_force.set_xlabel(f'{symbol} [{unit}]')
        ax_force.set_title(f'{quantity}-Comparison for phase {phase}')
    elif typPlot == 'time':
        ax_time.set_ylabel('Dimension' if dimension_axis else 'Force evaluation method')
        ax_time.set_xlabel(f'Time [{unitT}]')
        ax_time.set_title(f'Time-Comparison for phase {phase}')
    # legends
    handles_color = []
    for description, color in sorted(legendColor.items(),
                                     key=lambda item: fct_sortKey(item[0])):
        if (use_shades and
            len(color) == 4):
            color_leg = (color[0], color[1], color[2], 1.0)
        else:
            color_leg = color
        handles_color.append(Patch(facecolor=color_leg,
                                   label=description)
                             )
    original_lw = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.35
    handles_hatch = [Patch(facecolor='white',
                           edgecolor='black',
                           hatch=hatch*2, # fine legend strokes
                           label=description)
                     for description, hatch in legendHatch.items()]
    plt.rcParams['hatch.linewidth'] = original_lw #  back to normal
    handles_shades = []
    if use_shades and len(group_shades) > 1:
        handles_shades = [
            Patch(
                facecolor=(0.3, 0.3, 0.3, float(alpha)),
                edgecolor='black',
                linewidth=0.6,
                label=group
            )
            for group, alpha in group_shades.items()
        ]
    handles_lines = []
    if multiple_dim and not dimension_axis and typPlot in ['dual', 'force']:
        handles_lines = [Line2D([0], [0],
                                color=valueDict['color'],
                                linestyle='--',
                                linewidth=1.5,
                                label=(
                                    keyModel.split("_")[1]
                                    if "_" in keyModel else keyModel
                                    )
                                )
                         for keyModel, valueDict in modelsRef.items()
                         ]
        if modelsRef and perc_plot:
            handles_lines.append(Line2D([0], [0],
                                        color='k',
                                        linestyle=':',
                                        linewidth=1.5,
                                        label='±5%'
                                        )
                                 )
    if typPlot == 'dual':
        left_cm   = 3
        right_cm  = 2.7
        top_cm    = 0.7
        bottom_cm = 1.2
        fig_width_cm  = fig.get_figwidth() * CM_PER_INCH
        fig_height_cm = fig.get_figheight() * CM_PER_INCH
        fct_adjustCm(fig, left=left_cm, right=right_cm,
                 top=top_cm, bottom=bottom_cm)
    else:
        fct_adjustCm(fig, left=1.3, right=3.5, bottom=1.4, top=1.0)
    gap     = 0.02
    xLegend = 1.00+gap
    yLegend = 1.00
    legends = []
    for title, handles in [(category,        handles_color),
                       ('Dimension',    [] if dimension_axis else handles_hatch),
                       ('Method group', handles_shades),
                       ('UC2 values',   handles_lines)
                       ]:
        if len(handles) > 0:
            legend = fct_formatLegend(ax,
                                      typ        = 'handles',
                                      handles    = handles,
                                      xLegend    = xLegend,
                                      yLegend    = yLegend,
                                      title      = title)
            ax.add_artist(legend)
            legends.append(legend) # for adjust window
            fig.canvas.draw()
            yLegend = (legend.get_window_extent()
                       .transformed(ax.transAxes.inverted()).y0
                       - gap)
    if legends:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_width = max(legend.get_window_extent(renderer).width
                           for legend in legends) / fig.dpi
        fig_width = fig.get_figwidth()
        axes_right = ax.get_position().x1
        available_width = fig_width * (1 - axes_right)
        required_width = legend_width + gap*2
        if required_width > available_width:
            fig.set_size_inches(fig_width + required_width
                                - available_width,
                                fig.get_figheight(),
                                forward=True
                                )
    pathData = os.path.join(path,
                            f'plot_variations_{category}',
                            f'plot_variations_{typForce}_'
                            f'{typPlot}_p{phase}_{key.split("_")[1]}')
    filename = f'{typPlot}_{typForce}_p{phase}_{key}'
    pathFile = fct_formatSave(save, fig, pathData, filename)
    return pathFile

###############################################################################
#------------------------------ Time per Model --------------------------------
###############################################################################

def fct_runtimeRead(pathFile):
    runtime = {'model': None,
               'eval':  None,
               'plot':  None}
    try:
        with open(pathFile, 'r') as f: lines = f.readlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if  'RUNTIME_MODEL' in line: current_section = 'model'
            elif 'RUNTIME_EVAL' in line: current_section = 'eval'
            elif 'RUNTIME_PLOT' in line: current_section = 'plot'
            elif 'runtime_seconds' in line and current_section:
                runtime[current_section] = float(line.split('=')[1])
        return runtime
    except Exception as e:
        print("[Runtime read ERROR]", e)
        traceback.print_exc()
        return 0

def fct_runtimeTable(folder_infos, runtimes, pathPlot, save=True):
    import csv
    # Fixed unit for comparable tables across different plots
    factor = 60.0
    headers = [ 'Variant', 'Dimension',
               'model [min]', 'eval [min]',
               'model+eval [min]'
               ]
    def read_seconds(runtime, part):
        value = runtime.get(part)
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    rows = []
    for info, runtime in zip(folder_infos, runtimes):
        model = read_seconds(runtime, 'model')
        evaluation = read_seconds(runtime, 'eval')
        total = (model + evaluation
                 if model is not None and evaluation is not None
                 else None
                 )
        rows.append([str(info['number']),
                     str(info['dimension']),
                     model / factor if model is not None else None,
                     evaluation / factor if evaluation is not None else None,
                     total / factor if total is not None else None,
                     ])
    if not rows: return
    display_rows = [row[:2] + [f'{value:.2f}'
                               if value is not None else '—'
                               for value in row[2:]
                               ]
                    for row in rows
                    ]
    widths = [max(len(headers[j]), *(len(row[j])
                                     for row in display_rows))
              for j in range(len(headers))
              ]
    def format_row(row):
        return ' | '.join(value.ljust(widths[j]) if j < 2
                          else value.rjust(widths[j])
                          for j, value in enumerate(row)
                          )
    print('\nRuntime table')
    print(format_row(headers))
    print('-+-'.join('-' * width for width in widths))
    for row in display_rows: print(format_row(row))
    fig_table, ax_table = plt.subplots(figsize=(9,
                                                max(2.0, 0.36 * (len(rows) + 2))),
                                       dpi=dpiDefault
                                       )
    ax_table.axis('off')
    table = ax_table.table(cellText=display_rows,
                           colLabels=headers,
                           cellLoc='right',
                           colLoc='center',
                           loc='center'
                           )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table.auto_set_column_width(col=list(range(len(headers))))
    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        elif col_index < 2:
            cell.set_text_props(ha='left')
    ax_table.set_title('Runtime comparison', pad=12)
    fig_table.tight_layout()
    if save:
        os.makedirs(pathPlot, exist_ok=True)
        with open(os.path.join(pathPlot,
                               'RuntimeTable.csv'),
                  'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(headers)
            writer.writerows(rows)
            # LaTeX export
            def latex_escape(value):
                replacements = {'\\': r'\textbackslash{}',
                                '&': r'\&',
                                '%': r'\%',
                                '$': r'\$',
                                '#': r'\#',
                                '_': r'\_',
                                '{': r'\{',
                                '}': r'\}',
                                '~': r'\textasciitilde{}',
                                '^': r'\textasciicircum{}',
                                }
                return ''.join(replacements.get(char, char)
                               for char in str(value)
                               )
            latex_lines = [r'% Requires \usepackage{booktabs}',
                           r'\begin{table}[htbp]',
                           r'    \centering',
                           r'    \begin{tabular}{llrrr}',
                           r'        \toprule',
                           (r'        Variant & Dimension & model [min] '
                            r'& eval [min] & model+eval [min] \\'
                            ),
                           r'        \midrule',
                           ]
            for row in rows:
                cells = [latex_escape(row[0]),
                         latex_escape(row[1]),
                         *[f'{value:.2f}'
                           if value is not None else '--'
                           for value in row[2:]
                           ],
                         ]
                latex_lines.append('        ' + ' & '.join(cells) + r' \\')
            latex_lines.extend([r'        \bottomrule',
                                r'    \end{tabular}',
                                (r'    \caption{Model and evaluation runtimes '
                                 r'in minutes. Totals exclude plotting.}'
                                 ),
                                r'    \label{tab:runtime_comparison}',
                                r'\end{table}',
                                ])
            with open(os.path.join(pathPlot, 'RuntimeTable.tex'),
                      'w', encoding='utf-8') as file:
                file.write('\n'.join(latex_lines) + '\n')
    fct_formatSave(save, fig_table, pathPlot, 'RuntimeTable')

def plot_runtime(path,
                 folders,
                 category,
                 typPlot = 'parts', #total
                 parts=('model','eval','plot'),
                 save=True):
    pathPlot = os.path.join(path,
                            f'plot_variations_{category}',
                             'plot_variations_00_runtime')
    folders = sorted(folders,
                     key=lambda folder:
                         (
                         fct_sortKey(fct_parseKey(os.path.basename(folder))[1]),
                         fct_sortKey(os.path.basename(folder))
                         )
                     )
    # PARSE KEYS
    folder_infos = []
    for folder in folders:
        lastfolder = os.path.basename(folder)
        typ, number, dimension, suffix = fct_parseKey(lastfolder)
        folder_infos.append({'folder': folder,
                             'number': number,
                             'dimension': dimension})
    # GROUPING LOGIC
    _, colors = fct_setPlotstyle()
    n_groups = len(colors)
    n_levels = len(colors[0])
    hatch_list = ['', '/', '.', 'x', '\\\\', '+', '-']
    color_map = {}
    hatch_map = {}
    color_counter = 0
    n_models = len({info['number'] for info in folder_infos})
    shades_per_color = min(n_levels,
                           max(1, math.ceil(n_models / (n_groups))))
    for info in folder_infos:
        number = info['number']
        dimension = info['dimension']
        if number not in color_map:
            g = (color_counter // shades_per_color) % n_groups
            l = color_counter % shades_per_color
            color_map[number] = colors[g][l]
            color_counter += 1
        if dimension not in hatch_map:
            hatch_map[dimension] = hatch_list[len(hatch_map)
                                              % len(hatch_list)]
    # collect runtimes in seconds
    runtimes = [fct_runtimeRead(os.path.join(path,
                                             folder,
                                             'input_parameters.txt'))
                for folder in folders]
    # Additional runtime table: model, eval and model+eval
    fct_runtimeTable(folder_infos=folder_infos,
                     runtimes=runtimes,
                     pathPlot=pathPlot,
                     save=save
                     )
    runtime_data = {part: [runtime.get(part, 0) or 0
                           for runtime in runtimes]
                    for part in parts}
    # determine best unit
    if typPlot in ['total']:
        runtime_total = [sum(value or 0
                             for value in runtime.values())
                         for runtime in runtimes]
        max_time_sec = max(runtime_total)
    else:
        max_time_sec = max([max(vals) if len(vals) > 0 else 0
                            for vals in runtime_data.values()])
    if   max_time_sec > 3600: factor = 3600; unit = 'hour'; #unitShort = 'h'
    elif max_time_sec > 60:   factor = 60;   unit = 'min';  #unitShort = 'm'
    else:                     factor = 1;    unit = 'sec';  #unitShort = 's'
    # convert units
    if typPlot in ['total']:
        totals = [t / factor for t in runtime_total]
    else:
        runtime_data = {part: [value / factor
                               for value in values]
                        for part, values in runtime_data.items()}
    # ALPHA = PARTS
    alpha_map = {'model': 1.0, 'eval': 0.7, 'plot': 0.4}
    # plotting
    plt.rcParams['hatch.linewidth'] = 0.35
    x = np.arange(len(folders))
    fig, ax = plt.subplots(figsize=(15/CM_PER_INCH, 8/CM_PER_INCH), dpi=dpiDefault)
    if typPlot in ['total']:
        for i, info in enumerate(folder_infos):
            color = color_map[info['number']]
            hatch = hatch_map[info['dimension']]
            ax.bar(x[i], totals[i], color=color,
                   hatch=hatch, edgecolor='black')
        # labels
        for i, v in enumerate(totals):
            ax.text(i, v,
                    # f'{v:.2f} {unitShort}',
                    f'{v:.2f}',
                    ha='center',
                    va='bottom')
    else:
        width = 0.25
        offsets = np.linspace(-width * (len(parts) - 1) / 2,
                              width * (len(parts) - 1) / 2,
                              len(parts))
        for i, info in enumerate(folder_infos):
            color = color_map[info['number']]
            hatch = hatch_map[info['dimension']]
            for j, part in enumerate(parts):
                value = runtime_data[part][i]
                ax.bar(x[i] + offsets[j],
                       value,
                       width,
                       color=color,
                       alpha=alpha_map.get(part, 1.0),
                       hatch=hatch,
                       edgecolor='black')
                # ax.text(x[i] + offsets[j],
                #         value,
                #         # f'{value:.2f} {unitShort}',
                #         # f'{value:.2f}',
                #         ha='center',
                #         va='bottom',
                #         fontsize=5)
    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels([fct_categoryValue(category, info['number'])
                        for info in folder_infos],
                       rotation=45,
                       ha='right')
    # ax.set_xticklabels(folders, rotation=30, ha='right')
    ax.set_xlabel(fct_categoryInfo(category)[0] if category != 'total'
                  else 'Model / variation')
    ax.set_ylabel(f'Runtime [{unit}]')
    ax.set_title('Runtime Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # LEGENDS
    handles_color = [Patch(facecolor=color, edgecolor='black', label=number)
                     for number, color in color_map.items()]
    handles_hatch = [Patch(facecolor='white', edgecolor='black',
                           hatch=hatch, label=dim)
                     for dim, hatch in hatch_map.items()]
    if typPlot not in ['total']:
        handles_alpha = [Patch(facecolor='gray',
                               alpha=alpha_map[part],
                               label=part)
                         for part in parts]
    # fig.subplots_adjust(right=0.80)
    yLegend = 1.00
    gap     = 0.02
    xLegend = 1.00+gap
    fct_adjustCm(fig, left=1.3, right=3.3, bottom=1.7, top=0.9)
    legend = fct_formatLegend(ax,
                              typ        = 'handles',
                              handles    = handles_color,
                              xLegend    = xLegend,
                              yLegend    = yLegend,
                              title      = fct_categoryInfo(category)[0])
    ax.add_artist(legend)
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    yLegend = (legend.get_window_extent().transformed(inv).y0
               - gap)
    legend = fct_formatLegend(ax,
                              typ        = 'handles',
                              handles    = handles_hatch,
                              xLegend    = xLegend,
                              yLegend    = yLegend,
                              title      = 'Dimension')
    ax.add_artist(legend)
    fig.canvas.draw()
    if typPlot not in ['total']:
        yLegend = (legend.get_window_extent().transformed(inv).y0
                   - gap)
        fct_formatLegend(ax,
                         typ        = 'handles',
                         handles    = handles_alpha,
                         xLegend    = xLegend,
                         yLegend    = yLegend,
                         title      = 'Parts')
    fct_formatSave(save, fig, pathPlot, f'RuntimeComparison{typPlot}')
    return

def plot_manualComparison():
    pathPlot = r'C:\Users\annal\Downloads'
    _, colors = fct_setPlotstyle()
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(12/CM_PER_INCH, 5/CM_PER_INCH),
                           dpi=dpiDefault)
    fct_adjustCm(fig, left=1.5, right=4.8, bottom=1.5, top=0.6)
    ax.plot(x, [205.92, 212, 192.786],
            'o-',
            color=colors[0][0],
            label='fa_SM_quarter3')
    ax.plot(x, [243, 264, 233],
            's-',
            color=colors[1][0],
            label='if_tri_quarter3')
    ax.axhline(196.77,
               color=colors[0][0],
               linestyle='--',
               label='fa_SM_unitcell2')
    ax.axhline(226,
               color=colors[1][0],
               linestyle='--',
               label='if_tri_unitcell2')
    ax.set_xticks(x, ['Middle', 'Edge', 'Corner'])
    ax.set_ylabel(r'$F_{\mathrm{elem}}$ [kN]')
    ax.grid(True, alpha=0.3)
    gap=0.02
    ax.legend(loc            = 'upper left',
              bbox_to_anchor = (1+gap, 1),
              title          = 'Evaluation method',
              edgecolor      = 'black')
    fct_formatSave(True, fig, pathPlot, 'Felem_comparison')
# plot_manualComparison()

if __name__ == "__main__":
    from plaxis_00calling_02helpers import (open_plaxisData)
    device = 27
    phase = 10
    paths = []
    for typName in ['elem',
                    'soil']:
        for modelcat in [
                         'medium',
                         'fineMedium',
                         'fine',
                         # '0',
                         # '30000000.0'
                         ]:
            _, _, path = open_plaxisData(device,
                                         'spacer',
                                         os.path.join(
                     'Optimization - 2026-09-26 - Sensitivity Study',
                     f'meshing_{modelcat}_unitcell2',
                     f'results_{typName}',
                     f'results_{typName}_p{phase}'))
            paths.append(path)
            plot_pointsDistribution_comp(path, typName, phase)
        plot_pointsDistribution_comp(paths, typName, phase)
