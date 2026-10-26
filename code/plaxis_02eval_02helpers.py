# -*- coding: utf-8 -*-
'''
Created on Wed Apr  8 12:04:36 2026
@author: annaleoniesalzmann
'''

###############################################################################
#-------------------------------- USER-INPUT ----------------------------------
###############################################################################

dpiDefault        = 900 # most figures
internalTesting   = False # surfIG (i      v1       v2      sig         w)
printDistribution = True # surfIG for plot_pointsDistribution3D
digitsDefault     = 3 # prepareGrid for surfIG (how many digits)
CM_PER_INCH = 2.54

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import numpy as np
import pandas as pd
import math
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import os
import traceback
from scipy.spatial import Delaunay
from plaxis_00calling_02helpers import (fct_setPlotstyle,
                                        fct_formatSave,
                                        fct_mathLabel)

from plaxis_03plot_02helpers import (fct_setPlottext)

###############################################################################
#------------------------------ Helpers PLOTTER -------------------------------
###############################################################################

def fct_axisScaler(ax):
    from matplotlib.ticker import ScalarFormatter
    # x
    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((-3, 3))
    formatter_x.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter_x)
    # y
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-3, 3))
    formatter_y.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter_y)
    return ax

def plot_pointsDistribution2D(x,
                              y,
                              xAxis,
                              yAxis,
                              filler    = True,
                              pathFile  = None,
                              maxInfos  = None,
                              legend    = None):
    # print(f'plot_pointsDistribution2D for\nx:\n{x}\nand\ny:\n{y}')
    # maxInfos = [x, y, angle]
    fct_setPlottext()
    fig, ax = plt.subplots(figsize=(7.5/CM_PER_INCH,
                                    7.5/CM_PER_INCH))
    markers, colors = fct_setPlotstyle(flat=True, r=1)
    multi = (isinstance(x, (list, tuple)) and
             isinstance(y, (list, tuple)))
    # MULTI-LINE HANDLING
    if multi:
        for i, (xi, yi) in enumerate(zip(x, y)):
            xi  = np.array(xi)
            yi  = np.array(yi)
            idx = np.argsort(xi)
            xi  = xi[idx]
            yi  = yi[idx]
            color  = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            label  = legend[i] if legend is not None else None
            ax.plot(xi, yi,
                    marker      = marker[0],
                    markersize  = marker[1],
                    linewidth   = 1.5,
                    color       = color,
                    label       = label)
            if filler:
                if i == 0:
                    for j in range(len(xi) - 1):
                        ax.fill_between(xi[j:j+2],
                                        yi[j:j+2],
                                        color = colors[j % 2],
                                        alpha = 0.25,
                                        )
    # SINGLE-LINE HANDLING
    else:
        x   = np.array(x)
        y   = np.array(y)
        idx = np.argsort(x)
        x   = x[idx]
        y   = y[idx]
        ax.plot(x, y,
                linewidth   = 1.5,
                color       = colors[0],
                marker      = markers[0][0],
                markersize  = markers[0][1]
                )
        if filler:
            for i in range(len(x) - 1):
                ax.fill_between(x[i:i+2],
                                y[i:i+2],
                                color = colors[i%2],
                                alpha = 0.5,
                                )
    if maxInfos is not None:
        if multi:
            x_all = np.concatenate([np.asarray(xi)
                                    for xi in x])
            y_all = np.concatenate([np.asarray(yi)
                                    for yi in y])
            dx = 0.02 * (np.max(x_all) - np.min(x_all))
            dy = 0.02 * (np.max(y_all) - np.min(y_all))
        else:
            dx = 0.02 * (np.max(x) - np.min(x))
            dy = 0.02 * (np.max(y) - np.min(y))
        if isinstance(maxInfos, (tuple)):
            maxInfos = [maxInfos]
        for x_max, y_max, value in maxInfos:
            ax.scatter(x_max,
                       y_max,
                       color    = 'red',
                       s        = 40,
                       zorder   = 5,
                       )
            ax.annotate(value,
                        xy      = (x_max, y_max),
                        xytext  = (x_max + dx, y_max + dy),
                        color   = 'black',
                        arrowprops = dict(arrowstyle ='->',
                                          color      ='red',
                                          lw         =1,
                                          ),
                        bbox = dict(boxstyle         = 'round,pad=0.2',
                                    fc               = 'white',
                                    ec               = 'red',
                                    alpha            = 0.7,
                                    )
                        )
    ax = fct_axisScaler(ax)
    ax.set_xlabel(xAxis)
    ax.set_ylabel(yAxis)
    xAxis = xAxis.split(' ')[0]
    yAxis = yAxis.split(' ')[0]
    ax.set_title(f'{fct_mathLabel(yAxis)} over {xAxis}')
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.22)
    if pathFile is None:
        plt.show()
        file_path = None
    else:
        file_path = fct_formatSave(save = True,
                                   fig  = fig,
                                   path = os.path.dirname(pathFile),
                                   name = os.path.basename(pathFile),
                                   )
    return file_path

def plot_pointsDistribution3D(points,
                              values,
                              typ               = 'weight',
                              area              = 'Circle',
                              pile_x_c          = 0,
                              pile_y_c          = 0,
                              pile_r            = 1, # Unit Circle
                              v1_min            = 0,
                              v1_max            = 2*math.pi,
                              v2_min            = -12,
                              v2_max            = 0,
                              tri               = None,
                              contour           = False,
                              pathSave          = None,
                              valuesTotal       = None,
                              valuesTotalprint  = False,
                              dpi               = dpiDefault,
                              vmin              = None,
                              vmax              = None,
                              ):
    fct_setPlottext()
    n_points = max(len(points), 1)
    point_size = float(np.clip(15 * 100 / n_points, 2, 40))
    cross_size = 2 * point_size
    cross_linewidth = float(np.clip(np.sqrt(point_size / 15), 0.3, 1.5))
    # Scatter plot with color mapping
    fig, ax = plt.subplots(figsize=(7.5/CM_PER_INCH,
                                    6/CM_PER_INCH)
                           )
    _, colorMap = fct_setPlotstyle(flat     = True,
                                   r        = 1,
                                   colorMap = True,
                                   )
    if contour:
        if tri is not None: cont = tri
        else: cont = spatial.Delaunay(points)
        ax.tricontourf(points[:,0],
                       points[:,1],
                       cont.simplices,
                       values,
                       levels   = np.linspace(vmin
                                              if vmin is not None
                                              else np.min(values),
                                              vmax
                                              if vmax is not None
                                              else np.max(values),
                                              50
                                              ),
                       extend   = 'both',
                       cmap     = colorMap,
                       alpha    = 0.7,
                       zorder   = 1,
                       )
        sc = ax.scatter(points[:,0],
                        points[:,1],
                        c       = values,
                        cmap    = colorMap,
                        s       = point_size,
                        zorder  = 1,
                        vmin    = vmin,
                        vmax    = vmax,
                        )
        ax.scatter(points[:,0],
                   points[:,1],
                   c            = 'k',
                   marker       = 'x',
                   s            = cross_size,
                   linewidths   = cross_linewidth,
                   zorder       = 4,
                   )
    else:
        sc = ax.scatter(points[:,0],
                        points[:,1],
                        c       = values,
                        cmap    = colorMap,
                        s       = point_size,
                        zorder  = 4,
                        vmin    = vmin,
                        vmax    = vmax,
                        )
    if tri is not None:
        ax.triplot(points[:,0],
                   points[:,1],
                   tri.simplices,
                   color    = 'gray',
                   alpha    = 0.5,
                   zorder   = 3,
                   )
    # Label points
    if len(points) <= 30: # else to much information, not readable
        shift = 0.03
        for i, (x, y) in enumerate(points, start=1):
            ax.text(x+shift,
                    y+shift,
                    str(i),
                    fontsize    = 8,
                    ha          = 'left',
                    va          = 'center')
    if area == 'Circle':
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(pile_x_c + pile_r*np.cos(theta),
                pile_y_c + pile_r*np.sin(theta),
                'k--',
                label  = 'Pile Boundary',
                zorder = 2)
    elif (area == 'Rectangle' or
          area == 'Shaft'):
        # rectangle boundary (closed loop)
        ax.plot([v1_min, v1_max, v1_max, v1_min, v1_min],
                 [v2_min, v2_min, v2_max, v2_max, v2_min],
                 'k--',
                 label  = 'Boundary',
                 zorder = 2)
    # Colorbar
    from matplotlib.ticker import ScalarFormatter
    cbar = fig.colorbar(sc,
                        shrink = 0.8,
                        extend = 'both')
    if typ == 's_n':
        unitBar = ' [kN/m²]'
        typMath = r"$\sigma'_n$"
    elif typ in ('sig_zz', 'sigma'):
        unitBar = ' [kN/m²]'
        typMath = r"$\sigma'_{zz}$"
    elif typ == 'ref. weight':
        unitBar = ' [-]'
        typMath = r'$w_\mathrm{ref}$'
    elif typ == 'weight':
        unitBar = ' [m²]'
        typMath = r'$w$'
    else:
        unitBar = ''
        typMath = typ
    cbar.set_label(f'{typMath}{unitBar}')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_ha('left')
    offset_text.set_x(0.0)
    # Formatting
    ax.set_aspect('equal', adjustable='box')
    if area == 'Circle':
        is_unit = (pile_x_c == 0 and pile_y_c == 0 and pile_r == 1)
        unit = 'unitcircle'                    if is_unit else 'pile'
        xlabel, ylabel = (r'$\xi$', r'$\eta$') if is_unit else ('x', 'y')
    elif area == 'Rectangle':
        unit = area
        xlabel, ylabel = 'x', 'y'
    elif area == 'Shaft':
        unit = area
        xlabel, ylabel = r'$\theta$', 'z'
    else: raise ValueError(f'Area Typ {area} not known. '
                           'Use Circle, Rectangle or Shaft.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if (('weight' in typ) or ('area' in typ)):
        summation = np.sum(values)
        ax.set_title(f'Cubature Points\nSum of {typ}s = {summation:.4f}')
    else:
        ax.set_title('Cubature Points')
    if (valuesTotalprint is True and
        valuesTotal is not None):
        if isinstance(valuesTotal, (list, tuple, np.ndarray)):
            if len(valuesTotal) == 2:
                text = (f'Triangle total area = {valuesTotal[0]:.4f}\n'
                        f'Expected total area = {valuesTotal[1]:.4f}')
            else:
                text = '\n'.join([f'Value {i+1} = {v:.4f}'
                                  for i, v in enumerate(valuesTotal)])
        else:
            text = f'Expected total area = {valuesTotal:.4f}'
        fig.text(0.98, 0.98, text,
                 ha='right', va='top',
                 color='gray')
    # Figure dimensions [cm]
    fig_width, fig_height = fig.get_size_inches() * CM_PER_INCH
    
    # Margins [cm]
    left_cm   = 0.9
    right_cm  = 0.4
    bottom_cm = 0.8
    top_cm    = 0.5
    
    # Convert cm to relative figure coordinates
    fig.subplots_adjust(
        left   = left_cm / fig_width,
        right  = 1 - right_cm / fig_width,
        bottom = bottom_cm / fig_height,
        top    = 1 - top_cm / fig_height
    )
    if pathSave is None:
        plt.show()
        file_path = None
    else:
        filename = (os.path.splitext(os.path.basename(pathSave))[0]
                    + (f"_{unit}" if unit is not None else "")
                    + (f"_{typ}"  if typ  is not None else ""))
        file_path = fct_formatSave(save     = True,
                                   fig      = fig,
                                   path     = os.path.dirname(pathSave),
                                   name     = filename)
    return file_path

def plot_pointsDistribution3D_compare(csv_files,
                                      pile_center,
                                      pile_radius,
                                      typName = 'soil',
                                      typArea = 'Rectangle',
                                      perc = (0, 100),
                                      tol  = 1e-6,
                                      idx  = None,
                                      ):
    xc, yc = pile_center
    datasets = {}
    scale_values = []
    # Read saved results and identify soil stresses
    # Read saved results and identify soil stresses
    for method, csv_file in csv_files.items():
        df = pd.read_csv(csv_file, sep=';')
        df.columns = df.columns.str.strip()
        # Automatically detect stress column
        if 'sig_zz' in df.columns: sig = 'sig_zz'
        elif 's_n' in df.columns: sig = 's_n'
        elif 'sigma' in df.columns: sig = 'sigma'
        else:
            raise ValueError(f"No stress column in {csv_file}. \n"
                             f"Available columns: {list(df.columns)}")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['x', 'y', sig])
        if df.empty: raise ValueError(f'No valid values in {csv_file}')
        datasets[method] = (df, csv_file, sig)
        r = np.sqrt((df['x'] - xc)**2 + (df['y'] - yc)**2)
        if typName.startswith('soil'):
            mask = r > pile_radius + tol
        elif typName.startswith('elem'):
            mask = r < pile_radius + tol
        else:
            raise ValueError(f'Unknown typName: {typName}')
        selected_values = df.loc[mask, sig].to_numpy(dtype=float)
        if len(selected_values) == 0:
            raise ValueError(f'No values for color scale in {csv_file}. '
                             'Check pile_center and pile_radius.')
        scale_values.extend(selected_values)
        datasets[method] = (df, csv_file, sig)
    # Common color scale based on soil stresses only
    vmin, vmax = np.percentile(scale_values, perc)
    if vmin == vmax: raise ValueError('Common color scale has zero range.')
    print(f'Common color scale: {vmin:.2f} to {vmax:.2f} kN/m²')
    # Replot individual methods
    for method, (df, csv_file, sig) in datasets.items():
        # Detect triangulated methods
        is_tri = method.endswith(('if_tri', 'cs_tri'))
        
        # Remove duplicate coordinates before triangulation
        if is_tri:
            df_plot = (df.groupby(['x', 'y'], as_index=False)
                        .agg({sig: 'mean'}))
        else:
            df_plot = df
        points = df_plot[['x', 'y']].to_numpy(dtype=float)
        values = df_plot[sig].to_numpy(dtype=float)
        tri = None
        if is_tri:
            if len(points) >= 3:
                tri = Delaunay(points)
            else:
                print(f'Not enough points for triangulation: {method}')
        base = os.path.splitext(csv_file)[0]
        if idx is None:
            pathSave = base + '_compare.pdf'
        else:
            pathSave = f'{base}_compare_case{idx}.pdf'
        plot_pointsDistribution3D(points,
                                  values,
                                  typ       = sig,
                                  area      = typArea,
                                  v1_min    = df_plot['x'].min(),
                                  v1_max    = df_plot['x'].max(),
                                  v2_min    = df_plot['y'].min(),
                                  v2_max    = df_plot['y'].max(),
                                  pile_x_c          = xc,
                                  pile_y_c          = yc,
                                  pile_r            = pile_radius,
                                  tri       = tri,
                                  pathSave  = pathSave,
                                  vmin      = vmin,
                                  vmax      = vmax
                                  )

def plot_pointsDistribution3D_csv(csv_file,
                                  quantities  = ("s_n",
                                                 "s_1"),
                                  pile_center = None,
                                  ):
    from pathlib import Path
    csv_file = Path(csv_file)
    df = pd.read_csv(csv_file, sep="\t")
    df.columns = df.columns.str.strip()
    if pile_center is None:
        xc = (df["x"].min() + df["x"].max()) / 2
        yc = (df["y"].min() + df["y"].max()) / 2
    else:
        xc, yc = pile_center
    theta = np.mod(np.arctan2(df["y"] - yc,
                              df["x"] - xc),
                   2 * np.pi
                   )
    points = np.column_stack([theta.to_numpy(),
                              df["z"].to_numpy()
                              ])
    for quantity in quantities:
        values = df[quantity].to_numpy(dtype=float)
        pathSave = csv_file.with_name(f"{csv_file.stem}_replot.pdf")
        plot_pointsDistribution3D(points,
                                  values,
                                  typ=quantity,
                                  area="Shaft",
                                  pile_x_c=xc,
                                  pile_y_c=yc,
                                  v1_min=0,
                                  v1_max=2 * np.pi,
                                  v2_min=df["z"].min(),
                                  v2_max=df["z"].max(),
                                  contour=False,
                                  pathSave=str(pathSave),
                                  )

###############################################################################
#---------------------------- Helpers EXCTRACTER ------------------------------
###############################################################################

def fct_getInterfOld(g_o, interf, phas, typ):
    return list(g_o.getresults(interf, phas, typ, 'node', True))

def fct_getInterf(g_o, interf, phas, typ):
    if isinstance(interf, (list, tuple)):
        data = []
        for interf_i in interf:
            data.extend(g_o.getresults(interf_i, phas, typ, 'node', True))
        return data
    return list(g_o.getresults(interf, phas, typ, 'node', True))

def fct_getCS(g_o, plot, phas, typ):
    return list(g_o.getcrosssectionresults(plot, phas, typ, True))

def fct_getSingle(g_o, phas, soil, x, y, z=None):
    var = 1e-3
    diffs = [( 0.0,  0.0,  0.0),
             ( 0.0, -var,  0.0),
             (-var,  0.0,  0.0),
             ( 0.0,  0.0, -var),
             ( 0.0, +var,  0.0),
             (+var,  0.0,  0.0),
             (+var, +var,  0.0),
             (-var, -var,  0.0),
             (+var, -var,  0.0),
             (-var, +var,  0.0),
             ] # more possible
    for xdiff, ydiff, zdiff in diffs[:6]:
        try:
            if z is None: # 2D case
                point = (x+xdiff, y+ydiff)
                ux = g_o.getsingleresult(phas, soil.Ux,     point)
                uy = g_o.getsingleresult(phas, soil.Uy,     point)
                sy = g_o.getsingleresult(phas, soil.SigyyE, point)
                points = [x,y]
                values = [ux, uy, sy]
            else: # 3D case
                point = (x+xdiff, y+ydiff, z+zdiff)
                ux = g_o.getsingleresult(phas, soil.Ux,     point)
                uy = g_o.getsingleresult(phas, soil.Uy,     point)
                uz = g_o.getsingleresult(phas, soil.Uz,     point)
                sz = g_o.getsingleresult(phas, soil.SigzzE, point)
                points = [x,y,z]
                values = [ux, uy, uz, sz]
            if any(v is None for v in values): continue
            values = [float(v) for v in values]
            if not all(np.isfinite(v) for v in values): continue
            return [*points, *values] # x,y,z,Ux,Uy,Uz,SIG
        except Exception:
            continue
    return None

def fct_applyLimits(df, boundaries):
    if boundaries is None:
        df_new = df
    elif len(boundaries) == 3: # circle [3D with constant z]
        x_c, y_c, radius = boundaries
        df_new = df[(df['x'] - x_c) ** 2 +
                    (df['y'] - y_c) ** 2 <= radius ** 2].copy()
    elif len(boundaries) == 4: # rectangular [3D with constant z]
        xmin, xmax, ymin, ymax = boundaries
        df_new = df[df['x'].between(xmin, xmax) & # xmin ≤ x ≤ xmax
                    df['y'].between(ymin, ymax)].copy()
    elif len(boundaries) == 5: # shaft
        x_c, y_c, z_top, z_bottom, radius = boundaries
        # radius
        tol = 1e-2
        r_dist = np.sqrt((df['x'] - x_c)**2 + (df['y'] - y_c)**2)
        df_filtered = df[np.abs(r_dist - radius) <= tol].copy()
        if len(df_filtered) == 0:
            raise ValueError(f'No points found near radius {radius} ± {tol}')
        # height
        df_new = df_filtered[df_filtered['z'].between(z_bottom, z_top)].copy()
    else: raise ValueError("Value 'boundaries' has wrong dimension.")
    return df_new.reset_index(drop=True)

def fct_getShearmantle(g_o, phas, soil, x, y, z, xc, yc):
    diffs = [(  0.0,   0.0,   0.0),
             (  0.0, -1e-6,   0.0),
             (-1e-6,   0.0,   0.0),
             (  0.0,   0.0, -1e-6),
             (  0.0, +1e-6,   0.0),
             (+1e-6,   0.0,   0.0),
             (+1e-6, +1e-6,   0.0),
             (-1e-6, -1e-6,   0.0),
             (+1e-6, -1e-6,   0.0),
             (-1e-6, +1e-6,   0.0),
             ] # more possible
    for xdiff, ydiff, zdiff in diffs[:6]:
        # get stresses
        point = (x+xdiff, y+ydiff, z+zdiff)
        try:
            sxx = g_o.getsingleresult(phas, soil.SigxxE, point)
            syy = g_o.getsingleresult(phas, soil.SigyyE, point)
            szz = g_o.getsingleresult(phas, soil.SigzzE, point)
            sxy = g_o.getsingleresult(phas, soil.Sigxy,  point)
            syz = g_o.getsingleresult(phas, soil.Sigyz,  point)
            szx = g_o.getsingleresult(phas, soil.Sigzx,  point)
            values = [sxx, syy, szz, sxy, syz, szx]
            if any(v is None for v in values):
                continue
            if not all(np.isfinite(v) for v in values):
                continue
            break # successfull
        except Exception:
            continue
    else: return None
    sigma = np.array([[sxx, sxy, szx],
                      [sxy, syy, syz],
                      [szx, syz, szz]])
    # theta-angle
    dx = x - xc
    dy = y - yc
    theta = np.arctan2(dy, dx)
    theta = np.mod(theta, 2 * np.pi)
    # normal vector
    n = np.array([np.cos(theta),
                  np.sin(theta),
                  0.0])
    # traction
    t = sigma @ n
    # normal stress
    sigma_n = np.dot(t, n)
    tau_vec = t - sigma_n * n
    tau = np.hypot(np.hypot(tau_vec[0],
                            tau_vec[1]),
                   tau_vec[2])
    # could also be tau = np.linalg.norm(tau_vec)
    return [x, y, z, tau]

###############################################################################
#------------------------------------ Main ------------------------------------
###############################################################################

# Classify interface to get correct stress points also on the boundaries
def fct_classify_interface(g_o, phase, interfaces, center, radius,
                           testing = False):
    # pillar_center = np.array([xc, yc]) or xc
    # pillar_radius = r
    if testing: print(f'\ninput: {interfaces}')
    try:
        is_3D = np.ndim(center) > 0 and len(np.asarray(center)) > 1
        interf_elem = []
        interf_soil = []
        for interf in interfaces:
            if testing: print(f'input for fct_classify_interface is {interf}')
            try:
                x = np.asarray(fct_getInterf(g_o, interf,
                                             g_o.Phases[int(phase)],
                                             g_o.ResultTypes.Interface.X))
                x = x.astype(float)
                if is_3D:
                    y = np.asarray(fct_getInterf(g_o, interf,
                                                 g_o.Phases[int(phase)],
                                                 g_o.ResultTypes.Interface.Y))
                    y = y.astype(float)
            except:
                if testing:print('fct_classify_interface not possible due to STRING')
                traceback.print_exc()
                return None, None
            if is_3D:
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            else: dist = np.abs(x - center) # 2d Unitcell
            if testing: print(interf.Name.value)
            if testing: print(dist)
            if np.all(dist <= radius + 1e-3): # within pile
                interf_elem.append(interf)
            else: # outside pile
                interf_soil.append(interf)
        if testing:
            # necessary for correct input in 'fct_getInterf':
            # print("before conversion:")ea
            # print(type(interf_elem), len(interf_elem))
            # print(type(interf_soil), len(interf_soil))
            print(f'interf_elem is {interf_elem}')
            for interf in interf_elem: interf.Name.value
            print(f'interf_soil is {interf_soil}')
            for interf in interf_soil: interf.Name.value
        if len(interf_elem) == 1: interf_elem = interf_elem[0]
        if len(interf_soil) == 1: interf_soil = interf_soil[0]
        return interf_elem, interf_soil
    except:
        print('fct_classify_interface not possible')
        traceback.print_exc()
        return None, None

# Stress along a defined interface [2D]
def fct_data2D_interface(g_o, phase, interf):
    inte = g_o.ResultTypes.Interface
    phas = g_o.Phases[int(phase)]
    data = [fct_getInterf(g_o, interf, phas, inte.X),
            fct_getInterf(g_o, interf, phas, inte.Y),
            fct_getInterf(g_o, interf, phas, inte.Ux),
            fct_getInterf(g_o, interf, phas, inte.Uy),
            fct_getInterf(g_o, interf, phas,
                          inte.InterfaceEffectiveNormalStress),
            fct_getInterf(g_o, interf, phas, inte.InterfaceShearStress)]
    cols = ['x', 'y', 'ux', 'uy', 's_n', 's_1']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    return df.reset_index(drop=True)

# Stress along a defined interface [3D]
def fct_data3D_interface(g_o, phase, interf, boundaries=None):
    inte = g_o.ResultTypes.Interface
    phas = g_o.Phases[int(phase)]
    data = [fct_getInterf(g_o, interf, phas, inte.X),
            fct_getInterf(g_o, interf, phas, inte.Y),
            fct_getInterf(g_o, interf, phas, inte.Z),
            fct_getInterf(g_o, interf, phas, inte.Ux),
            fct_getInterf(g_o, interf, phas, inte.Uy),
            fct_getInterf(g_o, interf, phas, inte.Uz),
            fct_getInterf(g_o, interf, phas,
                          inte.InterfaceEffectiveNormalStress),
            fct_getInterf(g_o, interf, phas, inte.InterfaceShearStress)]
    cols = ['x', 'y', 'z', 'ux', 'uy', 'uz', 's_n', 's_1']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    df   = fct_applyLimits(df, boundaries)
    return df.reset_index(drop=True)

# Stress and deformation between two points (line) [2D]
def fct_data2D_lineCS(g_o, phase, start, end):
    plot = g_o.linecrosssectionplot(g_o.Plots[-1], start, end)
    soil = g_o.ResultTypes.Soil
    phas = g_o.Phases[int(phase)]
    # values due to intern coordinate system
    data = [fct_getCS(g_o, plot, phas, soil.X),
            fct_getCS(g_o, plot, phas, soil.Y),
            fct_getCS(g_o, plot, phas, soil.Ux),
            fct_getCS(g_o, plot, phas, soil.Uy),
            fct_getCS(g_o, plot, phas, soil.SigyyE),
            fct_getCS(g_o, plot, phas, soil.SigxxE),
            fct_getCS(g_o, plot, phas, soil.Sigxy)]
    cols = ['x', 'y', 'ux', 'uy', 'sig_yy', 'sig_xx', 'sig_xy']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    g_o.delete(plot)
    return df.reset_index(drop=True)

# Stress and deformation for a surface/polygon [3D]
def fct_data3D_horiCS(g_o, phase, z, boundaries=None):
    plot = g_o.crosssectionplot(g_o.Plots[-1], 0, 0, z,
                                               1, 0, z,
                                               0, 1, z)
    soil = g_o.ResultTypes.Soil
    phas = g_o.Phases[int(phase)]
    # g_o.Plots[-1].export(r"C:\Plaxis\Plots\my_plot.png", 1920, 1080)
    # g_o.Plots[-1].DrawRulers = False
    # g_o.Plots[-1].DrawAxes= False
    # g_o.Plots[-1].DrawLegend = True
    # g_o.Plots[-1].PlotType = "ContourLines"
    data = [fct_getCS(g_o, plot, phas, soil.X),
            fct_getCS(g_o, plot, phas, soil.Y),
            fct_getCS(g_o, plot, phas, soil.Z),
            fct_getCS(g_o, plot, phas, soil.Uz),
            fct_getCS(g_o, plot, phas, soil.SigzzE)]
    cols = ['x', 'y', 'z','uz', 'sig_zz']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    df   = fct_applyLimits(df, boundaries)
    g_o.delete(plot)
    return df.reset_index(drop=True)

# Stress and deformation for defined points [3D]
def fct_data3D_gridIP(g_o, phase, nx, ny, height, boundaries):
    soil = g_o.ResultTypes.Soil
    phas = g_o.Phases[int(phase)]
    data = []
    if len(boundaries) == 4: # rectangle
        xmin, xmax, ymin, ymax = boundaries
        x_points = np.linspace(xmin, xmax, nx)
        y_points = np.linspace(ymin, ymax, ny)
        z = height
        for x in x_points:
            for y in y_points:
                result = fct_getSingle(g_o, phas, soil, x, y, z)
                if result: data.append(result) # 'None' not considered
        cols = ['x', 'y', 'z','ux', 'uy', 'uz', 'sig_zz']
    elif len(boundaries) == 3: # circle
        xc, yc, radius = boundaries # for clean grid
        r_points = np.linspace(0.0001, 0.9999*radius, nx)
        if ny % 2 == 0: ntheta = ny
        else:           ntheta = ny - 1
        if (ntheta % 8) != 0: # has to be 8,16,24,32
            ntheta = int(math.ceil(ntheta/8) * 8)
        # print(f'ntheta: {ntheta}')
        theta_points = np.linspace(0,2*math.pi,ntheta+1)
        # print(f'Radius:\n{r_points}')
        # print(f'Angle: \n{theta_points}')
        # print(f'Height:\n{height}')
        for r in r_points:
            for theta in theta_points:
                x = xc - r * np.cos(theta)
                y = yc - r * np.sin(theta)
                z = height
                result = fct_getSingle(g_o, phas, soil, x, y, z)
                if result: data.append(result) # 'None' not considered
        cols = ['x', 'y', 'z','ux', 'uy', 'uz', 'sig_zz']
    elif len(boundaries) == 5: # shaft = [x, y, top, bottom, radius]
        xc, yc, z_top, z_bottom, radius = boundaries
        # print(f'Z1: {z_bottom}')
        # print(f'Z2: {z_top}')
        z_points      = np.linspace(z_bottom, z_top, nx)
        if ny % 2 == 0: ntheta = ny
        else:           ntheta = ny - 1 # to eliminate overlap
        if (ntheta % 4) != 0:
            ntheta = int(math.ceil(ntheta/4) * 4)
        theta_points  = np.linspace(0, 2*math.pi, ntheta+1)
        radius = 0.999*np.round(radius, 3) # transf. 0.3500000000000014
        # print(f'Radius:\n{radius}')
        # print(f'Angle: \n{theta_points}')
        # print(f'Height:\n{z_points}')
        for z in z_points:
            for theta in theta_points:
                x = xc - radius * np.cos(theta)
                y = yc - radius * np.sin(theta)
                result = fct_getShearmantle(g_o, phas, soil,
                                            x, y, z,
                                            xc, yc)
                if result: data.append(result) # 'None' not considered
        cols = ['x', 'y', 'z','shear']
    else:  raise ValueError("Value 'boundaries' has wrong dimension.")
    df   = pd.DataFrame(data, columns=cols)
    # print(f'Results:\n{df}')
    return df.reset_index(drop=True)

# Stress and deformation for defined line [2D]
def fct_data2D_lineIP(g_o, phase, coords):
    # Activate phase view
    phas = g_o.Phases[int(phase)]
    soil = g_o.ResultTypes.Soil
    data = []
    # Detect variable axis
    var_axes = [typ for typ, value in coords.items()
                if (hasattr(value, '__iter__') and
                    not isinstance(value, (str, bytes)))]
    if len(var_axes) != 1:
        raise ValueError('One coordinate must be iterable.')
    var_axis = var_axes[0]
    # Loop over variable coordinate
    for val in coords[var_axis]:
        c = {**coords, var_axis: val}
        x, y = c['x'], c['y']
        # Get 'filtered' results from PLAXIS
        result = fct_getSingle(g_o, phas, soil, x, y)
        if result: data.append(result) # 'None' not considered
    cols = ['x', 'y', 'ux', 'uy', 'sig_yy']
    df   = pd.DataFrame(data, columns=cols)
    return df.reset_index(drop=True)

def fct_calc2D_tilt(g_o, phase, coords, pathSave=None):
    import time
    t0 = time.time()
    # read results
    if isinstance(coords, (dict)): # regular grid
        results = fct_data2D_lineIP(g_o,
                                    int(phase),
                                    coords)
    elif isinstance(coords, (list)): # crosssection
        results = fct_data2D_lineCS(g_o,
                                    int(phase),
                                    coords[0],
                                    coords[1])
    else: # interface
        results = fct_data2D_interface(g_o,
                                       int(phase),
                                       coords)
    if len(results) < 2: return results, None # needs 2 values for diff.
    results = results.sort_values(by='x').reset_index(drop=True)
    x = results["x"].to_numpy()
    u = results["uy"].to_numpy()
    # Settlement
    s_max          = np.max(u)
    s_min          = np.min(u)
    s_max_idx      = np.argmax(u)
    s_min_idx      = np.argmin(u)
    settlementDiff = s_max - s_min
    settlement     = max(abs(s_max), abs(s_min))
    # Overall tilt
    dx_total = x[-1] - x[0]
    du_total = u[-1] - u[0]
    tilt     = abs(du_total / dx_total) * 100
    # deflection
    u_chord = u[0] + (du_total / dx_total) * (x - x[0])
    relativeDeflection = np.max(np.abs(u - u_chord))  # [m]
    span = dx_total
    deflectionRatio = relativeDeflection / span # [-]
    # Local tilt
    dx_diff         = np.diff(x)
    du_diff         = np.diff(u)
    tiltDiff = np.divide(np.abs(du_diff), dx_diff,
                         out=np.zeros_like(du_diff),
                         where=dx_diff > 0) * 100
    tiltDiffMax     = np.max(tiltDiff)
    tiltDiffMax_idx = np.argmax(tiltDiff)
    # Distortion
    distortion = max(abs(tiltDiff[0] - tilt),
                     abs(tiltDiff[-1] - tilt))
    if pathSave is not None:
        os.makedirs(os.path.dirname(pathSave),
                    exist_ok=True)
        results.to_csv(pathSave.replace(".pdf", ".csv"),
                       index=False,
                       sep=";")
        maxInfos = [(x[0],  u[0],
                     f"tilt = {tiltDiff[0]:.2f} %"),
                    (x[-1], u[-1],
                     f"tilt = {tiltDiff[-1]:.2f} %"),
                    (x[tiltDiffMax_idx], u[tiltDiffMax_idx],
                     f"max. tilt = {tiltDiff[tiltDiffMax_idx]:.2f} %"),
                    (x[s_max_idx], u[s_max_idx],
                     f"max. settlement = {s_max*1000:.0f} mm"),
                    (x[s_min_idx], u[s_min_idx],
                     f"min. settlement = {s_min*1000:.0f} mm"),
                    ]
        plot_pointsDistribution2D(results['x'],
                                  results['uy'],
                                  'distance x [m]',
                                  'deformation uy [m]',
                                  filler = False,
                                  pathFile=pathSave,
                                  maxInfos=maxInfos)
    return (settlement,
            settlementDiff,
            tilt,
            tiltDiffMax,
            distortion,
            relativeDeflection,
            deflectionRatio,
            time.time() - t0)

def fct_calc_globalTilt(x, y, u):
    """
    Calculate global tilt from characteristic lines of a rectangular area.
    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    # ----------------------------------------------------------
    # Function to get closest available point
    # because Plaxis points may not lie exactly on
    # xmin/xmax/ymin/ymax
    # ----------------------------------------------------------
    def get_point(x_target, y_target):
        idx = np.argmin((x-x_target)**2 + (y-y_target)**2)
        return idx
    # ----------------------------------------------------------
    # Characteristic points
    # ----------------------------------------------------------
    P_xmin_ymin = get_point(xmin, ymin)
    P_xmax_ymin = get_point(xmax, ymin)
    P_xmin_ymax = get_point(xmin, ymax)
    P_xmax_ymax = get_point(xmax, ymax)
    # Middle points
    P_xmid_ymin = get_point((xmin+xmax)/2,ymin)
    P_xmid_ymax = get_point((xmin+xmax)/2,ymax)
    P_xmin_ymid = get_point(xmin, (ymin+ymax)/2)
    P_xmax_ymid = get_point(xmax,(ymin+ymax)/2)
    # ----------------------------------------------------------
    # Define investigated lines
    # ----------------------------------------------------------
    lines = [(P_xmin_ymin, P_xmax_ymax), # diagonals
             (P_xmax_ymin, P_xmin_ymax),
             (P_xmin_ymin, P_xmax_ymin), # horizontal boundaries
             (P_xmin_ymax, P_xmax_ymax),
             (P_xmin_ymin, P_xmin_ymax), # vertical boundaries
             (P_xmax_ymin, P_xmax_ymax),
             (P_xmin_ymid, P_xmax_ymid), # centre horizontal
             (P_xmid_ymin, P_xmid_ymax), # centre vertical
             ]
    # ----------------------------------------------------------
    # Calculate tilt for each line
    # ----------------------------------------------------------
    tilt_values = []
    for i,j in lines:
        dx = x[j]-x[i]
        dy = y[j]-y[i]
        distance = np.sqrt(dx**2 + dy**2)
        if distance == 0:
            continue
        du = u[j]-u[i]
        tilt_local = abs(du / distance)
        tilt_values.append(tilt_local)
    # Maximum characteristic tilt
    tilt = np.max(tilt_values)
    return tilt

def fct_calc3D_tilt(g_o, phase, typ, inputdata, pathSave=None):
    """
    Calculate settlement, tilt and distortion from a 3D Plaxis surface.
    """
    from scipy.spatial import Delaunay
    import time
    t0 = time.time()
    phase = int(phase)
    # read results
    if typ == 'line':
        (x0, y0, z0, x_dis, y_dis, delta, a_extra) = inputdata
        # directions
        angles = [0, 90] # base
        for angleN in np.arange(1, a_extra + 1):
            anglePart = (90 / (a_extra + 1)) * angleN
            angles.extend([anglePart, anglePart+90])
        angles     = np.unique(np.round(angles, 3))
        s_list     = []
        u_list     = []
        settlement     = -np.inf
        settlementDiff = -np.inf
        tilt           = -np.inf
        tiltDiffMax    = -np.inf
        distortion     = -np.inf
        relativeDeflection = -np.inf
        deflectionRatio    = -np.inf
        for angle in angles:
            # print(angle)
            alphaR = np.radians(angle)
            dx     = np.cos(alphaR)
            dy     = np.sin(alphaR)
            eps    = 1e-12
            cos_a  = np.where(np.abs(dx) < eps, eps, dx)
            sin_a  = np.where(np.abs(dy) < eps, eps, dy)
            Lx     = (x_dis/2) / np.abs(cos_a)
            Ly     = (y_dis/2) / np.abs(sin_a)
            L      = min(Lx, Ly)
            n_steps = max(int(np.floor((L) / delta)), 3) * 2
            s_vals  = np.linspace(-L, L, n_steps)
            # print(s_vals)
            u_list_i = []
            s_list_i = []
            for s in s_vals:
                x = x0 + s * np.cos(alphaR)
                y = y0 + s * np.sin(alphaR)
                result = fct_getSingle(g_o,
                                       g_o.Phases[int(phase)],
                                       g_o.ResultTypes.Soil,
                                       x, y, z0)
                if result is not None:
                    resultU = round(result[5],5)
                    u_list_i.append(resultU)
                    s_list_i.append(s)
                else: print(f'Not found ({x},{y})')
            if len(u_list_i)<1:
                print(f'No results for {angle}.')
                continue
            s_list.append(s_list_i)
            u_list.append(u_list_i)
            # print(s_list_i)
            # print(u_list_i)
            # Settlement
            s_maxA          = np.max(u_list_i)
            s_minA          = np.min(u_list_i)
            s_max_idx       = np.argmax(u_list_i)
            s_min_idx       = np.argmin(u_list_i)
            settlementDiffA = s_maxA - s_minA
            settlementA     = max(abs(s_maxA), abs(s_minA))
            # print(settlementDiffA)
            # print(settlementA)
            if abs(settlementDiffA)>settlementDiff:
                settlementDiff = abs(settlementDiffA)
                s_max=s_maxA
                x_s_max=s_list_i[s_max_idx]
                u_s_max=u_list_i[s_max_idx]
                s_min=s_minA
                x_s_min=s_list_i[s_min_idx]
                u_s_min=u_list_i[s_min_idx]
            if abs(settlementA)>settlement:
                settlement = abs(settlementA)
            # Overall tilt
            dx_total = s_list_i[-1] - s_list_i[0]
            du_total = u_list_i[-1] - u_list_i[0]
            tiltA    = abs(du_total / dx_total) * 100
            # print(dx_total)
            # print(du_total)
            # print(tiltA)
            if abs(tiltA)>tilt:
                tilt = abs(tiltA)
            # Local tilt
            dx_diff         = np.diff(s_list_i)
            du_diff         = np.diff(u_list_i)
            tiltDiffA       = np.divide(np.abs(du_diff), dx_diff,
                                        out=np.zeros_like(du_diff),
                                        where=dx_diff > 0) * 100
            tiltDiffMaxA    = np.max(tiltDiffA)
            tiltDiffMax_idx = np.argmax(tiltDiffA)
            # print(tiltDiffMaxA)
            if abs(tiltDiffMaxA)>tiltDiffMax:
                tiltDiffMax = abs(tiltDiffMaxA)
                x_t_max     = s_list_i[tiltDiffMax_idx]
                u_t_max     = u_list_i[tiltDiffMax_idx]
            # Distortion
            distortionA = max(abs(tiltDiffA[0] - tiltA),
                             abs(tiltDiffA[-1] - tiltA))
            if abs(distortionA)>distortion:
                distortion = abs(distortionA)
            # print(distortionA)
            # Relative deflection
            dx_total = s_list_i[-1] - s_list_i[0]
            du_total = u_list_i[-1] - u_list_i[0]
            if dx_total != 0:
                u_chord = (u_list_i[0]
                           + (du_total / dx_total)
                           * (np.array(s_list_i) - s_list_i[0]))
                relativeDeflectionA = np.max(np.abs(np.array(u_list_i)
                                                    - u_chord)
                                             )
                span = abs(dx_total)
                deflectionRatioA = relativeDeflectionA / span
            
                if relativeDeflectionA > relativeDeflection:
                    relativeDeflection = relativeDeflectionA
                    deflectionRatio = deflectionRatioA
        if pathSave is not None:
            maxInfos = [(x_t_max, u_t_max,
                         f"max. tilt = {tiltDiffMax:.2f} %"),
                        (x_s_max, u_s_max,
                         f"max. settlement = {s_max*1000:.0f} mm"),
                        (x_s_min, u_s_min,
                         f"min. settlement = {s_min*1000:.0f} mm"),
                        ]
            plot_pointsDistribution2D(x        = s_list,
                                      y        = u_list,
                                      xAxis    = "distance to SI center [m]",
                                      yAxis    = "deformation u [m]",
                                      filler   = False,
                                      maxInfos = maxInfos,
                                      pathFile = pathSave,
                                      legend   = angles)
        return (settlement,
                settlementDiff,
                tilt,
                tiltDiffMax,
                distortion,
                relativeDeflection,
                deflectionRatio,
                time.time() - t0)
    else:
        if typ == 'grid':
            (nx, ny, height, boundaries) = inputdata
            results = fct_data3D_gridIP(g_o, phase, nx, ny, height, boundaries)
        elif typ == 'CS':
            (z, boundaries) = inputdata
            results = fct_data3D_horiCS(g_o, phase, z,
                                        boundaries=boundaries)
        elif typ == 'IF':
            (interf, boundaries) = inputdata
            results = fct_data3D_interface(g_o, phase, interf,
                                           boundaries=boundaries)
        else:                return None, None, None, None, None, None
        if len(results) < 3: return None, None, None, None, None, None
        x = results["x"].to_numpy()
        y = results["y"].to_numpy()
        u = results["uz"].to_numpy()
        settlement = np.max(np.abs(u))
        settlementDiff = (np.max(u) - np.min(u))
        edges = set()
        tri = Delaunay(np.column_stack([x, y]))
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1,3):
                    edge = tuple(sorted([simplex[i],simplex[j]]))
                    edges.add(edge)
                    # edges = {(0,1),(0,2),(0,3),(1,3),(2,3)}
        edge_rows = []
        tilt_values = []
        for i,j in edges:
            dx = x[j]-x[i]
            dy = y[j]-y[i]
            du = u[j]-u[i]
            distance = np.sqrt(dx**2 + dy**2)
            if distance == 0: continue
            tilt_local = abs(du/distance)
            tilt_values.append(tilt_local)
            edge_rows.append([x[i], y[i], u[i],
                              x[j], y[j], u[j],
                              distance, du, tilt_local])
        tiltDiffMax = np.max(tilt_values)
        tilt = fct_calc_globalTilt(x, y, u)
        distortion = np.max(np.abs(tilt_values - tilt))
        # deflection
        dx_total = x[-1] - x[0]
        dy_total = y[-1] - y[0]
        dl_total = np.sqrt(dx_total**2 + dy_total**2)
        du_total = u[-1] - u[0]
        u_chord = u[0] + (du_total / dl_total) * (x - x[0])
        relativeDeflection = np.max(np.abs(u - u_chord))  # [m]
        span = dx_total
        deflectionRatio = relativeDeflection / span # [-]
        if pathSave is not None:
            os.makedirs(os.path.dirname(pathSave),
                        exist_ok=True)
            df_edges = pd.DataFrame(edge_rows,
                                    columns=["x1", "y1", "u1",
                                             "x2", "y2", "u2",
                                             "distance", "du",
                                             "tilt_local"])
            df_edges.to_csv(pathSave.replace(".pdf", ".csv"),
                            sep=";", index=False)
            plot_pointsDistribution3D(np.column_stack([x, y]),
                                      u,
                                      typ='deformation u [m]',
                                      area = 'Rectangle',
                                      v1_min=boundaries[0],
                                      v1_max=boundaries[1],
                                      v2_min=boundaries[2],
                                      v2_max=boundaries[3],
                                      tri=tri,
                                      pathSave=pathSave)
        return (settlement,
                settlementDiff,
                tilt,
                tiltDiffMax,
                distortion,
                relativeDeflection,
                deflectionRatio,
                time.time() - t0)

# Stress and deformation for defined line [3D]
def fct_data3D_lineIP(g_o, phase, coords):
    # Activate phase view
    phas = g_o.Phases[int(phase)]
    soil = g_o.ResultTypes.Soil
    data = []
    # Detect variable axis
    var_axes = [typ for typ, value in coords.items()
                if (hasattr(value, '__iter__') and
                    not isinstance(value, (str, bytes)))]
    if len(var_axes) != 1:
        raise ValueError('One coordinate must be iterable.')
    var_axis = var_axes[0]
    # Loop over variable coordinate
    for val in coords[var_axis]:
        c = {**coords, var_axis: val}
        x, y, z = c['x'], c['y'], c['z']
        # Get 'filtered' results from PLAXIS
        result = fct_getSingle(g_o, phas, soil, x, y, z)
        if result: data.append(result) # 'None' not considered
    cols = ['x', 'y', 'z','ux', 'uy', 'uz', 'sig_zz']
    df   = pd.DataFrame(data, columns=cols)
    return df.reset_index(drop=True)

# results = fct_data3D_lineIP(g_o, int(phase),
#                             {'x': x_values, 'y': si_y, 'z': si_top})

# Forces along a line [2D]
# 'https://bentleysystems.service-now.com/community?id=
#  community_question&sys_id=5d05b43647194a909091861f536d4386'
def fct_data2D_InternalForce(g_o, phase,
                             material=None, soil=None, line=None):
    phas = g_o.Phases[int(phase)]
    plot = g_o.structuralforcesinvolumeplatesplot(g_o.Plots[-1])
    if soil is not None:
        center_cfg = g_o.centerlineconfig(material, soil)
    elif material is not None:
        center_cfg = g_o.centerlineconfig(material)
    else: raise ValueError('Case without material not defined.')
    if line is None:                       # symmetry configuration
        center_cfg.xmin_symmetry = True    # (defaults to False)
    g_o.generate(phas, center_cfg)         # generates centerlines
    if line is not None:
        g_o.centerline(line[0], line[1])   # specifies the geometry
    CL = g_o.CenterLines[-1]
    g_o.add(g_o.Plots[-1], CL)             # adds configuration to plot
    data = [g_o.getresults(CL, phas, typ)
            for typ in (g_o.ResultTypes.CenterLine.X,
                        g_o.ResultTypes.CenterLine.Y,
                        g_o.ResultTypes.CenterLine.Nx2D,
                        g_o.ResultTypes.CenterLine.Q2D,
                        g_o.ResultTypes.CenterLine.M2D)]
    g_o.delete(plot)
    cols = ['x','y','N','Q','M']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    return df.reset_index(drop=True)

# Forces along a line [3D]
def fct_data3D_InternalForce(g_o, phase, material=None, soil=None, line=None):
    phas = g_o.Phases[int(phase)]
    plot = g_o.structuralforcesinvolumepilesplot(g_o.Plots[-1])
    if soil is not None:
        center_cfg = g_o.centerlineconfig(material, soil)
    elif material is not None:
        center_cfg = g_o.centerlineconfig(material)
    else: raise ValueError('Case without material not defined.')
    if line is None:                       # symmetry configuration
        center_cfg.xmax_symmetry = True    # (defaults to False)
        center_cfg.ymax_symmetry = True
    g_o.generate(phas, center_cfg)
    if line is not None:                   # line = [[x,y,z],[x,y,z]]
        g_o.centerline(line[0], line[1])
    CL = g_o.CenterLines[-1]               # generates centerlines
    g_o.add(plot, CL)                      # adds configuration to plot
    data = [g_o.getresults(CL, phas, typ)
            for typ in (g_o.ResultTypes.CenterLine.X,
                        g_o.ResultTypes.CenterLine.Y,
                        g_o.ResultTypes.CenterLine.Z,
                        g_o.ResultTypes.CenterLine.N,
                        g_o.ResultTypes.CenterLine.Q13,
                        g_o.ResultTypes.CenterLine.Q23,
                        g_o.ResultTypes.CenterLine.M2,
                        g_o.ResultTypes.CenterLine.M3)]
    g_o.delete(plot)
    cols = ['x','y','z','N','Q13','Q23','M2','M3']
    df   = pd.DataFrame(np.column_stack(data), columns=cols)
    return df.reset_index(drop=True)

# Resistance of the pile SHAFT [2D]
def fct_calc2D_shaftForce(df, top=np.inf, depth=-np.inf, pathSave=None):
    # determine stress column for different datatypes
    if   's_1'    in df.columns: sig = 's_1'
    elif 'sig_xy' in df.columns: sig = 'sig_xy'
    elif 'sig_xx' in df.columns: sig = 'sig_xx'
    elif 'sig_yy' in df.columns: sig = 'sig_yy'
    else: raise ValueError('Dataset missing required stress column.')
    # calculate total force
    df      = df[df['y'] >= depth].copy()
    df      = df[df['y'] <= top].copy()
    integr  = np.trapz(df[sig], x = df['y'])
    radius  = df['x'].mean()
    area    = 2 * math.pi * radius
    F_total = integr * area
    if pathSave is not None:
        df.sort_values(by='x')
        plot_pointsDistribution2D(df['y'], df[sig],
                                  'depth [m]',
                                  r'stress $\sigma$ [kPa]',
                                  pathSave)
    return F_total

# Resistance of the pile BASE/SOIL/ELEM [2D]
def fct_calc2D_circleForce(df, radius=np.inf, pathSave=None):
    # determine stress column for different datatypes
    if 'sig_yy' in df.columns: sig = 'sig_yy'
    elif 's_n'  in df.columns: sig = 's_n'
    else: raise ValueError('Dataset missing required stress column.')
    # calculate total force
    # print(df['x'])
    # print(radius)
    df          = df[df['x'] <= radius].copy()
    df['A_i']   = math.pi * (df['x'].shift(-1)**2 - df['x']**2) # Trapezoidal
    df['sig_i'] = 0.5 * (df[sig].shift(-1) + df[sig])
    df['F_i']   = df['A_i'] * df['sig_i']
    F_total     = df['F_i'].sum()
    if pathSave is not None:
        df.sort_values(by='x')
        plot_pointsDistribution2D(df['x'], df[sig],
                                  'distance [m]',
                                  r'stress $\sigma$ [kPa]',
                                  pathSave)
    return F_total

# Settlement Data for certain curve point
def fct_data3D_settlement(g_o, phase, number):
    uz = list(g_o.getcurveresultspath(g_o.CurvePoints[number],
                                      g_o.Phases[int(phase)],
                                      g_o.Phases[int(phase)],
                                      g_o.ResultTypes.Soil.Uz))
    uz.insert(0,0)
    m_stage = [0]
    for step in g_o.Phases[int(phase)].Steps:
        m_stage.append(step.Reached.SumMstage.value)
    return pd.DataFrame(np.column_stack([uz, m_stage]),
                        columns = ['uz', 'Mstage'])

def fct_calc3D_surfIG(df, sig, method='simple',
                      boundaries=None, area=None, hole=None,
                      pathSave=None,
                      internalTesting=internalTesting,
                      printDistribution=printDistribution):
    # Filter and transform data
    # print(f'Results:\n{df}')
    if sig not in df.columns: raise ValueError(f"'{sig}' missing")
    df  = fct_applyLimits(df, boundaries)
    x   = df['x'].values
    y   = df['y'].values
    z   = df['z'].values if 'z' in df.columns else np.zeros_like(x)
    sig = df[sig].values
    # Compute area if not provided
    if area is None:
        if (boundaries is not None and
            len(boundaries) == 3): # circle
            area = math.pi * boundaries[2]**2
        else:  # rectangle or default
            area = (x.max() - x.min()) * (y.max() - y.min())
    # Simple integration
    if method == 'simple':
        sigma = np.mean(sig)
        F_z   = sigma * area
    # Triangular / Delaunay
    elif method in ['triangularRectangle',
                    'triangularCircle',
                    'triangularShaft']:
        # transform data
        if method in ['triangularShaft']:
            if boundaries is not None:
                if len(boundaries) == 5: # shaft
                    si_x, si_y, si_top, si_bottom, si_radius = boundaries
                    area_factor = si_radius # since 2pi * r = area
                    # --- angle computation ---
                    dx = x - si_x
                    dy = y - si_y
                    theta = np.arctan2(dy, dx)
                    # delete shift in the dataset
                    theta = fct_recalcTheta(theta)
                    # add points for full circle
                    (z, theta, sig) = fct_closeCircle(z, theta, sig)
                    idx   = np.argsort(theta)
                    theta = theta[idx]
                    z     = z[idx]
                    sig   = sig[idx]
                    # final point cloud
                    points = np.column_stack((theta, z))
                else:
                    raise ValueError('5 boundaries are necessary for triShaft')
            else:
                raise ValueError('boundaries are necessary for triShaft')
        else:
            area_factor = 1
            points = np.column_stack((x, y))
        # main calculation
        (F_z, sigma, tri, areas) = fct_calc3D_surfTriangle(points,
                                                           sig,
                                                           area,
                                                           area_factor,
                                                           hole=hole)
        # plot data
        if printDistribution:
            if boundaries is not None:
                if len(boundaries) == 3: # circle
                    xc, yc, radius = boundaries
                    plot_pointsDistribution3D(points, sig, typ='sigma',
                                              area = 'Circle',
                                              pile_x_c=xc, pile_y_c=yc,
                                              pile_r=radius,
                                              tri=tri, pathSave=pathSave,
                                              valuesTotal=[np.sum(areas),
                                                           area])
                elif len(boundaries) == 5: # shaft
                    si_x, si_y, si_top, si_bottom, si_radius = boundaries
                    plot_pointsDistribution3D(points, sig, typ='sigma',
                                              area = 'Shaft',
                                              v1_min=0, v1_max=2*math.pi,
                                              v2_min=si_bottom, v2_max=si_top,
                                              tri=tri, pathSave=pathSave,
                                              valuesTotal=[np.sum(areas),
                                                           area])
            if method in ['triangularRectangle']:
                plot_pointsDistribution3D(points, sig, typ='sigma',
                                          area = 'Rectangle',
                                          v1_min=min(x), v1_max=max(x),
                                          v2_min=min(y), v2_max=max(y),
                                          tri=tri, pathSave=pathSave,
                                          valuesTotal=[np.sum(areas),
                                                       area])
    elif method in ['simpsonRectangle', 'simpsonCircle', 'simpsonShaft']:
        if method == 'simpsonRectangle': # CASE 1: RECTANGULAR (x, y)
            if hole is not None:
                if internalTesting:
                    print('Hole not considered.')
            v1 = x
            v2 = y
            v2o = v2
            radius=None
        else: # method in [simpsonCircle', 'simpsonCylinder']:
            if method == 'simpsonCircle': # CASE 2: CIRCLE (r, theta)
                xc, yc, radius = boundaries
                r_inner = hole[0] if hole is not None else 0.0
                dx = x - xc # distance to the middle of the circle
                dy = y - yc
                v1 = np.sqrt(dx**2 + dy**2) # r
            elif method == 'simpsonShaft': # CASE 3: CYLINDER (z,th)
                xc, yc, z_top, z_bottom, radius = boundaries
                if hole is not None:
                    if internalTesting:
                        print('Hole not considered.')
                v1 = z
                dx = x - xc # distance to the middle of the circle
                dy = y - yc
                ri = np.sqrt(dx**2 + dy**2)
                r  = np.mean(ri) # r
                if abs(r-radius) > 1e-3:
                    raise ValueError('Wrong df.')
            else: raise ValueError('ValueError.')
            theta = np.arctan2(dy, dx)
            v2  = fct_recalcTheta(theta)
            # without theta - theta.min()
            v2o = fct_recalcTheta(theta, old=True)
        # transform data
        (v1, v2, sig, v1_uni, _, _, _, v2o) = fct_prepareGrid(v1,v2,sig, v2o)
        # control data
        (v1, v2, sig, v2o) = fct_calc3D_controlData(v1,v2,sig,v2o)
        # simpson calculation
        (F_z, w1, w2,
         factorWeight, grid) = fct_calc3D_surfSimpson(v1,v2,sig,method,
                                                      radius=radius,
                                                      savePath=pathSave)
        if method == 'simpsonCircle':
            area = (radius**2 - r_inner**2) * v2.max() / 2
        elif method == 'simpsonShaft':
            area = radius * abs(z.max() - z.min()) * v2.max()
        sigma = F_z / area

        # plot data
        if printDistribution:
            if method == 'simpsonRectangle':
                typ = 'Rectangle'
                points = np.column_stack([v1.ravel(), v2.ravel()])
                W = np.outer(w1, w2)
                weights = (W * factorWeight).ravel()
                plot_pointsDistribution3D(points, weights, typ='weight',
                                        area=typ,
                                        v1_min=v1.min(), v1_max=v1.max(),
                                        v2_min=v2.min(), v2_max=v2.max(),
                                        pathSave=pathSave,
                                        valuesTotal=area)
                weights = W.ravel()
                plot_pointsDistribution3D(points, weights, typ='ref. weight',
                                        area=typ,
                                        v1_min=v1.min(), v1_max=v1.max(),
                                        v2_min=v2.min(), v2_max=v2.max(),
                                        pathSave=pathSave)
                sigmas = grid.ravel()
                plot_pointsDistribution3D(points, sigmas, typ='sigma',
                                        area=typ,
                                        v1_min=v1.min(), v1_max=v1.max(),
                                        v2_min=v2.min(), v2_max=v2.max(),
                                        pathSave=pathSave)
            elif method == 'simpsonCircle':
                typ = 'Circle'
                thetamin = v2o.min() # account for theta - theta.min()
                X = v1 * np.cos(v2+thetamin) + xc
                Y = v1 * np.sin(v2+thetamin) + yc
                points = np.column_stack([X.ravel(), Y.ravel()])
                W = np.outer(w1, w2)
                weights = W.ravel()
                plot_pointsDistribution3D(points, weights, typ='ref. weight',
                                        area=typ,
                                        pile_x_c=xc, pile_y_c=yc, pile_r=radius,
                                        pathSave=pathSave)
                weights = (W * v1_uni[:, None] * factorWeight).ravel() # V1=r_i
                plot_pointsDistribution3D(points, weights, typ='weight',
                                        area=typ,
                                        pile_x_c=xc, pile_y_c=yc, pile_r=radius,
                                        pathSave=pathSave,
                                        valuesTotal=area)
                sigmas = grid.ravel()
                plot_pointsDistribution3D(points, sigmas, typ='sigma',
                                        area=typ,
                                        pile_x_c=xc, pile_y_c=yc, pile_r=radius,
                                        pathSave=pathSave)
            elif method == 'simpsonShaft':
                typ = 'Shaft'
                points = np.column_stack([v2.ravel(), v1.ravel()])
                W = np.outer(w1, w2)
                weights = (W * radius * factorWeight).ravel()
                plot_pointsDistribution3D(points, weights, typ='weight',
                                        area=typ,
                                        v1_min=0,        v1_max=2*math.pi,
                                        v2_min=z_bottom, v2_max=z_top,
                                        pathSave=pathSave,
                                        valuesTotal=area)
                weights = W.ravel()
                plot_pointsDistribution3D(points, weights, typ='ref. weight',
                                        area=typ,
                                        v1_min=0,        v1_max=2*math.pi,
                                        v2_min=z_bottom, v2_max=z_top,
                                        pathSave=pathSave)
                sigmas = grid.ravel()
                plot_pointsDistribution3D(points, sigmas, typ='sigma',
                                        area=typ,
                                        v1_min=0,        v1_max=2*math.pi,
                                        v2_min=z_bottom, v2_max=z_top,
                                        pathSave=pathSave)
        if internalTesting:
            print('   i       v1        v2       sig         w')
            print('---------------------------------------------')
            for i, (a, b, c, w) in enumerate(zip(v1, v2, sig, weights)):
                print(f'{i:3d}  {a:9.3f}  {b:9.3f}  {c:10.3f}   {w:2.0f}')
    else:
        raise ValueError("Unknown method. Use 'simple', 'triangularRectangle',"
                         "' triangularCircle', 'triangularShaft', "
                         "'simpsonRectangle', 'simpsonCircle' "
                         "or 'simpsonShaft'.")
    # print(f'Sigma is {sigma:.4f}.')
    return F_z, sigma

def fct_calc3D_controlData(v1,v2,sig, v2o=None,
                           internalTesting = internalTesting):
    # transform data
    (v1, v2, sig,
     v1_uni, v2_uni,
     v1_num, v2_num, v2o) = fct_prepareGrid(v1,v2,sig, v2o)

    # we need 5,7,9,11,
    if v1_num % 2 == 0: 
        # delete one row not possible since dvy have to be the same
        raise ValueError('Simpson requires odd v1_num.'
                         f'V1_u with v1_num {v1_num}: {v1_uni}'
                         f'Simpson requires odd v1_num {v1_num}.')

    # delete or change double values
    if v2_num % 2 == 0:
        if internalTesting:
            print(f'Old unique v2 values:\n{v2_uni}.')
        seen = {}
        duplicates_idx = []
        for i, (a, b) in enumerate(zip(v1, v2)):
            if (a, b) in seen:
                duplicates_idx.append(i)
            else:
                seen[(a, b)] = i

        # print(duplicates_idx)
        if duplicates_idx:
            if internalTesting:
                print(f'The searched duplicated positions are\n{duplicates_idx}.')
            # delete douple points for quarter or half
            mask = np.ones(len(v1), dtype=bool)
            mask[duplicates_idx] = False
            v1  = v1[mask]
            v2  = v2[mask]
            sig = sig[mask]
            # add points for full circle
            (v1, v2, sig) = fct_closeCircle(v1, v2, sig)
            # transform data
            (v1, v2, sig,
             v1_uni, v2_uni,
             v1_num, v2_num, v2o) = fct_prepareGrid(v1,v2,sig, v2o)
            if internalTesting:
                print(f'New unique v2 values:\n{v2_uni}.')
        else:
            raise ValueError('Simpson requires odd v2_num.\n'
                             f'V1_u before with v1_num {v1_num}: {v1_uni}\n'
                             f'V2_u before with v2_num {v2_num}: {v2_uni}\n'
                             'No duplicates found.')
        # print(f'V1_u after with v1_num {v1_num}: {v1_uni}')
        # print(f'V2_u after with v2_num {v2_num}: {v2_uni}')

    # in case full circle was already complete
    if v2_num % 2 == 0:
        if internalTesting:
            print(f'Old unique v2 values: {v2_uni}.')
        seen = {}
        duplicates_idx = []
        for i, (a, b) in enumerate(zip(v1, v2)):
            if (a, b) in seen:
                duplicates_idx.append(i)
            else:
                seen[(a, b)] = i
        if duplicates_idx:
            v1  = [v
                   for i, v in enumerate(v1)
                   if i not in duplicates_idx]
            v2  = [v
                   for i, v in enumerate(v2)
                   if i not in duplicates_idx]
            sig = [v
                   for i, v in enumerate(sig)
                   if i not in duplicates_idx]
            v2o = [v
                   for i, v in enumerate(v2o)
                   if i not in duplicates_idx]
            (v1, v2, sig,
             v1_uni, v2_uni,
             v1_num, v2_num, v2o) = fct_prepareGrid(v1,v2,sig, v2o)

    if v1_num * v2_num != len(sig):
        seen = {}
        duplicates_idx = []
        for i, (a, b, s) in enumerate(zip(v1, v2, sig)):
            if (a, b, s) in seen: duplicates_idx.append(i)
            else: seen[(a, b, s)] = i
        # print(duplicates_idx)
        if duplicates_idx:
            v1  = [v
                   for i, v in enumerate(v1)
                   if i not in duplicates_idx]
            v2  = [v
                   for i, v in enumerate(v2)
                   if i not in duplicates_idx]
            sig = [v
                   for i, v in enumerate(sig)
                   if i not in duplicates_idx]
            v2o = [v
                   for i, v in enumerate(v2o)
                   if i not in duplicates_idx]
            # print('Duplicated theta removed.')
            # transform data
            (v1, v2, sig,
             v1_uni, v2_uni,
             v1_num, v2_num, v2o) = fct_prepareGrid(v1,v2,sig, v2o)
        else:
            raise ValueError('v1_num * v2_num != len(sig)\n'
                             f'{v1_num} * {v2_num} != {len(sig)}\n'
                             'Simpson requires structured grid.')
        if v1_num * v2_num != len(sig):
            raise ValueError('v1_num * v2_num != len(sig)\n'
                             f'{v1_num} * {v2_num} != {len(sig)}\n'
                             f'v1_uni:\n{v1_uni}\n'
                             f'v2_uni:\n{v2_uni}\n'
                             'Simpson requires structured grid.')
        # print('v1_num * v2_num = len(sig)')
        # print(f'{v1_num} * {v2_num} = {len(sig)}')
    return v1,v2,sig,v2o

def fct_calc3D_surfSimpson(v1,v2,sig, method, radius=None, savePath=None):
    # transform data
    (v1, v2, sig,
     v1_uni, v2_uni, v1_num, v2_num, _) = fct_prepareGrid(v1,v2,sig,v2)
    # sort values at the grid point
    grid = np.array(sig).reshape((v1_num, v2_num))#v1=row,v2=spalten
    # to get weights = [1,4,2,4,2,4...1]
    w1 = np.ones(v1_num)
    w2 = np.ones(v2_num)
    w1[1:-1:2], w1[2:-1:2] = 4, 2
    w2[1:-1:2], w2[2:-1:2] = 4, 2
    # intergate over the grid
    integral = 0.0
    for i in range(len(w1)):
        if method == 'simpsonRectangle': factor = 1      # dA=dx*dy*1
        elif method == 'simpsonCircle':  factor = v1_uni[i]  # r_i
        elif method == 'simpsonShaft':   factor = radius # constant!
        for j in range(len(w2)):
            integral += w1[i] * w2[j] * grid[i,j] * factor
    d_v1 = (v1_uni[-1] - v1_uni[0]) / (v1_num - 1) # dx, dr
    d_v2 = (v2_uni[-1] - v2_uni[0]) / (v2_num - 1) # dy, dtheta
    factorWeight = d_v1 * d_v2 / 9
    F_z = integral * factorWeight
    return (F_z, w1, w2, factorWeight, grid)
# (F_z, w1, w2, factorWeight, grid) = fct_calc3D_surfSimpson(v1,v2,sig,
#                                                            method,
#                                                            radius=None)

def fct_prepareGrid(v1, v2, sig, v2o,
                    digits=digitsDefault,
                    internalTesting=internalTesting):
    v1  = np.round(np.asarray(v1),  digits)
    v2  = np.round(np.asarray(v2),  digits)
    sig = np.round(np.asarray(sig), digits)
    v2o = np.round(np.asarray(v2o), digits)
    idx = np.lexsort((v2, v1)) # (second prio, first prio)
    v1  = v1[idx]
    v2  = v2[idx]
    sig = sig[idx]
    v2o = v2o[idx]
    tol = 10**(-digits)
    v1_uni = fct_uniquewithTol(v1, tol=tol) # v1_unique
    v2_uni = fct_uniquewithTol(v2, tol=tol) # v2_unique
    v1_num    = len(v1_uni)
    v2_num    = len(v2_uni)
    if internalTesting:
        print(f'v1:\n{v1}')
        print(f'v2:\n{v2}')
        print(f'v1_uni:\n{v1_uni}')
        print(f'v2_uni:\n{v2_uni}')
        print(f'v1_num: {v1_num}')
        print(f'v2_num: {v2_num}')
    return (v1, v2, sig, v1_uni, v2_uni, v1_num, v2_num, v2o)
# (v1, v2, sig, v1_uni, v2_uni, v1_num, v2_num, v2_orig) = fct_prepareGrid(v1, v2, sig)

def fct_closeCircle(val, theta, sig,
                    digits=digitsDefault,
                    internalTesting=internalTesting):
    quadrants = [np.any((theta >= 0.0)       & (theta < 0.5*np.pi)),
                 np.any((theta >= 0.5*np.pi) & (theta < 1.0*np.pi)),
                 np.any((theta >= 1.0*np.pi) & (theta < 1.5*np.pi)),
                 np.any((theta >= 1.5*np.pi) & (theta <= 2*np.pi))]
    if all(quadrants): # for full circle = [True, True, True, True]
        if internalTesting:
            print("Full circle detected. Extending boundary row only.")
        # --- find unique v1 values and select minimum level ---
        tol = 10**(-digits)
        theta_min = np.min(fct_uniquewithTol(theta, tol=tol))
        # --- identify rows belonging to the boundary (v1_min) ---
        mask_starter  = np.isclose(theta, theta_min, atol=tol)
        val_starter   = val[mask_starter]
        theta_starter = theta[mask_starter] + 2*np.pi
        sig_starter   = sig[mask_starter]
        # --- append only boundary extension ---
        val   = np.concatenate([val,   val_starter])
        theta = np.concatenate([theta, theta_starter])
        sig   = np.concatenate([sig,   sig_starter])
    return (val, theta, sig)
# (v1, v2, sig) = fct_closeCircle(v1, v2, sig)
# (z, theta, sig) = fct_closeCircle(z, theta, sig)

# Compute resultant vertical force by integrating stress over a surface
def fct_uniquewithTol(arr, tol=1e-4): # for the next function necessary
    arr_sorted = np.sort(arr)
    groups = []
    current_group = [arr_sorted[0]]
    for val in arr_sorted[1:]:
        if abs(val - current_group[-1]) <= tol:
            current_group.append(val)
        else:
            groups.append(current_group)
            current_group = [val]
    groups.append(current_group)
    values = np.array([np.mean(g) for g in groups])
    values = np.sort(values)
    return values

def fct_recalcTheta(theta, old=False,
                    internalTesting=internalTesting):
    theta = np.mod(theta, 2*np.pi)
    theta_sorted = np.sort(theta)  # detect jump in theta
    dtheta = np.diff(theta_sorted) # distance to next value
    jump_idx = np.argmax(dtheta)   # pos. of maximal distance
    jump_size = dtheta[jump_idx]   # maximal distance
    if jump_size >= np.pi/2:       # missing quarter or more
        threshold = theta_sorted[jump_idx + 1]
        theta[theta >= threshold] -= 2*np.pi
        if internalTesting:
            print(f'Data transfered with threshold {threshold}.')
    if len(dtheta) > 0:
        jump_idx = np.argmax(dtheta)
        if dtheta[jump_idx] > np.pi:
            theta = np.mod(theta - theta_sorted[jump_idx+1], 2*np.pi)
            if internalTesting:
                print('Data transfered.')
    if old==False:
        theta =  theta - theta.min() # v2 starts with 0
    return theta
# theta = np.array([5.8, 5.9, 6.0, 0.1, 0.2, 1.0]) # example
# theta = fct_recalcTheta(theta)

def fct_calc3D_surfTriangle(points, sig, area, area_factor, hole=None):
    tri = spatial.Delaunay(points)
    simplices = tri.simplices
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    # one area for every triangle
    areas = 0.5 * np.abs((p1[:,0]-p0[:,0]) * (p2[:,1]-p0[:,1]) -
                         (p2[:,0]-p0[:,0]) * (p1[:,1]-p0[:,1]))
    areas = areas * area_factor
    # one sigma for every triangle
    sigmaTri = sig[simplices].mean(axis=1)
    # only values outside the hole [NOT TESTED]
    if hole is not None:
        hole_x, hole_y, hole_r = hole
        tri_centers = (p0 + p1 + p2) / 3
        distances = np.sqrt((tri_centers[:, 0] - hole_x)**2 +
                            (tri_centers[:, 1] - hole_y)**2)
        mask = distances >= hole_r
        areas = areas[mask]
        sigmaTri = sigmaTri[mask]
        area = area - math.pi * hole_r**2
    else: area = area - 0
    F_z = np.sum(sigmaTri * areas)
    sigma = F_z / np.sum(areas)
    return (F_z, sigma, tri, areas)
# (F_z, sigma, tri, areas) = fct_calc3D_surfTriangle(points, sig, area,
#                                                    area_factor, hole=hole)

def fct_getForcebyWissmann(g_o, phase,
                           pile_x_c, pile_y_c, pile_z_c, pile_r,
                           pathSave=None,
                           printDistribution=printDistribution,
                           internalTesting=internalTesting):
    soil = g_o.ResultTypes.Soil
    phas = g_o.Phases[int(phase)]

    # Points in unit coordinates (xi, eta)
    points = np.array([(+0,                 +0),
                       (+0,                 +0.953321175521807),
                       (+0,                 -0.882458098900126),
                       (+0,                 +0.582334188120499),
                       (+0.906662316102171, +0.294592444333741),
                       (-0.906662316102171, +0.294592444333741),
                       (+0.560348127669843, -0.771253032094644),
                       (-0.560348127669843, -0.771253032094644),
                       (+0.518695856299547, +0.713923598834010),
                       (-0.518695856299547, +0.713923598834010),
                       (+0.839267525316398, -0.272694549383947),
                       (-0.839267525316398, -0.272694549383947),
                       (+0.553832724273449, +0.179951160534772),
                       (-0.553832724273449, +0.179951160534772),
                       (+0.342287447682940, -0.471118254595022),
                       (-0.342287447682940, -0.471118254595022)])
    # corresponding weights by https://doi.org/10.1137/0723043
    weights = np.array([0.334521439965580,  0.087911387704639,
                        0.167628039106469,  0.305874815913735,
                        0.087911387704639,  0.087911387704639,
                        0.087911387704639,  0.087911387704639,
                        0.167628039106469,  0.167628039106469,
                        0.167628039106469,  0.167628039106469,
                        0.305874815913735,  0.305874815913735,
                        0.305874815913735,  0.305874815913735])

    # Integrate over the unit disk
    data = []
    sigma = np.empty(points.shape[0])
    for i in range(points.shape[0]):
        # Original point
        if internalTesting:
            print(f'Si position ({pile_x_c}, {pile_y_c})')
        xi, eta = points[i]
        x       = pile_x_c + pile_r * xi
        y       = pile_y_c + pile_r * eta
        z       = pile_z_c
        x_input = x
        y_input = y
        result  = fct_getSingle(g_o, phas, soil, x_input, y_input, z)
        if internalTesting:
            print(f'Wissmann point ({x_input}, {y_input}) for step 0')
        # Fallback 1: mirror at y-axis (-x, y)
        if result is None:
            x_input = pile_x_c + (pile_x_c - x)
            y_input = y
            result  = fct_getSingle(g_o, phas, soil, x_input, y_input, z)
            if result is None:
                result  = fct_getSingle(g_o, phas, soil,
                                        x_input, y_input,
                                        z - 0.01)
            if internalTesting:
                print(f'Wissmann point (x_mirrow) ({x_input}, {y_input})'
                      'for step 1')
        # Fallback 2: mirror at x-axis (x, -y)
        if result is None:
            x_input = x
            y_input = pile_y_c + (pile_y_c - y)
            result  = fct_getSingle(g_o, phas, soil, x_input, y_input, z)
            if result is None:
                result  = fct_getSingle(g_o, phas, soil,
                                        x_input, y_input,
                                        z - 0.01)
            if internalTesting:
                print(f'Wissmann point (x_mirrow) ({x_input}, {y_input})'
                      'for step 2')
        # Fallback 2: mirror at x&y-axis (-x, -y)
        if result is None:
            x_input = pile_x_c + (pile_x_c - x)
            y_input = pile_y_c + (pile_y_c - y)
            result  = fct_getSingle(g_o, phas, soil, x_input, y_input, z)
            if result is None:
                result  = fct_getSingle(g_o, phas, soil,
                                        x_input, y_input,
                                        z - 0.01)
            if internalTesting:
                print(f'Wissmann point (x_mirrow) ({x_input}, {y_input})'
                      'for step 3')
        # Fallback 3
        if result is None:
            x_input = x - 0.001
            y_input = y - 0.001
            result  = fct_getSingle(g_o, phas, soil, x_input, y_input, z)
            if internalTesting:
                print(f'Wissmann point (x_mirrow) ({x_input}, {y_input}) '
                      'for step 4')
        # No valid point found
        if result is None:
            raise ValueError('No valid result found for point '
                             f'(xi={xi:.3f}, eta={eta:.3f}) or '
                             f'(x={x:.3f}, y={y:.3f}, z={z:.3f})'
                             f'-> original / x-mirror / y-mirror failed.')
        else: sigma[i] = result[6]
        data.append([x, y, xi, eta, x_input, y_input,
                     weights[i], sigma[i]])
    df = pd.DataFrame(data, columns=['x', 'y', 'xi', 'eta', 'xdata', 'ydata',
                                     'weight', 'sigma'])
    F = np.dot(weights, sigma) * (pile_r**2)
    sigma_mean = F / (math.pi * pile_r**2)

    # Plots
    if printDistribution:
            plot_pointsDistribution3D(np.column_stack((df['xi'],df['eta'])),
                                    df['weight'],
                                    typ='ref-weight',
                                    area = 'Circle',
                                    pathSave=pathSave,
                                    valuesTotal=math.pi*1**2)
            plot_pointsDistribution3D(np.column_stack((df['x'],df['y'])),
                                    df['weight']* (pile_r**2),
                                    typ='weight', area = 'Circle',
                                    pile_x_c=pile_x_c,
                                    pile_y_c=pile_y_c,
                                    pile_r=pile_r,
                                    pathSave=pathSave,
                                    valuesTotal=math.pi*pile_r**2)
            plot_pointsDistribution3D(np.column_stack((df['x'],df['y'])),
                                    df['sigma'],
                                    typ='sigma',
                                    area = 'Circle',
                                    pile_x_c=pile_x_c,
                                    pile_y_c=pile_y_c,
                                    pile_r=pile_r,
                                    pathSave=pathSave)
            plot_pointsDistribution3D(np.column_stack((df['xdata'],df['ydata'])),
                                    df['sigma'],
                                    typ='sigmaPosition',
                                    area = 'Circle',
                                    pile_x_c=pile_x_c,
                                    pile_y_c=pile_y_c,
                                    pile_r=pile_r,
                                    pathSave=pathSave) # evaluation point
    return F, sigma_mean, df

###############################################################################
#------------------------------- Bending Moment -------------------------------
###############################################################################

# not used!

def fct_computeMoment(g_o, soil_volume, phase, z_max, z_min,
                          points, direction="x",
                          y_fixed=None, x_fixed=None, pathPlot=None,
                          title="Bending moment distribution"):
    import numpy as np
    # 1. Neutral axis (important assumption for bending)
    z0 = 0.5 * (z_max + z_min)
    # 2. Extract raw FEM results
    sigma_xx = np.array(g_o.getresults(soil_volume, phase,
                                       g_o.ResultTypes.Soil.SigmaXX,
                                       "node"))
    x = np.array(g_o.getresults(soil_volume, phase,
                                g_o.ResultTypes.Soil.X, "node"))
    y = np.array(g_o.getresults(soil_volume, phase,
                                g_o.ResultTypes.Soil.Y, "node"))
    z = np.array(g_o.getresults(soil_volume, phase,
                                g_o.ResultTypes.Soil.Z, "node"))
    # Element weight (in real use: replace with element volume weighting)
    weights = np.ones_like(sigma_xx)
    # 3. Define evaluation line
    if direction == "x":
        line = np.linspace(np.min(x), np.max(x), points)
    elif direction == "y":
        line = np.linspace(np.min(y), np.max(y), points)
    else:
        raise ValueError("direction must be 'x' or 'y'")
    M_line = np.zeros_like(line)
    for i, xi in enumerate(line):
        mask = np.abs(x - xi) < 1e-3
        if np.sum(mask) == 0:
            M_line[i] = 0
            continue
        M_line[i] = np.sum(sigma_xx[mask] * (z[mask] - z0) * weights[mask])
    # 4. Extract extrema
    M_max = np.max(M_line)
    M_min = np.min(M_line)
    # 5. Ouput
    if pathPlot is not None:
        plt.figure()
        plt.plot(line, M_line)
        plt.xlabel("Position along line")
        plt.ylabel("Equivalent bending moment")
        plt.title(title)
        plt.grid(True)
        plt.savefig(pathPlot, dpi=300, bbox_inches='tight')
        plt.show()
    return {"positions": line,
            "M":         M_line,
            "M_max":     M_max,
            "M_min":     M_min}

# BENDING DUE TO DUMMY PLATE
def fct_plateMoment(g_o, g_i,
                    dimension,
                    phaseSlab,
                    phase,
                    plateI,
                    si_x, si_y,
                    dis_x, dis_y,
                    E_dummy_factor=1e-5,
                    path_save=None,
                    export_plot=True,
                    si_r = 1):
    if phase >= phaseSlab:
        try:
            # print('INPUT PLATE:', plateI.Name.value)
            # print('OUTPUT PLATES:', len(g_o.Plates))
            # for plate in g_o.Plates:
            #     print('OUTPUT PLATE:', plate.Name.value)
            #     for plate_part in plate:
            #         print('OUTPUT PLATE INNER:', plate_part.Name.value)
            version = 1
            if version == 1:
                plO = next((plate_part
                            for plate in g_o.Plates
                            for plate_part in plate
                            if plate_part.Name.value == plateI.Name.value),
                           None
                           )
                # print(plO)
            elif version == 2:
                plO = next((plate
                            for plate in g_o.Plates
                            if plate.Name.value == plateI.Name.value),
                           None
                           )
            ph = g_o.Phases[int(phase)]
            pl = g_o.ResultTypes.Plate
            # 1. Get plate results
            if dimension.endswith('2'):
                x  = np.asarray(g_o.getresults(plO, ph, pl.X,   'node', True))
                y  = np.asarray(g_o.getresults(plO, ph, pl.Y,   'node', True))
                m  = np.asarray(g_o.getresults(plO, ph, pl.M2D, 'node', True))
                m  = m / E_dummy_factor
                df = pd.DataFrame({"x": x,
                                   "y": y,
                                   "M": m,
                                   "Mmax_abs": np.abs(m)})
            elif dimension.endswith('3'):
                x    = np.asarray(g_o.getresults(plO, ph, pl.X,   'node', True))
                y    = np.asarray(g_o.getresults(plO, ph, pl.Y,   'node', True))
                z    = np.asarray(g_o.getresults(plO, ph, pl.Z,   'node', True))
                m11  = np.asarray(g_o.getresults(plO, ph, pl.M11, 'node', True))
                m22  = np.asarray(g_o.getresults(plO, ph, pl.M22, 'node', True))
                m12  = np.asarray(g_o.getresults(plO, ph, pl.M12, 'node', True))
                m11  = m11/E_dummy_factor
                m22  = m22/E_dummy_factor
                m12  = m12/E_dummy_factor
                df   = pd.DataFrame({"x": x, "y": y, "z": z,
                                     "M11": m11,
                                     "M22": m22,
                                     "M12": m12}
                                    )
                Mavg = 0.5 * (df["M11"] + df["M22"])
                R    = np.sqrt(((df["M11"] - df["M22"]) / 2)**2
                               + df["M12"]**2)
                df["Mmax_abs"] = np.maximum(np.abs(Mavg + R),
                                            np.abs(Mavg - R)
                                            )
            else: raise ValueError(f'Dimension {dimension} not defined!')
            df = df.sort_values(by='x')
            # 2. Apply dummy stiffness correction
            # 3. GLOBAL MAX
            Mmax_global = np.nanmax(np.abs(df["Mmax_abs"].values))
            # 4. LOCAL DOMAIN
            if 'Unitcell' in dimension:
                Mmax_local = Mmax_global
            else:
                x_min, x_max = si_x - dis_x/2, si_x + dis_x/2
                y_min, y_max = si_y - dis_y/2, si_y + dis_y/2
                if dimension.endswith("3"):
                    df_local = df[(df["x"] >= x_min) & (df["x"] <= x_max) &
                                  (df["y"] >= y_min) & (df["y"] <= y_max)]
                elif dimension.endswith("2"): # for half2d
                    df_local = df[(df["x"] >= x_min) & (df["x"] <= x_max)]
                else: raise ValueError("Unknown dimension")
                if len(df_local) == 0: Mmax_local = Mmax_global
                else: Mmax_local = np.nanmax(np.abs(df_local
                                                    ["Mmax_abs"].values))
            # 6. EXPORT
            if path_save:
                os.makedirs(path_save, exist_ok=True)
                pathFile = os.path.join(path_save,
                                        f'Moment_p{phase}.csv')
                df.to_csv(pathFile, sep=";", index=False)
                pathFile = os.path.join(path_save,
                                        f'Moment_p{phase}.pdf')
                if dimension.endswith("3"):
                    for momentTyp in ['M11', 'M22', 'M12','Mmax_abs']:
                        plot_pointsDistribution3D(np.column_stack((df['x'],
                                                                   df['y'])),
                                                  df[momentTyp],
                                                  typ  = momentTyp,#kNm/m
                                                  area = 'Circle',
                                                  pile_x_c = si_x,
                                                  pile_y_c = si_y,
                                                  pile_r   = si_r,
                                                  contour  = True,
                                                  pathSave = pathFile)
                    # https://bentleysystems.service-now.com/community?
                    # id=kb_article&sysparm_article=KB0108661
                elif dimension.endswith("2"):
                    for momentTyp in ['M', 'Mmax_abs']:
                        pathFile = os.path.join(path_save,
                                   f'Moment_p{phase}_{momentTyp}.pdf')
                        plot_pointsDistribution2D(df['x'],
                                                  df[momentTyp],
                                                  'distance from the center',
                                                  momentTyp,#kNm/m
                                                  filler = False,
                                                  pathFile=pathFile)
                else: raise ValueError("Unknown dimension")
            # 7. RETURN RESULTS
            return {"Mmax_global": Mmax_global,
                    "Mmax_local":  Mmax_local,
                    "data":        df}
        except Exception as e:
            print("[EVAL Moment ERROR]", e)
            traceback.print_exc()
            return {"Mmax_global": 0, "Mmax_local": 0}
    else: return {"Mmax_global": 0, "Mmax_local": 0}

if __name__ == "__main__":
    from plaxis_00calling_02helpers import open_plaxisData
    device = 27
    phase = 10
    modeltyp = 'unitcell3'
    _, _, path = open_plaxisData(device,
                                 'spacer',
                                 os.path.join('Optimization - 2026-09-26 - Sensitivity Study')
                                 )
    cases = [
             (#0
                 'soil', 'Rectangle',#fig4.33
                 ['fineMedium'],
                 {'if_tri': 0, 'gr_sim21': 3, 'gr_sim11': 4, 'cs_tri': 5},
                 (5, 95),
             ),
            (#1
                'soil', 'Rectangle',#fig4.33
                ['fine'],
                {'if_tri': 0, 'gr_sim21': 3, 'gr_sim11': 4, 'cs_tri': 5},
                (5, 95),
            ),
             (#2
                 'elem', 'Circle', #fig4.27
                 ['medium', 'fineMedium', 'fine'],
                 {'cs_tri': 2, 'if_tri': 0},
                 (0, 100),
             ),
             (#3
                 'elem', 'Circle', #fig4.21
                 ['medium', 'fineMedium', 'fine'],
                 {'if_tri': 0,},
                 (0, 100),
             ),
             (#4
                 'elem', 'Circle', #fig4.27 add
                 ['medium', 'fineMedium', 'fine'],
                 {'gr_wis': 3, 'gr_sim21': 4, 'gr_sim15': 5, 'gr_sim11': 6,},
                 (0, 100),
             ),
             (#5
                 'elem', 'Circle', #fig4.20
                 ['coarse', 'mediumCoarse'],
                 {'if_tri': 0,},
                 (0, 100),
             ),
            ]
    for idx, (typName, typArea, modelcats, methods, perc) in enumerate(cases):
        subfolder = f'results_{typName}_Middle'
        csv_files = {}
        for modelcat in modelcats:
            for method, var_i in methods.items():
                key = f'{modelcat}_{method}'
                csv_files[key] = os.path.join(
                    path,
                    f'meshing_{modelcat}_{modeltyp}',
                    subfolder,
                    f'{subfolder}_p{phase}',
                    f'{subfolder}_var{var_i}_p{phase}.csv'
                    )
        plot_pointsDistribution3D_compare(csv_files   = csv_files,
                                          pile_center = (0.75, 0.75),
                                          pile_radius = 0.3,
                                          typName     = typName,
                                          typArea     = typArea,
                                          perc        = perc,
                                          idx         = idx,
                                          )
