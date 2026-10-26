

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from plaxis_02eval_02helpers import (fct_calc3D_surfIG,
                                     plot_pointsDistribution3D,
                                     plot_pointsDistribution2D,
                                     )

METHODS = {'elem': {0: ('triangularCircle', 's_n'),
                    2: ('triangularCircle', 's_n'),
                    3: ('Wissmann', None),
                    4: ('simpsonCircle', 'sig_zz'),
                    5: ('simpsonCircle', 'sig_zz'),
                    6: ('simpsonCircle', 'sig_zz'),
                    7: ('triangularCircle', 'sig_zz'),
                    },
           'soil': {0: ('triangularRectangle', 's_n'),
                    2: ('triangularRectangle', 's_n'),
                    3: ('simpsonRectangle', 'sig_zz'),
                    4: ('simpsonRectangle', 'sig_zz'),
                    5: ('triangularRectangle', 'sig_zz'),
                    },
           'base': {0: ('triangularCircle', 's_n'),
                    2: ('triangularCircle', 's_n'),
                    3: ('Wissmann', None),
                    4: ('simpsonCircle', 'sig_zz'),
                    5: ('simpsonCircle', 'sig_zz'),
                    6: ('simpsonCircle', 'sig_zz'),
                    7: ('triangularCircle', 'sig_zz'),
                    },
           'shaft': {0: ('triangularShaft', 's_1'),
                     2: ('simpsonShaft', 'shear'),
                     },
           }
RESULT_RE = re.compile(r'^results_(elem|soil|base|shaft)'
                       r'(_(?:Middle|Corner|EdgeX|EdgeY))?'
                       r'_var(\d+)_p(\d+)$'
                       )

def read_results(file):
    df = pd.read_csv(file, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    return df

def get_geometry(model, suffix, cache):
    key = (model, suffix)
    if key not in cache:
        stem = f'results_shaft{suffix}'
        candidates = sorted(model.glob(f'{stem}/{stem}_p*/{stem}_var0_p*.csv'))
        geom = None
        if candidates:
            df = read_results(candidates[0])
            if {'x', 'y'}.issubset(df.columns):
                xy = np.unique(df[['x', 'y']].to_numpy(dtype=float), axis=0)
                if len(xy) >= 3:
                    x, y = xy.T
                    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
                    xc, yc, c = np.linalg.lstsq(A, x*x + y*y, rcond=None)[0]
                    radius = np.median(np.hypot(x-xc, y-yc))
                    if np.isfinite(radius) and radius > 0:
                        geom = (float(xc), float(yc), float(radius))
        cache[key] = geom
    return cache[key]

def fallback_geometry(df):
    """Approximation only, for models without stored shaft var0 CSV."""
    xc = (df['x'].min() + df['x'].max()) / 2
    yc = (df['y'].min() + df['y'].max()) / 2
    r = np.max(np.hypot(df['x']-xc, df['y']-yc))
    return float(xc), float(yc), float(r)

def replot_wissmann(df, outfile, xc, yc, radius):
    needed = {'xi', 'eta', 'x', 'y', 'xdata', 'ydata', 'weight', 'sigma'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Wissmann CSV missing {needed-set(df.columns)}')
    unit_points = df[['xi', 'eta']].to_numpy(dtype=float)
    real_points = df[['x', 'y']].to_numpy(dtype=float)
    used_points = df[['xdata', 'ydata']].to_numpy(dtype=float)
    w = df['weight'].to_numpy(dtype=float)
    sigma = df['sigma'].to_numpy(dtype=float)
    plot_pointsDistribution3D(unit_points, w, typ='ref-weight',
                              area='Circle', pathSave=str(outfile),
                              valuesTotal=np.pi)
    plot_pointsDistribution3D(real_points, w * radius**2, typ='weight',
                              area='Circle', pile_x_c=xc, pile_y_c=yc,
                              pile_r=radius, pathSave=str(outfile),
                              valuesTotal=np.pi*radius**2)
    plot_pointsDistribution3D(real_points, sigma, typ='sigma',
                              area='Circle', pile_x_c=xc, pile_y_c=yc,
                              pile_r=radius, pathSave=str(outfile))
    plot_pointsDistribution3D(used_points, sigma, typ='sigmaPosition',
                              area='Circle', pile_x_c=xc, pile_y_c=yc,
                              pile_r=radius, pathSave=str(outfile))

def replot_all(basepath):
    basepath = Path(basepath)
    geometry_cache = {}
    n_ok, n_skip, failures = 0, 0, []
    for csv_file in sorted(basepath.rglob('results_*_var*_p*.csv')):
        match = RESULT_RE.fullmatch(csv_file.stem)
        if not match: continue
        kind, suffix, variant, phase = match.groups()
        suffix = suffix or ''
        variant = int(variant)
        try:
            df = read_results(csv_file)
            outfile = csv_file.with_suffix('.pdf')
            # 2D evaluation curves
            if 'z' not in df.columns:
                if variant >= 100:  # Derived force differences: no point plot.
                    n_skip += 1
                    continue
                stress_cols = (('s_1', 'sig_xy', 'sig_xx', 'sig_yy')
                               if kind == 'shaft' else ('sig_yy', 's_n'))
                stress = next((c
                               for c in stress_cols
                               if c in df.columns),
                              None)
                if stress is None:
                    n_skip += 1
                    continue
                coord = 'y' if kind == 'shaft' else 'x'
                plot_pointsDistribution2D(df[coord], df[stress],
                                          'depth [m]'
                                              if coord == 'y'
                                              else 'distance [m]',
                                          r'stress $\sigma$ [kPa]',
                                          filler=True, pathFile=str(outfile)
                )
            else:
                method_sig = METHODS.get(kind, {}).get(variant)
                if method_sig is None:
                    n_skip += 1  # Simple mean, IFA, etc.: no original plot.
                    continue
                method, stress = method_sig
                model = csv_file.parents[2]
                geo = get_geometry(model, suffix, geometry_cache)
                if geo is None:
                    geo = fallback_geometry(df)
                    print(f'[WARN] Approximated centre/radius for {csv_file}')
                xc, yc, radius = geo
                if method == 'Wissmann':
                    replot_wissmann(df, outfile, xc, yc, radius)
                else:
                    if stress not in df.columns:
                        raise ValueError(f'Missing column {stress!r}')
                    kwargs = {}
                    if method in ('triangularCircle', 'simpsonCircle'):
                        # Allow a little tolerance at the original interface.
                        dist_max = float(np.max(np.hypot(df['x']-xc,
                                                         df['y']-yc)))
                        r_lim = max(radius + 0.001, dist_max + 1e-5)
                        kwargs['boundaries'] = [xc, yc, r_lim]
                    elif method in ('triangularShaft', 'simpsonShaft'):
                        kwargs['boundaries'] = [xc, yc,
                                                float(df['z'].max()),
                                                float(df['z'].min()), radius
                                                ]
                    elif kind == 'soil' and variant == 0:
                        kwargs['hole'] = [xc, yc, radius]
                    fct_calc3D_surfIG(df,
                                      stress,
                                      method=method,
                                      pathSave=str(outfile),
                                      printDistribution=True,
                                      **kwargs
                                      )
            n_ok += 1
            print(f'[OK] {csv_file}')
        except Exception as e:
            failures.append((csv_file, str(e)))
            print(f'[ERROR] {csv_file}: {e}')
        finally:
            plt.close('all')

    # Moment plots: M/Mmax_abs in 2D; M11/M22/M12/Mmax_abs in 3D.
    for csv_file in sorted(basepath.rglob('Moment_p*.csv')):
        try:
            df = read_results(csv_file)
            if not {'x', 'y'}.issubset(df.columns):
                n_skip += 1
                continue
            moment3D = [name for name in ('M11', 'M22', 'M12', 'Mmax_abs')
                        if name in df.columns]
            if moment3D and 'M11' in moment3D:
                model = csv_file.parent.parent
                suffix = csv_file.parent.name.removeprefix('summary_plot')
                geo = get_geometry(model, suffix, geometry_cache)
                if geo is None:
                    geo = fallback_geometry(df)
                xc, yc, radius = geo
                for typ in moment3D:
                    plot_pointsDistribution3D(df[['x',
                                                  'y']].to_numpy(dtype=float),
                                              df[typ].to_numpy(dtype=float),
                                              typ=typ,
                                              area='Circle',
                                              pile_x_c=xc,
                                              pile_y_c=yc,
                                              pile_r=radius,
                                              contour=True,
                                              pathSave=str(csv_file.
                                                           with_suffix('.pdf'))
                                              )
            else:
                for typ in ('M', 'Mmax_abs'):
                    if typ in df.columns:
                        pathFile = str(csv_file.with_name(f'{csv_file.stem}_{typ}.pdf'))
                        plot_pointsDistribution2D(df['x'],
                                                  df[typ],
                                                  'distance from the center',
                                                  typ, filler=False,
                                                  pathFile=pathFile
                                                  )
            n_ok += 1
        except Exception as e:
            failures.append((csv_file, str(e)))
        finally:
            plt.close('all')

    # Tilt CS/IF/grid: the CSV contains the original triangulation edges.
    for csv_file in sorted(basepath.rglob('results_*_tilt*_p*.csv')):
        try:
            df = read_results(csv_file)
            if {'x1', 'y1', 'u1', 'x2', 'y2', 'u2'}.issubset(df.columns):
                pt1 = df[['x1', 'y1', 'u1']].copy()
                pt2 = df[['x2', 'y2', 'u2']].copy()
                pt1.columns = pt2.columns = ['x', 'y', 'u']
                verts = pd.concat((pt1, pt2), ignore_index=True)
                verts = verts.drop_duplicates(subset=['x', 'y'])
                if len(verts) < 3:
                    raise ValueError('Not enough vertices in edge file.')
                pts = verts[['x', 'y']].to_numpy(dtype=float)
                pathSave = str(csv_file.with_suffix('.pdf'))
                plot_pointsDistribution3D(pts, verts['u'].to_numpy(dtype=float),
                                          typ='deformation u [m]',
                                          area='Rectangle',
                                          v1_min=float(verts['x'].min()),
                                          v1_max=float(verts['x'].max()),
                                          v2_min=float(verts['y'].min()),
                                          v2_max=float(verts['y'].max()),
                                          tri=Delaunay(pts),
                                          pathSave=pathSave
                                          )
                n_ok += 1
            elif {'x', 'uy'}.issubset(df.columns):
                pathFile = str(csv_file.with_suffix('.pdf'))
                plot_pointsDistribution2D(df['x'], df['uy'],
                                          'distance x [m]',
                                          'deformation uy [m]',
                                          filler=False,
                                          pathFile=pathFile
                                          )
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            failures.append((csv_file, str(e)))
        finally:
            plt.close('all')

    print(f'Finished: {n_ok} input CSV plotted, {n_skip} skipped, '
          f'{len(failures)} errors.')
    for file, error in failures:
        print(f'[ERROR] {file}: {error}')

if __name__ == '__main__':
    device = 27
    from plaxis_00calling_02helpers import open_plaxisData
    _, _, BASE = open_plaxisData(device, 'spacer',
                 'Optimization - 2026-09-26 - Sensitivity Study')
    for model in ['unitcell2', 'unitcell3']:
        for mesh in [
                     'coarse',
                     'mediumCoarse',
                     'medium',
                     'fineMedium',
                     'fine',
                     ]:
            path = Path(BASE) / f'meshingFiner_{mesh}_{model}'
            if not path.is_dir():
                print(f'[SKIP] Folder not found: {path}')
                continue
            print(path)
            replot_all(path)
