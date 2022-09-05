import warnings
import colorsys
import webcolors
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interp
import matplotlib.colors as colors
from colorspacious import cspace_converter, cspace_convert

labfn = cspace_converter('sRGB1', 'CAM02-UCS')
revfn = cspace_converter('CAM02-UCS', 'sRGB1')

def modify_color(hc, lightdark=1, saturate=1):

    rgb = np.array(webcolors.hex_to_rgb(hc)) / 255
    h, l, s = colorsys.rgb_to_hls(*rgb)
    rgb = colorsys.hls_to_rgb(h, min(1, lightdark * l), s=min(1, s * saturate))
    return webcolors.rgb_to_hex((np.array(rgb) * 255).astype(int))

def params_to_standard(p, hxs):
    p1 = np.exp(p)
    return np.reshape(p1, (2, len(hxs)))

def params_to_rgb(p, hxs):
    p1 = params_to_standard(p, hxs)
    mod = [modify_color(hx, lightdark=ld, saturate=sat) \
        for hx, ld, sat in zip(hxs, p1[0], p1[1])]
    return np.array(list(map(webcolors.hex_to_rgb, mod))) / 255

def params_to_cam(p, hxs):
    rgb = params_to_rgb(p, hxs)
    return np.vstack([labfn(tup) for tup in rgb])

def find_optimal_params(hxs, target):

    def objective(p):
        cam = params_to_cam(p, hxs)
        return np.sum((cam[:,0] - target)**2)

    popsize = 100
    p0 = np.zeros((2 * len(hxs),))
    bnd = [(np.log(0.25), np.log(2.5))] * (2 * len(hxs))
    init = np.tile(p0, (popsize, 1)) + \
        np.random.uniform(-0.01, 0.01, size=(popsize, 2*len(hxs)))
    sol = opt.differential_evolution(objective, bnd, popsize=popsize, init=init)

    if not sol.success:
        warnings.warn(sol.message)

    return sol.x, sol.fun

def find_optimal_params_sequential(hxs, min_intensity=55, max_intensity=85):

    target = np.linspace(min_intensity, max_intensity, len(hxs))
    return find_optimal_params(hxs, target)

def find_optimal_params_diverging(hxs, min_intensity=55, max_intensity=85):

    if len(hxs) % 2:
        tmp = np.linspace(min_intensity, max_intensity, len(hxs) // 2 + 1)
        target = np.concatenate((tmp[::-1], tmp[1:]))
    else:
        tmp = np.linspace(min_intensity, max_intensity, len(hxs) // 2)
        target = np.concatenate((tmp[::-1], tmp))
    return find_optimal_params(hxs, target)

def interpolate_sequential(hxs, params):

    cam = params_to_cam(params, hxs)

    if np.any(np.diff(cam[:,0]) <= 0):
        warnings.warn('Fixing non-monotonicity in intensity')
        arg = np.argsort(cam[:,0])
        cam = cam[arg]

    x = cam[:,0]
    xx = np.linspace(np.amin(x), np.amax(x), 250)
    return interp.Akima1DInterpolator(x, cam)(xx)

def interpolate_diverging(hxs, params):

    cam = params_to_cam(params, hxs)
    n = len(hxs)

    x = cam[:n//2+1,0][::-1]
    xx = np.linspace(np.amin(x), np.amax(x), 125)

    if n // 2 >= 3:
        xx1 = interp.Akima1DInterpolator(x, cam[:n//2+1][::-1])(xx)
    else:
        xx1 = interp.interp1d(x, cam[:n//2+1][::-1], kind='slinear', axis=0)(xx)

    x = cam[n//2:,0]
    xx = np.linspace(np.amin(x), np.amax(x), 125)

    if n // 2 >= 3:
        xx2 = interp.Akima1DInterpolator(x, cam[n//2:])(xx)
    else:
        xx2 = interp.interp1d(x, cam[n//2:], kind='slinear', axis=0)(xx)

    return np.concatenate((xx1[::-1], xx2))

def cmap_from_params(name, hxs, params, kind):

    if kind == 'sequential':
        cam = interpolate_sequential(hxs, params)
    elif kind == 'diverging':
        cam = interpolate_diverging(hxs, params)
    else:
        raise NotImplementedError

    cs = np.vstack(list(map(revfn, cam)))
    cs = np.clip(cs, 0, 1)
    return colors.LinearSegmentedColormap.from_list(name, cs)

def simulate_cvd(fig, kind, severity):

    from PIL import Image

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb = np.fromstring(buf, dtype=np.uint8).astype(np.float64)
    rgb = rgb.reshape(nrows, ncols, 3) / 255

    cvd_space = dict(name='sRGB1+CVD', cvd_type=kind, severity=severity)
    sim = np.clip(cspace_convert(rgb, cvd_space, 'sRGB1'), 0, 1)
    sim = (sim * 255).astype(np.uint8)
    return Image.frombuffer('RGB', (ncols, nrows), sim, 'raw', 'RGB', 0, 3)

def simulate_grayscale(fig):

    from PIL import Image

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb = np.fromstring(buf, dtype=np.uint8).astype(np.float64)
    rgb = rgb.reshape(nrows, ncols, 3) / 255

    sim = np.clip(cspace_convert(rgb, 'sRGB1', 'CAM02-UCS')[:,:,0], 0, 100)
    sim = np.tile(np.expand_dims(sim * 2.55, 2), (1, 1, 3)).astype(np.uint8)
    return Image.frombuffer('RGB', (ncols, nrows), sim, 'raw', 'RGB', 0, 3)

# vim: set ft=python:
