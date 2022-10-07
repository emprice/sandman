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
    ''' Modifies the given color by lightening and saturating according
    to the input parameters

    :param hc: Hex code for the input color
    :type hc: str
    :param lightdark: Lightening parameter. For values greater than unity,
        the color will be lightened, and, for values less than unity, the
        color will be darkened. The lightness of the color in HLS space
        is scaled by the value of `lightdark`, but it will not exceed
        unity when the color is reconstructed
    :type lightdark: float, optional
    :param saturate: Saturation parameter. For values greater than unity,
        the color will be saturated, and, for values less than unity, the
        color will be desaturated. The saturation of the color in HLS space
        is scaled by the value of `saturate`, but it will not exceed
        unity when the color is reconstructed
    :type saturate: float, optional
    :return: The modified, reconstructed color as a hex code
    :rtype: str
    '''
    rgb = np.array(webcolors.hex_to_rgb(hc)) / 255
    h, l, s = colorsys.rgb_to_hls(*rgb)
    rgb = colorsys.hls_to_rgb(h, max(0, min(1, lightdark * l)),
        s=max(0, min(1, s * saturate)))
    return webcolors.rgb_to_hex((np.array(rgb) * 255).astype(int))

def params_to_standard(p, hxs, ld0, sat0):
    ''' Converts the input parameters into a standard form

    :param p: Input parameters as fed into objective function,
        with shape :code:`(2 * len(hxs),)`
    :type p: :class:`numpy.ndarray`
    :param hxs: User-specified colors for the current optimization
    :type hxs: list of str
    :param ld0: Baseline lightness to add after the transform
    :type ld0: float
    :param sat0: Baseline saturation to add after the transform
    :type sat0: float
    :return: Parameters that can be converted to colors,
        with shape :code:`(2, len(hxs))`
    :rtype: :class:`numpy.ndarray`
    '''
    p1 = np.exp(p)
    p1 = np.reshape(p1, (2, len(hxs)))
    p1[0,:] += ld0
    p1[1,:] += sat0
    return p1

def params_to_rgb(p, hxs, ld0, sat0):
    ''' Converts the input parameters and colors into RGB triples

    :param p: Input parameters as fed into objective function,
        with shape :code:`(2 * len(hxs),)`
    :type p: :class:`numpy.ndarray`
    :param hxs: User-specified colors for the current optimization
    :type hxs: list of str
    :param ld0: Baseline lightness to add after the transform
    :type ld0: float
    :param sat0: Baseline saturation to add after the transform
    :type sat0: float
    :return: RGB triples mapped to the [0,1] interval,
        with shape :code:`(len(hxs), 3)`
    :rtype: :class:`numpy.ndarray`
    '''
    p1 = params_to_standard(p, hxs, ld0, sat0)
    mod = [modify_color(hx, lightdark=ld, saturate=sat) \
        for hx, ld, sat in zip(hxs, p1[0], p1[1])]
    return np.array(list(map(webcolors.hex_to_rgb, mod))) / 255

def params_to_cam(p, hxs, ld0, sat0):
    ''' Converts the input parameters and colors into CAM02 triples

    :param p: Input parameters as fed into objective function,
        with shape :code:`(2 * len(hxs),)`
    :type p: :class:`numpy.ndarray`
    :param hxs: User-specified colors for the current optimization
    :type hxs: list of str
    :param ld0: Baseline lightness to add after the transform
    :type ld0: float
    :param sat0: Baseline saturation to add after the transform
    :type sat0: float
    :return: CAM02 triples on their native intervals,
        with shape :code:`(len(hxs), 3)`
    :rtype: :class:`numpy.ndarray`
    '''
    rgb = params_to_rgb(p, hxs, ld0, sat0)
    return np.vstack([labfn(tup) for tup in rgb])

def find_optimal_params(hxs, target, ld0, sat0):
    ''' Optimizes the given colors such that they have perceptual lightness
    values as close as possible to the given target values

    :param hxs: Colors to be optimized
    :type hxs: list of str
    :param target: Proposed target values for the lightness parameter,
        with shape :code:`(len(hxs),)`
    :type target: :class:`numpy.ndarray`
    :param ld0: Baseline lightness to add after the transform
    :type ld0: float
    :param sat0: Baseline saturation to add after the transform
    :type sat0: float
    :return: Optimized parameters and value of the objective function
    :rtype: tuple of :class:`numpy.ndarray` and float
    '''
    def objective(p):
        cam = params_to_cam(p, hxs, ld0, sat0)
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

def find_optimal_params_sequential(hxs, lightness_base=0,
saturation_base=0, min_intensity=55, max_intensity=85):
    ''' Optimizes the given colors by modifying their lightness
    and saturation to reach a sequential perceived intensity.

    :param hxs: Hex color codes of colors to optimize
    :type hxs: list of str
    :param lightness_base: Additive boost for the lightness parameter
    :type lightness_base: float
    :param saturation_base: Additive boost for the saturation parameter
    :type saturation_base: float
    :param min_intensity: Minimum target perceived intensity
    :type min_intensity: float
    :param max_intensity: Maximum target perceived intensity
    :type max_intensity: float
    :return: Optimized parameters and value of the objective function
    :rtype: tuple of :class:`numpy.ndarray` and float
    '''
    target = np.linspace(min_intensity, max_intensity, len(hxs))
    return find_optimal_params(hxs, target, lightness_base, saturation_base)

def find_optimal_params_diverging(hxs, lightness_base=0,
saturation_base=0, min_intensity=55, max_intensity=85):
    ''' Optimizes the given colors by modifying their lightness
    and saturation to reach a diverging perceived intensity.

    :param hxs: Hex color codes of colors to optimize
    :type hxs: list of str
    :param lightness_base: Additive boost for the lightness parameter
    :type lightness_base: float
    :param saturation_base: Additive boost for the saturation parameter
    :type saturation_base: float
    :param min_intensity: Minimum target perceived intensity
    :type min_intensity: float
    :param max_intensity: Maximum target perceived intensity
    :type max_intensity: float
    :return: Optimized parameters and value of the objective function
    :rtype: tuple of :class:`numpy.ndarray` and float
    '''
    assert len(hxs) % 2 == 1
    tmp = np.linspace(max_intensity, min_intensity, len(hxs) // 2 + 1)
    target = np.concatenate((tmp[::-1], tmp[1:]))
    return find_optimal_params(hxs, target, lightness_base, saturation_base)

def interpolate_sequential(hxs, params, lightness_base, saturation_base):
    ''' Performs an interpolation in intensity space between the given colors,
    assuming the map is sequential.

    :param hxs: Hex color codes of colors to interpolate
    :type hxs: list of str
    :param params: Parameters of the colormap
    :type params: :class:`numpy.ndarray`
    :param lightness_base: Additive boost for the lightness parameter
    :type lightness_base: float
    :param saturation_base: Additive boost for the saturation parameter
    :type saturation_base: float
    :return: Interpolated colors in CAM space
    :rtype: :class:`numpy.ndarray`
    '''
    cam = params_to_cam(params, hxs, lightness_base, saturation_base)

    if np.any(np.diff(cam[:,0]) <= 0):
        warnings.warn('Fixing non-monotonicity in intensity')
        arg = np.argsort(cam[:,0])
        cam = cam[arg]

    x = cam[:,0]
    xx = np.linspace(np.amin(x), np.amax(x), 250)
    return interp.Akima1DInterpolator(x, cam)(xx)

def interpolate_diverging(hxs, params, lightness_base, saturation_base):
    ''' Performs an interpolation in intensity space between the given colors,
    assuming the map is diverging.

    :param hxs: Hex color codes of colors to interpolate
    :type hxs: list of str
    :param params: Parameters of the colormap
    :type params: :class:`numpy.ndarray`
    :param lightness_base: Additive boost for the lightness parameter
    :type lightness_base: float
    :param saturation_base: Additive boost for the saturation parameter
    :type saturation_base: float
    :return: Interpolated colors in CAM space
    :rtype: :class:`numpy.ndarray`
    '''
    assert len(hxs) % 2 == 1
    cam = params_to_cam(params, hxs, lightness_base, saturation_base)
    n = len(hxs)
    N = 250 // (n - 1)

    x = cam[:n//2+1,0]
    xx = np.linspace(np.amin(x), np.amax(x), N * ((n - 1) // 2))

    if len(x) >= 3:
        xx1 = interp.Akima1DInterpolator(x, cam[:n//2+1])(xx)
    else:
        xx1 = interp.interp1d(x, cam[:n//2+1], kind='slinear', axis=0)(xx)

    x = cam[n//2:,0][::-1]
    xx = np.linspace(np.amin(x), np.amax(x), N * ((n - 1) // 2))

    if len(x) >= 3:
        xx2 = interp.Akima1DInterpolator(x, cam[n//2:][::-1])(xx)
    else:
        xx2 = interp.interp1d(x, cam[n//2:][::-1], kind='slinear', axis=0)(xx)

    return np.concatenate((xx1, xx2[::-1]))

def cmap_from_params(name, hxs, params, kind, lightness_base, saturation_base):
    ''' Generates a :mod:`matplotlib` colormap from the given parameters.

    :param name: Name to assign to this colormap
    :type name: str
    :param hxs: Hex color codes of colors to interpolate
    :type hxs: list of str
    :param params: Parameters of the colormap
    :type params: :class:`numpy.ndarray`
    :param kind: One of :code:`sequential`, :code:`diverging`, or :code:`cyclic`
    :type kind: str
    :param lightness_base: Additive boost for the lightness parameter
    :type lightness_base: float
    :param saturation_base: Additive boost for the saturation parameter
    :type saturation_base: float
    :return: The final colormap
    :rtype: :class:`matplotlib.colors.LinearSegmentedColormap`
    '''
    if kind == 'sequential':
        cam = interpolate_sequential(hxs, params, lightness_base, saturation_base)
    elif kind in ['diverging', 'cyclic']:
        cam = interpolate_diverging(hxs, params, lightness_base, saturation_base)
    else:
        raise NotImplementedError

    cs = np.vstack(list(map(revfn, cam)))
    cs = np.clip(cs, 0, 1)
    return colors.LinearSegmentedColormap.from_list(name, cs)

def simulate_cvd(fig, kind, severity):
    ''' Applies the given color vision deficiency to the given figure
    and returns a modified image.

    :param fig: Figure to modify
    :type fig: :class:`matplotlib.figure`
    :param kind: One of :code:`protanomaly`, :code:`deuteranomaly`,
        or :code:`tritanomaly`
    :type kind: str
    :param severity: Severity of the CVD, in the range [0, 100]
    :type severity: float
    :return: An approximate image perceived with this CVD
    :rtype: :class:`PIL.Image.Image`
    '''
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
    ''' Applies a grayscale intensity mapping to the given figure
    and returns a modified image. This is intended to approximate the
    appearance either in black-and-white print media or perceived with
    no color vision whatsoever.

    .. important::

       The returned image has the perceived lightness assigned to the
       red, green, and blue channels equally.

    :param fig: Figure to modify
    :type fig: :class:`matplotlib.figure`
    :return: An approximate image with no color
    :rtype: :class:`PIL.Image.Image`
    '''
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
