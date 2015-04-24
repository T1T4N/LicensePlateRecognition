from PIL import Image

# Constants for indicating coordinates in a pixel's context
TOP_LEFT = 0
TOP = 1
TOP_RIGHT = 2
LEFT = 3
CENTER = 4
RIGHT = 5
BOTTOM_LEFT = 6
BOTTOM = 7
BOTTOM_RIGHT = 8

# There are eight flags: the cells in the context, skipping the center
context_flag = {}
cur_flag = 1
for i in xrange(9):
    if i == CENTER:
        continue
    context_flag[i] = cur_flag
    cur_flag = cur_flag << 1

# Constants defining how far apart two components of YUV colors must be to be
# considered different
Y_THRESHHOLD = 48
U_THRESHHOLD = 7
V_THRESHHOLD = 6

rgb_yuv_cache = {}  # memoization


def rgb_to_yuv(rgb):
    """Takes a tuple of (r, g, b) and returns a tuple (y, u, v).  Both must be
    24-bit color!

    This is the algorithm from the original hq2x source; it doesn't seem to
    match any other algorithm I can find, but whatever.
    """

    if rgb in rgb_yuv_cache:
        return rgb_yuv_cache[rgb]

    r, g, b = rgb

    y = (r + g + b) >> 2
    u = 128 + ((r - b) >> 2)
    v = 128 + ((-r + g * 2 - b) >> 3)

    rgb_yuv_cache[rgb] = y, u, v
    return y, u, v


def yuv_equal(a, b):
    """Takes two tuples of (y, u, v).  Returns True if they are equal-ish,
    False otherwise.  "Equal-ish" is defined arbitrarily as tolerating small
    differences in the components of the two colors.
    """
    ay, au, av = a
    by, bu, bv = b
    if abs(ay - by) > Y_THRESHHOLD:
        return False
    if abs(au - bu) > U_THRESHHOLD:
        return False
    if abs(av - bv) > V_THRESHHOLD:
        return False

    return True


# Various interpolations of colors done by hq2x
def interpolate(func, *args):
    return tuple(map(func, *args))


def interp1(*args):
    return interpolate(lambda a, b: (a * 3 + b) / 4, *args)


def interp2(*args):
    return interpolate(lambda a, b, c: (a * 2 + b + c) / 4, *args)


def interp5(*args):
    return interpolate(lambda a, b: (a + b) / 2, *args)


def interp6(*args):
    return interpolate(lambda a, b, c: (a * 5 + b * 2 + c) / 8, *args)


def interp7(*args):
    return interpolate(lambda a, b, c: (a * 6 + b + c) / 8, *args)


def interp9(*args):
    return interpolate(lambda a, b, c: (a * 2 + b * 3 + c * 3) / 8, *args)


def interp10(*args):
    return interpolate(lambda a, b, c: (a * 14 + b + c) / 16, *args)


def hq2x(source):
    """Upscales a sprite image using the hq2x algorithm.

    Argument is an Image object containing the source image.  Returns another
    Image object containing the upscaled image.
    """

    w, h = source.size
    mode = source.mode  # XXX use this for the target I guess somehow; palette?
    source = source.convert('RGB')
    dest = Image.new('RGB', (w * 2, h * 2))

    # These give direct pixel access via grid[x, y]
    sourcegrid = source.load()
    destgrid = dest.load()

    # Wrap sourcegrid in a function to cap the coordinates; we need a 3x3 array
    # centered on the current pixel, and factoring out the capping is simpler
    # than a ton of ifs
    def get_px(x, y):
        if x < 0:
            x = 0
        elif x >= w:
            x = w - 1

        if y < 0:
            y = 0
        elif y >= h:
            y = h - 1

        return sourcegrid[x, y]

    for x in xrange(w):
        for y in xrange(h):
            # This is a flattened 3x3 grid with the current pixel in the
            # middle; if the pixel is on an edge, the row/column in the void is
            # just a copy of the edge
            context = [
                get_px(x - 1, y - 1), get_px(x, y - 1), get_px(x + 1, y - 1),
                get_px(x - 1, y), get_px(x, y), get_px(x + 1, y),
                get_px(x - 1, y + 1), get_px(x, y + 1), get_px(x + 1, y + 1),
            ]

            tl, tr, bl, br = hq2x_pixel(context)

            destgrid[x * 2, y * 2] = tl
            destgrid[x * 2 + 1, y * 2] = tr
            destgrid[x * 2, y * 2 + 1] = bl
            destgrid[x * 2 + 1, y * 2 + 1] = br

    return dest


def hq2x_pixel(context):
    """Applies the hq2x algorithm to a single pixel, given the 3x3 context
    around it.  This is the gigantic switch() from the original source.

    Returns the four corresponding pixels to be put in the new image: upper
    left, upper right, lower left, lower right.
    """

    # The massive lookup table is keyed on a bitstring where each bit
    # corresponds to an element in the context.  The top left is 0x1,
    # top middle is 0x2, and so on across and down.  Bits turned on
    # indicate a pixel different from the current pixel
    yuv_context = [rgb_to_yuv(rgb) for rgb in context]
    yuv_px = rgb_to_yuv(context[CENTER])

    pattern = 0
    for bit in xrange(9):
        if bit != CENTER and not yuv_equal(yuv_context[bit], yuv_px):
            pattern = pattern | context_flag[bit]

    if pattern in (0, 1, 4, 32, 128, 5, 132, 160, 33, 129, 36, 133, 164, 161, 37, 165):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (2, 34, 130, 162):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (16, 17, 48, 49):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (64, 65, 68, 69):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (8, 12, 136, 140):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (3, 35, 131, 163):
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (6, 38, 134, 166):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (20, 21, 52, 53):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (144, 145, 176, 177):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern in (192, 193, 196, 197):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern in (96, 97, 100, 101):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (40, 44, 168, 172):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (9, 13, 137, 141):
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (18, 50):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (80, 81):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (72, 76):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (10, 138):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 66:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 24:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (7, 39, 135):
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (148, 149, 180):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern in (224, 228, 225):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern in (41, 169, 45):
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (22, 54):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (208, 209):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (104, 108):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (11, 139):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (19, 51):
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tl = interp1(context[CENTER], context[LEFT])
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tl = interp6(context[CENTER], context[TOP], context[LEFT])
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (146, 178):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
            br = interp1(context[CENTER], context[BOTTOM])
        else:
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
            br = interp6(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
    elif pattern in (84, 85):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            tr = interp1(context[CENTER], context[TOP])
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            tr = interp6(context[CENTER], context[RIGHT], context[TOP])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
    elif pattern in (112, 113):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            bl = interp1(context[CENTER], context[LEFT])
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            bl = interp6(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (200, 204):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
            br = interp1(context[CENTER], context[RIGHT])
        else:
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp6(context[CENTER], context[BOTTOM], context[RIGHT])
    elif pattern in (73, 77):
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            tl = interp1(context[CENTER], context[TOP])
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            tl = interp6(context[CENTER], context[LEFT], context[TOP])
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (42, 170):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
            bl = interp1(context[CENTER], context[BOTTOM])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            bl = interp6(context[CENTER], context[LEFT], context[BOTTOM])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (14, 142):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
            tr = interp1(context[CENTER], context[RIGHT])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            tr = interp6(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 67:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 70:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 28:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 152:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 194:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 98:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 56:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 25:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (26, 31):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (82, 214):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (88, 248):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (74, 107):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 27:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 86:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 216:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 106:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 30:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 210:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 120:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 75:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 29:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 198:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 184:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 99:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 57:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 71:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 156:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 226:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 60:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 195:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 102:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 153:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 58:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 83:
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 92:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 202:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 78:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 154:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 114:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 89:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 90:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (55, 23):
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tl = interp1(context[CENTER], context[LEFT])
            tr = context[CENTER]
        else:
            tl = interp6(context[CENTER], context[TOP], context[LEFT])
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern in (182, 150):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
            br = interp1(context[CENTER], context[BOTTOM])
        else:
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
            br = interp6(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
    elif pattern in (213, 212):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            tr = interp1(context[CENTER], context[TOP])
            br = context[CENTER]
        else:
            tr = interp6(context[CENTER], context[RIGHT], context[TOP])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
    elif pattern in (241, 240):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            bl = interp1(context[CENTER], context[LEFT])
            br = context[CENTER]
        else:
            bl = interp6(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (236, 232):
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
            br = interp1(context[CENTER], context[RIGHT])
        else:
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp6(context[CENTER], context[BOTTOM], context[RIGHT])
    elif pattern in (109, 105):
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            tl = interp1(context[CENTER], context[TOP])
            bl = context[CENTER]
        else:
            tl = interp6(context[CENTER], context[LEFT], context[TOP])
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern in (171, 43):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
            bl = interp1(context[CENTER], context[BOTTOM])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            bl = interp6(context[CENTER], context[LEFT], context[BOTTOM])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (143, 15):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
            tr = interp1(context[CENTER], context[RIGHT])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            tr = interp6(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 124:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 203:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 62:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 211:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 118:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 217:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 110:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 155:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 188:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 185:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 61:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 157:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 103:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 227:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 230:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 199:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 220:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 158:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 234:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 242:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 59:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 121:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 87:
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 79:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 122:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 94:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 218:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 91:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 229:
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 167:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 173:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 181:
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 186:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 115:
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 93:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 206:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern in (205, 201):
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        else:
            bl = interp7(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern in (174, 46):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = interp1(context[CENTER], context[TOP_LEFT])
        else:
            tl = interp7(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (179, 147):
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = interp1(context[CENTER], context[TOP_RIGHT])
        else:
            tr = interp7(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern in (117, 116):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = interp1(context[CENTER], context[BOTTOM_RIGHT])
        else:
            br = interp7(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 189:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 231:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 126:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 219:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 125:
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            tl = interp1(context[CENTER], context[TOP])
            bl = context[CENTER]
        else:
            tl = interp6(context[CENTER], context[LEFT], context[TOP])
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
        tr = interp1(context[CENTER], context[TOP])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 221:
        tl = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            tr = interp1(context[CENTER], context[TOP])
            br = context[CENTER]
        else:
            tr = interp6(context[CENTER], context[RIGHT], context[TOP])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
    elif pattern == 207:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
            tr = interp1(context[CENTER], context[RIGHT])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            tr = interp6(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 238:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
            br = interp1(context[CENTER], context[RIGHT])
        else:
            bl = interp9(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp6(context[CENTER], context[BOTTOM], context[RIGHT])
    elif pattern == 190:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
            br = interp1(context[CENTER], context[BOTTOM])
        else:
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
            br = interp6(context[CENTER], context[RIGHT], context[BOTTOM])
        bl = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 187:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
            bl = interp1(context[CENTER], context[BOTTOM])
        else:
            tl = interp9(context[CENTER], context[LEFT], context[TOP])
            bl = interp6(context[CENTER], context[LEFT], context[BOTTOM])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 243:
        tl = interp1(context[CENTER], context[LEFT])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            bl = interp1(context[CENTER], context[LEFT])
            br = context[CENTER]
        else:
            bl = interp6(context[CENTER], context[BOTTOM], context[LEFT])
            br = interp9(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 119:
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tl = interp1(context[CENTER], context[LEFT])
            tr = context[CENTER]
        else:
            tl = interp6(context[CENTER], context[TOP], context[LEFT])
            tr = interp9(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern in (237, 233):
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern in (175, 47):
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern in (183, 151):
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern in (245, 244):
        tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 250:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 123:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 95:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 222:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 252:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 249:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 235:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp2(context[CENTER], context[TOP_RIGHT], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 111:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[RIGHT])
    elif pattern == 63:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp2(context[CENTER], context[BOTTOM_RIGHT], context[BOTTOM])
    elif pattern == 159:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 215:
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp2(context[CENTER], context[BOTTOM_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 246:
        tl = interp2(context[CENTER], context[TOP_LEFT], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 254:
        tl = interp1(context[CENTER], context[TOP_LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 253:
        tl = interp1(context[CENTER], context[TOP])
        tr = interp1(context[CENTER], context[TOP])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 251:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[TOP_RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 239:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        tr = interp1(context[CENTER], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[RIGHT])
    elif pattern == 127:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp2(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp2(context[CENTER], context[BOTTOM], context[LEFT])
        br = interp1(context[CENTER], context[BOTTOM_RIGHT])
    elif pattern == 191:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM])
        br = interp1(context[CENTER], context[BOTTOM])
    elif pattern == 223:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp2(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[BOTTOM_LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp2(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 247:
        tl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        bl = interp1(context[CENTER], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])
    elif pattern == 255:
        if not yuv_equal(yuv_context[LEFT], yuv_context[TOP]):
            tl = context[CENTER]
        else:
            tl = interp10(context[CENTER], context[LEFT], context[TOP])
        if not yuv_equal(yuv_context[TOP], yuv_context[RIGHT]):
            tr = context[CENTER]
        else:
            tr = interp10(context[CENTER], context[TOP], context[RIGHT])
        if not yuv_equal(yuv_context[BOTTOM], yuv_context[LEFT]):
            bl = context[CENTER]
        else:
            bl = interp10(context[CENTER], context[BOTTOM], context[LEFT])
        if not yuv_equal(yuv_context[RIGHT], yuv_context[BOTTOM]):
            br = context[CENTER]
        else:
            br = interp10(context[CENTER], context[RIGHT], context[BOTTOM])

    return tl, tr, bl, br