"""Synthetic SEN0628 / VL53L7CX frames from a scene description.

Behaviours reproduced from the 2026-09-26/27 recordings:
  - 8x8 zones, 60 deg square field, zone half-angle 3.75 deg
  - perpendicular distance reported (wall reads flat apart from tilt)
  - solid-surface noise ~5 mm (8 mm under 200 mm)
  - zones straddling an edge: intermediate value, extra jitter, dropouts
  - invalid = 4000, min range 20 mm
  - optional wall tilt ramp (measured ~35 mm across the field at 33 cm)

Scene units: mm. Camera view: x to the sensor's left->right, y top->bottom,
origin on the optical axis.
"""
import numpy as np

INVALID = 4000
FOV_DEG = 60.0
NZ = 8
SUB = 12                      # sub-rays per zone edge (SUB*SUB per zone)

class Box:
    """Rectangular block standing off the wall. x,y = centre of its face (mm), w,h = face size, depth = stand-off from wall."""
    def __init__(self, x, y, w, h, depth):
        self.x, self.y, self.w, self.h, self.depth = x, y, w, h, depth
    def covers(self, X, Y):
        return (np.abs(X - self.x) <= self.w/2) & (np.abs(Y - self.y) <= self.h/2)

def zone_rays():
    """Tangent-plane coordinates of sub-rays for every zone: arrays (8,8,SUB,SUB)."""
    half = np.tan(np.radians(FOV_DEG/2))
    edges = np.linspace(-half, half, NZ+1)
    tx = np.empty((NZ, SUB)); 
    for i in range(NZ):
        tx[i] = edges[i] + (np.arange(SUB)+0.5)/SUB * (edges[i+1]-edges[i])
    TX = tx[None, :, None, :]          # (1, col, 1, sub)
    TY = tx[:, None, :, None]          # (row, 1, sub, 1)
    return np.broadcast_to(TX, (NZ,NZ,SUB,SUB)), np.broadcast_to(TY, (NZ,NZ,SUB,SUB))

_TX, _TY = zone_rays()

def render_ideal(wall, boxes=(), tilt_x=0.0, tilt_y=0.0):
    """Per-sub-ray perpendicular depth and per-zone coverage stats.
    tilt_x/tilt_y: mm of extra wall distance per mm of lateral offset (small)."""
    # depth of wall along each sub-ray: intersect z = wall + tilt terms
    Z = np.full(_TX.shape, float(wall))
    X = _TX * Z; Y = _TY * Z
    Z = wall + tilt_x * X + tilt_y * Y
    X = _TX * Z; Y = _TY * Z
    for b in boxes:
        zf = wall - b.depth
        Xf, Yf = _TX * zf, _TY * zf            # where the ray crosses the face plane
        hit = b.covers(Xf, Yf)
        Z = np.where(hit, zf, Z)
    return Z

def frames(wall, boxes=(), n=30, tilt_x=0.0, tilt_y=0.0, sigma=5.0, mix_jitter=20.0, flicker=0.15, rng=None):
    """Return (n,8,8) int16 frames."""
    rng = np.random.default_rng(rng)
    Z = render_ideal(wall, boxes, tilt_x, tilt_y)                  # (8,8,SUB,SUB)
    zmean = Z.mean(axis=(2,3))
    zmin, zmax = Z.min(axis=(2,3)), Z.max(axis=(2,3))
    mixed = (zmax - zmin) > 15.0                                   # two surfaces in the zone
    # coverage fraction of the nearer surface
    f = ((Z - zmin[...,None,None]) < 15.0).mean(axis=(2,3))
    mixness = 4*f*(1-f)                                            # 0 solid .. 1 half/half
    # per-zone dropout propensity for mixed zones (observed 0..90% of frames)
    p_drop = np.where(mixed, mixness * rng.uniform(0.0, 0.9, (NZ,NZ)), 0.0)
    out = np.empty((n, NZ, NZ))
    for k in range(n):
        s = np.where(zmean < 200, 8.0, sigma) + mix_jitter * mixness   # extra jitter on straddlers
        v = zmean + rng.normal(0, 1, (NZ,NZ)) * s
        # straddlers flicker toward one of the two surfaces
        pick = rng.uniform(size=(NZ,NZ))
        v = np.where(mixed & (pick < flicker), zmin + rng.normal(0,sigma,(NZ,NZ)), v)
        v = np.where(mixed & (pick > 1-flicker), zmax + rng.normal(0,sigma,(NZ,NZ)), v)
        drop = rng.uniform(size=(NZ,NZ)) < p_drop
        v = np.where(drop | (v < 20) | (v >= INVALID), INVALID, v)
        out[k] = v
    return out.astype(np.int16)

def zone_mm(distance):
    """Width of one zone (mm) at a given distance."""
    return 2*distance*np.tan(np.radians(FOV_DEG/2)) / NZ

if __name__ == "__main__":
    import grid
    np.set_printoptions(linewidth=120)
    # reproduce beacon8: wall ~275 mm, box 69x81 mm face, ~60 mm deep, centred left/up of axis
    z = zone_mm(275 - 60)
    fr = frames(275, [Box(x=-0.9*z, y=-0.6*z, w=69, h=81, depth=60)], n=30, tilt_x=0.03, rng=1)
    m, s, nv = grid.average_frames(fr)
    print("synthetic 8x8 mean:"); print(np.round(m).astype(int))
    print("synthetic 8x8 std:");  print(np.round(s).astype(int))
    real = np.load("beacon8.npy"); rm, rs, _ = grid.average_frames(real)
    print("real beacon8 mean:");  print(np.round(rm).astype(int))
    print("real beacon8 std:");   print(np.round(rs).astype(int))
