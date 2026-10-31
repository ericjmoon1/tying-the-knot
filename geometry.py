"""Numerical section geometry used by the existing poster model.

The geometry definitions below are kept equal to the poster source. They need
NumPy and SciPy, and can be evaluated without a graphics context.
"""
from pathlib import Path
import numpy as np
"""Compact band with four travelling face turnovers and thin raised creases.
Rounded rectangular section with material coordinates transported around the band.
"""
import numpy as np
RADIUS=2.20
HALF_WIDTH=.60
HALF_THICKNESS=.16
EDGE_RADIUS=.018
TURNOVER_RELIEF=.055
RIM_SHAVE_DEPTH=.09
RIM_SHAVE_HEIGHT=.10
HANDEDNESS=1
# Small face/rim refinements; both vanish on the two defining link curves.
FACE_RELIEF=.030
RIM_WAVE=.065
CONTOUR_PHASE=.70

def twist(theta):
    return 2*HANDEDNESS*np.asarray(theta)

# Fixed rounded rectangle in radial/axial (x,z) coordinates. True straight
# wall segments; only the edges have a small circular rounding.
def _perimeter():
    d,w,r=HALF_THICKNESS,HALF_WIDTH,EDGE_RADIUS
    xz=[]
    for cx,cz,a in [(d-r,w-r,0),(-d+r,w-r,np.pi/2),
                    (-d+r,-w+r,np.pi),(d-r,-w+r,3*np.pi/2)]:
        b=np.linspace(a,a+np.pi/2,257)
        xz.extend(np.c_[cx+r*np.cos(b),cz+r*np.sin(b)])
    xz=np.array(xz);xz=np.vstack([xz,xz[:1]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xz,axis=0),axis=1))]
    # Place zero at the centre of the outside wall, rather than a corner.
    zero=s[-1]-(w-r)
    return s,s[-1],zero,xz
_ARC,_LENGTH,_ZERO,_XZ=_perimeter()

def base_section(alpha):
    arc=(_ZERO+float(alpha)/(2*np.pi)*_LENGTH)%_LENGTH
    return np.interp(arc,_ARC,_XZ[:,0]),np.interp(arc,_ARC,_XZ[:,1])

def _crease(theta):
    # Two opposite material edges travel around the section. Select the
    # outside edge; the inside edge is its centrally opposite counterpart.
    x,z=base_section(twist(theta)+np.pi/2)
    if x<0:x,z=-x,-z
    return z,(x/HALF_THICKNESS)**2

def _tent(z,peak):
    # Two broad almost-planar walls meet at a thin ridge, not a hollow.
    # The endpoints stay on the original band envelope.
    w=HALF_WIDTH
    if z<=peak:return max(0.,min(1.,(z+w)/max(peak+w,1e-12)))
    return max(0.,min(1.,(w-z)/max(w-peak,1e-12)))

def section(theta,beta):
    x,z=base_section(float(beta)+twist(theta))
    peak,strength=_crease(theta)
    target=peak if x>=0 else -peak
    # Relief is outward on both walls. There are no concave dimples or
    # axial height functions, and the major wall shape remains cylindrical.
    relief=TURNOVER_RELIEF*strength*_tent(z,target)*(abs(x)/HALF_THICKNESS)**4
    # Straight, shallow edge shaving where the turnover reaches the rim.
    # It only affects the edge zone, not the broad wall or band height.
    edge_zone=max(0.,(abs(z)-(HALF_WIDTH-RIM_SHAVE_HEIGHT))/RIM_SHAVE_HEIGHT)
    shave=RIM_SHAVE_DEPTH*(1-strength)*edge_zone*abs(x)/HALF_THICKNESS
    x+=np.sign(x)*(relief-shave)
    return x,z

def point(theta,beta):
    x,z=section(theta,beta)
    return np.array([(RADIUS+x)*np.cos(theta),(RADIUS+x)*np.sin(theta),z])

def face_coordinates(theta,beta):
    # Literal material labels: independent of theta and camera position.
    # Geometry transports these coordinates around the compact section.
    return HALF_WIDTH*np.sin(beta),HALF_THICKNESS*np.cos(beta)


TAU=2*np.pi
def perimeter(a):
    arc=(_ZERO+np.asarray(a)/TAU*_LENGTH)%_LENGTH
    return np.interp(arc,_ARC,_XZ[:,0]),np.interp(arc,_ARC,_XZ[:,1])

def compact_section(theta,beta,n):
    a=HANDEDNESS*n*np.asarray(theta)/2
    x,z=perimeter(a+beta)
    if n==0:return x,z
    px,pz=perimeter(a+np.pi/2)
    pz=np.where(px<0,-pz,pz); strength=(px/HALF_THICKNESS)**2
    peak=np.where(x>=0,pz,-pz)
    tent=np.where(z<=peak,(z+HALF_WIDTH)/np.maximum(peak+HALF_WIDTH,1e-12),
                  (HALF_WIDTH-z)/np.maximum(HALF_WIDTH-peak,1e-12))
    relief=TURNOVER_RELIEF*strength*np.clip(tent,0,1)*(abs(x)/HALF_THICKNESS)**4
    zone=np.maximum(0,(abs(z)-(HALF_WIDTH-RIM_SHAVE_HEIGHT))/RIM_SHAVE_HEIGHT)
    shave=RIM_SHAVE_DEPTH*(1-strength)*zone*abs(x)/HALF_THICKNESS
    return x+np.sign(x)*(relief-shave),z


def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-14)
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
# Smooth periodic material coordinates, preserving closure and the band shape.
_NT,_NB=2048,512
_tt,_bb=np.meshgrid(np.arange(_NT)*2*np.pi/_NT,np.arange(_NB)*2*np.pi/_NB,indexing="ij")
_x,_z=compact_section(_tt,_bb,4)
_fields=[spline_filter(gaussian_filter(v,(10,3),mode="wrap"),order=3,mode="grid-wrap") for v in [_x,_z]]
def ring_points(t,b,n=4):
 """Evaluate the closed refined surface in periodic material coordinates."""
 t,b=np.broadcast_arrays(t,b)
 coords=np.array([(t%(2*np.pi))*_NT/(2*np.pi),(b%(2*np.pi))*_NB/(2*np.pi)])
 x,z=[map_coordinates(v,coords,order=3,mode="grid-wrap",prefilter=False) for v in _fields]
 return np.stack([(RADIUS+x)*np.cos(t),(RADIUS+x)*np.sin(t),z],axis=-1)


def section_coordinates(theta, beta):
    """Radial displacement and height of the existing smoothed model."""
    theta, beta = np.broadcast_arrays(theta, beta)
    coords = np.array([(theta % TAU)*_NT/TAU, (beta % TAU)*_NB/TAU])
    return tuple(map_coordinates(v, coords.reshape(2,-1), order=3, mode='grid-wrap',
                                 prefilter=False).reshape(theta.shape) for v in _fields)


def lifted_angle(x, z, reference):
    """Continuous polar angle close to a known winding reference.

    atan2 returns only a principal angle. Subtracting the reference direction
    before atan2, then adding it back, keeps the full turns in the labels.
    """
    c, s = np.cos(reference), np.sin(reference)
    return reference + np.arctan2(z*c-x*s, x*c+z*s)


def thick_ribbon(theta, u, v):
    """Actual rotated rectangular solid; v=0 gives the simplified ribbon S."""
    theta, u, v = np.broadcast_arrays(theta, u, v)
    phi = twist(theta)
    x = -u*np.sin(phi) + v*np.cos(phi)
    z = u*np.cos(phi) + v*np.sin(phi)
    return np.stack([(RADIUS+x)*np.cos(theta),
                     (RADIUS+x)*np.sin(theta), z], axis=-1)


def construction_points(theta, beta, progress):
    """Move the boundary of the thick ribbon to the refined compact model.

    progress=0: rounded rectangular cross-section rotated through 2*theta.
    progress=1: the refined surface with the same fixed link curves.
    Intermediate stages interpolate section polar angles and radii, keeping
    section points in their circular order rather than clipping them together.
    This is reshaping, not a material-removal or manufacturing simulation.
    """
    theta, beta = np.broadcast_arrays(theta, beta)
    if not 0 <= progress <= 1:
        raise ValueError('progress must be between 0 and 1')
    if progress == 1:
        if theta.ndim == 0:
            return refined_points(theta, beta)
        return refined_points(theta, beta)
    v, u = perimeter(beta)
    phi = twist(theta)
    first_angle = lifted_angle(v, u, beta) + phi
    first_radius = np.hypot(v, u)
    x, z = refined_section_coordinates(theta, beta)
    final_angle = lifted_angle(x, z, beta+phi)
    final_radius = np.hypot(x, z)
    angle = (1-progress)*first_angle + progress*final_angle
    radius = (1-progress)*first_radius + progress*final_radius
    x, z = radius*np.cos(angle), radius*np.sin(angle)
    return np.stack([(RADIUS+x)*np.cos(theta),
                     (RADIUS+x)*np.sin(theta), z], axis=-1)


def refined_section_coordinates(theta, beta, amount=None, rim_amount=None):
    """Add modest face relief and wavy rims without moving the link curves.

    The cos(beta)^2 gate vanishes at both fixed-beta interface curves. The
    periodic cosine varies four times around the band. The height term is
    concentrated near the top and bottom using (z/w)^3.
    """
    theta, beta = np.broadcast_arrays(theta, beta)
    x, z = section_coordinates(theta, beta)
    amount = FACE_RELIEF if amount is None else amount
    rim_amount = RIM_WAVE if rim_amount is None else rim_amount
    gate = np.cos(beta)**2
    # Set the two exactly represented interface labels to zero also in floats.
    gate = np.where((beta % np.pi) == np.pi/2, 0., gate)
    wave = np.cos(4*theta+CONTOUR_PHASE)
    dx = amount*gate*np.tanh(x/.04)*wave
    dz = rim_amount*gate*(z/HALF_WIDTH)**3*wave
    return x+dx, z+dz


def face_adjusted_points(theta, beta, amount=0., rim_amount=0.):
    """Evaluate optional adjustments; zero strengths return the old geometry."""
    theta, beta = np.broadcast_arrays(theta, beta)
    # Add displacements to the old evaluator, retaining its rounding at K1,K2.
    original = (ring_points(theta[None], beta[None])[0] if theta.ndim == 0
                else ring_points(theta, beta))
    x, z = section_coordinates(theta, beta)
    xn, zn = refined_section_coordinates(theta, beta, amount, rim_amount)
    dx, dz = xn-x, zn-z
    return original + np.stack([dx*np.cos(theta),dx*np.sin(theta),dz],axis=-1)


def refined_points(theta, beta):
    """The selected surface with small noncylindrical face and rim contours."""
    return face_adjusted_points(theta, beta, FACE_RELIEF, RIM_WAVE)
