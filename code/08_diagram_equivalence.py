"""08_diagram_equivalence.py — five smooth stages

Produces and saves the following figures in figures/:
  - L-to-mirror-smooth-01-original.png
  - L-to-mirror-smooth-02-deformation.png
  - L-to-mirror-smooth-03-deformation.png
  - L-to-mirror-smooth-04-deformation.png
  - L-to-mirror-smooth-05-standard.png

Run from the project folder: python code/08_diagram_equivalence.py
"""
"""Recreate all five transparent PNGs and editable SVGs.

Run from the project folder: python code/08_diagram_equivalence.py
Requires: numpy, scipy, matplotlib. No LaTeX or Manim needed for these exports.
"""
from pathlib import Path
from io import BytesIO
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
"""Five-stage smooth planar deformation, with actual crossing-centered gaps.

The original and final diagrams are unchanged. Smooth periodic filtering
gradually rounds the intermediate curves; there is no change of handedness
or crossing information.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter1d

BLUE, RED = '#1952D9', '#D91F35'
COLORS = (BLUE, RED)
GAP = .072
STAGES = (
    ('L-to-mirror-smooth-01-original', 0.),
    ('L-to-mirror-smooth-02-deformation', .25),
    ('L-to-mirror-smooth-03-deformation', .50),
    ('L-to-mirror-smooth-04-deformation', .75),
    ('L-to-mirror-smooth-05-standard', 1.),
)

@lru_cache(maxsize=1)
def data():
    with np.load(Path(__file__).parent/'geometry/deformation-data.npz') as z:
        return z['start'],z['end']

def curves(alpha):
    """Interpolate the closed curves; round intermediate stages periodically."""
    start,end=data()
    pair=(1-alpha)*start+alpha*end
    sigma=32*np.sin(np.pi*alpha)
    if sigma>1e-7:
        p=gaussian_filter1d(pair[:,:-1],sigma=sigma,axis=1,mode='wrap')
        pair=np.concatenate([p,p[:,:1]],axis=1)
    return pair

def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]

def intersections(a,b,self_cross=False):
    """Polyline crossings. The deduplication tolerance is 1e-7 data units."""
    p=a[:-1];r=np.diff(a,axis=0);q=b[:-1];s=np.diff(b,axis=0)
    den=cross(r[:,None],s[None]);v=q[None]-p[:,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        t=cross(v,s[None])/den;u=cross(v,r[:,None])/den
    good=(abs(den)>1e-14)&(t>=0)&(t<1)&(u>=0)&(u<1)
    if self_cross:
        i,j=np.indices(good.shape)
        good&=(j>i+1)&~((i==0)&(j==len(p)-1))
    result=[]
    for i,j in zip(*np.where(good)):
        xy=p[i]+t[i,j]*r[i]
        if not any(np.linalg.norm(xy-z[2])<1e-7 for z in result):
            result.append((i+t[i,j],j+u[i,j],xy))
    return result

def ordered_crossings(pair):
    """Find each moving crossing in a small cyclic window near its anchor."""
    n=pair.shape[1]-1
    result=[]
    for k in range(4):
        ids=np.arange(k*(n//4)-105,k*(n//4)+106)
        local=pair[:,ids%n]
        found=intersections(*local)
        if len(found)!=1:
            raise ValueError(('Expected one crossing in local window',k,found))
        i,j,xy=found[0]
        result.append(((ids[0]+i)%n,(ids[0]+j)%n,xy))
    return result

def visible_paths(alpha,gap=GAP):
    """Split the underpassing strands to leave clear gaps at actual crossings."""
    pair=curves(alpha)
    crossings=ordered_crossings(pair)
    n=pair.shape[1]-1
    for component,p in enumerate(pair):
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        length=s[-1]
        cuts=[np.interp(c[component],np.arange(n+1),s)
              for k,c in enumerate(crossings) if k%2==component]
        boundaries=[0.,length]
        for cut in cuts:
            for shift in [-length,0,length]:
                boundaries.extend([max(0.,min(length,cut+shift-gap)),
                                   max(0.,min(length,cut+shift+gap))])
        boundaries=sorted(set(boundaries))
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            mid=(lo+hi)/2
            if min(abs((mid-c+length/2)%length-length/2) for c in cuts)<gap:
                continue
            ss=np.r_[lo,s[(s>lo)&(s<hi)],hi]
            xy=np.column_stack([np.interp(ss,s,p[:,j]) for j in range(2)])
            yield component,xy


ROOT=Path(__file__).resolve().parent.parent/'figures';ROOT.mkdir(exist_ok=True)

def render(name,alpha):
    """Frame one stage tightly and export transparent PNG and editable SVG."""
    p=curves(alpha).reshape(-1,2)
    lo=p.min(axis=0)-.065
    hi=p.max(axis=0)+.065
    span=hi-lo
    fig=plt.figure(figsize=span/span.max()*5.4)
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
    ax.axis('off')
    for component,path in visible_paths(alpha):
        ax.plot(*path.T,color=COLORS[component],linewidth=3.0,
                solid_capstyle='round',solid_joinstyle='round')
    for ext in ['png','svg']:
        b=BytesIO()
        fig.savefig(b,format=ext,dpi=360,transparent=True,pad_inches=0)
        out=ROOT/(name+'.'+ext)
        temporary=out.with_suffix(out.suffix+'.tmp')
        with open(temporary,'wb') as stream:
            stream.write(b.getvalue());stream.flush();os.fsync(stream.fileno())
        temporary.replace(out)
    plt.close(fig)

if __name__=='__main__':
    for name,alpha in STAGES:
        # Figure: blue/red link at this stage of the smooth planar deformation.
        render(name,alpha)
        print(name)
