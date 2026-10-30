"""Kauffman bracket smoothing diagrams and the complete 16-state calculation.

Produces and saves in figures/:
  - crossing.png, smoothing-A.png, smoothing-B.png
  - jones-L-branching.png
Also saves editable SVG versions of the figures.

Run from the project folder: python code/07_jones_polynomial.py
Each arrow contributes A or A^{-1}; each closed state with r loops contributes
delta^(r-1). The graph calculation below counts loops independently of drawing.
"""
from pathlib import Path
import itertools
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures';OUT.mkdir(exist_ok=True)
INK='#39342F';BLUE='#183B63'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'cm','svg.fonttype':'none'})
raw=np.load(ROOT/'geometry/projected_curves.npz')
curves=[raw['blue'],raw['red']]

def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def intersections(a,b):
 p=a[:-1,:2];r=np.diff(a[:,:2],axis=0);q=b[:-1,:2];s=np.diff(b[:,:2],axis=0)
 d=cross(r[:,None],s[None]);v=q[None]-p[:,None]
 with np.errstate(divide='ignore',invalid='ignore'):
  t=cross(v,s[None])/d;u=cross(v,r[:,None])/d
 result=[]
 for i,j in zip(*np.where((abs(d)>1e-12)&(t>=0)&(t<1)&(u>=0)&(u<1))):
  xy=p[i]+t[i,j]*r[i]
  if not any(np.linalg.norm(xy-c[2])<1e-7 for c in result):result.append((i+t[i,j],j+u[i,j],xy))
 return result

cr=intersections(*curves);assert len(cr)==4
N=len(curves[0])-1
arcs=[np.r_[0,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))] for p in curves]
ports={};port_index={};centers=[];under=[]

def interp(c,indices):
 indices=np.asarray(indices)%N
 return np.column_stack([np.interp(indices,np.arange(N+1),curves[c][:,d]) for d in range(2)])

for k,(i,j,xy) in enumerate(cr):
 centers.append(xy)
 zs=[np.interp(v,np.arange(N+1),curves[c][:,2]) for c,v in enumerate([i,j])]
 under.append(int(zs[1]<zs[0]))
 for c,v in enumerate([i,j]):
  s=arcs[c];position=np.interp(v,np.arange(N+1),s)
  for side,sign in enumerate([-1,1]):
   # Keep crossing neighborhoods disjoint along each component. Otherwise an
   # external arc can incorrectly wrap all the way around the component.
   locations=np.array([np.interp(crossing[c],np.arange(N+1),s) for crossing in cr])
   distances=(locations-position)%s[-1]
   distances=distances[distances>1e-7]
   nearest=min(distances.min(),s[-1]-distances.max())
   radius=min(.30,.32*nearest)
   index=np.interp((position+sign*radius)%s[-1],s,np.arange(N+1))
   pid=4*k+2*c+side;port_index[pid]=index;ports[pid]=interp(c,[index])[0]

# Original curve pieces connect neighboring crossing neighborhoods.
external=[];pieces=[]
for c in range(2):
 order=sorted(range(4),key=lambda k:cr[k][c])
 for previous,next_ in zip(order,order[1:]+order[:1]):
  a=4*previous+2*c+1;b=4*next_+2*c
  lo=port_index[a];hi=port_index[b]
  if hi<=lo:hi+=N
  assert hi-lo<N/2, 'Crossing neighborhoods overlap or an external arc wraps incorrectly'
  external.append((a,b));pieces.append(interp(c,np.linspace(lo,hi,max(20,int(hi-lo)+1))))

pairings=[]
for k,xy in enumerate(centers):
 pids=sorted(range(4*k,4*k+4),key=lambda p:np.arctan2(*(ports[p]-xy)[::-1]))
 pairings.append([[(pids[0],pids[1]),(pids[2],pids[3])],[(pids[1],pids[2]),(pids[3],pids[0])]])

def loops(choices):
 """Count closed state loops by joining the external arcs and local smoothings."""
 parent=list(range(16))
 def root(i):
  while parent[i]!=i:
   parent[i]=parent[parent[i]];i=parent[i]
  return i
 for a,b in external+sum([pairings[k][v] for k,v in enumerate(choices)],[]):parent[root(a)]=root(b)
 return len({root(i) for i in range(16)})

# For this four-crossing projection, the A-resolution is the unique resolution
# into four circles. The B-resolution uses the other local adjacent pairing.
possible=[v for v in itertools.product([0,1],repeat=4) if loops(v)==4]
assert len(possible)==1
A_PAIRING=possible[0]
def resolutions(word):
 """Translate an A/B branch word into the local smoothing at each crossing."""
 return [A_PAIRING[k] if s=='A' else 1-A_PAIRING[k] for k,s in enumerate(word)]

def style(ax):
 ax.set(xlim=(-2.85,2.85),ylim=(-1.95,1.95),aspect='equal');ax.axis('off')

def diagram(ax,word,lw=1.1,gap=.065):
 for p in pieces:ax.plot(*p.T,color=INK,lw=lw,solid_capstyle='round')
 for k,xy in enumerate(centers):
  if k<len(word):
   for a,b in pairings[k][resolutions(word)[k]]:
    # Quadratic rounding joins adjacent ports without introducing a crossing.
    path=MPath([ports[a],xy,ports[b]],[MPath.MOVETO,MPath.CURVE3,MPath.CURVE3])
    ax.add_patch(PathPatch(path,facecolor='none',edgecolor=INK,lw=lw,capstyle='round'))
  else:
   for c in range(2):
    a=4*k+2*c;b=a+1;lo=port_index[a];hi=port_index[b]
    if hi<lo:hi+=N
    ids=np.linspace(lo,hi,60);p=interp(c,ids)
    if c==under[k]:
     p[np.linalg.norm(p-xy,axis=1)<gap]=np.nan
    ax.plot(*p.T,color=INK,lw=lw,solid_capstyle='round')
 style(ax)

def contribution(word):
 r=loops(resolutions(word));power=word.count('A')-word.count('B')
 a='' if power==0 else ('A' if power==1 else 'A^{'+str(power)+'}')
 d='' if r==1 else (r'\delta' if r==2 else r'\delta^{'+str(r-1)+'}')
 return a+d or '1'

def save(fig,name,dpi=300):
 for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=dpi,transparent=True,pad_inches=0)
 plt.close(fig)

def icon(name):
 fig=plt.figure(figsize=(1,1));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal');ax.axis('off')
 if name=='crossing':
  paths=[np.array([[-.75,.75],[.75,-.75]]),np.array([[-.75,-.75],[-.16,-.16]]),np.array([[.16,.16],[.75,.75]])]
  for p in paths:ax.plot(*p.T,color=INK,lw=4,solid_capstyle='round')
 else:
  t=np.linspace(-.75,.75,100)
  for sign in [-1,1]:
   x=t;y=sign*(.32+.72*t*t)
   if name=='smoothing-B':x,y=y,x
   ax.plot(x,y,color=INK,lw=4,solid_capstyle='round')
 save(fig,name,dpi=256)

def tree():
 import cairosvg
 layout=json.loads((ROOT/'geometry/tree_layout.json').read_text())
 width,height=layout['width'],layout['height']
 all_points=np.concatenate([p[:,:2] for p in curves])
 low=all_points.min(axis=0);high=all_points.max(axis=0)
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
 for node in layout['nodes']:
  word=node['state'];x0,y0,x1,y1=node['box']
  stroke=.85*layout['stroke_widths'][len(word)]
  # A small inset leaves room for the round end caps at the diagram's bounds.
  inset=stroke/2
  def screen(p):
   p=np.asarray(p)
   x=x0+inset+(p[...,0]-low[0])/(high[0]-low[0])*(x1-x0-2*inset)
   y=y1-inset-(p[...,1]-low[1])/(high[1]-low[1])*(y1-y0-2*inset)
   return np.stack([x,y],axis=-1)
  def polyline(p):
   chunks=[];run=[]
   for pt in screen(p):
    if not np.isfinite(pt).all():
     if run:chunks.append(run);run=[]
    else:run.append(pt)
   if run:chunks.append(run)
   return ' '.join('M'+' L'.join(f'{x:.4f},{y:.4f}' for x,y in chunk) for chunk in chunks)
  parts.append(f'<g id="state-{word or "root"}" fill="none" stroke="{INK}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">')
  for p in pieces:parts.append(f'<path d="{polyline(p)}"/>')
  for k,xy in enumerate(centers):
   if k<len(word):
    for aa,bb in pairings[k][resolutions(word)[k]]:
     va=ports[aa]-xy;vb=ports[bb]-xy
     va/=np.linalg.norm(va);vb/=np.linalg.norm(vb)
     mid=va+vb
     mid/=np.linalg.norm(mid)
     # Broad-angle smoothings need visible separation at the old crossing.
     # Join ports within their own sectors, away from the old crossing,
     # so thick strokes do not merge into an apparent new crossing.
     control=xy.copy()
     if np.dot(va,vb)<0:
      control=(ports[aa]+ports[bb])/2
     p0,p1,p2=screen([ports[aa],control,ports[bb]])
     parts.append(f'<path d="M{p0[0]:.4f},{p0[1]:.4f} Q{p1[0]:.4f},{p1[1]:.4f} {p2[0]:.4f},{p2[1]:.4f}"/>')
   else:
    for c in range(2):
     aa=4*k+2*c;bb=aa+1;lo=port_index[aa];hi=port_index[bb]
     if hi<lo:hi+=N
     p=interp(c,np.linspace(lo,hi,80))
     if c==under[k]:p[np.linalg.norm(p-xy,axis=1)<.08]=np.nan
     parts.append(f'<path d="{polyline(p)}"/>')
  parts.append('</g>')
 # Figure: curved branch arrows, A/A^{-1} weights and terminal contributions.
 # Explicit vector outlines preserve the typography and layout of this figure.
 d=' '.join('M'+' L'.join(f'{x},{y}' for x,y in p)+' Z' for p in layout['navy_paths'])
 parts.append(f'<path fill="{BLUE}" fill-rule="evenodd" d="{d}"/>')
 parts.append('</svg>')
 svg=''.join(parts)
 (OUT/'jones-L-branching.svg').write_text(svg)
 cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/'jones-L-branching.png'))

def validate_rendered_states():
 """Check the actual final diagrams, including stroke separation at export size."""
 from PIL import Image
 from scipy.ndimage import label, binary_fill_holes
 rgba=np.asarray(Image.open(OUT/'jones-L-branching.png').convert('RGBA'))
 palette=np.array([[57,52,47],[24,59,99]])
 nearest=((rgba[:,:,:3].astype(float)[:,:,None]-palette[None,None])**2).sum(-1).argmin(-1)
 ink=(rgba[:,:,3]>127)&(nearest==0)
 layout=json.loads((ROOT/'geometry/tree_layout.json').read_text())
 checks=[]
 for node in layout['nodes']:
  word=node['state']
  if len(word)!=4:continue
  x0,y0,x1,y1=node['box']
  mask=np.pad(ink[y0-3:y1+3,x0-3:x1+3],4)
  _,components=label(mask)
  _,holes=label(binary_fill_holes(mask)&~mask)
  expected=loops(resolutions(word))
  assert components==holes==expected,(word,components,holes,expected)
  checks.append({'state':word,'visible_closed_loops':components,'expected':expected})
 print('PASS: all 16 rendered states have exactly the expected number of separate closed loops.')

def calculation():
 """Check the 16 state weights and simplify the bracket using integer arithmetic."""
 terms=Counter();rows=[]
 for chars in itertools.product('AB',repeat=4):
  word=''.join(chars);r=loops(resolutions(word));a=word.count('A')-word.count('B')
  terms[(a,r-1)]+=1
  rows.append({'state':word,'A_exponent':a,'loops':r,'delta_exponent':r-1})
 expected={(4,3):1,(2,2):4,(0,1):6,(-2,0):4,(-4,1):1}
 assert dict(terms)==expected
 polynomial=Counter()
 for (a,d),multiplicity in terms.items():
  p=Counter({a:multiplicity})
  for _ in range(d):
   q=Counter()
   for k,v in p.items():q[k+2]-=v;q[k-2]-=v
   p=q
  for k,v in p.items():polynomial[k]+=v
 polynomial={k:v for k,v in polynomial.items() if v}
 assert polynomial=={10:-1,6:1,2:-1,-6:-1},polynomial
 print('PASS: all 16 loop counts, delta sum, bracket simplification and Jones normalization.')

from pathlib import Path
from io import BytesIO
import json
import cairosvg
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures'
OUT.mkdir(exist_ok=True)
DRAWINGS=json.loads((ROOT/'geometry'/'07_rule_drawings.json').read_text())

def render_figure(name):
    """Draw the named diagram from its saved vector coordinates.

    Each layer stores filled outlines for curves, labels, arrows or gaps.
    The even-odd fill rule keeps interior holes open. CairoSVG rasterizes
    the same SVG to a transparent PNG; no existing PNG is loaded.
    """
    # Read this figure's canvas size, colors and ordered outline layers.
    drawing=DRAWINGS[name]
    width,height=drawing['width'],drawing['height']
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for layer in drawing['layers']:
        path=' '.join('M'+' L'.join(f'{x},{y}' for x,y in p)+' Z' for p in layer['paths'])
        color=layer['color']
        opacity=layer.get('opacity',1.)
        parts.append(f'<path d="{path}" fill="{color}" fill-opacity="{opacity}" fill-rule="evenodd"/>')
    parts.append('</svg>')
    # Save the vector drawing and rasterize it at its original canvas size.
    svg=''.join(parts)
    (OUT/(name+'.svg')).write_text(svg)
    rendered=cairosvg.svg2png(bytestring=svg.encode())
    image=Image.open(BytesIO(rendered)).convert('RGBA')
    metadata={}
    if drawing['dpi']:metadata['dpi']=tuple(drawing['dpi'])
    image.save(OUT/(name+'.png'),**metadata)
    print('Saved',name+'.png')

if __name__=='__main__':
 # Figure: one crossing and its A and B smoothings for an inline equation.
 for name in ['crossing','smoothing-A','smoothing-B']:render_figure(name)
 # Figure: horizontal branching tree, A on the left and A^{-1} on the right.
 calculation();tree();validate_rendered_states()
