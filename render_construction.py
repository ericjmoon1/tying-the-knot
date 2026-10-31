"""Render the thick ribbon, reshaping stages, and the refined final ring.

Run: python code/render_construction.py
Produces transparent PNGs in figures/:
  01-thick-twisted-ribbon.png -- a genuinely rotated rounded section.
  02-reshaping.png           -- halfway through the section deformation.
  03-compact-ring.png        -- the adjusted compact model with fixed link curves.
  04-compact-front.png       -- the same final geometry from a lower view.
  05-compact-top.png         -- the same final geometry from above.
  06-compact-with-curves.png -- the unchanged blue and red interface curves.
  construction-overview.png  -- all six views with descriptive captions.
  contour-comparison.png     -- old/new contours from two matching views.
  baseline-compact.png and baseline-front.png -- inputs for that comparison.

Uses the existing poster renderer and its gold materials. Only the geometric
function passed to the renderer changes between construction stages.
"""
from pathlib import Path
import importlib.util
from io import BytesIO
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import geometry as model_geometry
from geometry import construction_points

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('poster_renderer',ROOT/'03_modeling_ring.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.prog['surface_opacity'].value=1.
m.prog['simplified'].value=0

def save(name):
    """Tightly frame the complete silhouette with a little transparent padding."""
    im=m.grab();box=im.getchannel('A').getbbox()
    if box is None:raise RuntimeError('empty render')
    left,top,right,bottom=box;pad=80
    im=im.crop((max(0,left-pad),max(0,top-pad),min(m.W,right+pad),min(m.H,bottom+pad)))
    im.thumbnail((2300,1800),Image.Resampling.LANCZOS)
    data=BytesIO();im.save(data,format='PNG');p=m.OUT/(name+'.png');tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(data.getvalue());f.flush();os.fsync(f.fileno())
    tmp.replace(p);print('Saved',name,flush=True)

def surface(progress,polar,name):
    m.set_camera(polar);m.fb.clear(0,0,0,0,depth=1);m.prog['diagnostic_blend'].value=0
    m.draw(lambda t,b:construction_points(t,b,progress),m.face_coordinates,nt=1201,nb=257)
    save(name)

def overview():
    """Figure: arrange all six generated views with descriptive captions."""
    names=['01-thick-twisted-ribbon','02-reshaping','03-compact-ring',
           '04-compact-front','05-compact-top','06-compact-with-curves']
    labels=['Thick ribbon: actual rotating section','Reshaping halfway',
            'Adjusted compact model','Lower view','View from above',
            'Same defining curves']
    canvas=Image.new('RGB',(1800,1040),'white');pen=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',22)
    except OSError:font=ImageFont.load_default(size=22)
    for i,(name,label) in enumerate(zip(names,labels)):
        im=Image.open(m.OUT/(name+'.png')).convert('RGBA');im.thumbnail((550,440))
        x=(i%3)*600+(600-im.width)//2;y=(i//3)*520+20+(440-im.height)//2
        canvas.paste(im,(x,y),im);pen.text(((i%3)*600+20,(i//3)*520+475),label,font=font,fill='#222')
    canvas.save(m.OUT/'construction-overview.png')

def comparison():
    """Figure: compare old and selected contours at identical camera settings."""
    amounts=model_geometry.FACE_RELIEF,model_geometry.RIM_WAVE
    model_geometry.FACE_RELIEF=0.;model_geometry.RIM_WAVE=0.
    surface(1.,50,'baseline-compact')
    surface(1.,78,'baseline-front')
    model_geometry.FACE_RELIEF,model_geometry.RIM_WAVE=amounts
    canvas=Image.new('RGB',(1500,1000),'white');pen=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',24)
    except OSError:font=ImageFont.load_default(size=24)
    names=['baseline-compact','03-compact-ring','baseline-front','04-compact-front']
    for i,name in enumerate(names):
        im=Image.open(m.OUT/(name+'.png')).convert('RGBA');im.thumbnail((690,420))
        x=(i%2)*750+(750-im.width)//2;y=(i//2)*500+20+(420-im.height)//2
        canvas.paste(im,(x,y),im)
        pen.text(((i%2)*750+25,(i//2)*500+460),
                 'Previous contours' if i%2==0 else 'Adjusted contours',font=font,fill='#222')
    canvas.save(m.OUT/'contour-comparison.png')

if __name__=='__main__':
    # Figure: an actual rounded rectangular section rotating with the ribbon.
    surface(0.,50,'01-thick-twisted-ribbon')
    # Figure: the same material coordinates halfway through compact reshaping.
    surface(.5,50,'02-reshaping')
    # Figure: adjusted faces and rims, retaining the defining link curves.
    surface(1.,50,'03-compact-ring')
    # Figures: alternate views for comparison with the supplied photographs.
    surface(1.,78,'04-compact-front')
    surface(1.,12,'05-compact-top')
    # Figure: the same final model with the same two colored interface curves.
    m.set_camera(58);m.fb.clear(0,0,0,0,depth=1)
    m.prog['diagnostic_blend'].value=0
    m.draw(lambda t,b:construction_points(t,b,1.),m.face_coordinates,nt=1201,nb=257)
    m.prog['diagnostic_blend'].value=1
    for side in [1,-1]:
        def edge(t):
            p=m.accepted_edge(t,side)
            center=np.stack([m.RADIUS*np.cos(t),m.RADIUS*np.sin(t),np.zeros_like(t)],axis=-1)
            return p+.026*m.unit(p-center)
        def tube(t,b):
            p=edge(t);dt=1e-4;T=m.unit(edge(t+dt)-edge(t-dt))
            radial=np.stack([np.cos(t),np.sin(t),np.zeros_like(t)],axis=-1)
            A=m.unit(radial-np.sum(radial*T,axis=-1,keepdims=True)*T);B=m.unit(np.cross(T,A))
            return p+.028*(np.cos(b)[...,None]*A+np.sin(b)[...,None]*B)
        m.draw(tube,lambda t,b:(np.zeros_like(t),np.full_like(t,m.HALF_THICKNESS*side)),nb=17)
    save('06-compact-with-curves')

    overview()

    comparison()
