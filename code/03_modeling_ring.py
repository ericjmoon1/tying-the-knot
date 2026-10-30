"""03_modeling_ring.py — parametric ribbon and refined ring

Produces and saves the following figures in figures/:
  - ribbon-construction.png
  - simplified-ribbon-matte.png
  - ring-solid.png
  - ring-curves.png
  - link-camera-smooth.png

Run from the project folder: python code/03_modeling_ring.py
"""
"""Render the original Manim gold shader offscreen through EGL, no window required.
Builds the surface mesh and renders its gold materials on a transparent background.
"""
from pathlib import Path
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
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
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures';OUT.mkdir(exist_ok=True)
# Render at high resolution before downsampling for smooth transparent edges.
W,H=6000,4800
try:
    ctx=moderngl.create_standalone_context(backend='egl')
except Exception:
    ctx=moderngl.create_standalone_context()
vertex='''#version 330
uniform mat3 camera_rotation;
uniform vec2 frame_shape;
in vec3 point; in vec3 normal; in vec2 material;
out vec3 model_point; out vec3 world_normal; out vec3 view_point;
out vec4 vertex_color; out vec2 material_coordinates;
void main(){model_point=point;world_normal=normal;material_coordinates=material;
 view_point=camera_rotation*point;vertex_color=vec4(1.);
 gl_Position=vec4(view_point.xy*2./frame_shape,-view_point.z*.1,1.);}
'''
frag=r'''
#version 330
uniform mat3 camera_rotation;
uniform float focal_distance;
uniform float ring_radius;
uniform float ring_thickness;
uniform float ring_width;
uniform float clay_preview;
uniform float diagnostic_blend;
uniform float single_face;
uniform float cut_progress;
uniform float surface_opacity;
uniform float simplified;
in vec3 model_point;
in vec2 material_coordinates;
in vec3 world_normal;
in vec3 view_point;
in vec4 vertex_color;
out vec4 frag_color;
float softbox(vec3 r,vec3 l,vec3 axis,float rough){
    vec3 side=normalize(cross(l,axis)),up=normalize(cross(side,l));
    float edge=.012+rough*.30;
    return smoothstep(0.,.2,dot(r,l))
        *(1.-smoothstep(.28-edge,.28+edge,abs(dot(r,side))))
        *(1.-smoothstep(.65-edge,.65+edge,abs(dot(r,up))));
}
vec3 studio(vec3 r,float rough){
    vec3 e=mix(vec3(.66,.67,.70),vec3(.10,.11,.13),smoothstep(-.7,.9,r.z));
    e+=vec3(5.0)*softbox(r,normalize(vec3(-.5,-1.,-.25)),vec3(0,0,1),rough);
    e+=vec3(3.0)*softbox(r,normalize(vec3(.65,.3,.5)),vec3(0,0,1),rough);
    e*=1.-.87*softbox(r,normalize(vec3(.35,-1.,-.12)),vec3(0,0,1),rough);
    return e;
}
vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
float signedPower(float x,float p){return sign(x)*pow(abs(x),p);}
float grain(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float etchNoise(vec3 p){
    vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
    return mix(mix(mix(grain(i),grain(i+vec3(1,0,0)),f.x),
                   mix(grain(i+vec3(0,1,0)),grain(i+vec3(1,1,0)),f.x),f.y),
               mix(mix(grain(i+vec3(0,0,1)),grain(i+vec3(1,0,1)),f.x),
                   mix(grain(i+vec3(0,1,1)),grain(i+vec3(1,1,1)),f.x),f.y),f.z);
}
void main(){
    vec3 V=normalize(transpose(camera_rotation)*(vec3(0,0,focal_distance)-view_point));
    vec3 N=normalize(world_normal);
    if(dot(N,V)<0.)N=-N;
    float v=material_coordinates.y;
    if(simplified>.5) v=dot(normalize(world_normal),V)>=0.?ring_thickness:-ring_thickness;
    float etched=1.-smoothstep(-ring_thickness*.06,ring_thickness*.06,v);
    float rough=mix(.075,.88,etched);
    vec3 F0=vec3(1.,.63,.20);
    float nv=max(dot(N,V),0.);
    vec3 F=F0+(1.-F0)*pow(1.-nv,5.);
    vec3 reflected=studio(reflect(-V,N),rough)*F*.57;
    float illumination=.30+.38*max(0.,dot(N,normalize(vec3(-.5,-1.,1.5))))
        +.17*max(0.,dot(N,normalize(vec3(.8,.5,1.))));
    if(clay_preview>.5){
        frag_color=vec4(pow(aces(vec3(.38,.40,.44)*illumination),vec3(1./2.2)),1.);
        return;
    }
    // Dry etched gold: diffuse-only, flatter and darker than the polished face.
    // Neither studio reflections nor Fresnel highlights enter this branch.
    float dryLight=.46+.20*max(0.,dot(N,normalize(vec3(-.5,-1.,1.5))))
        +.08*max(0.,dot(N,normalize(vec3(.8,.5,1.))));
    float pits=etchNoise(model_point*155.);
    float fine=etchNoise(model_point*440.);
    // Broken, irregular micro-scratches, not continuous brushed stripes.
    float theta=atan(model_point.y,model_point.x);
    vec2 uv=vec2(theta*ring_radius,material_coordinates.x);
    vec2 cell=floor(uv*vec2(85.,210.));
    vec2 local=fract(uv*vec2(85.,210.));
    float offset=.20+.60*grain(vec3(cell,19.));
    float aa=max(fwidth(local.y),.035);
    float scratch=(1.-smoothstep(.035,.035+aa,abs(local.y-offset)))
        *smoothstep(.08,.2,local.x)*(1.-smoothstep(.65,.90,local.x));
    float etchTone=.66+.22*pits+.12*fine-.13*scratch;
    vec3 matte=F0*.72*dryLight*etchTone;
    vec3 color=mix(reflected+F0*.025,matte,etched);
    vec3 finished=pow(aces(color),vec3(1./2.2));
    vec3 blue=vec3(.10,.32,.90),red=vec3(.90,.10,.16);
    float flatLight=.68+.32*max(0.,dot(N,normalize(vec3(-.5,-1.,1.5))));
    vec3 diagnostic=mix(vec3(25.,82.,217.)/255.,vec3(217.,31.,53.)/255.,etched);
    finished=mix(finished,diagnostic,diagnostic_blend);
    float around=mod(atan(model_point.y,model_point.x)+6.283185307,6.283185307);
    if(cut_progress>=0. && around<=cut_progress*6.283185307 && abs(material_coordinates.x)<.018)
        finished=vec3(.98);
    frag_color=vec4(finished,diagnostic_blend>.5?1.:surface_opacity);
}
'''
# Matte gold lighting, expressed as a smooth function of the surface normal.
import json
_material=json.loads((ROOT/'geometry/matte_material.json').read_text())
_terms=[]
for powers,coefficient in zip(_material['powers'],_material['coefficients']):
    factors=[f'N.{axis}' for axis,power in zip('xyz',powers) for _ in range(power)]
    monomial='*'.join(factors) or '1.'
    coefficient_glsl=','.join(f'{value:.12g}' for value in coefficient)
    _terms.append(f'vec3({coefficient_glsl})*({monomial})')
_matte='if(matte_ribbon>.5){vec3 c='+('+'.join(_terms))+';frag_color=vec4(clamp(c,0.,1.),.82);return;}\n'
frag=frag.replace('uniform float simplified;', 'uniform float simplified;\nuniform float matte_ribbon;')
frag=frag.replace('float etched=',_matte+'float etched=')
prog=ctx.program(vertex_shader=vertex,fragment_shader=frag)
phi,theta=np.deg2rad([58,-30])
view=np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
right=np.array([-np.sin(theta),np.cos(theta),0.]);up=np.cross(view,right)
rot=np.stack([right,up,view]).astype('f4')
prog['camera_rotation'].write(rot.T.tobytes());prog['frame_shape'].value=(6.4,5.12)
prog['surface_opacity'].value=.82
for k,v in dict(focal_distance=10000.,ring_radius=RADIUS,ring_thickness=HALF_THICKNESS,ring_width=HALF_WIDTH,clay_preview=0.,diagnostic_blend=0.,single_face=0.,cut_progress=-1.).items():
 if k in prog:prog[k].value=v
tex=ctx.texture((W,H),4);depth=ctx.depth_renderbuffer((W,H));fb=ctx.framebuffer(tex,depth);fb.use()
ctx.enable(moderngl.DEPTH_TEST)
def set_camera(polar):
 """Set the elevation while retaining the common azimuth and framing."""
 global rot
 phi,theta=np.deg2rad([polar,-30])
 view=np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
 right=np.array([-np.sin(theta),np.cos(theta),0.]);up=np.cross(view,right)
 rot=np.stack([right,up,view]).astype('f4')
 prog['camera_rotation'].write(rot.T.tobytes())

def draw(function,labels,nt=2001,nb=257,bounds=(0,2*np.pi)):
 """Sample a parametric surface, estimate normals and render its triangle mesh."""
 t,b=np.meshgrid(np.linspace(0,2*np.pi,nt),np.linspace(bounds[0],bounds[1],nb),indexing='ij')
 p=function(t,b);e=1e-4
 normal=unit(np.cross(function(t+e,b)-p,function(t,b+e)-p))
 data=np.concatenate([p,normal,np.stack(labels(t,b),axis=-1)],axis=-1).astype('f4')
 ix=np.arange(nt*nb).reshape(nt,nb);a=ix[:-1,:-1].ravel();c=ix[1:,:-1].ravel();d=ix[1:,1:].ravel();f=ix[:-1,1:].ravel()
 indices=np.stack([a,c,d,a,d,f],axis=-1).astype('u4').ravel()
 vb=ctx.buffer(data.tobytes());ib=ctx.buffer(indices.tobytes());vao=ctx.vertex_array(prog,[(vb,'3f 3f 2f','point','normal','material')],ib)
 vao.render();vao.release();vb.release();ib.release()
def grab():return Image.frombytes('RGBA',(W,H),fb.read(components=4)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
def metal():
 fb.clear(0,0,0,0,depth=1)
 prog['diagnostic_blend'].value=0
 draw(lambda t,b:ring_points(t,b,4),lambda t,b:(HALF_WIDTH*np.sin(b),HALF_THICKNESS*np.cos(b)))
 return grab()
def accepted_edge(t,side):return ring_points(t,np.pi/2 if side==1 else 3*np.pi/2,4)
def model(t,u):return np.stack([(RADIUS-u*np.sin(2*t))*np.cos(t),(RADIUS-u*np.sin(2*t))*np.sin(t),u*np.cos(2*t)],axis=-1)
def save(im,name):
 # Common output frame; the camera profile is chosen for each view.
 im.crop((360,440,5640,4360)).resize((2640,1960),Image.Resampling.LANCZOS).save(OUT/name)
if __name__=='__main__':
    fb.clear(0,0,0,0,depth=1)
    prog['simplified'].value=1
    draw(model,lambda t,u:(u,np.full_like(u,HALF_THICKNESS)),nb=65,bounds=(-HALF_WIDTH,HALF_WIDTH))
    # Figure: gold ribbon surface with four half-twists.
    save(grab(),'ribbon-construction.png')
    # Figure: the same four-half-twist ribbon with a matte gold material.
    prog['matte_ribbon'].value=1
    fb.clear(0,0,0,0,depth=1)
    draw(model,lambda t,u:(u,np.full_like(u,HALF_THICKNESS)),nb=65,bounds=(-HALF_WIDTH,HALF_WIDTH))
    save(grab(),'simplified-ribbon-matte.png')
    prog['matte_ribbon'].value=0
    prog['simplified'].value=0
    # Figure: refined solid ring, with polished and etched material faces.
    set_camera(50)
    prog['surface_opacity'].value=1.
    gold=metal();save(gold,'ring-solid.png')
    set_camera(58)
    prog['surface_opacity'].value=.82
    metal()
    # Retain the metal depth buffer, so hidden curves remain hidden.
    prog['diagnostic_blend'].value=1
    for side in [1,-1]:
        def edge(t):
            p=accepted_edge(t,side)
            center=np.stack([RADIUS*np.cos(t),RADIUS*np.sin(t),np.zeros_like(t)],axis=-1)
            return p+.026*unit(p-center)
        def tube(t,b):
            p=edge(t);e=1e-4;T=unit(edge(t+e)-edge(t-e))
            radial=np.stack([np.cos(t),np.sin(t),np.zeros_like(t)],axis=-1)
            A=unit(radial-np.sum(radial*T,axis=-1,keepdims=True)*T);B=unit(np.cross(T,A))
            return p+.028*(np.cos(b)[...,None]*A+np.sin(b)[...,None]*B)
        draw(tube,lambda t,b:(np.zeros_like(t),np.full_like(t,HALF_THICKNESS*side)),nb=17)
    # Figure: refined ring with the blue and red material boundary curves.
    save(grab(),'ring-curves.png')
    print('Rendered two gold textures, transparent backgrounds, translucent surfaces, and depth-tested curves.')


# Figure: blue and red projected link, with the supplied framing and crossing gaps.
from pathlib import Path
from io import BytesIO
import json
import cairosvg
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures'
OUT.mkdir(exist_ok=True)
DRAWINGS=json.loads((ROOT/'geometry'/'03_projection.json').read_text())

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

if __name__=="__main__":
    render_figure("link-camera-smooth")
