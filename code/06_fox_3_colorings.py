"""06_fox_3_colorings.py

Produces and saves the following figures in figures/:
  - link-L-arcs-numbered.png (and .svg)
  - link-L-gold-dotted.png (and .svg)

Run from the project folder: python code/06_fox_3_colorings.py
Keep the geometry/ folder beside this script.
"""
from pathlib import Path
from io import BytesIO
import json
import cairosvg
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'figures'
OUT.mkdir(exist_ok=True)
DRAWINGS=json.loads((ROOT/'geometry'/'06_drawings.json').read_text())

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

def check_colorings():
    """Check the four arc-color equations over Z_3."""
    from itertools import product
    solutions=[]
    for a,b,c,d in product(range(3),repeat=4):
        if all(v%3==0 for v in [a+b-2*d,c+d-2*a,a+b-2*c,c+d-2*b]):
            solutions.append((a,b,c,d))
    assert solutions==[(0,0,0,0),(1,1,1,1),(2,2,2,2)]
    return solutions

if __name__=="__main__":
    check_colorings()
    # Figure: Four black arcs numbered 1–4; dotted ends mark undercrossing approaches.
    render_figure('link-L-arcs-numbered')
    # Figure: Constant gold coloring with variables a,b,c,d and dotted undercrossing approaches.
    render_figure('link-L-gold-dotted')
