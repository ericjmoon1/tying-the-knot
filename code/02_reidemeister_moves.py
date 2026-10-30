"""02_reidemeister_moves.py

Produces and saves the following figures in figures/:
  - reidemeister-1.png (and .svg)
  - reidemeister-2.png (and .svg)
  - reidemeister-3.png (and .svg)

Run from the project folder: python code/02_reidemeister_moves.py
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
DRAWINGS=json.loads((ROOT/'geometry'/'02_drawings.json').read_text())

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
    # Figure: Type I: red kink diagrams on either side of a straight strand.
    render_figure('reidemeister-1')
    # Figure: Type II: vertical red and blue strands, with the two-crossing cancellations.
    render_figure('reidemeister-2')
    # Figure: Type III: two rows of red, blue and green strands passing across a crossing.
    render_figure('reidemeister-3')
