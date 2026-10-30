"""04_candidate_links.py

Produces and saves the following figures in figures/:
  - hopf-link.png (and .svg)
  - whitehead-link.png (and .svg)
  - whitehead-link-mirror.png (and .svg)
  - solomons-link.png (and .svg)
  - solomons-link-mirror.png (and .svg)
  - torus-2-6.png (and .svg)
  - torus-2-6-mirror.png (and .svg)
  - link-6-2-2.png (and .svg)
  - link-6-2-2-mirror.png (and .svg)
  - link-6-2-3.png (and .svg)
  - link-6-2-3-mirror.png (and .svg)
  - torus-2-8.png (and .svg)
  - link-7-2-5.png (and .svg)
  - link-7-2-5-mirror.png (and .svg)

Run from the project folder: python code/04_candidate_links.py
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
DRAWINGS=json.loads((ROOT/'geometry'/'04_drawings.json').read_text())

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
    # Figure: Hopf link: red and blue components and the link-table label.
    render_figure('hopf-link')
    # Figure: Whitehead link: red and blue components and the link-table label.
    render_figure('whitehead-link')
    # Figure: Mirror Whitehead link, with reversed over/under information.
    render_figure('whitehead-link-mirror')
    # Figure: Solomon’s link: red vertical and blue horizontal ovals.
    render_figure('solomons-link')
    # Figure: Mirror Solomon’s link, with reversed over/under information.
    render_figure('solomons-link-mirror')
    # Figure: Star of David link: triangular red and blue components.
    render_figure('torus-2-6')
    # Figure: Mirror Star of David link: triangular red and blue components.
    render_figure('torus-2-6-mirror')
    # Figure: Link 6^2_2: red and blue component diagram and its label.
    render_figure('link-6-2-2')
    # Figure: Mirror of 6^2_2: red and blue component diagram and its label.
    render_figure('link-6-2-2-mirror')
    # Figure: Link 6^2_3: red and blue component diagram and its label.
    render_figure('link-6-2-3')
    # Figure: Mirror of 6^2_3: red and blue component diagram and its label.
    render_figure('link-6-2-3-mirror')
    # Figure: Torus link T(2,8): red and blue components with eight crossings.
    render_figure('torus-2-8')
    # Figure: Link 7^2_5: red and blue component diagram and its label.
    render_figure('link-7-2-5')
    # Figure: Mirror of 7^2_5: red and blue component diagram and its label.
    render_figure('link-7-2-5-mirror')
