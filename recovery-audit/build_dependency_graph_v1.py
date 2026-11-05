#!/usr/bin/env python3
"""Make the dated, letter-width v1 of the anonymous dependency graph.

Run: python3 recovery-audit/build_dependency_graph_v1.py
Pinned DOT positions are in points; render with neato -n2, not dot's ranked layout.
The original DOT, audit, and professional-vault report are not rewritten.
"""
from datetime import date
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = HERE / 'notebook-dependencies'
OUT = HERE / 'notebook-dependencies-v1'
FONT = 'Helvetica'
FONT_SIZE = 12
EDGE_WIDTH = .45
TIMELINE_WIDTH = EDGE_WIDTH * 5
# Physical coordinates in points. The plot plus left labels fits inside
# US letter's 612-point portrait width without shrinking the 12-point text.
LEFT, RIGHT = 110.0, 580.0
ROW_Y = {'top': 254.0, 'middle': 146.0, 'bottom': 38.0}
START, END = date(2021, 1, 1), date(2026, 1, 1)


def q(value):
    return json.dumps(str(value))


def date_x(value):
    d = date.fromisoformat(value)
    return LEFT + (d - START).days / (END - START).days * (RIGHT - LEFT)


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout


def main():
    mapping = json.loads(BASE.with_suffix('.mapping.json').read_text())
    # Read the actual source geometry and edges, so v1 remains a modification
    # of the source DOT rather than silently substituting a different graph.
    source = json.loads(run('dot', '-Tjson', str(BASE.with_suffix('.dot'))))
    objects = {o['name']: o for o in source['objects']}
    by_gvid = {o['_gvid']: o['name'] for o in source['objects']}
    nodes = {n['id']: n for n in mapping['nodes']}
    source_data = {e['id']: (by_gvid[e['tail']], by_gvid[e['head']]) for e in source['edges'] if e.get('id', '').startswith('e')}
    expected_data = {e['id']: (e['source'], e['target']) for e in mapping['edges']}
    assert source_data == expected_data, 'DOT and identity mapping disagree; inspect before rebuilding'
    assert set(nodes) <= set(objects), 'Missing mapped nodes in source DOT'
    file_ids = mapping['rows']['top'] + mapping['rows']['bottom']
    old_xs = [float(objects[i]['pos'].split(',')[0]) for i in file_ids]
    xmin, xmax = min(old_xs), max(old_xs)
    coords = {}
    for nid, node in nodes.items():
        if node['kind'] == 'notebook':
            x = date_x(node['date'])
        else:
            old_x = float(objects[nid]['pos'].split(',')[0])
            x = LEFT + (old_x - xmin) / (xmax - xmin) * (RIGHT - LEFT)
        coords[nid] = (x, ROW_Y[node['row']])

    lines = [
        '// Date-scaled v1 of notebook-dependencies.dot.',
        '// Render: neato -n2 -Tsvg notebook-dependencies-v1.dot -o notebook-dependencies-v1.svg',
        '// Identical dates share horizontal positions, including checkpoint copies.',
        '// Date exceptions follow the original mapping: 2022-21-10 -> 2022-10-21 (inferred);',
        '// undated example notebooks use their study directory date.',
        'digraph dependencies_v1 {',
        '  graph [layout=neato, inputscale=72, overlap=true, splines=line, outputorder=edgesfirst, bgcolor="white", pad=0.10, margin=0, size="8.5,11"];',
        f'  node [shape=circle, label="", width=0.025, height=0.025, fixedsize=true, style=filled, fillcolor="black", color="black", penwidth=0.4, pin=true, fontname="{FONT}", fontsize={FONT_SIZE}];',
        f'  edge [color="black", penwidth={EDGE_WIDTH}, arrowsize=0.20];',
    ]
    for nid, (x, y) in coords.items():
        lines.append(f'  {nid} [id="{nid}", pos="{x:.8f},{y:.8f}!"];')
    for year in range(2021, 2027):
        x, y = date_x(f'{year}-01-01'), ROW_Y['middle']
        for part, py in [('axis', y), ('lo', y - 5), ('hi', y + 5)]:
            lines.append(f'  year_{year}_{part} [shape=point, width=0, height=0, style=invis, pos="{x:.8f},{py:.8f}!"];')
        lines.append(f'  year_{year}_label [label="{year}", shape=box, fixedsize=false, width=0, height=0, margin="0.025,0.01", style=filled, fillcolor="white", color="white", penwidth=0, fontname="{FONT}", fontsize={FONT_SIZE}, pos="{x:.8f},{y - 20:.8f}!"];')
    for row, label in [('top', 'inputs'), ('middle', 'notebooks'), ('bottom', 'outputs')]:
        lines.append(f'  label_{row} [label="{label}", shape=plaintext, fixedsize=false, width=0, height=0, style="", fontname="{FONT}", fontsize={FONT_SIZE}, pos="43,{ROW_Y[row]:.8f}!"];')
    for eid, (a, b) in source_data.items():
        lines.append(f'  {a} -> {b} [id="{eid}"];')
    # Replace the old notebook-to-notebook chronology guide with a single axis.
    lines.append(f'  year_2021_axis -> year_2026_axis [id="timeline", dir=none, color="gray", penwidth={TIMELINE_WIDTH}];')
    for year in range(2021, 2027):
        lines.append(f'  year_{year}_lo -> year_{year}_hi [id="tick_{year}", dir=none, color="gray", penwidth={TIMELINE_WIDTH}];')
    lines.append('}')
    OUT.with_suffix('.dot').write_text('\n'.join(lines) + '\n')
    run('neato', '-n2', '-Tsvg', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.svg')))
    run('neato', '-n2', '-Tpng', '-Gdpi=160', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.png')))

    # Check the produced geometry, not merely the requested positions.
    rendered = json.loads(run('neato', '-n2', '-Tjson', str(OUT.with_suffix('.dot'))))
    actual = {o['name']: tuple(map(float, o['pos'].split(','))) for o in rendered['objects']}
    x0 = actual['year_2021_axis'][0]
    x1 = actual['year_2026_axis'][0]
    for nid, node in nodes.items():
        if node['kind'] == 'notebook':
            expected_fraction = (date.fromisoformat(node['date']) - START).days / (END - START).days
            actual_fraction = (actual[nid][0] - x0) / (x1 - x0)
            assert abs(expected_fraction - actual_fraction) < .00002
            assert actual[nid][1] == actual['year_2021_axis'][1]
    for row, ids in mapping['rows'].items():
        assert len({actual[nid][1] for nid in ids}) == 1
    visible_edges = {e.get('id'): e for e in rendered['edges']}
    for eid in source_data:
        assert visible_edges[eid]['color'] == 'black'
    assert float(visible_edges['timeline']['penwidth']) == TIMELINE_WIDTH
    svg = ET.parse(OUT.with_suffix('.svg')).getroot()
    ns = '{http://www.w3.org/2000/svg}'
    texts = svg.findall('.//' + ns + 'text')
    assert sorted(t.text for t in texts) == sorted(['inputs', 'notebooks', 'outputs'] + list(map(str, range(2021, 2027))))
    assert all(float(t.attrib['font-size']) == FONT_SIZE for t in texts)
    assert len({t.attrib['font-family'] for t in texts}) == 1
    assert float(svg.attrib['width'].removesuffix('pt')) <= 612.1
    # The SVG scales the graph at most slightly for padding. Check that text
    # stays approximately 12 points when the whole figure fits letter portrait.
    import re
    transform = svg.find(ns + 'g').attrib['transform']
    scale = float(re.search(r'scale\(([0-9.]+)', transform)[1])
    assert 11.5 <= FONT_SIZE * scale <= 12.5
    print(json.dumps(dict(notebooks=sum(n['kind'] == 'notebook' for n in nodes.values()), dependency_edges=len(source_data), year_ticks=6, timeline_penwidth=TIMELINE_WIDTH, dependency_penwidth=EDGE_WIDTH, font=FONT, svg_font_size=FONT_SIZE, fitted_font_size=FONT_SIZE * scale, svg_width=svg.attrib['width'], svg_height=svg.attrib['height'], validation='passed'), indent=2))


if __name__ == '__main__':
    main()
