#!/usr/bin/env python3
"""Create v3 from v2: white notebook interiors and red outgoing edges.
Run: python3 recovery-audit/build_dependency_graph_v3.py
"""
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'notebook-dependencies-v2.dot'
OUT = HERE / 'notebook-dependencies-v3'


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout


def main():
    mapping = json.loads((HERE / 'notebook-dependencies.mapping.json').read_text())
    nodes = {n['id']: n for n in mapping['nodes']}
    before = json.loads(run('neato', '-n2', '-Tjson', str(SOURCE)))
    originals = {o['name']: o for o in before['objects']}
    by_gvid = {o['_gvid']: o['name'] for o in before['objects']}
    red_notebooks = {nid for nid, n in nodes.items() if n['kind'] == 'notebook' and originals[nid]['color'] == 'red'}
    edges_to_recolor = {e['id'] for e in before['edges'] if e.get('id') and by_gvid[e['tail']] in red_notebooks}
    changed_nodes, changed_edges = set(), set()
    lines = ['// v3: white-filled notebooks; red outgoing edges from red notebooks.', '// Render: neato -n2 -Tsvg notebook-dependencies-v3.dot -o notebook-dependencies-v3.svg']
    for line in SOURCE.read_text().splitlines():
        if line.startswith('// Render:'):
            continue
        if line.startswith('digraph '):
            line = 'digraph dependencies_v3 {'
        match = re.match(r'^\s*(\w+) \[id="([^"]+)"', line)
        if match and match[1] in nodes and nodes[match[1]]['kind'] == 'notebook':
            line = line.replace('style="solid"', 'style="filled"').replace('fillcolor="none"', 'fillcolor="white"')
            changed_nodes.add(match[1])
        else:
            match = re.search(r'\[id="(e[0-9]+)"', line)
            if match and match[1] in edges_to_recolor:
                line = re.sub(r'color="(?:black|red)"', 'color="red"', line)
                changed_edges.add(match[1])
        lines.append(line)
    assert len(changed_nodes) == sum(n['kind'] == 'notebook' for n in nodes.values())
    assert changed_edges == edges_to_recolor
    OUT.with_suffix('.dot').write_text('\n'.join(lines) + '\n')
    run('neato', '-n2', '-Tsvg', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.svg')))
    run('neato', '-n2', '-Tpng', '-Gdpi=160', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.png')))
    after = json.loads(run('neato', '-n2', '-Tjson', str(OUT.with_suffix('.dot'))))
    rendered = {o['name']: o for o in after['objects']}
    for nid, n in nodes.items():
        for key in ['pos', 'width', 'height', 'penwidth', 'color']:
            assert rendered[nid][key] == originals[nid][key], (nid, key)
        if n['kind'] == 'notebook':
            assert rendered[nid]['style'] == 'filled' and rendered[nid]['fillcolor'] == 'white'
        else:
            assert rendered[nid]['style'] == originals[nid]['style']
            assert rendered[nid]['fillcolor'] == originals[nid]['fillcolor']
    original_edges = {e['id']: e for e in before['edges']}
    for edge in after['edges']:
        eid = edge['id']
        assert edge['color'] == ('red' if eid in edges_to_recolor else original_edges[eid]['color'])
    ns = '{http://www.w3.org/2000/svg}'
    svg = ET.parse(OUT.with_suffix('.svg')).getroot()
    for group in svg.findall('.//' + ns + 'g'):
        nid = group.get('id')
        if nid in nodes:
            assert group.find(ns + 'ellipse').get('fill') == ('white' if nodes[nid]['kind'] == 'notebook' else 'none')
    print(json.dumps(dict(white_notebook_nodes=len(changed_nodes), red_notebooks=len(red_notebooks), red_notebook_outgoing_edges=len(edges_to_recolor), validation='passed'), indent=2))


if __name__ == '__main__':
    main()
