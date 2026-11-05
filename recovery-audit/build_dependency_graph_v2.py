#!/usr/bin/env python3
"""Style v1 with hollow markers and missing-file impact; leave v1 unchanged.

Run: python3 recovery-audit/build_dependency_graph_v2.py
Render the resulting DOT with neato -n2. Unknown paths and unchecked remote
resources stay black; red denotes a resolved local path absent on disk.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'notebook-dependencies-v1.dot'
OUT = HERE / 'notebook-dependencies-v2'


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout


def radius(obj):
    ellipse = next(d for d in obj['_draw_'] if d['op'] in {'e', 'E'})
    return float(ellipse['rect'][2])


def main():
    mapping = json.loads((HERE / 'notebook-dependencies.mapping.json').read_text())
    nodes = {n['id']: n for n in mapping['nodes']}
    edges = {e['id']: e for e in mapping['edges']}
    before = json.loads(run('neato', '-n2', '-Tjson', str(SOURCE)))
    originals = {o['name']: o for o in before['objects']}
    gvids = {o['_gvid']: o['name'] for o in before['objects']}
    original_edges = {e['id']: e for e in before['edges'] if e.get('id') in edges}
    assert set(original_edges) == set(edges)
    for eid, edge in edges.items():
        actual = original_edges[eid]
        assert (gvids[actual['tail']], gvids[actual['head']]) == (edge['source'], edge['target'])
    file_radii = {radius(originals[nid]) for nid, n in nodes.items() if n['kind'] == 'file'}
    assert len(file_radii) == 1, 'Input/output nodes in v1 have inconsistent sizes'
    file_radius = file_radii.pop()
    notebook_radius = file_radius * 1.5
    status = {}
    for nid, node in nodes.items():
        if node['kind'] != 'file':
            continue
        path = node.get('resolved_path')
        if not path or node.get('unresolved') or node.get('pattern'):
            state = 'unknown'
            reason = 'Unresolved file identity or pattern; not classified as missing'
        elif re.match(r'^[A-Za-z]+://', path):
            state = 'unknown'
            reason = 'Remote resource availability not checked'
        elif Path(path).is_file():
            state, reason = 'present', 'Exact local path exists'
        elif Path(path).exists():
            state, reason = 'unknown', 'Path exists but is not a regular file'
        else:
            state, reason = 'missing', 'Exact local path is absent'
        status[nid] = dict(row=node['row'], path=path, state=state, reason=reason, color='red' if state == 'missing' else 'black')
    red_edges = {eid for eid, edge in edges.items() if status.get(edge['source'], {}).get('state') == 'missing'}
    red_notebooks = {edges[eid]['target'] for eid in red_edges if nodes[edges[eid]['target']]['kind'] == 'notebook'}
    for nid, node in nodes.items():
        if node['kind'] == 'notebook':
            incoming = sorted(eid for eid in red_edges if edges[eid]['target'] == nid)
            status[nid] = dict(row='middle', notebook=node['notebook'], state='receives missing input' if incoming else 'no confirmed missing incoming file', color='red' if nid in red_notebooks else 'black', incoming_red_edges=incoming)

    lines = [
        '// v2: hollow markers and confirmed missing local files.',
        '// Render: neato -n2 -Tsvg notebook-dependencies-v2.dot -o notebook-dependencies-v2.svg',
        '// Missing output nodes color their outgoing consumption edges, not incoming production edges.',
        '// Notebook color follows incoming red edges; outputs are checked independently on disk.',
        '// Unresolved paths and unchecked URLs stay black; see notebook-dependencies-v2.status.json.',
    ]
    modified_nodes, modified_edges = set(), set()
    for line in SOURCE.read_text().splitlines():
        if line.startswith('// Render:'):
            continue
        if line.startswith('digraph '):
            line = 'digraph dependencies_v2 {'
        match = re.match(r'^\s*(\w+) \[id="([^"]+)"', line)
        if match and match[1] in nodes:
            nid = match[1]
            is_notebook = nodes[nid]['kind'] == 'notebook'
            r = notebook_radius if is_notebook else file_radius
            stroke = r / 3 if is_notebook else r / 2
            # Use the rendered v1 radius rather than its nominal width: Graphviz
            # rounds the original 1.8-point diameter to 2 points when rendering.
            attrs = f', width={2*r/72:.10f}, height={2*r/72:.10f}, style="solid", fillcolor="none", color="{status[nid]["color"]}", penwidth={stroke:.10f}'
            line = line.removesuffix('];') + attrs + '];'
            modified_nodes.add(nid)
        else:
            match = re.search(r'\[id="(e[0-9]+)"', line)
            if match and match[1] in edges:
                eid = match[1]
                line = line.removesuffix('];') + f', color="{"red" if eid in red_edges else "black"}"];'
                modified_edges.add(eid)
        lines.append(line)
    assert modified_nodes == set(nodes)
    assert modified_edges == set(edges)
    OUT.with_suffix('.dot').write_text('\n'.join(lines) + '\n')
    run('neato', '-n2', '-Tsvg', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.svg')))
    run('neato', '-n2', '-Tpng', '-Gdpi=160', str(OUT.with_suffix('.dot')), '-o', str(OUT.with_suffix('.png')))

    after = json.loads(run('neato', '-n2', '-Tjson', str(OUT.with_suffix('.dot'))))
    rendered = {o['name']: o for o in after['objects']}
    for nid, node in nodes.items():
        r = radius(rendered[nid])
        if node['kind'] == 'file':
            assert r == radius(originals[nid])
            assert abs(float(rendered[nid]['penwidth']) - r / 2) < 1e-8
        else:
            assert r == file_radius * 1.5
            assert abs(float(rendered[nid]['penwidth']) - r / 3) < 1e-8
        assert rendered[nid]['color'] == status[nid]['color']
        assert rendered[nid]['pos'] == originals[nid]['pos'], 'Layout changed'
        assert any(d['op'] == 'e' for d in rendered[nid]['_draw_']), 'Node is not hollow'
    for edge in after['edges']:
        eid = edge.get('id')
        if eid in edges:
            assert edge['color'] == ('red' if eid in red_edges else 'black')
        elif eid == 'timeline' or eid.startswith('tick_'):
            assert edge['color'] == 'gray'
    ns = '{http://www.w3.org/2000/svg}'
    svg = ET.parse(OUT.with_suffix('.svg')).getroot()
    for group in svg.findall('.//' + ns + 'g'):
        if group.get('id') in nodes:
            assert group.find(ns + 'ellipse').get('fill') == 'none'
    before_svg = ET.parse(SOURCE.with_suffix('.svg')).getroot()
    assert [(t.text, t.get('font-family'), t.get('font-size')) for t in svg.findall('.//' + ns + 'text')] == [(t.text, t.get('font-family'), t.get('font-size')) for t in before_svg.findall('.//' + ns + 'text')]
    summary = dict(missing_inputs=sum(s['row'] == 'top' and s['state'] == 'missing' for s in status.values()), missing_outputs=sum(s['row'] == 'bottom' and s['state'] == 'missing' for s in status.values()), red_notebooks=len(red_notebooks), red_edges=len(red_edges), unknown_files=sum(s['state'] == 'unknown' for s in status.values()), file_radius_points=file_radius, file_contour_points=file_radius/2, notebook_radius_points=notebook_radius, notebook_contour_points=notebook_radius/3, validation='passed')
    OUT.with_suffix('.status.json').write_text(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), summary=summary, node_status=status, red_edges=sorted(red_edges), conventions='Red: confirmed absent exact local path, its outgoing consumption edges, and notebooks receiving those edges. Black does not imply verified availability for unknown paths or remote URLs. Missing generated files do not color their incoming production edges.'), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
