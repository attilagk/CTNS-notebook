#!/usr/bin/env python3
"""Build the anonymous three-row notebook dependency graph from the audit.

Usage: python3 recovery-audit/build_dependency_graph.py
Reads notebooks and inventories without executing notebook or helper code.
Does not run audit_inputs.main() or write through the professional-vault symlink.
"""
import ast
import collections
import csv
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEM = HERE / 'notebook-dependencies'
EXCLUDED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.eps', '.ps', '.gif', '.webp', '.tif', '.tiff', '.bmp', '.ico', '.gv', '.dot', '.tex', '.html', '.py', '.r', '.ipynb'}
NON_FILE_INPUTS = {'shell/magic', 'unparsed cell', 'shell path expression', 'pd.read_clipboard'}


def read_csv(name):
    with (HERE / name).open(newline='') as f:
        return list(csv.DictReader(f))


def eligible(row, output=False):
    loader = row['loader']
    path = row['resolved_path']
    leaf = loader.rsplit('.', 1)[-1]
    if 'savefig' in loader or loader.startswith('gv.') or leaf == 'to_latex':
        return False, 'figure, diagram, or presentation output'
    if not output and loader in NON_FILE_INPUTS:
        return False, 'non-file operation or whole-cell review entry'
    if Path(path).suffix.lower() in EXCLUDED_EXTENSIONS:
        return False, 'image, presentation, or code file'
    if not output and row.get('status') == 'directory present; members need review':
        return False, 'directory enumeration rather than individual data file'
    return True, ''


def notebook_date(path):
    """Directory fallback handles examples and checkpoint snapshots."""
    stem = Path(path).stem
    directory = Path(path).parent
    if directory.name == '.ipynb_checkpoints':
        directory = directory.parent
    source = stem[:10] if re.match(r'^\d{4}-\d{2}-\d{2}', stem) else directory.name[:10]
    note = 'date from notebook filename' if source == stem[:10] else 'date from containing study directory'
    if source == '2022-21-10':
        source = '2022-10-21'
        note = 'inferred: malformed 2022-21-10 interpreted as 2022-10-21; unconfirmed'
    return date.fromisoformat(source).isoformat(), note


def supplemental_outputs():
    """Follow ExcelWriter destinations the first audit did not classify as writes."""
    spec = importlib.util.spec_from_file_location('static_audit', HERE / 'audit_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Writers(module.Audit):
        def call(self, node, env):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'ExcelWriter':
                val = self.ev(node.args[0], env) if node.args else next((self.ev(k.value, env) for k in node.keywords if k.arg in {'path', 'path_or_buf'}), module.UNKNOWN)
                self.record(node, module.expr(node.func), val, 'write', 'Workbook container; worksheet writes share this file')
                return module.UNKNOWN
            return super().call(node, env)

    analyzer = Writers()
    for p in sorted(ROOT.rglob('*.ipynb')):
        analyzer.notebook(p)
    return [row for row in analyzer.writes if row['loader'].endswith('ExcelWriter')]


def main():
    coverage = read_csv('notebook-coverage.csv')
    inputs = read_csv('input-inventory.csv')
    outputs = read_csv('output-producers.csv') + supplemental_outputs()
    notebook_paths = {row['notebook'] for row in coverage}
    actual_paths = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*.ipynb')}
    if notebook_paths != actual_paths:
        raise ValueError('Notebook coverage is stale; inspect the audit before generating the graph')
    ordered = sorted(notebook_paths, key=lambda p: (notebook_date(p)[0], p))
    notebook_ids = {p: f'n{i:03d}' for i, p in enumerate(ordered)}
    positions = {p: i for i, p in enumerate(ordered)}
    nodes = {}
    edges = {}
    excluded = []
    files = {}

    def add_edge(source, target, relation, evidence):
        key = (source, target, relation)
        edge = edges.setdefault(key, dict(source=source, target=target, relation=relation, evidence=[]))
        if evidence not in edge['evidence']:
            edge['evidence'].append(evidence)

    for p in ordered:
        dt, note = notebook_date(p)
        nid = notebook_ids[p]
        nodes[nid] = dict(id=nid, kind='notebook', row='middle', notebook=p, date=dt, date_evidence=note, checkpoint='.ipynb_checkpoints' in Path(p).parts)

    # A workbook created through ExcelWriter is one file, not one per worksheet.
    workbook_cells = {(r['notebook'], str(r['cell'])) for r in outputs if r['loader'].endswith('ExcelWriter') and r['resolved_path']}

    def file_node(row, output):
        path = row['resolved_path']
        if path:
            key = ('path', path if re.match(r'^[A-Za-z]+://', path) else os.path.normpath(path))
        else:
            # Unknown filenames remain anonymous placeholders; no invented filenames.
            key = ('unresolved_output' if output else 'unresolved_input', row['notebook'], row['cell'], row['source'], row['line'], row['expression'])
        if key not in files:
            fid = 'f' + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:14]
            files[key] = fid
            exists = Path(path).is_file() if path and not re.match(r'^[A-Za-z]+://', path) else None
            nodes[fid] = dict(id=fid, kind='file', row='top', resolved_path=path or None, unresolved=not bool(path), pattern=bool(path and any(c in path for c in '*?[')), present_on_disk=exists, consumers=[], producers=[], evidence=[], recovery_candidates=[], uncertainty=[])
        fid = files[key]
        node = nodes[fid]
        evidence = {k: row.get(k, '') for k in ['notebook', 'cell', 'source', 'line', 'loader', 'expression', 'expected_path', 'status', 'notes']}
        if evidence not in node['evidence']:
            node['evidence'].append(evidence)
        for field in ['local_candidates', 'dropbox_candidates']:
            for candidate in row.get(field, '').split(' | '):
                if candidate and candidate not in node['recovery_candidates']:
                    node['recovery_candidates'].append(candidate)
        if row.get('candidate_evidence') and row['candidate_evidence'] not in node['uncertainty']:
            node['uncertainty'].append(row['candidate_evidence'])
        bucket = node['producers' if output else 'consumers']
        if row['notebook'] not in bucket:
            bucket.append(row['notebook'])
        if output:
            node['row'] = 'bottom'
        nid = notebook_ids[row['notebook']]
        add_edge(nid if output else fid, fid if output else nid, 'produces' if output else 'consumes', evidence)

    for output, rows in [(False, inputs), (True, outputs)]:
        for row in rows:
            ok, reason = eligible(row, output)
            # Worksheet calls with a known writer object do not create extra
            # unresolved file nodes beside the resolved workbook container.
            if output and not row['resolved_path'] and '.to_excel(writer' in row['expression'] and (row['notebook'], str(row['cell'])) in workbook_cells:
                ok, reason = False, 'worksheet write linked through its resolved ExcelWriter container'
            if not ok:
                excluded.append(dict(direction='output' if output else 'input', reason=reason, **row))
                continue
            file_node(row, output)

    # Store reconstruction provenance separately: no speculative data-flow edge
    # or fabricated cleaning notebook is added to the anonymous graph.
    cleaned = str(ROOT.parent / 'resources/QPS/5xFAD-mouse/CO28154-behavioral-data-cleaned.xlsx')
    for node in nodes.values():
        if node.get('resolved_path') == cleaned:
            node['reconstruction'] = dict(raw_local_path=str(ROOT.parent / 'resources/QPS/5xFAD-mouse/CO28154-rawdata-in-vivo.xlsx'), raw_dropbox_url='https://www.dropbox.com/scl/fi/tfj5cq637eq7md7f01lcj/CO28154-rawdata-in-vivo.xlsx?rlkey=bmuemeuw7bxhfe9xmac9kqocc&dl=0', method='Reconstruct cleaning using CO28152 raw/cleaned reference pair, then adapt to CO28154 cohort and sheets; not implemented or validated')

    rows = {'top': [], 'middle': list(notebook_ids.values()), 'bottom': []}
    def center(fid):
        node = nodes[fid]
        neighbors = node['producers'] or node['consumers']
        return sum(positions[p] for p in neighbors) / len(neighbors) if neighbors else 0
    for fid in files.values():
        rows[nodes[fid]['row']].append(fid)
    for row in ['top', 'bottom']:
        rows[row].sort(key=lambda fid: (center(fid), nodes[fid].get('resolved_path') or '', fid))

    lines = [
        'digraph dependencies {',
        '  graph [rankdir=TB, splines=line, nodesep=0.10, ranksep=2.0, outputorder=edgesfirst, bgcolor="white", pad=0.12];',
        '  node [shape=circle, label="", width=0.075, height=0.075, fixedsize=true, style=filled, fillcolor="#202020", color="#202020", penwidth=0.4];',
        '  edge [color="#7b879488", penwidth=0.45, arrowsize=0.28];',
    ]
    # Equal-width invisible grids let the chronology occupy the whole width
    # rather than compressing the shorter notebook row against the left edge.
    columns = max(len(ids) for ids in rows.values())
    grid = {}
    for row, ids in rows.items():
        slots = {round(i * (columns - 1) / (len(ids) - 1)): nid for i, nid in enumerate(ids)} if len(ids) > 1 else {columns // 2: ids[0]}
        grid[row] = [slots.get(i, f'gap_{row}_{i:03d}') for i in range(columns)]
        lines.append('  { rank=same; ' + '; '.join(grid[row]) + '; }')
        for nid in grid[row]:
            if nid in nodes:
                lines.append(f'  {nid} [id="{nid}"];')
            else:
                lines.append(f'  {nid} [style=invis];')
        for a, b in zip(grid[row], grid[row][1:]):
            lines.append(f'  {a} -> {b} [style=invis, weight=10000];')
    for i in range(columns):
        a, b, c = (grid[row][i] for row in ['top', 'middle', 'bottom'])
        lines.append(f'  {a} -> {b} -> {c} [style=invis, weight=10000, minlen=1];')
    for a, b in zip(rows['middle'], rows['middle'][1:]):
        lines.append(f'  {a} -> {b} [id="timeline_{a}_{b}", dir=none, color="#b0b0b0", penwidth=0.45, constraint=false];')
    for i, edge in enumerate(edges.values()):
        edge['id'] = f'e{i:04d}'
        lines.append(f'  {edge["source"]} -> {edge["target"]} [id="{edge["id"]}", constraint=false];')
    lines.append('}')
    dot = '\n'.join(lines) + '\n'
    STEM.with_suffix('.dot').write_text(dot)
    subprocess.run(['dot', '-Tsvg', str(STEM.with_suffix('.dot')), '-o', str(STEM.with_suffix('.svg'))], check=True, timeout=180)
    # Layout validation uses the same graph and coordinates as the SVG.
    plain = subprocess.run(['dot', '-Tplain', str(STEM.with_suffix('.dot'))], check=True, capture_output=True, text=True, timeout=180).stdout
    coords = {}
    for line in plain.splitlines():
        parts = shlex.split(line)
        if parts and parts[0] == 'node' and parts[1] in nodes:
            coords[parts[1]] = dict(x=float(parts[2]), y=float(parts[3]))
    assert set(coords) == set(nodes), 'Rendered graph dropped nodes'
    row_y = {}
    for row, ids in rows.items():
        values = [coords[n]['y'] for n in ids]
        assert max(values) - min(values) < .001, f'{row}: nodes are not on one row'
        row_y[row] = values[0]
    assert row_y['top'] > row_y['middle'] > row_y['bottom'], 'Incorrect row order'
    xs = [coords[n]['x'] for n in rows['middle']]
    assert all(a < b for a, b in zip(xs, xs[1:])), 'Chronological notebook order was not preserved'
    svg = ET.parse(STEM.with_suffix('.svg'))
    assert not svg.findall('.//{http://www.w3.org/2000/svg}text'), 'SVG contains visible labels'
    for nid, pos in coords.items():
        nodes[nid]['position_inches'] = pos
    summary = dict(notebooks=len(ordered), checkpoints=sum(nodes[n]['checkpoint'] for n in notebook_ids.values()), external_input_nodes=len(rows['top']), generated_data_nodes=len(rows['bottom']), unresolved_file_placeholders=sum(n.get('unresolved', False) for n in nodes.values()), dataflow_edges=len(edges), chronological_guide_segments=len(ordered)-1, excluded_audit_records=len(excluded))
    mapping = dict(summary=summary, conventions=dict(rows={'top': 'external inputs with no identified notebook producer', 'middle': 'notebooks in chronological order', 'bottom': 'generated data, including outputs consumed by other notebooks'}, visible_labels=False, style='identical nodes; file status and identities appear only in this mapping', chronology='evenly ordered notebook positions, not proportional elapsed time; same-row guide has no arrowheads', unresolved='placeholders correspond to unresolved loading/writing sites, not confirmed distinct physical files', identity='normalized exact paths only; renamed or relocated candidates are not automatically merged', graph_limit='static audit, including conditional branches; unresolved operations and untracked runtime dependencies may require manual review'), rows=rows, nodes=list(nodes.values()), edges=list(edges.values()), excluded_records=excluded)
    STEM.with_suffix('.mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    subprocess.run(['dot', '-Tpng', '-Gsize=24,6', '-Gdpi=120', str(STEM.with_suffix('.dot')), '-o', str(STEM.with_name(STEM.name+'-preview').with_suffix('.png'))], check=True, timeout=180)
    print(json.dumps(summary, indent=2))
    print('Validation passed: all notebooks, exact three rows, chronological order, no SVG text labels.')


if __name__ == '__main__':
    main()
