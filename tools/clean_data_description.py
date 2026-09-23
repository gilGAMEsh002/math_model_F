#!/usr/bin/env python3
"""Remove only the inspected, isolated near-white 5pt text blocks from this PDF."""
import argparse
import hashlib
import json
from pathlib import Path

import pymupdf
from pypdf import PdfReader, PdfWriter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_overlay(ops):
    allowed = {b'q', b'Q', b'BT', b'ET', b'Tm', b'Tf', b'RG', b'rg', b'TJ', b'Tj'}
    return (
        all(op in allowed for _, op in ops)
        and sum(op == b'BT' for _, op in ops) == 1
        and any(op == b'Tf' and abs(float(args[1]) - 5) < 1e-6 for args, op in ops)
        and any(op == b'rg' and len(args) == 3 and all(abs(float(x) - .988) < 1e-6 for x in args)
                for args, op in ops)
        and all(op not in {b'rg', b'RG'} or (len(args) == 3 and all(abs(float(x) - .988) < 1e-6 for x in args))
                for args, op in ops)
    )


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] for line in b.get('lines', []) for s in line['spans']]


def visible_signature(page):
    return [(s['text'], s['color'], round(s['size'], 4), tuple(round(x, 4) for x in s['bbox']))
            for s in spans(page) if s['color'] != 0xFCFCFC]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Keep the original file; choose a separate output path.')
    reader = PdfReader(args.source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    removed = []
    for page in writer.pages:
        stream = page.get_contents()
        ops = stream.operations
        stack, intervals = [], []
        for i, (_, op) in enumerate(ops):
            if op == b'q':
                stack.append(i)
            elif op == b'Q':
                if not stack:
                    raise ValueError('Unbalanced graphics state')
                start = stack.pop()
                if is_overlay(ops[start:i + 1]):
                    intervals.append((start, i + 1))
        if stack:
            raise ValueError('Unbalanced graphics state')
        skip = {i for a, b in intervals for i in range(a, b)}
        stream.operations = [item for i, item in enumerate(ops) if i not in skip]
        page.replace_contents(stream)
        page.compress_content_streams()
        removed.append(len(intervals))
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as out:
        writer.write(out)
    before, after = pymupdf.open(args.source), pymupdf.open(args.output)
    assert len(before) == len(after)
    checks = []
    for i, (p, q) in enumerate(zip(before, after)):
        assert p.rect == q.rect
        assert visible_signature(p) == visible_signature(q), f'Visible text changed on page {i + 1}'
        assert all(s['color'] != 0xFCFCFC for s in spans(q))
        assert len(p.get_links()) == len(q.get_links()), f'Links changed on page {i + 1}'
        # At 72 dpi every changed pixel must lie inside one of the removed spans.
        a, b = p.get_pixmap(alpha=False), q.get_pixmap(alpha=False)
        assert (a.width, a.height, a.n) == (b.width, b.height, b.n)
        boxes = [pymupdf.Rect(s['bbox']) + (-2, -2, 2, 2) for s in spans(p) if s['color'] == 0xFCFCFC]
        sa, sb = a.samples, b.samples
        changed = 0
        for pos in range(0, len(sa), a.n):
            if sa[pos:pos + a.n] != sb[pos:pos + a.n]:
                pixel = pos // a.n
                x, y = pixel % a.width, pixel // a.width
                assert any(box.contains(pymupdf.Point(x, y)) for box in boxes), (i + 1, x, y)
                changed += 1
        checks.append({'page': i + 1, 'removed_blocks': removed[i],
                       'visible_text_and_positions_unchanged': True,
                       'changed_pixels_at_72dpi': changed,
                       'changes_confined_to_removed_text': True})
    report = {'source': args.source.name, 'output': args.output.name,
              'source_sha256': digest(args.source), 'output_sha256': digest(args.output),
              'pages': len(before), 'removed_blocks': sum(removed),
              'rule': 'isolated q/BT...ET/Q block, 5pt font, RGB=(0.988,0.988,0.988)',
              'pypdf_version': __import__('pypdf').__version__,
              'pymupdf_version': pymupdf.VersionBind, 'checks': checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pages': len(before), 'removed_blocks': sum(removed), 'verification': 'passed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
