#!/usr/bin/env python3
# loops.py DIS FUNC_SUBSTR : list loops (backward branches, zero-overhead loops) with body sizes
import re, sys
dis, pat = sys.argv[1], sys.argv[2]
lines = open(dis).read().split('\n')
funcs = {}; cur = None
for l in lines:
    m = re.match(r'^([0-9a-f]+) <(.*)>:$', l)
    if m: cur = m.group(2); funcs[cur] = []; continue
    m = re.match(r'^\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)$', l)
    if cur and m: funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3)))
for name, ins in funcs.items():
    if pat not in name: continue
    addrs = [a for a, _, _ in ins]
    idx = {a: i for i, a in enumerate(addrs)}
    print(f'{name[:100]}: {len(ins)} instructions')
    for i, (a, mn, ops) in enumerate(ins):
        tm = re.search(r'\b([0-9a-f]+) <', ops)
        if not tm: continue
        t = int(tm.group(1), 16)
        if mn.startswith('loop'):
            if t in idx: print(f'  zloop @{a:x}: body {idx[t]-i-1} insns ({a:x}..{t:x})')
        elif t <= a and t in idx and (mn.startswith('b') or mn in ('j',)):
            body = ins[idx[t]:i+1]
            nst = sum(1 for _, m2, o2 in body if (m2.startswith('l32i') or m2 == 'lsi') and 'a1,' in o2.replace(' ', '') )
            nss = sum(1 for _, m2, o2 in body if (m2.startswith('s32i') or m2 == 'ssi') and 'a1,' in o2.replace(' ', ''))
            print(f'  back-branch @{a:x} -> {t:x}: {i-idx[t]+1} insns span (stack loads {nst}, stores {nss})')
