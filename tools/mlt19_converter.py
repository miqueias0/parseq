#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print(f"Uso: python {sys.argv[0]} <diretório_dataset>")
    sys.exit(1)

root = sys.argv[1]

with open(root + '/gt.txt', 'r', encoding='utf-8') as f:
    d = f.readlines()

with open(root + '/lmdb.txt', 'w', encoding='utf-8') as f:
    for line in d:
        line_s = line.strip()
        if not line_s:
            continue
        parts = line_s.split(',', maxsplit=2)
        if len(parts) < 3:
            continue
        img, script, label = parts
        label = label.strip()
        if label and script in ['Latin', 'Symbols']:
            f.write('\t'.join([img, label]) + '\n')
