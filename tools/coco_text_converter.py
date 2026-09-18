#!/usr/bin/env python3
import os

for s in ['train', 'val']:
    gt_path = f'{s}_words_gt.txt'
    if not os.path.exists(gt_path):
        print(f"Arquivo {gt_path} não encontrado, pulando...")
        continue
    with open(gt_path, 'r', encoding='utf8') as f:
        d = f.readlines()

    with open('{}_lmdb.txt'.format(s), 'w', encoding='utf8') as f:
        for line in d:
            try:
                fname, label = line.split(',', maxsplit=1)
            except ValueError:
                continue
            fname = '{}_words/{}.jpg'.format(s, fname.strip())
            label = label.strip().strip('|')
            f.write('\t'.join([fname, label]) + '\n')
