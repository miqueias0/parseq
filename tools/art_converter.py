import os
import sys
import json

labels_file = sys.argv[1] if len(sys.argv) > 1 else 'train_task2_labels.json'
if not os.path.exists(labels_file):
    print(f"Arquivo não encontrado: {labels_file}. Uso: python tools/art_converter.py [caminho_labels.json]")
    sys.exit(0)

with open(labels_file, 'r', encoding='utf8') as f:
    d = json.load(f)

with open('gt.txt', 'w', encoding='utf8') as f:
    for k, v in d.items():
        if len(v) != 1:
            print('error', v)
        v = v[0]
        if v['language'].lower() != 'latin':
            # print('Skipping non-Latin:', v)
            continue
        if v['illegibility']:
            # print('Skipping unreadable:', v)
            continue
        label = v['transcription'].strip()
        if not label:
            # print('Skipping blank label')
            continue
        if '#' in label and label != 'LocaL#3':
            # print('Skipping corrupted label')
            continue
        f.write('\t'.join(['train_task2_images/' + k + '.jpg', label]) + '\n')
