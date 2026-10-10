
import os, zipfile, shutil

datasets_map = {
    'acm.zip': 'acm',
    'webkb.zip': 'webkb',

    'sst1.zip': 'sst1',
    'dblp.zip': 'dblp',
    'books.zip': 'books',
    '20ng.zip': '20ng',
}

os.makedirs('datasets', exist_ok=True)
zips_dir = 'zips' if os.path.exists('zips') else '.'

for zip_file, dname in datasets_map.items():
    full_zip = os.path.join(zips_dir, zip_file)
    if not os.path.exists(full_zip):
        continue
    target_dir = os.path.join('datasets', dname)
    print(f'Extraindo {zip_file} -> {target_dir}...')
    with zipfile.ZipFile(full_zip, 'r') as zf:
        zf.extractall(target_dir)

    # Se extraiu com uma subpasta interna com mesmo nome, achata
    inner = [os.path.join(target_dir, x) for x in os.listdir(target_dir)]
    if len(inner) == 1 and os.path.isdir(inner[0]):
        sub = inner[0]
        for f in os.listdir(sub):
            shutil.move(os.path.join(sub, f), target_dir)
        os.rmdir(sub)

    # Organiza splits
    splits_dir = os.path.join(target_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    for f in os.listdir(target_dir):
        if f.startswith('split_') and f.endswith('.pkl'):
            shutil.move(os.path.join(target_dir, f), os.path.join(splits_dir, f))

print('Todos os datasets descompactados e organizados com sucesso!')

   python info_scripts/generate_jina_v5.py \
      --datasets webkb reuters90 dblp books acm 20ng sst1 \
      --folds 10