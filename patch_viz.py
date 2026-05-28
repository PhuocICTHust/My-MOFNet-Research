import pathlib

f = pathlib.Path(r'E:\MOFNet\可视化柱状图.py')
lines = f.read_text(encoding='utf-8').splitlines(keepends=True)

out = []
skip = False
for line in lines:
    stripped = line.strip()
    # Detect start of the bad if/elif block
    if stripped == "if disease == 'STAD':":
        skip = True
        out.append('    num_pool = 2\n')
        continue
    # Stop skipping after the last elif block's last line
    if skip:
        # The block ends after "num_pool = 2" inside the elif disease=='BRCA' branch
        # We skip until we hit a line that is NOT part of this if/elif
        if stripped.startswith('if ') or stripped.startswith('elif ') or stripped.startswith('test_pth_file_name') or stripped.startswith('num_pool'):
            continue
        else:
            skip = False
            out.append(line)
    else:
        out.append(line)

f.write_text(''.join(out), encoding='utf-8')
print('Patched OK')

# Verify
remaining = [l for l in pathlib.Path(r'E:\MOFNet\可视化柱状图.py').read_text(encoding='utf-8').splitlines() if '1650' in l or ('模态' in l and 'pth' in l)]
if remaining:
    print('WARNING - still found old lines:')
    for r in remaining:
        print(' ', r)
else:
    print('Clean - no old paths remain')
