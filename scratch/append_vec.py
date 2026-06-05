import os

with open('tests/test_vec_scan.py', 'r') as f:
    content = f.read()

parts = content.split('# FORWARD KERNEL\n# ──────────────────────────────────────────────────────────────────\n')
kernels = '# ──────────────────────────────────────────────────────────────────\n# VECTOR SCAN KERNELS\n# ──────────────────────────────────────────────────────────────────\n' + parts[1].split('# TEST HARNESS')[0]

with open('kernel_nitro.py', 'a') as f:
    f.write('\n\n')
    f.write(kernels)
    
print("Appended successfully.")
