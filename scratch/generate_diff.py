import subprocess
import os

def main():
    cwd = "c:/Users/iamsa/Documents/HTSPC/HTSPC-H3"
    cmd = "git diff 4f673c882eb74a12858b0f71c7318422ad9dca90..HEAD -- hgdm_omega.py"
    res = subprocess.check_output(cmd, shell=True, cwd=cwd).decode("utf-8", errors="ignore")
    out_path = os.path.join(cwd, "scratch/diff_omega_utf8.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(res)
    print("Diff saved to scratch/diff_omega_utf8.txt")

if __name__ == "__main__":
    main()
