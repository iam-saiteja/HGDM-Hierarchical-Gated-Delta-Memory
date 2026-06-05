import sys
import os

try:
    import paramiko
except ImportError:
    import subprocess
    print("Installing paramiko...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko

def run_remote_command(cmd):
    print(f"Running on remote: {cmd}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect("100.79.235.83", username="csd23rb8", password="Saiteja@1000", timeout=10)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        # Read outputs safely
        stdout_str = stdout.read().decode(errors='replace')
        stderr_str = stderr.read().decode(errors='replace')
        
        # Determine encoding
        enc = sys.stdout.encoding or 'utf-8'
        
        print("--- STDOUT ---")
        sys.stdout.write(stdout_str.encode(enc, errors='replace').decode(enc))
        print("\n--- STDERR ---")
        sys.stdout.write(stderr_str.encode(enc, errors='replace').decode(enc))
        print("")
        
        ssh.close()
        return exit_status
    except Exception as e:
        print(f"SSH Exception: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ssh.py <command>")
        sys.exit(1)
    cmd = " ".join(sys.argv[1:])
    sys.exit(run_remote_command(cmd))
