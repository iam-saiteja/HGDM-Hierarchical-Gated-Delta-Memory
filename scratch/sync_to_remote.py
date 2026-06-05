import os
import sys
import paramiko

def sync():
    local_dir = r"c:\Users\iamsa\Documents\HTSPC\HTSPC-H3"
    remote_dir = "htspc"
    
    print("Connecting to remote server...", flush=True)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect("100.79.235.83", username="csd23rb8", password="Saiteja@1000", timeout=15)
        sftp = ssh.open_sftp()
        
        # Ensure remote directory exists
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass
            
        exclude_dirs = {'.git', '.venv', '__pycache__', 'one_billion'}
        
        for root, dirs, files in os.walk(local_dir):
            # Filter directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # ONLY upload .py, .gitignore, .md files
                if not (file.endswith('.py') or file == '.gitignore' or file.endswith('.md')):
                    continue
                    
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir).replace('\\', '/')
                remote_path = f"{remote_dir}/{rel_path}"
                
                # Ensure remote parent directory exists
                remote_parent = os.path.dirname(remote_path)
                if remote_parent:
                    parts = remote_parent.split('/')
                    current = ""
                    for part in parts:
                        if not part:
                            continue
                        current = f"{current}/{part}" if current else part
                        try:
                            sftp.mkdir(current)
                        except IOError:
                            pass
                
                print(f"Uploading {rel_path} -> {remote_path}", flush=True)
                sftp.put(local_path, remote_path)
                
        sftp.close()
        ssh.close()
        print("Sync complete!", flush=True)
    except Exception as e:
        print(f"Sync failed: {e}", flush=True)

if __name__ == "__main__":
    sync()
