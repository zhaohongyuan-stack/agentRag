"""
部署脚本：同步修改的后端代码 + 前端模板 + nginx 配置到服务器，重建 agent 容器。

上传文件:
  - agent_platform/gateway/request_handler/event_callback.py (新)
  - agent_platform/gateway/request_handler/handler.py
  - agent_platform/server.py
  - agent_platform/runtime/llm_client.py
  - deployment/frontend/agent_template.html (新)
  - deployment/nginx/nginx.conf

随后:
  - docker compose up -d --build agent   (重建 agent 镜像)
  - docker compose restart nginx         (应用 nginx 配置)
"""
import paramiko
import os
import time

HOST, PORT, USER, PASSWORD = "101.201.59.211", 22, "root", "@Zhy123456"
BASE = r"c:\Users\zhaoh\OneDrive\桌面\研究生大创\ragagent"

FILES = [
    # (本地相对路径, 服务器相对路径)
    (r"agent_platform\gateway\request_handler\event_callback.py", "agent_platform/gateway/request_handler/event_callback.py"),
    (r"agent_platform\gateway\request_handler\handler.py", "agent_platform/gateway/request_handler/handler.py"),
    (r"agent_platform\server.py", "agent_platform/server.py"),
    (r"agent_platform\runtime\llm_client.py", "agent_platform/runtime/llm_client.py"),
    (r"agent_platform\agents\verifier_agent.py", "agent_platform/agents/verifier_agent.py"),
    (r"deployment\frontend\agent_template.html", "deployment/frontend/agent_template.html"),
    (r"deployment\nginx\nginx.conf", "deployment/nginx/nginx.conf"),
]

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("连接服务器...")
    ssh.connect(HOST, PORT, USER, PASSWORD, timeout=30)
    sftp = ssh.open_sftp()

    # 1. 上传文件
    for local_rel, remote_rel in FILES:
        local = os.path.join(BASE, local_rel)
        remote = f"/opt/ace-rag/{remote_rel}"
        if not os.path.exists(local):
            print(f"  [跳过] 本地不存在: {local_rel}")
            continue
        sftp.put(local, remote)
        size = os.path.getsize(local)
        print(f"  [上传] {remote_rel} ({size/1024:.1f} KB)")

    sftp.close()

    # 2. 重建 agent 容器
    print("\n重建 agent 容器 (docker compose up -d --build agent)...")
    stdin, stdout, stderr = ssh.exec_command(
        "cd /opt/ace-rag && docker compose up -d --build agent 2>&1 | tail -20"
    )
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    print(out)

    # 3. 重启 nginx 应用配置
    print("\n重启 nginx...")
    stdin, stdout, stderr = ssh.exec_command("cd /opt/ace-rag && docker compose restart nginx 2>&1 | tail -5")
    stdout.channel.recv_exit_status()
    print(stdout.read().decode().strip())

    # 4. 验证 agent 容器状态
    print("\n等待 agent 服务启动...")
    time.sleep(12)
    stdin, stdout, stderr = ssh.exec_command("docker ps --filter name=agent --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
    stdout.channel.recv_exit_status()
    print(stdout.read().decode().strip())

    ssh.close()
    print("\n部署完成！")

if __name__ == "__main__":
    main()