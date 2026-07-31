"""
统一启动脚本 — 一键启动 A组检索服务 + B组 Agent 服务

用法:
    python scripts/start_servers.py              # 启动两个服务
    python scripts/start_servers.py --retrieval  # 仅启动 A组检索服务
    python scripts/start_servers.py --agent      # 仅启动 B组 Agent 服务
    python scripts/start_servers.py --check      # 检查服务状态

端口:
    A组检索服务: 8000  (http://127.0.0.1:8000/docs)
    B组 Agent:   8002  (http://127.0.0.1:8002/docs)

Ctrl+C 停止全部服务
"""

import os
import sys
import time
import signal
import socket
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RETRIEVAL_PORT = int(os.environ.get("RETRIEVAL_PORT", 8000))
AGENT_PORT = int(os.environ.get("AGENT_PORT", 8002))
RETRIEVAL_HOST = "127.0.0.1"
AGENT_HOST = "0.0.0.0"

# 服务定义
SERVICES = {
    "retrieval": {
        "name": "A组检索服务",
        "port": RETRIEVAL_PORT,
        "host": RETRIEVAL_HOST,
        "cmd": [sys.executable, "-m", "retrieval_service.server"],
        "cwd": str(PROJECT_ROOT / "knowledge_platform" / "retrieval"),
        "health_url": f"http://127.0.0.1:{RETRIEVAL_PORT}/health",
        "env": {**os.environ},
    },
    "agent": {
        "name": "B组Agent服务",
        "port": AGENT_PORT,
        "host": AGENT_HOST,
        "cmd": [sys.executable, "-m", "agent_platform.server"],
        "cwd": str(PROJECT_ROOT),
        "health_url": f"http://127.0.0.1:{AGENT_PORT}/health",
        "env": {**os.environ, "AGENT_PORT": str(AGENT_PORT), "AGENT_HOST": AGENT_HOST},
    },
}

# 运行中的进程
_processes = {}


def print_banner():
    print()
    print("=" * 60)
    print("  ACE-RAG 统一启动脚本")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  A组检索服务: http://127.0.0.1:{RETRIEVAL_PORT}")
    print(f"  B组Agent:   http://127.0.0.1:{AGENT_PORT}")
    print("=" * 60)
    print()


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def check_health(url: str, timeout: int = 30) -> bool:
    """等待服务健康检查通过"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def start_service(key: str) -> bool:
    """启动单个服务"""
    svc = SERVICES[key]
    name = svc["name"]
    port = svc["port"]

    print(f"\n[{name}] 正在启动 (端口 {port}) ...")

    # 检查端口是否已占用
    if is_port_open(port):
        if check_health(svc["health_url"], timeout=3):
            print(f"  [{name}] 端口 {port} 已有服务运行且健康，跳过启动")
            return True
        else:
            print(f"  [{name}] 端口 {port} 被占用但健康检查失败，请先释放端口")
            return False

    # 启动子进程
    print(f"  [{name}] 执行: {' '.join(svc['cmd'])}")
    print(f"  [{name}] 工作目录: {svc['cwd']}")

    try:
        proc = subprocess.Popen(
            svc["cmd"],
            cwd=svc["cwd"],
            env=svc["env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _processes[key] = proc
        print(f"  [{name}] 进程已创建 (PID={proc.pid})")
    except Exception as e:
        print(f"  [{name}] 启动失败: {e}")
        return False

    # 等待健康检查
    print(f"  [{name}] 等待服务就绪 (最多 60 秒) ...")
    ready = False
    start_time = time.time()

    while time.time() - start_time < 60:
        # 检查进程是否意外退出
        if proc.poll() is not None:
            # 读取输出
            output = proc.stdout.read() if proc.stdout else ""
            print(f"  [{name}] 进程意外退出 (code={proc.returncode})")
            if output:
                # 打印最后 20 行输出
                lines = output.strip().split("\n")
                for line in lines[-20:]:
                    print(f"    {line}")
            return False

        # 检查端口
        if is_port_open(port):
            if check_health(svc["health_url"], timeout=5):
                ready = True
                break

        time.sleep(1)

    if ready:
        elapsed = time.time() - start_time
        print(f"  [{name}] 服务就绪! (耗时 {elapsed:.1f}s)")
        print(f"  [{name}] API文档: http://127.0.0.1:{port}/docs")
        return True
    else:
        print(f"  [{name}] 服务启动超时 (60s)")
        return False


def stop_service(key: str):
    """停止单个服务"""
    proc = _processes.get(key)
    if proc is None:
        return

    name = SERVICES[key]["name"]
    if proc.poll() is None:  # 进程仍在运行
        print(f"  [{name}] 正在停止 (PID={proc.pid}) ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print(f"  [{name}] 已停止")
        except subprocess.TimeoutExpired:
            print(f"  [{name}] 强制终止 ...")
            proc.kill()
            proc.wait(timeout=3)
            print(f"  [{name}] 已强制终止")
    _processes.pop(key, None)


def stop_all():
    """停止所有服务"""
    print("\n正在停止所有服务 ...")
    for key in list(_processes.keys()):
        stop_service(key)
    print("全部服务已停止。")


def check_status():
    """检查服务状态"""
    print("\n服务状态检查:")
    print("-" * 50)
    for key, svc in SERVICES.items():
        port = svc["port"]
        name = svc["name"]
        running = is_port_open(port)
        healthy = False
        if running:
            try:
                with urllib.request.urlopen(svc["health_url"], timeout=2) as resp:
                    healthy = resp.status == 200
            except Exception:
                pass

        status = "健康运行" if healthy else ("端口占用(不健康)" if running else "未运行")
        print(f"  {name:12s} | 端口 {port} | {status}")
    print("-" * 50)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACE-RAG 统一启动脚本")
    parser.add_argument("--retrieval", action="store_true", help="仅启动 A组检索服务")
    parser.add_argument("--agent", action="store_true", help="仅启动 B组 Agent 服务")
    parser.add_argument("--check", action="store_true", help="检查服务状态")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    print_banner()

    # 确定要启动的服务
    if args.retrieval and not args.agent:
        keys = ["retrieval"]
    elif args.agent and not args.retrieval:
        keys = ["agent"]
    else:
        keys = ["retrieval", "agent"]  # 默认启动全部

    # 注册信号处理（Ctrl+C 优雅退出）
    def signal_handler(sig, frame):
        print("\n收到中断信号 (Ctrl+C) ...")
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    # 按顺序启动服务
    success = True
    for key in keys:
        if not start_service(key):
            success = False
            print(f"\n[警告] {SERVICES[key]['name']} 启动失败")
            break  # A组失败则不启动 B组

    if not success:
        print("\n启动失败，正在清理已启动的服务 ...")
        stop_all()
        sys.exit(1)

    # 全部启动成功
    print()
    print("=" * 60)
    print("  全部服务已启动!")
    print("=" * 60)
    for key in keys:
        svc = SERVICES[key]
        print(f"  {svc['name']}: http://127.0.0.1:{svc['port']}/docs")
    print()
    print("  按 Ctrl+C 停止全部服务")
    print("=" * 60)
    print()

    # 持续监控进程
    try:
        while True:
            all_alive = True
            for key, proc in list(_processes.items()):
                if proc.poll() is not None:
                    name = SERVICES[key]["name"]
                    print(f"\n[警告] {name} 进程已退出 (code={proc.returncode})")
                    all_alive = False
                    _processes.pop(key, None)

            if not all_alive:
                print("\n有服务异常退出，正在停止剩余服务 ...")
                stop_all()
                sys.exit(1)

            time.sleep(2)

    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
