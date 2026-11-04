"""作业运行器的安全边界:名字不在白名单里就 404,同名作业没跑完不许重入。

这两条是「页面上有重跑按钮但没有 shell」的实证。另加一条注册表自检:每个作业都得是
仓库 Makefile 里真有的目标,免得页面按钮指向一个不存在的配方。
"""
import re
import pathlib

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import jobs as jobs_api
from app.core import jobs


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(jobs_api.router)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_unknown_job_is_rejected(client):
    """前端只传作业名。名字对不上就到此为止——命令片段根本没有地方可以传进来。"""
    for path, method in (("/api/jobs/rm -rf", "post"), ("/api/jobs/rm -rf", "get"),
                         ("/api/jobs/未注册/stop", "post")):
        resp = await getattr(client, method)(path)
        assert resp.status_code == 404, path
        assert "未注册的作业" in resp.json()["detail"]


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业在跑就拒绝再发起,否则两份进程抢同一批产物。"""
    monkeypatch.setitem(jobs._runs, "kb-build",
                        jobs.JobRun(status="running", pid=4242))
    resp = await client.post("/api/jobs/kb-build")
    assert resp.status_code == 409 and "正在运行中" in resp.json()["detail"]


async def test_status_all_covers_registry(client):
    body = (await client.get("/api/jobs")).json()
    assert {j["name"] for j in body["jobs"]} == set(jobs.JOBS)
    for j in body["jobs"]:
        assert j["cmd"].startswith("make ")     # 一律走 make,页面按的和终端敲的是同一条命令


def test_every_job_target_exists_in_makefile():
    text = (pathlib.Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z0-9_.-]+):", text, re.MULTILINE))
    missing = [spec.argv[1] for spec in jobs.JOBS.values() if spec.argv[1] not in targets]
    assert not missing, f"作业指向的 make 目标不存在:{missing}"


async def test_windows_original_recipe_runs_and_log_is_readable(tmp_path, monkeypatch):
    """真实执行只读配方:覆盖 Make/Bash 路径含空格、中文编码、退出码和日志。"""
    import os
    if os.name != "nt":
        pytest.skip("Windows integration")
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_runs", {})
    run = await jobs.start("kb-preview")
    await run.watcher
    assert run.status == "ok" and run.returncode == 0
    assert "总块数" in jobs.tail("kb-preview")


async def test_windows_stop_terminates_task_tree(tmp_path, monkeypatch):
    """终止实际父子进程,确认不会留下后台任务,而非只检查调用参数。"""
    import asyncio
    import os
    import sys
    if os.name != "nt":
        pytest.skip("Windows integration")
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_runs", {})
    child_file = tmp_path / "child.pid"
    parent = (
        "import subprocess,sys,time,pathlib;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid));time.sleep(120)"
    )
    monkeypatch.setattr(jobs, "job_command", lambda spec, env: (sys.executable, "-c", parent))
    run = await jobs.start("kb-preview")
    for _ in range(100):
        if child_file.exists():
            break
        await asyncio.sleep(.05)
    assert child_file.exists()
    child_pid = int(child_file.read_text())
    try:
        await jobs.stop("kb-preview")
        await run.watcher
        assert run.status == "stopped"
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, child_pid)
        if handle:
            try:
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                assert code.value != 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
    finally:
        if run.proc is not None:
            await jobs.stop("kb-preview")


def test_posix_job_command_keeps_original_argv(monkeypatch):
    monkeypatch.setattr(jobs, "IS_WINDOWS", False)
    spec = jobs.JOBS["calibrate-confidence"]
    assert jobs.job_command(spec, {}) == spec.argv


async def test_flywheel_start_route_uses_registered_recipe(client, monkeypatch):
    started = []
    async def fake_start(name):
        started.append(jobs.JOBS[name].argv)
        return jobs.JobRun(status="running")
    monkeypatch.setattr(jobs, "start", fake_start)
    assert (await client.post("/api/jobs/flywheel")).status_code == 200
    assert started == [("make", "flywheel")]


async def test_simultaneous_flywheel_starts_refuse_reentry(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_runs", {})
    monkeypatch.setattr(jobs, "job_command", lambda spec, env: ("unused-test-command",))
    entered, release = asyncio.Event(), asyncio.Event()
    async def fake_spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        raise FileNotFoundError("test spawn failure")
    monkeypatch.setattr(jobs.asyncio, "create_subprocess_exec", fake_spawn)
    first = asyncio.create_task(jobs.start("flywheel"))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="正在运行中"):
            await jobs.start("flywheel")
    finally:
        release.set()
        with pytest.raises(FileNotFoundError):
            await first
    assert jobs.status("flywheel")["status"] == "failed"
