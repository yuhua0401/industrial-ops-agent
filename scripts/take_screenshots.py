"""
take_screenshots.py - 生成 README 效果图

用 Playwright 驱动系统 Edge（无头）对运行中的演示页
（默认 http://localhost:8000/）做真实交互并截图：

    welcome.png          欢迎屏 + 预设问题
    login.png            JWT 登录弹窗
    chat-knowledge.png   知识问答（路由卡片 + RAG 来源）
    chat-diagnosis.png   故障诊断（置信度 + 诊断结论）
    chat-pipeline.png    全流程报修（Pipeline 时间线 + 工单 + 售后）
    chat-after-sale.png  售后配件查询（parts 库存）

前置：服务已启动（uvicorn backend.main:app --port 8000），
     数据面容器与种子数据就绪（python scripts/preflight_check.py 全绿）。

用法：
    python scripts/take_screenshots.py [--url http://localhost:8000/]
"""
import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "docs" / "images"

# 预设问题按钮的 data-msg 前缀（与 index.html / app.js 模板一致）
SEL_WELCOME = ".preset-grid .q-btn >> nth=0"
SEL_KNOWLEDGE = '.q-btn[data-msg^="CNC-1000 的主轴"]'
SEL_DIAGNOSIS = '.q-btn[data-msg^="设备报E001"]'
SEL_PIPELINE = '.q-btn[data-msg^="设备报E002"]'
SEL_PARTS = '.q-btn[data-msg^="主轴轴承 BRG-6204"]'


async def shot(page, name: str) -> None:
    await page.screenshot(path=str(OUT / name))
    print(f"  ✔ {name}")


async def wait_response_done(page, timeout_ms: int) -> None:
    """等待一轮回答结束：发送键经历 隐藏→重新可见。"""
    await page.wait_for_selector("#sendBtn", state="hidden", timeout=15000)
    await page.wait_for_selector("#sendBtn", state="visible", timeout=timeout_ms)
    await page.wait_for_timeout(1200)   # 等流式光标/进度行清理


async def new_session(page) -> None:
    await page.click("#newSessionBtn")
    await page.wait_for_selector(".preset-grid .q-btn", state="visible")
    await page.wait_for_timeout(700)


async def run(url: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # 优先系统 Edge/Chrome，免下载浏览器；都不可用则回退 Playwright 自带 Chromium
        launch_kwargs = {"headless": True}
        browser = None
        for channel in ("msedge", "chrome"):
            try:
                browser = await p.chromium.launch(channel=channel, **launch_kwargs)
                print(f"浏览器: 系统 {channel}")
                break
            except Exception:
                continue
        if browser is None:
            browser = await p.chromium.launch(**launch_kwargs)
            print("浏览器: Playwright Chromium")

        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
            locale="zh-CN",
        )
        page = await ctx.new_page()
        page.set_default_timeout(30000)

        print("打开演示页…")
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_selector(".preset-grid .q-btn", state="visible")

        # 预填设备上下文（pipeline 售后环节据此查保修），并断言确已写入
        await page.fill("#deviceModel", "CNC-1000")
        await page.fill("#deviceSn", "SN-AC-0001")
        sn_val = await page.evaluate("document.getElementById('deviceSn').value")
        assert sn_val == "SN-AC-0001", f"deviceSn 预填失败: {sn_val!r}"

        await page.wait_for_timeout(1500)
        await shot(page, "welcome.png")

        # ── 登录弹窗 ─────────────────────────────────────────
        await page.click("#loginBtn")
        await page.wait_for_selector("#loginModal:not(.hidden)", state="visible")
        await page.wait_for_timeout(400)
        await shot(page, "login.png")

        await page.click("#loginSubmitBtn")
        await page.wait_for_selector("#userInfo", state="visible")
        print("  登录成功（engineer）")

        # ── 场景：知识问答 ───────────────────────────────────
        print("场景：知识问答…")
        await new_session(page)
        await page.click(SEL_KNOWLEDGE)
        await wait_response_done(page, 150000)
        await shot(page, "chat-knowledge.png")

        # ── 场景：故障诊断 ───────────────────────────────────
        print("场景：故障诊断…")
        await new_session(page)
        await page.click(SEL_DIAGNOSIS)
        await wait_response_done(page, 150000)
        await shot(page, "chat-diagnosis.png")

        # ── 场景：全流程报修 ─────────────────────────────────
        print("场景：全流程报修（诊断→工单→售后，较慢）…")
        await new_session(page)
        await page.click(SEL_PIPELINE)
        await wait_response_done(page, 300000)
        await shot(page, "chat-pipeline.png")

        # ── 场景：售后配件 ───────────────────────────────────
        print("场景：售后配件…")
        await new_session(page)
        await page.click(SEL_PARTS)
        await wait_response_done(page, 150000)
        await shot(page, "chat-after-sale.png")

        await browser.close()

    print(f"\n完成，共 6 张 → {OUT}")
    return 0


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成 README 效果图")
    parser.add_argument("--url", default="http://localhost:8000/", help="演示页地址")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url)))
