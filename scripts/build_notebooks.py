"""Generate standalone Colab notebooks from the tested shared module."""

import argparse
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def code_cell(cell_id, source):
    metadata = {"id": cell_id}
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build_notebook(notebook_name):
    dependencies = [
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    cells = [
        {
            "cell_type": "markdown",
            "id": "intro",
            "metadata": {"id": "intro"},
            "source": [
                "# 英文音频 → 中英双语字幕（LRC / SRT）\n",
                "先填写下方必要参数，再依次运行：**1. 配置参数 → 2. 准备环境 → 3. 开始处理**。建议选择 GPU 运行时。\n",
                "字幕保存到音频所在目录。默认保存进度，补译时只需重跑 Step 3；重启运行时后依次运行三步。\n",
                '默认生成 LRC 歌词；在 Step 1 设置 `OUTPUT_FORMAT = "srt"` 可生成 SRT 字幕。\n',
                "设置 `OUTPUT_CHINESE = False` 仅输出英文；`SAVE_PROGRESS = False` 关闭进度保存与恢复。\n",
            ],
        }
    ]
    cells.append(
        code_cell(
            "config",
            """
        # [Step 1] 配置参数：先填写或确认下方必要参数
        import os
        from getpass import getpass

        # 必要参数：音频文件和目录必须正确；翻译接口仅在输出中文时需要
        AUDIO_FILENAME = "example.mp3"  # 必填：Drive 中的文件名，包含扩展名
        DRIVE_FOLDER = "/content/drive/MyDrive/podcast"  # 音频所在目录，按准备说明上传可不改
        API_KEY = os.environ.get("TRANSLATION_API_KEY", "")  # 留空时运行本步骤会提示隐藏输入
        BASE_URL = os.environ.get("TRANSLATION_BASE_URL", "https://api.openai.com/v1")  # API 根地址
        MODEL_NAME = os.environ.get("TRANSLATION_MODEL", "gpt-4.1-mini")  # 接口支持的翻译模型

        # 可选参数：一般保持默认即可
        OUTPUT_FORMAT = "lrc"  # "lrc"：歌词（默认）；"srt"：字幕
        OUTPUT_CHINESE = True  # False：仅输出英文，不需要翻译接口
        SAVE_PROGRESS = True  # 保存并恢复进度；False：不读写进度文件，每次重新处理
        AUTO_DISCONNECT = False  # True：仅在全部成功保存后自动断开
        MAX_WORKERS = 5
        MAX_RETRIES = 2  # 首次请求失败后的额外尝试次数；0 表示不重试
        REQUEST_TIMEOUT = 30  # 每次翻译请求的超时秒数
        WHISPER_MODEL = "turbo"

        if OUTPUT_CHINESE and not API_KEY:
            API_KEY = getpass("请输入翻译 API Key（输入不会显示）：")
        print(f"配置完成：{AUDIO_FILENAME}")
    """,
        )
    )
    cells.append(
        code_cell(
            "setup",
            f"""
        # [Step 2] 安装依赖并挂载 Google Drive（直接运行，无需修改）
        import shutil
        import subprocess
        import sys
        from google.colab import drive

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *{dependencies!r}],
            check=True,
        )
        if shutil.which("ffmpeg") is None:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
        drive.mount("/content/drive")
        print("环境准备完成。")
    """,
        )
    )
    core = (ROOT / "audio2text.py").read_text(encoding="utf-8")
    pipeline = dedent(
        """
        def run_task():
            import time

            required = (
                "DRIVE_FOLDER", "AUDIO_FILENAME", "WHISPER_MODEL", "OUTPUT_CHINESE",
                "API_KEY", "BASE_URL", "MODEL_NAME", "MAX_WORKERS", "MAX_RETRIES",
                "REQUEST_TIMEOUT", "SAVE_PROGRESS", "AUTO_DISCONNECT", "OUTPUT_FORMAT",
            )
            if any(name not in globals() for name in required):
                raise RuntimeError("请先运行 Step 1 配置参数；重启运行时后请依次运行三步。")
            if OUTPUT_FORMAT not in ("lrc", "srt"):
                raise ValueError('OUTPUT_FORMAT 只能填写 "lrc" 或 "srt"。请修改 Step 1 后重新运行。')
            source_path = resolve_audio_path(DRIVE_FOLDER, AUDIO_FILENAME)
            translation_config = None
            if OUTPUT_CHINESE:
                translation_config = TranslationConfig(
                    api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME,
                    max_workers=MAX_WORKERS, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT,
                )
            started = time.monotonic()
            print("正在准备转录（有匹配的进度时自动恢复）...")
            segments = transcribe_audio(source_path, model=WHISPER_MODEL, save_progress=SAVE_PROGRESS)
            print(f"转录已就绪：{len(segments)} 个片段。")
            if not any(segment["text"] for segment in segments):
                print("未识别到有效文本，请检查音频内容。")

            def show_progress(finished, total):
                if finished % 20 == 0 or finished == total:
                    print(f"本轮翻译进度：{finished}/{total}")

            output_path, result = export_subtitles(
                source_path, segments, subtitle_format=OUTPUT_FORMAT,
                config=translation_config, output_chinese=OUTPUT_CHINESE,
                progress=show_progress, save_progress=SAVE_PROGRESS,
            )
            print(f"字幕已保存：{output_path}，耗时 {time.monotonic() - started:.1f} 秒。")
            if OUTPUT_CHINESE:
                print(f"翻译覆盖率：{len(result.translations)}/{result.total}，复用 {result.reused} 条。")
            else:
                print("已导出英文字幕，未调用翻译 API。")
            if not SAVE_PROGRESS:
                print("进度保存已关闭，再次运行将重新处理全部内容。")
            if result.failed_indices:
                print(f"仍有 {len(result.failed_indices)} 条翻译失败，已保留英文。")
                if SAVE_PROGRESS:
                    print("检查接口后重跑 Step 3，即可继续补译。")
            elif AUTO_DISCONNECT:
                from google.colab import runtime
                print("字幕已完整保存，5 秒后断开运行时。")
                time.sleep(5)
                runtime.unassign()
            return output_path, result

        output_path, result = run_task()
        """
    )
    cells.append(
        code_cell(
            "run",
            "# [Step 3] 开始处理 / 继续补译（直接运行，无需修改）\n" + core + pipeline,
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": notebook_name, "gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing files")
    args = parser.parse_args()
    outdated = []
    for notebook_name in ("audio2lrc.ipynb", "audio2srt.ipynb"):
        path = ROOT / notebook_name
        content = json.dumps(build_notebook(notebook_name), ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                outdated.append(path.name)
        else:
            path.write_text(content, encoding="utf-8")
    if outdated:
        parser.exit(
            1, "Outdated notebooks: " + ", ".join(outdated) + "\nRun python scripts/build_notebooks.py\n"
        )


if __name__ == "__main__":
    main()
