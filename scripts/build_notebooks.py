"""Generate standalone Colab notebooks from the tested shared module."""

import argparse
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def code_cell(cell_id, source, *, hidden=False):
    metadata = {"id": cell_id}
    if hidden:
        metadata["cellView"] = "form"
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build_notebook(subtitle_format):
    label = subtitle_format.upper()
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
                f"# 英文音频 → 中英双语 {label}\n",
                "选择 GPU 运行时，依次执行各步骤，在 Step 2 设置音频文件及翻译接口。\n",
                "输出保存在音频所在目录，保留原音频文件名。API Key 可在运行时隐藏输入。\n",
                "关闭中文输出即可跳过 API。自动断开默认关闭，出错或翻译不全时保留运行时。\n",
                "维护说明：此文件由 `scripts/build_notebooks.py` 生成。\n",
            ],
        }
    ]
    cells.append(
        code_cell(
            "setup",
            f"""
        # [Step 1] 安装依赖并挂载 Google Drive
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
    cells.append(
        code_cell(
            "config",
            """
        # [Step 2] 配置：修改文件名、API 地址和模型
        import os
        from getpass import getpass

        AUDIO_FILENAME = "example.mp3"
        DRIVE_FOLDER = "/content/drive/MyDrive/podcast"
        OUTPUT_CHINESE = True
        BASE_URL = os.environ.get("TRANSLATION_BASE_URL", "https://api.openai.com/v1")
        MODEL_NAME = os.environ.get("TRANSLATION_MODEL", "gpt-4.1-mini")
        API_KEY = os.environ.get("TRANSLATION_API_KEY", "")
        MAX_WORKERS = 5
        MAX_RETRIES = 2  # 首次请求失败后的额外尝试次数；0 表示不重试
        REQUEST_TIMEOUT = 30
        WHISPER_MODEL = "turbo"
        AUTO_DISCONNECT = False  # 仅在全部成功保存后自动断开

        # 清除上次任务状态，防止失败后误用旧结果。
        segments = None
        source_path = None
        translation_config = None
        if OUTPUT_CHINESE and not API_KEY:
            API_KEY = getpass("请输入翻译 API Key（输入不会显示）：")
        print(f"配置完成：{AUDIO_FILENAME}")
    """,
        )
    )
    core = (ROOT / "audio2text.py").read_text(encoding="utf-8")
    cells.append(code_cell("helpers", "#@title [Step 3] 加载字幕处理函数（直接运行）\n" + core, hidden=True))
    cells.append(
        code_cell(
            "transcribe",
            """
        # [Step 4] 将音频复制到独立临时目录，执行英文转录
        import time

        segments = None
        source_path = None
        translation_config = None
        candidate_path = resolve_audio_path(DRIVE_FOLDER, AUDIO_FILENAME)
        if OUTPUT_CHINESE:
            translation_config = TranslationConfig(
                api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME,
                max_workers=MAX_WORKERS, max_retries=MAX_RETRIES,
                timeout=REQUEST_TIMEOUT,
            )
        started = time.monotonic()
        segments = transcribe_audio(candidate_path, model=WHISPER_MODEL)
        source_path = candidate_path
        print(f"转录完成：{len(segments)} 个片段，耗时 {time.monotonic() - started:.1f} 秒。")
        if not any(segment["text"] for segment in segments):
            print("未识别到有效文本，请检查音频内容。")
    """,
        )
    )
    cells.append(
        code_cell(
            "export",
            f'''
        # [Step 5] 并发翻译并保存 {label}；可单独重跑本步骤，无需再次转录
        import time

        if segments is None or source_path is None:
            raise RuntimeError("请先成功运行 Step 4，再生成字幕。")

        # 支持重跑本步骤时修改 API 配置或关闭中文输出。
        translation_config = None
        if OUTPUT_CHINESE:
            translation_config = TranslationConfig(
                api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME,
                max_workers=MAX_WORKERS, max_retries=MAX_RETRIES,
                timeout=REQUEST_TIMEOUT,
            )

        def show_progress(finished, total):
            if finished % 20 == 0 or finished == total:
                print(f"翻译进度：{{finished}}/{{total}}")

        started = time.monotonic()
        output_path, result = export_subtitles(
            source_path, segments, subtitle_format="{subtitle_format}",
            config=translation_config, output_chinese=OUTPUT_CHINESE,
            progress=show_progress,
        )
        print(f"字幕已保存：{{output_path}}")
        print(f"本步骤耗时：{{time.monotonic() - started:.1f}} 秒。")
        if OUTPUT_CHINESE:
            print(f"翻译覆盖率：{{len(result.translations)}}/{{result.total}}（不计空白片段）")
        else:
            print("已导出英文字幕，未调用翻译 API。")
        if result.failed_indices:
            print(f"仍有 {{len(result.failed_indices)}} 个片段翻译失败，已保留英文。")
            print(f"失败片段索引（从 0 开始）：{{result.failed_indices}}")
            print("可检查接口后重跑 Step 5；重跑会重新翻译所有非空片段。")
        elif AUTO_DISCONNECT:
            from google.colab import runtime
            print("字幕已完整保存，5 秒后断开运行时。")
            time.sleep(5)
            runtime.unassign()
    ''',
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"audio2{subtitle_format}.ipynb", "gpuType": "T4", "provenance": []},
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
    for subtitle_format in ("lrc", "srt"):
        path = ROOT / f"audio2{subtitle_format}.ipynb"
        content = json.dumps(build_notebook(subtitle_format), ensure_ascii=False, indent=2) + "\n"
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
