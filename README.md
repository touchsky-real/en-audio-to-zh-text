# 🎙️ 英文音频自动转中英字幕

在 Google Colab 中使用 Whisper Turbo 转录英文播客，再通过 OpenAI 兼容的翻译接口生成 **LRC 歌词**或 **SRT 字幕**。也可以关闭翻译，只导出英文。

| 用途 | 笔记本 | 在 Colab 中打开 |
| --- | --- | --- |
| 音乐播放器、播客歌词 | [audio2lrc.ipynb](audio2lrc.ipynb) | [打开 LRC 版](https://colab.research.google.com/github/touchsky-real/en-audio-to-zh-text/blob/main/audio2lrc.ipynb) |
| 视频播放器字幕 | [audio2srt.ipynb](audio2srt.ipynb) | [打开 SRT 版](https://colab.research.google.com/github/touchsky-real/en-audio-to-zh-text/blob/main/audio2srt.ipynb) |

## 快速开始

1. 在 Google Drive 中创建 `podcast` 文件夹，上传 MP3、WAV 等音频。
2. 打开上面的 Colab 笔记本，选择 GPU 运行时（例如 T4），执行 Step 1 安装依赖并挂载 Drive。
3. 修改 Step 2 的配置，执行时按提示输入 API Key。密钥通过隐藏输入读取，无需写入代码。
4. 依次运行 Step 3～5。字幕会保存到音频所在目录。

```python
AUDIO_FILENAME = "example.mp3"  # Drive 中的完整文件名，保留空格和特殊字符
DRIVE_FOLDER = "/content/drive/MyDrive/podcast"
OUTPUT_CHINESE = True           # False：仅输出英文，不需要 API Key
BASE_URL = "https://api.openai.com/v1"  # 可换成兼容接口的 API 根地址
MODEL_NAME = "gpt-4.1-mini"      # 改为接口实际支持的模型
MAX_WORKERS = 5
MAX_RETRIES = 2                 # 首次请求之外最多再尝试 2 次
AUTO_DISCONNECT = False         # True：全部成功保存后自动释放运行时
```

也可通过 `TRANSLATION_API_KEY`、`TRANSLATION_BASE_URL`、`TRANSLATION_MODEL` 环境变量提供接口配置。`BASE_URL` 通常以 `/v1` 结尾，不要包含 `/chat/completions`。

Colab 免费 GPU 的可用性和运行时长受平台配额限制；翻译接口的费用取决于你的服务商。

## 输出与重试

- LRC 版生成 `example.lrc`，英文和中文使用相同时间戳。与音频同名、放在同一目录即可供兼容的播放器读取。
- SRT 版生成 `example_en.srt` 和 `example.srt`，分别保存英文字幕和最终字幕。
- 原音频不改名。转录使用独立的本地临时目录，命令失败会直接报错，不会读取以前生成的字幕。
- 翻译会复用每个工作线程的 HTTP 连接；超时、限流、服务端错误和无效返回会按配置重试，等待时间逐步增加，单次最多 60 秒。
- API Key、地址等导致的永久 HTTP 错误会停止翻译并报出状态码。已在途的请求会等待结束。
- 重试耗尽的片段保留英文，并显示失败索引。空白片段不计入翻译覆盖率。
- 转录成功后可以单独重跑 Step 5，无需再次转录；**每次重跑会重新翻译所有非空片段**，可能产生新的 API 费用。
- 自动断开默认关闭；启用后也只在完整保存、没有翻译失败时执行。转录错误、API 错误和保存错误会保留运行时，便于处理。
- 同名字幕会在完整写入后替换。请先另存需要保留的旧版本。

## 如何更新 Colab 副本

上面的入口打开 GitHub `main` 分支中的笔记本。仓库更新后，重新从入口打开即可使用新版。

**已经保存到 Google Drive 的副本不会自动同步。** 请重新打开新版并保存副本，或手动替换旧副本中的代码，再填回自己的配置。两份笔记本均自带公共代码，运行时不需要额外下载本仓库的 Python 文件。

## 本地使用

需要 Python 3.10+ 和已加入 PATH 的 FFmpeg。首次转录会下载 Whisper 模型权重；GPU 可加速转录，CPU 也可运行但较慢。

```bash
python -m pip install -r requirements.txt
```

```python
import os
from audio2text import TranslationConfig, export_subtitles, transcribe_audio

audio = "example.mp3"
config = TranslationConfig(
    api_key=os.environ["TRANSLATION_API_KEY"],
    base_url="https://api.openai.com/v1",
    model="gpt-4.1-mini",
)
segments = transcribe_audio(audio)
path, result = export_subtitles(audio, segments, "lrc", config)
print(path, result.failed_indices)

# 仅英文：export_subtitles(audio, segments, "srt", output_chinese=False)
```

## 开发维护

公共逻辑位于 `audio2text.py`，笔记本配置与步骤模板位于 `scripts/build_notebooks.py`。修改后重新生成两份笔记本，避免分别维护导致行为不一致：

```bash
python scripts/build_notebooks.py
python scripts/build_notebooks.py --check
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m ruff check .
```

开发检查无需安装 Whisper、下载模型或提供 API Key。测试覆盖转录失败隔离、文件名冲突、时间戳进位、翻译重试、并发顺序、文件写入失败及两份笔记本的独立执行流程。GitHub Actions 在推送和 PR 时自动运行检查。

依赖使用 PyPI 发布版本，不再从 Whisper 的 Git 主分支安装。升级依赖时同步修改 `requirements.txt` 和 `requirements-dev.txt` 中的 `requests` 版本，并重新生成笔记本。

## 效果示例

![Demo](.img/demo.jpg)

## 数据说明

Whisper 在你的 Colab 或本地运行时中处理音频；翻译时会将识别出的文本发送到你配置的 API。音频会从 Drive 复制到运行时临时目录，处理后清理。分享笔记本前请检查自己添加的密钥和包含私人内容的输出。
