# 🎙️ Audio2LRC: 英文播客/音频自动转中文字幕

> "我想听英文播客的时候发现，这些播客都没有中文字幕，于是就有了这个项目。"

**LRC 歌词版** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/touchsky-real/en-audio-to-zh-text/blob/main/audio2lrc.ipynb) &emsp;&emsp; **SRT 字幕版** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/touchsky-real/en-audio-to-zh-text/blob/main/audio2srt.ipynb)


## 📖 项目介绍

本项目是一个运行在 Google Colab 上的自动化脚本，旨在将英文音频（MP3/WAV等）快速转换为 **中英对照的 LRC 歌词文件**。

**为什么选择 LRC 格式？**
对于播客（Podcast）或音频听众来说，LRC 格式比 SRT 更实用。只需将生成的 `.lrc` 文件与音频文件命名一致并放在同一目录下，大多数主流音乐播放器（如 Musicolet, Salt Player等）即可自动识别并滚动显示歌词。

### ✨ 核心特性

- **零成本**: 完全依托 [Google Colab](https://colab.google/) 的免费 T4 GPU 资源运行。
- **极速转录**: 使用 OpenAI 的 **Whisper (Turbo模型)**，转录速度极快且准确。
- **AI 翻译**: 支持接入 OpenAI 兼容格式的 API（如 GPT-4o-mini, DeepSeek 等）进行高质量翻译。
- **并发加速**: 内置多线程并发翻译机制，长音频处理效率高。
- **自动清洗**: 自动处理文件名中的特殊字符，防止转录出错。

---

## ⚙️ 准备工作

在使用脚本前，请先在您的 Google Drive 中准备好以下环境：

1.  **创建文件夹**：
    在 Google Drive 根目录下新建一个文件夹，命名为 `podcast`。
    - 完整路径应为：`/content/drive/MyDrive/podcast`
2.  **上传音频**：
    将需要转换的英文音频文件上传到该文件夹中。

---

## 🚀 快速开始

1.  点击上方的 **"Open in Colab"** 按钮打开笔记本。
2.  **连接运行时**：点击右上角 "连接"，建议选择 T4 GPU 环境。
3.  **先填写必要参数**：打开笔记本后，第一个代码块就是 **[Step 1] 配置参数**。按照下方说明填写文件名、确认目录和翻译接口；可选参数一般保持默认。
4.  **依次运行三个代码块**：**Step 1 配置参数 → Step 2 准备环境 → Step 3 开始处理**。API Key 留空时，Step 1 会提示隐藏输入；Step 2 会请求挂载 Google Drive。

生成的 `.lrc` 或 `.srt` 文件会保存到音频所在目录。默认开启进度保存，补译时只需重新运行 **Step 3**，已成功的片段不会重复翻译。修改参数后，重新运行 **Step 1、Step 3**；重启运行时后则需依次运行三步。

### 必要参数：先填写或确认这几项

| 参数 | 是否需要修改 | 填写说明 |
| --- | --- | --- |
| `AUDIO_FILENAME` | **必填** | Drive 中真实的音频文件名，包含扩展名，例如 `example.mp3`；只填文件名，不填目录。 |
| `DRIVE_FOLDER` | **确认路径** | 音频所在目录，默认 `/content/drive/MyDrive/podcast`；按上方准备说明上传则无需修改。 |
| `API_KEY` | **输出中文时需要** | 翻译服务商提供的密钥。可留空，运行 Step 1 时按提示隐藏输入。 |
| `BASE_URL` | **输出中文时确认** | 翻译 API 根地址，例如 `https://api.openai.com/v1`；使用其他服务商时填写其兼容接口地址，不要加 `/chat/completions`。 |
| `MODEL_NAME` | **输出中文时确认** | 翻译接口支持的模型名称，默认 `gpt-4.1-mini`；必须与所用服务商匹配。 |

```python
AUDIO_FILENAME = "example.mp3"
DRIVE_FOLDER = "/content/drive/MyDrive/podcast"
API_KEY = ""  # 留空，运行时按提示输入
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4.1-mini"
```

如果仅需英文字幕，将下方的 `OUTPUT_CHINESE` 改为 `False`，无需配置翻译接口。笔记本也支持通过 `TRANSLATION_API_KEY`、`TRANSLATION_BASE_URL`、`TRANSLATION_MODEL` 环境变量读取接口配置。

### 可选参数：一般保持默认

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `OUTPUT_CHINESE` | `True` | 输出中英双语；`False` 仅输出英文，不调用翻译 API。 |
| `SAVE_PROGRESS` | `True` | 保存并恢复转录、成功的翻译；`False` 不读写进度文件，每次重新处理，已有进度文件保留。 |
| `AUTO_DISCONNECT` | `False` | 设为 `True` 后，只有字幕全部成功保存才自动断开 Colab。 |
| `MAX_WORKERS` | `5` | 翻译并发数；遇到接口限流时可适当调低，例如 `1`。 |
| `MAX_RETRIES` | `2` | 首次请求失败后的额外重试次数；`0` 表示不重试。 |
| `REQUEST_TIMEOUT` | `30` | 每次翻译请求的超时秒数。 |
| `WHISPER_MODEL` | `"turbo"` | 音频转录模型，与上面的翻译模型 `MODEL_NAME` 不同。 |

进度恢复及旧字幕导入的详细说明见 [累积翻译与进度恢复](docs/progress.md)。

---

## 🖼️ 效果示例

![Demo](.img/demo.jpg)

## ⚠️ 隐私说明

- **API Key 安全**: 本 Notebook 运行在您的私人 Google Colab 环境中。但在分享修改后的副本给他人前，请务必删除代码框中填写的 `API_KEY`。
- **数据安全**: 脚本仅申请 Google Drive 挂载权限以读取音频和保存字幕，不会上传数据到其他服务器（除您配置的翻译 API 接口外）。
