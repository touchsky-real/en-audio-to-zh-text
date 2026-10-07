# 🎙️ Audio2LRC：英文音频 / 视频转中文字幕

在 Google Colab 中用 Whisper 识别英文，再调用翻译接口生成中英双语字幕，默认输出 LRC 歌词。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/touchsky-real/en-audio-to-zh-text/blob/main/audio2lrc.ipynb)

## 支持的输入格式

| 类型 | 常见格式 |
| --- | --- |
| 音频 | MP3、WAV、M4A、AAC、FLAC、OGG、OPUS、WMA |
| 视频（含音轨） | MP4、MKV、MOV、WEBM、AVI |

视频可直接输入，无需转成 MP3。Whisper 通过 FFmpeg 读取声音，不识别画面或画面中的字幕；实际支持范围取决于音轨能否解码。Step 2 会检查并安装 FFmpeg。

## 快速开始

1. 在 Google Drive 的“我的云端硬盘”中新建 `podcast` 文件夹，上传音频或视频。
2. 点击上方 **Open in Colab**，建议选择 **T4 GPU** 运行时。
3. 在 **Step 1** 填写下方参数。
4. 依次运行 **Step 1 配置 → Step 2 准备环境 → Step 3 识别英文 → Step 4 翻译并保存**。

字幕保存在原文件所在文件夹。LRC 用于音乐播放器的滚动歌词，SRT 用于视频字幕。

## 参数说明

在 **Step 1** 替换对应行，修改后重新运行该步骤才会生效。文字用英文引号，数字不用引号；`True` 是开启，`False` 是关闭。

### 先填这几项

```python
AUDIO_FILENAME = "example.mp4"                  # 真实文件名，包含扩展名，只填文件名
DRIVE_FOLDER = "/content/drive/MyDrive/podcast"  # 文件夹路径，按上述步骤上传则不用改
API_KEY = ""                                  # 留空，运行时按提示隐藏输入密钥
BASE_URL = "https://api.openai.com/v1"          # 翻译服务商提供的 API 地址
MODEL_NAME = "gpt-4.1-mini"                     # 该服务商支持的中文翻译模型
```

**密钥、地址、模型名称要与同一服务商匹配。** `BASE_URL` 不是聊天网页地址，末尾不要加 `/chat/completions`，程序会自动添加。

### 其余参数一般保持默认

| 参数 | 默认值 | 含义 / 什么时候改 |
| --- | --- | --- |
| `OUTPUT_FORMAT` | `"lrc"` | 滚动歌词选 `"lrc"`，视频字幕选 `"srt"`，均为小写。 |
| `OUTPUT_CHINESE` | `True` | 输出中英双语；改为 `False` 只输出英文，无需翻译密钥。 |
| `SAVE_PROGRESS` | `True` | 保存识别结果和成功翻译，重跑时只补缺失内容。关闭后不读写进度。 |
| `AUTO_DISCONNECT` | `False` | 改为 `True`，全部完成并保存后自动断开 Colab；有失败片段时不会断开。 |
| `MAX_WORKERS` | `3` | 最多同时翻译 3 条字幕；限流时可降为 `1`。 |
| `MAX_RETRIES` | `2` | 每条失败后最多再试 2 次，合计最多 3 次；`0` 表示不重试。 |
| `REQUEST_TIMEOUT` | `30` | 连接或等待接口数据的超时秒数；超时可试 `60`，不是整个视频的处理限时。 |
| `WHISPER_MODEL` | `"turbo"` | 负责把声音识别成英文的模型，一般不改。 |

例如，`example.mp4` 默认生成 `example.lrc`；选择 SRT 则生成 `example.srt` 和英文版 `example_en.srt`。

## 补译与重新运行

保持 `SAVE_PROGRESS = True`，已成功的翻译会继续复用。

| 情况 | 运行步骤 |
| --- | --- |
| 参数不变，只补失败片段 | **Step 4** |
| 修改翻译参数或输出格式 | **Step 1 → 4** |
| 更换文件或 `WHISPER_MODEL` | **Step 1 → 3 → 4**（环境已准备好） |
| 重启、切换 CPU 或打开新版笔记本 | **Step 1 → 2 → 4**（已有保存的转录；否则先运行 Step 3） |

翻译阶段只调用接口，可以在 Step 3 完成后切换 CPU。进度文件和旧字幕导入详见 [累积翻译与进度恢复](docs/progress.md)。

## 翻译失败时如何排查

“本轮已处理”包含成功和失败，达到 100% 不代表全部成功；“翻译覆盖率”是累计成功条数。失败日志会显示片段编号、尝试次数和最后原因。

| 日志原因 | 怎么处理 |
| --- | --- |
| HTTP 429：请求受限 | 检查配额和请求上限；限流时可将 `MAX_WORKERS` 降为 `1`，等待限制恢复。 |
| 请求超时 / HTTP 408 | 检查网络，可将 `REQUEST_TIMEOUT` 提高到 `60`。 |
| HTTP 5xx：服务端错误 | 稍后重试，检查服务商状态。 |
| HTTP 401 / 403 | 检查密钥、账号和模型权限。 |
| HTTP 400 / 404 / 422 | 检查接口地址、模型名称和参数是否受支持。 |
| 网络连接 / TLS 错误 | 检查网络、代理或证书。 |
| 返回格式异常 / 翻译为空 | 检查接口是否兼容 Chat Completions；HTTP 200 也可能没有有效译文。 |

排查后按上表补译。旧 Colab 副本不会自动更新，请从顶部入口打开新版并填回自己的配置。

## 效果示例

![Demo](.img/demo.jpg)

## 隐私与费用

识别在 Colab 内运行；翻译会将英文文本发送到你配置的服务商，并可能产生接口费用。分享笔记本前请移除填入的 API Key。
