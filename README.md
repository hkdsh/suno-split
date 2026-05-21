# suno-split

把一个音频文件按随机的时间-频率块拆成多条 MP3 轨道的本地工具。当前脚本默认输出 20 轨，适合做声音设计、解谜素材或私有归档实验。

## 功能

- 先用 `ffmpeg` 把输入音频解码成 float WAV，再用 STFT 做时间-频率切分。
- 每个时间-频率块会随机分配到一条输出轨道，默认 20 轨。
- 人声常见频段附近使用更细的频带划分，让拆分后的声部更碎。
- 每条输出轨道独立编码为 MP3。
- 同时写出 `manifest.csv`、`metadata.json`、`track_summary.csv`，方便复现和检查分配结果。

## 环境依赖

需要本机已安装：

- Python 3.10+
- `ffmpeg`
- Python 包：`librosa`、`numpy`、`soundfile`

如果使用 `uv`，可以在仓库目录里直接临时安装依赖运行：

```bash
uv run --with librosa --with numpy --with soundfile python main.py /path/to/input.mp3
```

如果使用普通虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install librosa numpy soundfile
python main.py /path/to/input.mp3
```

## 使用方式

最简单的运行方式：

```bash
python main.py /path/to/input.mp3
```

指定输出目录、轨道数和随机种子：

```bash
python main.py /path/to/input.wav \
  --output ./output_split \
  --tracks 20 \
  --seed 20260521
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 输入音频路径，支持 `ffmpeg` 可解码的格式 |
| `--output` | 自动生成 | 输出目录；不传时会在输入文件旁边创建带时间戳的目录 |
| `--tracks` | `20` | 输出轨道数量，范围 2 到 255 |
| `--seed` | `20260521` | 随机种子；同一输入和同一参数下可复现分配结果 |
| `--n-fft` | `4096` | STFT 窗口大小 |
| `--hop-length` | `512` | STFT hop 长度 |
| `--bitrate` | `320k` | 输出 MP3 码率 |

## 输出内容

输出目录中会包含：

- `track_01.mp3` 到 `track_NN.mp3`：拆分后的音频轨道。
- `manifest.csv`：每个时间-频率块被分配到哪条轨道，包括时间、频段和 STFT bin 信息。
- `metadata.json`：输入路径、采样率、声道数、轨道数、随机种子、STFT 参数等元数据。
- `track_summary.csv`：每条轨道的峰值、缩放系数、分配块数量和近似活跃时长。

## 注意事项

- 这是频域拆分工具，不是高保真分轨或人声/伴奏分离工具。
- 输出轨道单独听会是碎片化素材；它们更适合做实验素材或后续 DAW 编排。
- MP3 是有损编码，不能用输出 MP3 直接期待无损重建原音频。
- 如果输出轨道峰值过高，脚本会对单条轨道做保护性缩放，缩放值会记录在 `track_summary.csv`。
- 请只处理你有权使用的音频素材。
