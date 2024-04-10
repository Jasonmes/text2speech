import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
import os
# 读取文本文件
with open('/home/biw002/Desktop/bigspace/jason/text2speech/textfile/text.txt', 'r', encoding='utf-8') as file:
    text_to_speak = file.read().strip()


# 设置只使用特定编号的 GPU，例如只使用编号为 0 的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置环境变量以使用指定的 Hugging Face 镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 使用 GPU 初始化 TTS (tacotron2) 和 Vocoder (HiFIGAN)
# 确保在 'run_opts' 中添加 'device':'cuda'
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device":"cuda"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device":"cuda"})

# 在 GPU 上运行 TTS
mel_output, mel_length, alignment = tacotron2.encode_text(text_to_speak)

# 在 GPU 上运行 Vocoder (将 spectrogram 转换为 waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# 在保存 waveform 之前，确保将其移动到 CPU
waveforms_cpu = waveforms.squeeze(1).cpu()
torchaudio.save('example_TTS.wav', waveforms_cpu, 22050)


# 保存 waveform 到文件
# torchaudio.save('example_TTS.wav', waveforms.squeeze(1), 22050)
