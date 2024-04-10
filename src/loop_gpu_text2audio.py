import os
import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from pydub import AudioSegment
import string

def split_text_smartly(text, max_length):
    """
    智能分割文本，确保不在单词中间分割。
    :param text: 待分割的文本。
    :param max_length: 每段文本的最大长度。
    :return: 分割后的文本列表。
    """
    # 包含 ASCII 标点符号
    split_chars = list(string.punctuation) + ['\n', '—', '…']
    # 注意添加了额外的非 ASCII 标点符号，如长破折号和省略号
    segments = []
    while text:
        if len(text) <= max_length:
            segments.append(text)
            break
        # 找到最大长度附近的分割点
        split_point = max_length
        while split_point > 0 and text[split_point] not in split_chars:
            split_point -= 1
        # 如果没有找到合适的分割点，直接使用最大长度（避免无限循环）
        if split_point == 0:
            split_point = max_length
        # 分割文本并继续处理剩余部分
        segments.append(text[:split_point])
        text = text[split_point:].lstrip()  # 移除前导空格和标点符号
    return segments


# 读取文本文件
text_file_path = '/home/biw002/Desktop/bigspace/jason/text2speech/textfile/text.txt'
with open(text_file_path, 'r', encoding='utf-8') as file:
    text_to_speak = file.read().strip()

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 初始化 TTS 和 Vocoder
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device":"cuda"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device":"cuda"})

# 分割文本为192字符的片段
# text_segments = [text_to_speak[i:i+192] for i in range(0, len(text_to_speak), 192)]
max_length = 192
text_segments = split_text_smartly(text_to_speak, max_length)

temp_audio_path = '/home/biw002/Desktop/bigspace/jason/text2speech/textfile/tempaudio'
output_path = '/home/biw002/Desktop/bigspace/jason/text2speech/output'
final_audio = AudioSegment.empty()

# 确保临时音频文件夹存在
os.makedirs(temp_audio_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# 对每个文本片段进行处理并分别保存
for i, segment in enumerate(text_segments):
    print(segment)
    mel_output, mel_length, alignment = tacotron2.encode_text(segment)
    waveforms = hifi_gan.decode_batch(mel_output)
    waveforms_cpu = waveforms.squeeze(1).cpu()
    temp_file_path = os.path.join(temp_audio_path, f"temp_audio_{i}.wav")
    torchaudio.save(temp_file_path, waveforms_cpu, 22050)
    
    # 使用pydub读取每个片段并追加到最终音频中
    segment_audio = AudioSegment.from_wav(temp_file_path)
    final_audio += segment_audio

# 保存最终合并的音频文件
final_audio_file_path = os.path.join(output_path, "final_example_TTS.wav")
final_audio.export(final_audio_file_path, format="wav")
