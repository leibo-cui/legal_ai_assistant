import gradio as gr
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import requests
import json
import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import datetime
from typing import Tuple, List, Dict
import numpy as np
import tempfile
import subprocess
import os
from rag_llamaindex import get_rag_system
import threading


import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

init_lock = threading.Lock()

# 全局初始化模型
MODEL, PROCESSOR, MESSAGES = None, None, []
# 在全局变量中添加
RAG_SYSTEM = None

def init_models():
    global MODEL, PROCESSOR, MESSAGES, RAG_SYSTEM, INPUT_MODE  # 添加 RAG_SYSTEM
    with init_lock:
        if MODEL is None:
            model_name = CONFIG["models"]["whisper"]
            PROCESSOR = WhisperProcessor.from_pretrained(model_name)
            MODEL = WhisperForConditionalGeneration.from_pretrained(model_name)
        if RAG_SYSTEM is None:  # 初始化 RAG
            RAG_SYSTEM = get_rag_system()
        if not MESSAGES:
            MESSAGES = [{"role": "system", "content": "你是我的智能法律AI助手"}]

    INPUT_MODE= 'voice'
    return MODEL, PROCESSOR, MESSAGES

# 新增输入模式切换函数
def switch_input_mode(mode):
    new_mode = "text" if mode == "文本输入" else "voice"
    return [
        gr.Textbox(visible=(new_mode == "text"), value=""),
        gr.Audio(visible=(new_mode == "voice"), value=None),
        new_mode  # 返回更新后的模式状态
    ]

# 修改后的处理函数
def process_input(text_input: str, audio_input: Tuple[int, np.ndarray],  input_mode: str) -> str:
    # 初始化模型（仅首次调用时加载）
    model, processor, messages = init_models()

    # 非阻塞播放
    play_start_sound()

    # 输入验证
    print(f"[Debug] Current mode: {input_mode}")  # 调试日志
    print(f"[Debug] Audio input: {audio_input is not None}")  # 调试日志

    # 根据输入模式选择处理逻辑
    if input_mode == "text":
        if not text_input.strip():
            return "请输入文本内容"
        user_input = text_input.strip()
    elif input_mode == "voice":
        if audio_input is None:
            return "请录制语音内容"

        try:
            # 语音转文本逻辑（原 process_audio 的转换部分）
            sample_rate, waveform = audio_input
            waveform = torch.from_numpy(waveform).float()
            # ... (保留原语音转文本代码)
            # 重采样到16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)

            # 预处理音频
            inputs = processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True  # 新增：显式生成注意力掩码
            )

            input_features = inputs.input_features
            attention_mask = inputs.attention_mask  # 提取注意力掩码

            # 生成文本
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,)

            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            user_input = transcription
        except Exception as e:
            print(f"[Error] ASR failed: {e}")
            return "语音识别失败，请重试"
    else:
        return "请选择有效的输入方式并输入内容"

    # 调用 RAG 系统
    response = query_rag_model(user_input)
    system_message = response["response"]

    print(f"system message: {system_message}")
    #把system_messagee 内容中的* 号去掉
    system_message = system_message.replace("**", "").replace("*","")

    # 更新对话记录
    MESSAGES.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": system_message}
    ])

    # 语音合成
    # asyncio.run(text_to_speech(system_message))
    # 语音合成（直接调用同步函数）
    text_to_speech(system_message)  # 移除 asyncio.run()

    # 格式化对话记录
    return format_chat_history()

def process_audio(audio: Tuple[int, np.ndarray]) -> str:
    # 初始化模型（仅首次调用时加载）
    model, processor, messages = init_models()

    audio_to_text_start_time = datetime.datetime.now()

    # 语音转文本
    sample_rate, waveform = audio
    waveform = torch.from_numpy(waveform).float()

    # 重采样到16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    # 预处理音频
    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # 生成文本
    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    audio_to_text_end_time = datetime.datetime.now()
    print(f"audio to text time: {audio_to_text_end_time - audio_to_text_start_time}")
    print(f"after transcripte to text: {datetime.datetime.now()}")

    invoke_llm_start_time = datetime.datetime.now()
    # 调用大模型
    # response = query_ollama_model(transcription)
    # 调用RAG
    response = query_rag_model(transcription)

    # system_message = response.get("response", "请求失败")
    system_message = response["response"]
    print(f"system message: {system_message}")
    #把system_messagee 内容中的* 号去掉
    system_message = system_message.replace("**", "").replace("*","")

    invoke_llm_end_time = datetime.datetime.now()
    print(f"invoke llm time: {invoke_llm_end_time - invoke_llm_start_time}")
    print(f"system message: {system_message}")
    print(f"after invoke LLM: {datetime.datetime.now()}")



    # 更新对话历史
    # MESSAGES.append(
    #     {"role": "user", "content": transcription},
    #     {"role": "assistant", "content": system_message}
    # )
    MESSAGES.append({"role": "system", "content": system_message})

    # MESSAGES.extend([{"role": "user", "content": transcription},
    #     {"role": "assistant", "content": system_message}])

    # 异步语音合成
    # asyncio.run_coroutine_threadsafe(
    #     text_to_speech(system_message),
    #     asyncio.new_event_loop()
    # )
    print(f"messages: {MESSAGES}")
    text_to_speech_start_time = datetime.datetime.now()
    asyncio.run(text_to_speech(system_message))
    print(f"after text to speech: {datetime.datetime.now()}")
    text_to_speech_end_time = datetime.datetime.now()
    print(f"text to speech time: {text_to_speech_end_time - text_to_speech_start_time}")

    chat_transcription = ""

    for message in MESSAGES:
        if message["role"] != "system":
            chat_transcription += message["role"] + message["content"] + "\n\n"

            # 格式化输出文本
    # return format_chat_history()
    return chat_transcription

# define RAG query function
def query_rag_model(prompt: str) -> Dict:
    import time
    try:
        start = time.time()
        response = RAG_SYSTEM.query(prompt)
        print(f"RAG耗时: {time.time() - start:.2f}s")
        return {"response": response}
    except Exception as e:
        return {"response": f"法律查询失败：{str(e)}"}


def query_ollama_model(prompt: str) -> Dict:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5
            },
            # timeout=120
        )
        return response.json()
    except Exception as e:
        return {"response": f"模型请求失败：{str(e)}"}


async def async_play_start_sound():
    audio = AudioSegment.from_wav(CONFIG["audio"]["start_sound"])
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio.export(tmp.name, format="wav")
        safe_play(AudioSegment.from_wav(tmp.name))

def play_start_sound():
    asyncio.run(async_play_start_sound())


def text_to_speech(text: str):
    async def async_tts():
        # 原异步代码逻辑
        voice = "zh-CN-XiaoxiaoNeural"
        temp_file = "temp_part.mp3"
        # 播放开始提示音
        # AudioSegment.from_wav("start.wav").export("start_temp.wav", format="wav")
        # safe_play(AudioSegment.from_wav("start_temp.wav"))

        MAX_LEN = 1000
        parts = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]

        for part in parts:
            communicate = edge_tts.Communicate(
                part,
                voice=CONFIG["tts"]["voice"],
                rate="+10%",
                volume="+20%"
            )
            await communicate.save(temp_file)
            audio_segment = AudioSegment.from_file(temp_file, format="mp3")
            safe_play(audio_segment)
            try:
                os.remove(temp_file)
            except PermissionError:
                pass

    # 创建新事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_tts())
    loop.close()
    # 同步执行异步函数
    # loop = asyncio.get_event_loop()
    # if loop.is_running():
    #     loop.create_task(async_tts())
    # else:
    #     loop.run_until_complete(async_tts())

# async def text_to_speech(text: str):
#     voice = "zh-CN-XiaoxiaoNeural"
#     output_file = "output.mp3"
#     temp_file = "temp_part.mp3"
#
#     # 播放开始提示音
#     AudioSegment.from_wav("start.wav").export("start_temp.wav", format="wav")
#     safe_play(AudioSegment.from_wav("start_temp.wav"))
#
#     MAX_LEN = 1000  # 单次合成字符限制
#     parts = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]
#
#     for part in parts:
#         # communicate = edge_tts.Communicate(part, ...)
#         # await communicate.save(temp_file)
#         # safe_play(AudioSegment.from_mp3(temp_file))
#         # os.remove(temp_file)
#
#         communicate = edge_tts.Communicate(
#             part,
#             voice=CONFIG["tts"]["voice"],
#             rate="+10%",  # 语速加快10%
#             volume="+20%"  # 音量提高20%
#         )
#         await communicate.save(temp_file)
#         audio_segment = AudioSegment.from_file(temp_file, format="mp3")
#         print(f"after audio segment")
#         try:
#             safe_play(audio_segment)
#             # play_via_pipe(audio_segment)
#             os.remove(temp_file)
#         except Exception as e:
#             print(f"error playing audio: {e}")


def format_chat_history() -> str:
    return "\n\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in MESSAGES
        if msg["role"] != "system"
    ])


def safe_play(audio_segment):
    # 创建临时文件路径（不自动删除）
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_name = temp_file.name

    # 导出音频到临时文件（此时文件句柄已关闭）
    audio_segment.export(temp_name, format="wav")

    # 使用 ffplay 播放
    try:
        subprocess.run(
            ['ffplay', '-nodisp', '-autoexit', temp_name],
            # timeout=120, # 设置超时限制
            stdin=subprocess.PIPE,  # 避免 ffplay 占用输入流
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    finally:
        # 强制删除临时文件（即使播放失败）
        try:
            os.remove(temp_name)
        except PermissionError:
            print(f"警告：临时文件 {temp_name} 删除失败，可能仍在被占用。")

def play_via_pipe(audio_segment):
    # 确保音频格式与 ffplay 参数匹配
    raw_data = audio_segment.set_frame_rate(44100) \
                            .set_channels(2) \
                            .raw_data  # 导出为 PCM s16le 格式

    # 构造 ffplay 命令
    ffplay_cmd = [
        'ffplay',
        '-nodisp',       # 不显示窗口
        '-autoexit',     # 播放完成后自动退出
        '-f', 's16le',   # 强制指定输入格式为 PCM 16-bit little-endian
        '-ar', '44100',  # 采样率 44.1kHz
        '-ac', '2',      # 双声道（如果音频实际是单声道，需改为 1）
        '-'              # 从 stdin 读取数据
    ]

    # 启动 ffplay 并传递数据
    process = subprocess.Popen(
        ffplay_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    process.communicate(input=raw_data)

# 创建Gradio界面
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources="microphone", type="numpy"),
    outputs=gr.Textbox(label="对话记录", lines=10),
    title="智能法律咨询系统",
    description="""语音输入 → 文本转换 → 法律咨询 → 语音输出""",
    flagging_mode="never",
    examples=[[example] for example in CONFIG["ui"]["examples"]]
)

# 创建新界面布局
with gr.Blocks(title="智能法律咨询系统") as interface:
    gr.Markdown("## 智能法律咨询系统（语音/文本双模式）")

    # 状态组件
    input_mode_state = gr.State("voice")

    with gr.Row():
        # 左侧输入面板
        with gr.Column(scale=1):
            mode_radio = gr.Radio(
                choices=["语音输入", "文本输入"],
                value="语音输入",
                label="选择输入方式"
            )

            text_input = gr.Textbox(
                label="文本输入",
                visible=False,
                placeholder="请输入法律问题..."
            )

            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="语音输入",
                visible=True
            )

            submit_btn = gr.Button("提交问题", variant="primary")

            # 示例问题区
            gr.Markdown("### FAQ Examples")
            example_btns = gr.Dataset(
                components=[text_input],
                samples=[[example] for example in CONFIG["ui"]["examples"]],
                label="点击示例快速提问",
                elem_id="example_dataset"
            )

        # 右侧输出面板
        with gr.Column(scale=2):
            chat_history = gr.Textbox(
                label="对话记录",
                interactive=False,
                lines=20,
                placeholder="对话记录将在此显示..."
            )

    # 事件绑定（已修复）
    mode_radio.change(
        fn=switch_input_mode,
        inputs=mode_radio,
        outputs=[text_input, audio_input, input_mode_state]
    )

    submit_btn.click(
        process_input,
        inputs=[text_input, audio_input, input_mode_state],
        outputs=chat_history
    )

    example_btns.click(
        lambda example: [
            "文本输入",  # 切换为文本模式
            example[0],  # 填充文本
            None,  # 清空语音
            "text"  # 更新状态
        ],
        inputs=[example_btns],
        outputs=[mode_radio, text_input, audio_input, input_mode_state],
        show_progress=False
    ).then(
        process_input,
        inputs=[text_input, audio_input, input_mode_state],
        outputs=chat_history
    )

if __name__ == "__main__":
    # 预加载模型
    # 增加计算模型加载的时间
    model_start_time = datetime.datetime.now()
    init_models()
    model_end_time = datetime.datetime.now()
    print(f"model loading time: {model_end_time - model_start_time}")
    interface.launch(
        max_threads=2, # 限制并发数
        server_name="127.0.0.1",
        server_port=CONFIG["server"]["port"],
        share=False
    )