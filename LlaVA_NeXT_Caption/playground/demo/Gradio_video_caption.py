import sys
from datetime import datetime

import gradio as gr
import os, json

import argparse
import torch

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import time

import base64
from openai import OpenAI
import cv2
from moviepy.editor import VideoFileClip
import time
import base64

import os

import numpy as np

import asyncio


def fc_parse_args(videopath, model="lmms-lab/LLaVA-NeXT-Video-34B-DPO", output_dir="./temp_output_dir",
                  output_name="output.json",
                  conv_mode="mistral_direct", frames=24):
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default=videopath, required=False)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default=output_dir,
                        required=False)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default=output_name,
                        required=False)
    parser.add_argument("--model-path", type=str, default=model)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str,
                        default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=frames)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    return parser.parse_args()


def process_video_audio(video_path, seconds_per_frame=3):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


def process_video(video_path, frames_numbers_needed=5):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_time_length = int(total_frames / fps)
    frames_to_skip = int(fps * video_time_length / frames_numbers_needed)

    curr_frame = 0
    print(f'Total frames: {total_frames}')
    print(f'FPS: {fps}')

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    #
    # audio_path = f"{base_video_path}.mp3"
    # clip = VideoFileClip(video_path)
    # clip.audio.write_audiofile(audio_path, bitrate="32k")
    # clip.audio.close()
    # clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    # print(f"Extracted audio to {audio_path}")
    return base64Frames


async def gpt_video_inference(video_path, input_prompt, frames_numbers_needed=5):
    base64Frames = process_video(video_path, frames_numbers_needed=5)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": input_prompt},
            {"role": "user", "content": [
                "These are the frames from the video.",
                *map(lambda x: {"type": "image_url",
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
            ],
             }
        ],
        temperature=0,
    )
    caption = response.choices[0].message.content
    print(caption)
    return (caption)


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    # fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


async def run_inference(args, input_prompt):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["patchify_video_feature"] = False

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
        else:
            least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

        scaling_factor = math.ceil(least_token_number / 4096)
        # import pdb;pdb.set_trace()

        if scaling_factor >= 2:
            if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                print(float(scaling_factor))
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                               model_name, load_8bit=args.load_8bit,
                                                                               overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                               model_name)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    video_path = args.video_path
    sample_set = {}
    question = input_prompt
    # question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes"
    # question = "What does this video describe? A. Buiding B.Forest C.coutryside D.Moon \nAnswer with the option's letter from the given choices directly."
    sample_set["Q"] = question
    sample_set["video_name"] = args.video_path

    # Check if the video exists
    if os.path.exists(video_path):
        video = load_video(video_path, args)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        video = [video]

    # try:
    # Run inference on the video and add the output to the list

    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = question
    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        # import pdb;pdb.set_trace()
        start_time = time.time()
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video",
                                    do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True,
                                    stopping_criteria=[stopping_criteria])
        end_time = time.time()
        print(f"Time taken for inference: {end_time - start_time} seconds")
        # import pdb;pdb.set_trace()
        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"Question: {prompt}\n")
    print(f"Response: {outputs}\n")
    # import pdb;pdb.set_trace()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    sample_set["pred"] = outputs
    ans_file.write(json.dumps(sample_set) + "\n")
    ans_file.flush()

    ans_file.close()
    return outputs


async def dumb_response(input):
    return input


async def UX_inference(models, input_path, prompt, input_frames):
    # video_path=os.path.join(dir,input_path)
    llava_answer = ""
    gpt_answer = ""
    if "lmms-lab/LLaVA-NeXT-Video-34B-DPO" in models and "GPT4o" in models:
        args = fc_parse_args(input_path, frames=input_frames)
        llava_answer, gpt_answer = await asyncio.gather(run_inference(args, prompt),
                                                        gpt_video_inference(input_path, prompt, input_frames))
        print("both")
        return llava_answer, gpt_answer
    elif not "lmms-lab/LLaVA-NeXT-Video-34B-DPO" in models and not "GPT4o" in models:
        print("None")
        return llava_answer, gpt_answer
    elif "lmms-lab/LLaVA-NeXT-Video-34B-DPO" in models:
        args = fc_parse_args(input_path, frames=input_frames)
        llava_answer = await run_inference(args, prompt)
        gpt_answer = ""
        print("llava")
        return llava_answer, gpt_answer
    else:
        llava_answer = ""
        gpt_answer = await gpt_video_inference(input_path, prompt, input_frames)
        print("gpt")
        return llava_answer, gpt_answer


def show_json(filename):
    with open(filename, 'r') as file:
        # Read the entire file content
        data_string = file.read()

    # Parse the JSON data
    data = json.loads(data_string)
    data = json.dumps(data, indent=4)
    return data


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


def show_videos(input_video_dir):
    temp_row = gr.Row()
    with temp_row:
        for i in range(3):
            win_name = str(i)
            gr.Video(win_name)

    return temp_row


# def show_video(input_path, video_dir, prompt):
#     print(input_path)
#     print(video_dir)
#     filename = str(prompt)
#     video = os.path.join(input_path, video_dir, filename)
#     print(video)
#     return video


def show_video(dir, input_file_path):
    print(dir)
    print(input_file_path)
    test_path = os.path.join(dir, input_file_path)
    video = ""
    print(f'testpath is {test_path}')
    if os.path.isfile(test_path):
        video = test_path
        print(f"video is {video}")
        return gr.update(label="Video", value=video)


def generate_prompt_list(input_video_dir, company_name='Runway'):
    prompt_list = os.listdir(os.path.join(input_video_dir, 'Runway'))
    prompt_list.pop(0)
    count = 0
    for item in prompt_list:
        print(str(count) + item)
        count += 1
    return prompt_list


def list_video_files(directory):
    try:
        # Ensure the directory exists
        if not os.path.isdir(directory):
            return f"Error: {directory} is not a valid directory."

        # List all files in the directory
        files = os.listdir(directory)

        # Filter for .mp4 files
        mp4_files = [file for file in files if (file.endswith('.mp4') or file.endswith('.mov'))]

        if mp4_files is not None:
            return mp4_files
        else:
            return []
    except Exception as e:
        return str(e)


def update_prompt_list(input_video_dir):
    try:
        if os.path.isfile(input_video_dir):
            prompt_list = [input_video_dir]
        else:
            prompt_list = list_video_files(input_video_dir)
            print(prompt_list)
            if ".DS_Store" in prompt_list[0]:
                prompt_list.pop(0)
            count = 0
            for item in prompt_list:
                print(str(count) + item)
                count += 1
        return gr.update(choices=list(prompt_list), label='File list', value=prompt_list[0], interactive=True)
    except Exception as e:
        print(e)


sys.stdout = Logger("output.log")

with gr.Blocks() as ux_evaluator:
    default_path = "/raid/FC_project/LLaVA-NeXT/videos"
    filepath = gr.Text(value=default_path)
    file_list = gr.Dropdown(label="File list", choices=list_video_files(default_path),
                            value=list_video_files(default_path)[0], interactive=True)
    with gr.Row():
        fc_col_left = gr.Column()
        fc_col_right = gr.Column()
        with fc_col_left:
            prompt = gr.TextArea(label="Prompt", value="""You are a professional cinematographer. Please provide a description of the video. you need to capture some key information from it, including: 
*) camera position: e.g. Aerial, high, low, shoulder,.etc
*) camera movement: Is the camera is still or the camera is moving, e.g. Pan, Tilt,zoom,.etc
*) camera moving direction: is the camera moving vertically or horizontally, or not moving
*) camera zooming: is the camera zooming in or zooming out, or keep still
*) what is the time: day, dawn,sunset or night
*) main objects in the video
*) movement of the main objects in the video
*) if there are human in the video, how many of them, what are they doing
*) if there are animals in the video, what animals do they have, what are the animals doing 
*) if there is an moving object in the video, what it is, how it's moving
summarize the description in a 100 word narrative description""")
            frames = gr.Text(label="Frames for caption", value=24)
            filepath.change(fn=update_prompt_list, inputs=filepath, outputs=file_list)
        with fc_col_right:
            video_preview = gr.Video(label="Video", value=list_video_files(default_path)[0], height=400)
            file_list.change(fn=show_video, inputs=[filepath, file_list], outputs=video_preview)
    AI_video_model = gr.CheckboxGroup(["lmms-lab/LLaVA-NeXT-Video-34B-DPO", "GPT4o"], interactive=True,
                                      label="Caption Model")
    caption_btn = gr.Button(value="Run Caption")

    with gr.Row():
        fc_col_left1 = gr.Column()
        fc_col_right1 = gr.Column()
        with fc_col_left1:
            caption_output_llava = gr.TextArea(label="Llava Video Caption")
        with fc_col_right1:
            caption_output_GPT4o = gr.TextArea(label="GPT4o Video Caption")

    caption_btn.click(fn=UX_inference, inputs=[AI_video_model, video_preview, prompt, frames],
                      outputs=[caption_output_llava, caption_output_GPT4o])

    logs = gr.Textbox(label="Logs")
    ux_evaluator.load(read_logs, None, logs, every=1)

if __name__ == "__main__":
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    MODEL = "gpt-4o"
    openai_api_key = "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"
    client = OpenAI(api_key=openai_api_key)
    VIDEO_PATH = "/Users/fcfu/Downloads/Peak_test.mov"

    ux_evaluator.launch(share=True)
