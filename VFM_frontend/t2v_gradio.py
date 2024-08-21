import os
import textwrap
import uuid
import openai
from openai import OpenAI

import gradio as gr
import torch
import torchvision
import random

from demo.remote_models import RemoteTextEncoder
from ml_platform.accelerate.proxy import download_weights_from_cloud_storage

# from your_video_generation_model import VideoGenerationModel  # Replace with your actual model import
from models.generation.video_diffuser.pipeline import VideoDiffuserPipeline
from models.generation.video_diffuser.scheduler import (
    DerivedFlowMatchEulerDiscreteScheduler,
)
from models.generation.video_diffuser.video_dif import (
    VideoDiffusionTransformer,
)
from models.vae.model_components.causal_vae.model_accelerate import (
    Causal3DVAEModel,
)

inference_task_config = """"""

MODEL = "llama3.1:70b"
MODEL_BASE_URL = 'http://127.0.0.1:11434/v1'  # use my server
API_KEY_SET = 'ollama'
SYS_PROMT = {"role": "system",
             "content": """
         ## role
         you are a prompt expert who can help to add tags to a prompt

        ## skill
        you will review the input prompt and add key tags to it based on it's contents

        ## constraints
        - you will output the prompt according to the example format
        - you only output the prompt, don't output any other information
        - make sure the description starts with "This video depicts "

        ## example
        input prompt: "This video depicts a cat with brown hairs and a white apron washing dishes with its paws, 
                        we can see the head of the cat, the water is flowing rapidly, the dish is white and made of china, 
                        the background is a kitchen, the cat looks very happy"
        output prompt: caption: [
  {
    "description": "This video depicts a cat with brown hairs and a white apron washing dishes with its paws, we can see the head of the cat, the water is flowing rapidly, the dish is white and made of china, the background is a kitchen, the cat looks very happy",
    "tags": [
      "cat",
      "dishwashing",
      "faucet",
      "kitchen"
    ]
  }
]

         """}


def prompt_rewrite(input_msg):
    user_model = MODEL
    user_messages = [
        SYS_PROMT,
        # {"role": "user", "content": "Who won the world series in 2020?"},
        # {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": input_msg}
    ]
    client = client = OpenAI(
        base_url=MODEL_BASE_URL,  # <<<<< you need to do the port mapping kuberate desktop in VScode
        api_key=API_KEY_SET,  # required, but unused
    )
    response = client.chat.completions.create(
        model=user_model,
        messages=user_messages,

        # tools=tools,
        # tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message.content
    print(response_message)
    return response_message


class MockSPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


TRANSFORMER_MODEL_GCS = (
    "gs://predera-developer-testing/yuanjun/models/ptransformer_1b_v0_160k"
)
VAE_MODEL_GCS = "gs://pretrained-model-weights/vfm_vae/vfm_vae_iter1/"

TRANSFORMER_LOCAL = "/tmp/transformer_model/"
VAE_MODEL_LOCAL = "/tmp/vae_model/"

PROMPT_LENGTH = 150
VAE_SCALE_FACTOR = 0.2737
VAE_CONTEXT_LENGTH = 17
VAE_TEMPORAL_COMPRESSION = 4
GUIDANCE_SCALE = 4.0
VIDEO_PATH = "/tmp/gen_videos"
NEGATIVE_EMBEDDING_FILE = (
    "models/generation/data/DeepFloyd--t5-v1_1-xxl__null_embedding.safetensors"
)


def prepare_weights():
    os.makedirs(TRANSFORMER_LOCAL, exist_ok=True)
    os.makedirs(VAE_MODEL_LOCAL, exist_ok=True)
    print("downloading transformer weights")
    download_weights_from_cloud_storage(
        TRANSFORMER_MODEL_GCS, TRANSFORMER_LOCAL
    )
    transformer_path = os.path.join(
        TRANSFORMER_LOCAL, "/".join(TRANSFORMER_MODEL_GCS[5:].split("/")[1:])
    )
    print("finished downloading transformer weights, downloading vae weights")
    download_weights_from_cloud_storage(VAE_MODEL_GCS, VAE_MODEL_LOCAL)
    vae_path = os.path.join(
        VAE_MODEL_LOCAL, "/".join(VAE_MODEL_GCS[5:].split("/")[1:])
    )
    print("weights downloaded")
    return transformer_path, vae_path


def get_model():
    print("initializing model")
    print("connecting to text encoder")
    text_encoder = RemoteTextEncoder(token_cutoff=PROMPT_LENGTH)
    print("text encoder connected")

    transformer_path, vae_path = prepare_weights()
    print("initializing transformer model")
    model = VideoDiffusionTransformer.from_pretrained(transformer_path)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(torch.bfloat16)
    model.eval()
    model.device = torch.device("cuda:0")
    print("transformer model initialized")
    print("initializing vae model")
    vae_model = (
        Causal3DVAEModel.from_pretrained(vae_path)
        .to(torch.device("cuda:0"))
        .to(torch.bfloat16)
    )
    vae_model.eval()
    print("vae model initialized")
    print("initializing pipeline")
    pipeline = VideoDiffuserPipeline(
        model, vae_model, DerivedFlowMatchEulerDiscreteScheduler()
    )
    neg_info = pipeline.get_negative_prompt_embedding(
        null_embedding_file=NEGATIVE_EMBEDDING_FILE
    )
    print("pipeline initialized")

    return pipeline, text_encoder, neg_info


pipeline, text_encoder, neg_info = get_model()


def update_dropdown_choices():
    # This function returns the new choices
    new_choices = list_video_files(VIDEO_PATH);
    return gr.Dropdown(choices=new_choices, label="File list")


def show_video(input_file_path):
    full_path = os.path.join(VIDEO_PATH, input_file_path)
    print(f'Full path is {full_path}')
    if os.path.isfile(full_path):
        video = full_path
    else:
        video = ""  # or a default video path
        print(f"Warning: File not found at {full_path}")
    print(f"Video path is {video}")
    return gr.update(label="Video", value=video)


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
            return ["NA"]
    except Exception as e:
        return str(e)


def generate_video(prompt, guidance_scale, neg_prompt, steps, use_seed, seed):
    print(f"processing prompt: {prompt}, neg_prompt: {neg_prompt}")

    """ add prompt rewrite function """
    new_prompt = prompt_rewrite(prompt)

    te, tm = text_encoder.encode(new_prompt)
    if neg_prompt != "":
        print(f"using negative prompt: {neg_prompt}")
        neg_te, neg_tm = text_encoder.encode(neg_prompt)
        inf_neg_info = (neg_te, neg_tm)
    else:
        print("using default negative prompt")
        inf_neg_info = neg_info
    print(f"prompt embedding shape: {te.shape}")

    generator = torch.Generator()
    if use_seed:
        generator.manual_seed(seed)
    print(f"generating video: {steps} steps, seed: {generator.initial_seed()}")
    video = pipeline(
        te,
        tm,
        VAE_CONTEXT_LENGTH,
        VAE_TEMPORAL_COMPRESSION,
        VAE_SCALE_FACTOR,
        num_inference_steps=int(steps),
        negative_prompt=inf_neg_info[0][:, :PROMPT_LENGTH, :],
        negative_prompt_mask=inf_neg_info[1][:, :PROMPT_LENGTH],
        guidance_scale=guidance_scale,
        return_dict=False,
        sequence_parallel_group=MockSPGroup(),
        generator=generator,
    )[0]
    video = (video * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu()
    import einops

    video = einops.rearrange(video[0], "c t h w -> t h w c")
    print(f"video generated: {video.shape}")

    os.makedirs(VIDEO_PATH, exist_ok=True)
    filename = f"{VIDEO_PATH}/{uuid.uuid4()}.mp4"
    torchvision.io.write_video(filename, video, 15, options={"-crf": "20"})
    return prompts, filename


def batch_generate_video(prompt, guidance_scale, neg_prompt, steps, use_seed, seed, batch_number):
    print(f"processing prompt: {prompt}, neg_prompt: {neg_prompt}")
    te, tm = text_encoder.encode(prompt)
    """ add prompt rewrite function """
    new_prompt = prompt_rewrite(prompt)
    te, tm = text_encoder.encode(new_prompt)
    if neg_prompt != "":
        print(f"using negative prompt: {neg_prompt}")
        neg_te, neg_tm = text_encoder.encode(neg_prompt)
        inf_neg_info = (neg_te, neg_tm)
    else:
        print("using default negative prompt")
        inf_neg_info = neg_info
    print(f"prompt embedding shape: {te.shape}")

    for i in range(batch_number):
        generator = torch.Generator()
        if use_seed:
            seed = random.randint(0, 999999999)
            generator.manual_seed(seed)
        print(f"generating video: {steps} steps, seed: {generator.initial_seed()}")
        video = pipeline(
            te,
            tm,
            VAE_CONTEXT_LENGTH,
            VAE_TEMPORAL_COMPRESSION,
            VAE_SCALE_FACTOR,
            num_inference_steps=int(steps),
            negative_prompt=inf_neg_info[0][:, :PROMPT_LENGTH, :],
            negative_prompt_mask=inf_neg_info[1][:, :PROMPT_LENGTH],
            guidance_scale=guidance_scale,
            return_dict=False,
            sequence_parallel_group=MockSPGroup(),
            generator=generator,
        )[0]
        video = (video * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu()
        import einops

        video = einops.rearrange(video[0], "c t h w -> t h w c")
        print(f"video generated: {video.shape}")

        os.makedirs(VIDEO_PATH, exist_ok=True)
        filename = f"{VIDEO_PATH}/{seed}_{uuid.uuid4()}.mp4"
        torchvision.io.write_video(filename, video, 15, options={"-crf": "20"})

    filelist = list_video_files(VIDEO_PATH)
    gr.update(label="File list", choices=filelist)

    return new_prompt, filename


with gr.Blocks() as ux_t2v:
    with gr.Row():
        fc_col_left = gr.Column()
        fc_col_right = gr.Column()
        with fc_col_left:
            prompts = gr.Textbox(
                lines=3,
                value=textwrap.dedent("""caption: [
  {
    "description": "This video depicts a cat with brown hairs and a white apron washing dishes with its paws, we can see the head of the cat, the water is flowing rapidly, the dish is white and made of china, the background is a kitchen, the cat looks very happy",
    "tags": [
      "cat",
      "dishwashing",
      "faucet",
      "kitchen"
    ]
  }
]
                        """),
                placeholder="",
            )
            guidance_scale = gr.Slider(
                minimum=0, maximum=8, step=0.5, value=5, label="Guidance Scale"
            )
            negative_prompts = gr.Textbox(
                lines=1,
                value="""
                        """,
                label="Negative Prompt",
            )
            steps = gr.Slider(minimum=25, maximum=100, step=5, value=75, label="#Steps")
            use_seed = gr.Checkbox(
                label="Use the fixed random seed below", value=True, key="use_seed"
            )
            seed = gr.Number(value=42, label="Seed")
            video = gr.Video(autoplay=True, loop=True)
            genvideo = gr.Button("Generate Video")
            genvideo.click(fn=generate_video, inputs=[prompts, guidance_scale, negative_prompts, steps, use_seed, seed],
                           outputs=[prompts, video])

        with fc_col_right:
            # title="VFM Demo: text2video",
            # description="Generate videos from text prompts using a Predera VFM V0.01.\n HAVE FUN!",
            batch_quantity = gr.Number(value=10, label="How many video do you want")
            batch_button = gr.Button(value="Run batch", variant="primary")
            batch_button.click(fn=batch_generate_video,
                               inputs=[prompts, guidance_scale, negative_prompts, steps, use_seed, seed,
                                       batch_quantity],
                               outputs=[prompts])
            file_list = gr.Dropdown(
                label="File list",
                choices=list_video_files(VIDEO_PATH),
                value=list_video_files(VIDEO_PATH)[0] if list_video_files(VIDEO_PATH) else None,
                interactive=True
            )
            video_preview = gr.Video(label="Video", autoplay=True, loop=True)
            file_list.change(fn=show_video, inputs=[file_list], outputs=video_preview)
            update_button = gr.Button(value="update list")
            update_button.click(update_dropdown_choices, outputs=file_list)

if __name__ == "__main__":
    print("launch ux...")
    print(list_video_files(VIDEO_PATH))
    ux_t2v.launch()
