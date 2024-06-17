import gradio as gr
import pandas as pd

with gr.Blocks() as hil_ux:
    title = gr.Label("HIL evaluation tool for VFM")
    with gr.Row():
        col_1=gr.Column()
        col_2=gr.Column()
        with col_1:
            compare_version = gr.Dropdown(
                label="Baseline video, by default is example video, you can choose a VFM version to compare",
                choices=["Sample Video", "v1.0", "v1.3", "v1.4"], value="Sample Video", interactive=True)
        with col_2:
            video_version = gr.Dropdown(label="Target VFM version", choices=["v1.0","v1.3","v1.4"],value="v1.0",interactive=True)
    test_cases = gr.Dropdown(label="Test cases", choices=["TC1: A water fall","TC2: a car driving","TC3: A view of beach"],value="TC1: A water fall",interactive=True)
    with gr.Row():
        fc_col_left = gr.Column()
        fc_col_right = gr.Column()
        with fc_col_left:
            video_preview = gr.Video(label="Sample Video")
        with fc_col_right:
            video_preview = gr.Video(label="Generated Video")

    with gr.Group():
        with gr.Row():
            col_3 = gr.Column()
            col_4 = gr.Column()
            with col_3:
                gr.Slider(1, 50,step=1, label="Prompt adherence metrics",interactive=True)
            with col_4:
                gr.Text(label="Prompt adherence comments")
        with gr.Row():
            col_3 = gr.Column()
            col_4 = gr.Column()
            with col_3:
                gr.Slider(1, 50,step=1, label="Realism metrics",interactive=True)
            with col_4:
                gr.Text(label="Realism metrics comments")
        with gr.Row():
            col_3 = gr.Column()
            col_4 = gr.Column()
            with col_3:
                gr.Slider(1, 50,step=1, label="Physics Accuracy ",interactive=True)
            with col_4:
                gr.Text(label="Physics Accuracy metrics comments")
        with gr.Row():
            col_3 = gr.Column()
            col_4 = gr.Column()
            with col_3:
                gr.Slider(1, 50,step=1, label="Visual Quality(Artifacts)",interactive=True)
            with col_4:
                gr.Text(label="Visual Quality(Artifacts) metrics comments")
        total_score=gr.Label(label="Total Score is: 40")
        submit_btn = gr.Button(value="Submit metrics scores")

    tc_score_list_label=gr.Label("TC score list")
    TC_and_metrics_list={'TC name':["TC1: A water fall","TC2: a car driving","TC3: A view of beach"],"Prompt Adherence":["2","4","1",],"Realism":["2","4","10"],"Physics Accuracy":["21","34","41",],"Visual Quality":["2","4","10"]}
    TC_list_with_score = gr.DataFrame(value=pd.DataFrame(TC_and_metrics_list)),


if __name__ == "__main__":
    hil_ux.launch(share=True)
