import streamlit as st
from PIL import Image, ImageFilter
import onnxruntime as ort
import zipfile
import io
import cv2
from utils import YoloxONNX, draw
import numpy as np
from utils import YoloxONNX
import supervision as sv
import pandas as pd
import os
from supervision.detection.core import Detections

@st.cache_resource
def load_model(path):
    model = YoloxONNX(
        model_path="yolox.onnx",
        input_shape=(640,640),
        class_score_th=0.3,
        nms_th=0.45,
        nms_score_th=0.1,
        with_p6=False,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
    return model

@classmethod
def from_yolox(cls, bboxes, scores, class_ids) -> Detections:
    return cls(
        xyxy=bboxes,
        confidence=scores,
        class_id=class_ids.astype(int),
    )   
Detections.from_yolox = from_yolox

class_names = ["stm","s","m","l","w"]
@st.cache_resource
def load_model(path):
    model = YoloxONNX(
        model_path="yolox.onnx",
        input_shape=(640,640),
        class_score_th=0.3,
        nms_th=0.45,
        nms_score_th=0.1,
        with_p6=False,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
    return model

# Title of the Streamlit app
model = load_model("yolox.onnx")
st.title("Image Upload and Analysis")

# File uploader for images
uploaded_files = st.file_uploader("Choose an image...",accept_multiple_files=True, type=["jpg", "jpeg", "png","JPG","JPEG","PNG"])

ress = []
images = []
image_names = []
if uploaded_files is not None:
    with st.expander("annotated images"):
        for uploaded_file in uploaded_files:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            # convert to numpy
            image = np.array(image).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            bboxes, scores, class_ids = model.inference(image)
            # annotated = draw(image.copy(), 0.5, bboxes, scores, class_ids,
            #                  class_names)
            # filter
            indices = [i for i, value in enumerate(scores) if value > 0.4]
            bboxes = bboxes[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = Detections.from_yolox(bboxes, scores, class_ids)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_scale=1)

            annotated_image = box_annotator.annotate(
                scene=image, detections=detections)

            labels = [
                f"{class_names[int(idx)]} {conf:.2f}"
                for idx, conf in zip(class_ids, scores)
            ]

            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)
            st.image(annotated_image)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for i, image in enumerate(annotated_image):
                    # ndarray → バイナリ画像データに変換
                    is_success, buffer = cv2.imencode(".png", image)
                    if not is_success:
                        continue  # エンコード失敗したらスキップ

                    image_bytes = buffer.tobytes()

        # ZIPファイルに書き込む（ファイル名指定）
                    filename = image_names[i] if i < len(image_names) else f"image_{i}.png"
                    zip_file.writestr(filename, image_bytes)
            res = np.c_[[uploaded_file.name]*len(scores),bboxes, scores, class_ids]
            ress.extend(res)
    zip_buffer.seek(0)
    # Streamlitでダウンロードボタンを表示
    st.download_button(
        label="画像をZIPで一括ダウンロード",
        data=zip_buffer,
        file_name="images.zip",
        mime="application/zip"
    )
    df = pd.DataFrame(ress, columns=["file","xmin","ymin","xmax","ymax", "score","class_id"])
    st.write(df)


st.write("備考")
s = ["0, stm:stomata","1, s:s_trichome","2, m:m_trichome","3,l:l_trichome","4, w:wripped"]
st.write(s)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Apply a filter to the image
    # filtered_image = image.filter(ImageFilter.BLUR)
    
    # Display the filtered image
    # st.image(filtered_image, caption='Blurred Image.', use_column_width=True)
