from PIL import Image, ImageFilter
import onnxruntime as ort
import numpy as np
import cv2
from utils import YoloxONNX
import streamlit as st
import supervision as sv
import pandas as pd
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

from supervision.detection.core import Detections

@classmethod
def from_yolox(cls, bboxes, scores, class_ids) -> Detections:
    return cls(
        xyxy=bboxes,
        confidence=scores,
        class_id=class_ids.astype(int),
    )   
Detections.from_yolox = from_yolox

class_names = ["stm","s","m","l","w"]


# Title of the Streamlit app
model = load_model("yolox.onnx")

image = cv2.imread("images/0.jpg")
bboxes, scores, class_ids = model.inference(image)

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

st.write("test")
st.image(annotated_image)
file = "0.jpg"
res = np.c_[[file]*len(scores),bboxes, scores, class_ids]
s = ["0, stm:stomata","1, s:s_trichome","2, m:m_trichome","3,l:l_trichome","4, w:wripped"]
st.write(s)
df = pd.DataFrame(res, columns=["file","xmin","ymin","xmax","ymax", "score","class_id"])
st.write(df)