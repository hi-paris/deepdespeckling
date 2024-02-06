import streamlit as st
from PIL import Image
from io import BytesIO, TextIOWrapper
import numpy as np

from deepdespeckling.despeckling import get_denoiser, get_model_weights_path
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
from deepdespeckling.utils.load_cosar import cos2mat
from deepdespeckling.utils.utils import load_sar_image

st.set_page_config(layout="wide", page_title="Deepdespeckling")

st.write("## Despeckle your SAR images")
st.write(
    "Try to upload a light image "
)
st.sidebar.write("## Upload and download :gear:")

# Download the fixed image


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def preprocess_for_png(image, threshold):
    noisy_image = np.clip(image, 0, threshold)
    noisy_image = noisy_image / threshold * 255
    noisy_image = Image.fromarray(image.astype('float64')).convert('L')

    return noisy_image


def preprocess_noisy_image(upload_path, denoiser):
    image = cos2mat(upload_path)
    noisy_image = np.array(image).reshape(
        1, np.size(image, 0), np.size(image, 1), 2)
    noisy_image, _, _ = denoiser.preprocess_noisy_image(noisy_image)
    threshold = np.mean(noisy_image) + 3 * np.std(noisy_image)

    noisy_image = preprocess_for_png(noisy_image, threshold=threshold)

    return image, noisy_image, threshold


def fix_image(upload_path):
    model_name = "spotlight"
    denoiser = get_denoiser(model_name=model_name)

    image, noisy_image, threshold = preprocess_noisy_image(
        upload_path=upload_path, denoiser=denoiser)

    col1.write("Original Image :camera:")
    col1.image(noisy_image)

    model_weights_path = get_model_weights_path(model_name=model_name)
    despeckled_image = denoiser.denoise_image(
        image, model_weights_path, PATCH_SIZE, STRIDE_SIZE)["denoised"]["full"]

    despeckled_image = preprocess_for_png(
        despeckled_image, threshold=threshold)

    col2.write("Despeckled Image")
    col2.image(despeckled_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download despeckled image", convert_image(
        despeckled_image), "despeckled.png", "image/png")


def init_image():
    col1.write("Noisy Image ")
    col1.image("img/entire/noisy.png")

    col1.write("Despeckled Image")
    col1.image("img/entire/denoised.png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["cos"])

if my_upload is not None:
    b = my_upload.getvalue()
    with open("test.cos", "wb") as f:
        f.write(b)
    fix_image("test.cos")
else:
    init_image()
