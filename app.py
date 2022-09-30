import numpy as np
import cv2
from PIL import Image, ImageOps
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from requests_toolbelt.multipart.decoder import MultipartDecoder
import streamlit as st
from streamlit_image_comparison import image_comparison


def get_mask_from_image(
        _image: Image,
        _api: str = 'http://localhost:9009/predict') -> np.ndarray:
    image_file = io.BytesIO()
    _image.save(image_file, 'WebP')
    image_file.seek(0)

    mp_encoder = MultipartEncoder(
        fields={
            'image': (
                'image', image_file,
                'image/webp'
            )
        }
    )
    response = requests.post(
        url=_api,
        data=mp_encoder,
        headers={'Content-Type': mp_encoder.content_type}
    )
    response_status_code = response.status_code

    if response_status_code != 200:
        raise RuntimeError(
            'Status mask server: {}'.format(
                response.status_code)
        )

    mask = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    response.close()

    return mask


def draw_mask(_image: Image, mask: np.ndarray) -> Image:
    gt_color = np.array([0, 200, 20], dtype=np.float16)

    img = np.array(_image.convert('RGB'))

    img[mask == 255] = (
            img[mask == 255].astype(np.float16) * 0.3 + gt_color * 0.7
    ).astype(np.uint8)

    return Image.fromarray(img)


def main():
    st.markdown("""
        # Сегментация капилляров глаза человека по снимкам с офтальмологической щелевой лампы
        > Powered by [Kovalenko Alexey](https://github.com/AlexeySrus)
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(original_image)
        mask = get_mask_from_image(original_image)
        result = draw_mask(original_image, mask)

        image_comparison(
            img1=original_image,
            img2=result,
            label1="Original Image",
            label2="Image with Vessel Mask"
        )


if __name__ == "__main__":
    main()
