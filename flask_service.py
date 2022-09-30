from argparse import ArgumentParser, Namespace
import numpy as np
import cv2
from flask import Flask, abort, request, Response, send_file
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
import logging
import torch
from PIL import Image
from threading import Lock
import traceback
from inference_utils import SegmentationInference, SGLInference

logging.basicConfig(level=logging.INFO)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def args_parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=9009)
    parser.add_argument(
        '--model', required=False, type=str,
        default='data/eye_segment.pt',
        help='Path to traced MAnet PyTorch model'
    )
    parser.add_argument(
        '--sgl-model', required=False, type=str,
        default='data/SGL_model_latest.pt',
        help='Path to traced SGL PyTorch model'
    )
    return parser.parse_args()


class FunctionServingWrapper(object):
    """
    Class of wrapper for restriction count of simultaneous function calls
    """

    def __init__(self,
                 callable_function: callable,
                 count_of_parallel_users: int = 1):
        self.callable_function = callable_function
        self.resources = [Lock() for _ in range(count_of_parallel_users)]
        self.call_mutex = Lock()

    def __call__(self, *_args, **_kwargs):
        """
        Run call method of target callable function
        Args:
            *_args: args for callable function
            **_kwargs: kwargs for callable function
        Returns:
            Return callable function results
        """
        self.call_mutex.acquire()
        i = -1
        while True:
            for k in range(len(self.resources)):
                if not self.resources[k].locked():
                    i = k
                    break
            if i > -1:
                break

        self.resources[i].acquire()
        self.call_mutex.release()

        result = self.callable_function(*_args, **_kwargs)
        self.resources[i].release()

        return result


app = Flask(__name__)
app_log = logging.getLogger('werkzeug')
app_log.setLevel(logging.ERROR)
segment_model_serve: FunctionServingWrapper = None
sgl_segment_model_serve: FunctionServingWrapper = None


def serve_pil_image(pil_img: Image):
    return send_file(
        io.BytesIO(cv2.imencode('.png', np.array(pil_img.convert('RGB')))[1]), \
        mimetype='image/png'
    )


@app.route('/predict_manet', methods=['POST'])
def manet_server_inference():
    logging.info(f'{request.remote_addr}   predict ')
    global segment_model_serve

    result: Image = None

    try:
        image: Image = Image.open(request.files['image'])
        result = segment_model_serve(image)
    except Exception as e:
        logging.error(
            'server_inference: traced exception'
            '{}: \'{}'.format(
                e, traceback.format_exc()
            )
        )

        abort(400)

    return serve_pil_image(result)


@app.route('/predict_sgl', methods=['POST'])
def sgl_server_inference():
    logging.info(f'{request.remote_addr}   predict ')
    global sgl_segment_model_serve

    result: Image = None

    try:
        image: Image = Image.open(request.files['image'])
        result = sgl_segment_model_serve(image)
    except Exception as e:
        logging.error(
            'server_inference: traced exception'
            '{}: \'{}'.format(
                e, traceback.format_exc()
            )
        )

        abort(400)

    return serve_pil_image(result)


if __name__ == '__main__':
    args = args_parse()
    segment_model_serve = FunctionServingWrapper(
        SegmentationInference(
            model_path=args.model,
            device=DEVICE
        )
    )
    sgl_segment_model_serve = FunctionServingWrapper(
        SGLInference(
            model_path=args.sgl_model,
            device=DEVICE
        )
    )
    app.run(host=args.ip, debug=False, port=args.port)