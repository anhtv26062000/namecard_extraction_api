from libs.vietocr.vietocr.tool.predictor import Predictor
from libs.vietocr.vietocr.tool.config import Cfg
from utils import get_config
from PIL import Image
import numpy as np
from utils.image import perspective_transform
from utils.logger import logger


class VietOCR:
    def __init__(self, args_file):
        logger.info("Initialize VietOCR Recognizer")
        self.args = get_config(args_file)
        if self.args.config_name != "":
            self.config = Cfg.load_config_from_name(self.args.config_name)
        else:
            self.config = Cfg.load_config_from_file(self.args.config_file)
        self.config["device"] = self.args.device

        # Edit weights model in config/vietocr.yaml
        self.config["weights"] = get_config(args_file).weights
        print(self.config["weights"])

        # Prevent model to download pretrained
        self.config["cnn"]["pretrained"] = False
        logger.info("Loading recognizer...")
        self.recognizer = Predictor(self.config)
        logger.info("Recognizer loaded successfully!")

    def preprocess_input(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image

    def recognize(self, image, return_prob=False):
        """
        image: PIL.Image or np.ndarray
        return_prob: return score of recognizer
        """
        image = self.preprocess_input(image)

        return self.recognizer.predict(image, return_prob=return_prob)

    def recognize_batch(self, images, return_prob=False):
        """
        image: PIL.Image or np.ndarray
        return_prob: return score of recognizer
        """
        # image = self.preprocess_input(images)

        return self.recognizer.predict_batch(images, return_prob=return_prob)

    def crop_and_recognize(self, image, bboxes, return_prob=False):
        """
        image: PIL.Image or np.ndarray
        bboxes: List of bbox[xyxyxyxy]
        return_prob: return score of recognizer
        """
        # Crop text image by bbox
        text_images = []
        for bbox in bboxes:
            text_img = perspective_transform(
                image, [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox) - 1, 2)]
            )
            text_images.append(self.preprocess_input(text_img))

        return self.recognizer.predict_batch(text_images, return_prob=return_prob)
