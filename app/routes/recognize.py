import io
import os
import json
import time
import base64
from pathlib import Path

import cv2
import shutil
import aiofiles
import traceback
import numpy as np
from PIL import Image
from fastapi import APIRouter, Form, File, UploadFile, Form

from src.recognizer import VietOCR
from libs.mmocr.mmocr.utils.ocr import MMOCR
from src import hough_based_detection as hough
from src import postprocessing
from utils import transform
from utils import easyocr_group
from utils.logger import logger


## VARIABLE
input_path = "./temp/1_input/raw.jpg"
cropper_path = "./temp/2_cropper/crop.jpg"
detector_path = "./temp/3_detector/det.jpg"
recognizer_path = "./temp/4_recognizer/"
det = "PANet_IC15"
det_ckpt = "./weights/panet_r18_fpem_ffm_sbn_600e_vbc583_20221215.pth"
text_det_config_dir = os.path.join(str(Path.cwd()), "libs/mmocr/configs/")

## LOAD MODEL
detector = MMOCR(det=det, det_ckpt=det_ckpt, recog=None, config_dir=text_det_config_dir)
recognizer = VietOCR("config/vietocr.yaml")

## FOLDER
try:
    os.makedirs(recognizer_path, exist_ok=True)
    print("Created directory '%s' successfully" % recognizer_path)
except OSError as error:
    os.removedirs(recognizer_path)
    print("Reset directory '%s' successfully" % recognizer_path)
    os.makedirs(recognizer_path, exist_ok=True)
    print("Created directory '%s' successfully" % recognizer_path)

router = APIRouter()


@router.post("/recognize")
async def detect(file: UploadFile = File(...)):
    # Save file image
    async with aiofiles.open(input_path, "wb") as out_file:
        img = await file.read()  # async read
        await out_file.write(img)  # async write

    # Document Localization
    start_time = time.time()
    raw_img = cv2.imread(input_path)
    doc_bbox = hough.single_detector(raw_img)
    crop_img = transform.four_point_transform(raw_img, doc_bbox)
    cv2.imwrite(cropper_path, crop_img)
    print("Document Localization Runtime: ", time.time() - start_time)

    # Text Detetion
    det_results = detector.readtext(cropper_path, output=detector_path)
    m_pD = np.array(det_results[0]["boundary_result"]).astype(int)[:, :8]

    # Crop and Text Recognition
    word_imgs = []
    for i, word_bbox in enumerate(m_pD):
        img = cv2.imread(cropper_path)
        word_bbox = word_bbox.reshape((4, 2)).astype("float32")
        word_img = transform.four_point_transform_word(img, word_bbox)
        word_imgpath = recognizer_path + "crop_" + str(i).zfill(3) + ".jpg"
        cv2.imwrite(word_imgpath, word_img)

        word_img = Image.open(word_imgpath)

        word_imgs.append(word_img)

    text, score = recognizer.recognize_batch(word_imgs, return_prob=True)

    # Group words to block
    raw_result = []
    for i in range(len(m_pD)):
        bbox = m_pD[i].reshape((-1, 2))
        raw_result.append((bbox.tolist(), text[i]))

    rec_group = easyocr_group.get_paragraph(raw_result)
    crop_img = cv2.imread(cropper_path)
    # easyocr_group.show_in_order(crop_img, rec_group, raw_result, output=detector_path)
    with open(detector_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read())

    rec_result = []
    rec_result_height_len = []
    for i in range(len(rec_group)):
        rec_result.append(rec_group[i])
        rec_result_height_len.append(rec_group[i][2:])
    name, mobile, add, mail, web = postprocessing.post_process(
        rec_result, rec_result_height_len
    )

    return {
        "code": "1000",
        "data": {
            "name": name,
            "mobile": mobile,
            "address": add,
            "email": mail,
            "website": web,
            "bboxes": rec_result,
            "imgdata": img_base64,
        },
    }


@router.post("/recognize_crop")
async def detect(file: UploadFile = File(...)):

    # Save file image
    async with aiofiles.open(cropper_path, "wb") as out_file:
        img = await file.read()  # async read
        await out_file.write(img)  # async write

        # Text Detetion
    det_results = detector.readtext(cropper_path)
    m_pD = np.array(det_results[0]["boundary_result"]).astype(int)[:, :8]

    # Crop and Text Recognition
    word_imgs = []
    for i, word_bbox in enumerate(m_pD):
        img = cv2.imread(cropper_path)
        word_bbox = word_bbox.reshape((4, 2)).astype("float32")
        word_img = transform.four_point_transform_word(img, word_bbox)
        word_imgpath = recognizer_path + "crop_" + str(i).zfill(3) + ".jpg"
        cv2.imwrite(word_imgpath, word_img)

        word_img = Image.open(word_imgpath)

        word_imgs.append(word_img)

    text, score = recognizer.recognize_batch(word_imgs, return_prob=True)

    # Group words to block
    raw_result = []
    for i in range(len(m_pD)):
        bbox = m_pD[i].reshape((-1, 2))
        raw_result.append((bbox.tolist(), text[i]))

    rec_group = easyocr_group.get_paragraph(raw_result)
    crop_img = cv2.imread(cropper_path)
    easyocr_group.show_in_order(crop_img, rec_group, raw_result, output=detector_path)
    with open(detector_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read())

    rec_result = []
    rec_result_height_len = []
    for i in range(len(rec_group)):
        rec_result.append(rec_group[i])
        rec_result_height_len.append(rec_group[i][2:])
    name, mobile, add, mail, web = postprocessing.post_process(
        rec_result, rec_result_height_len
    )

    return {
        "code": "1000",
        "data": {
            "name": name,
            "mobile": mobile,
            "address": add,
            "email": mail,
            "website": web,
            "bboxes": rec_result,
            "imgdata": img_base64,
        },
    }


@router.post("/recognize_crop_test")
async def detect(file: UploadFile = File(...)):

    # # Save file image
    # async with aiofiles.open(cropper_path, "wb") as out_file:
    #     img = await file.read()  # async read
    #     await out_file.write(img)  # async write

    # Save file image
    async with aiofiles.open(input_path, "wb") as out_file:
        img = await file.read()  # async read
        await out_file.write(img)  # async write

    # Document Localization
    start_time = time.time()
    raw_img = cv2.imread(input_path)
    doc_bbox = hough.single_detector(raw_img)
    crop_img = transform.four_point_transform(raw_img, doc_bbox)
    cv2.imwrite(cropper_path, crop_img)
    print("Document Localization Runtime: ", time.time() - start_time)

    # Text Detetion
    det_results = detector.readtext(cropper_path)
    m_pD = np.array(det_results[0]["boundary_result"]).astype(int)[:, :8]

    # Crop and Text Recognition
    word_imgs = []
    for i, word_bbox in enumerate(m_pD):
        img = cv2.imread(cropper_path)
        word_bbox = word_bbox.reshape((4, 2)).astype("float32")
        word_img = transform.four_point_transform_word(img, word_bbox)
        word_imgpath = recognizer_path + "crop_" + str(i).zfill(3) + ".jpg"
        cv2.imwrite(word_imgpath, word_img)

        word_img = Image.open(word_imgpath)
        word_imgs.append(word_img)

    text, score = recognizer.recognize_batch(word_imgs, return_prob=True)

    # Group words to block
    raw_result = []
    for i in range(len(m_pD)):
        bbox = m_pD[i].reshape((-1, 2))
        raw_result.append((bbox.tolist(), text[i]))

    rec_group = easyocr_group.get_paragraph(raw_result)
    crop_img = cv2.imread(cropper_path)
    easyocr_group.show_in_order(crop_img, rec_group, raw_result, output=detector_path)

    test = []
    rec_result = []
    rec_result_height_len = []
    for i in range(len(rec_group)):
        test.append(rec_group[i][1:])
        rec_result.append(rec_group[i])
        rec_result_height_len.append(rec_group[i][2:])
    name, mobile, add, mail, web = postprocessing.post_process(
        rec_result, rec_result_height_len
    )

    return {
        "code": "1000",
        "data": {
            "rec_test": test,
            "name:": name,
            "mobile:": mobile,
            "address:": add,
            "email:": mail,
            "website:": web,
        },
    }
