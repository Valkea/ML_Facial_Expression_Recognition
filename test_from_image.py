#!/usr/bin/env python
# coding: utf-8

import os
import requests
import argparse

url = "http://0.0.0.0:5000/predict"


def send_pic(filepath):
    # data = {'url': 'data/Angry01.jpg'}
    files = {"media": open(f"{filepath}.jpg", "rb")}

    try:
        result = requests.post(url, files=files).json()
        # result = requests.post(url, json=test_face).json()
        print(f">>> {result}")
    except Exception as error_msg:
        print(f"AN ERROR OCCURED:\n{error_msg}")


if __name__ == "__main__":

    # Initialize arguments parser
    def file_choices(choices, fname):
        ext = os.path.splitext(fname)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return os.path.splitext(fname)[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jpg", type=lambda s: file_choices(("jpg"), s), help="The path to the jpg file"
    )
    args = parser.parse_args()

    print(f"\n>>> SENDING {args.jpg}.jpg\n")
    send_pic(args.jpg)
