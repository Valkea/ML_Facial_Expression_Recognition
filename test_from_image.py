#!/usr/bin/env python
# coding: utf-8

import os
import requests
import argparse


def send_pic(filepath, url):
    # data = {'url': 'data/Angry01.jpg'}
    files = {"media": open(f"{filepath}.jpg", "rb")}

    try:
        result = requests.post(url, files=files).json()
        # result = requests.post(url, json=test_face).json()
        print(f">>> {result}")
    except Exception as error_msg:
        print(f"AN ERROR OCCURED:\n{error_msg}")


if __name__ == "__main__":

    url = "http://0.0.0.0:5000/predict"

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
    parser.add_argument(
        "--url", type=str, help="The FER13 server URL",
    )
    args = parser.parse_args()

    if(args.url):
        url = args.url

    print(f"\n>>> SENDING {args.jpg}.jpg TO {url}\n")
    send_pic(args.jpg, url)
