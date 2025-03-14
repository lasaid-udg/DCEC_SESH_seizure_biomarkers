#!/usr/bin/python3.7
import os
import sys
import yaml
import pathlib
import logging.config
from dotenv import load_dotenv


load_dotenv()

current_path = str(pathlib.Path(__file__).parent.parent.resolve())

logging.config.fileConfig(current_path + "/etc/logging.ini")


def catch_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value,
                                                  exc_traceback))


sys.excepthook = catch_exception

with open(current_path + "/etc/settings.yaml", "r") as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if "chb-mit" in settings:
    settings["chb-mit"]["dataset"] = os.environ.get("CHB_DATASET_HOME")
if "siena" in settings:
    settings["siena"]["dataset"] = os.environ.get("SIENA_DATASET_HOME")
if "tusz" in settings:
    settings["tusz"]["dataset"] = os.environ.get("TUSZ_DATASET_HOME")
if "tuep" in settings:
    settings["tuep"]["dataset"] = os.environ.get("TUEP_DATASET_HOME")