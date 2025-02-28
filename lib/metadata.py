import os
import re
import glob
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple
from . import settings


class MetadataChb():

    def __init__(self, file: str):
        """
        :param file: path to the metadata file
        """
        self.seizure_ranges = file

    @property
    def seizure_ranges(self) -> dict:
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, file: str) -> None:
        """
        Read and parse a single metadata file
        :param file: path to the metadata file
        """
        self._seizure_ranges = {}
        clean_lines, single_file = [], []

        with open(file) as fp:
            lines = fp.readlines()
        lines = [re.sub("\s+", " ", x.lower()) for x in lines]

        skip = True
        clean_lines = []
        for line in lines:
            if "file name" in line:
                skip = False
            if "changed" in line:
                skip = True
            if not skip:
                clean_lines.append(line)

        clean_lines, text_lines = [], clean_lines
        for line in text_lines:
            if line == " ":
                clean_lines.append(single_file)
                single_file = []
            elif not any(x in line for x in ["file start", "file end"]):
                single_file.append(line)

        if len(single_file):
            clean_lines.append(single_file)

        for _file in clean_lines:
            filename, seizures = self.get_seizure_single_file(_file)
            self._seizure_ranges.update({filename: {"full_file": os.path.join(*(["/"] + file.split("/")[:-1] + [filename])),
                                                    "seizures": seizures}})

    def get_seizure_single_file(self, text_lines: list) -> Tuple[str, tuple]:
        """
        Detect the [start_time, end_time] for each seizure
        :param text_lines: content of the metadata file
        """   
        seizures = [(0.0, 0.0, "NULL")]
        filename = re.search("(?<=:)\s.+", text_lines[0]).group(0).strip()
        number_seizures = int(re.findall("(?<=:\s).+", text_lines[1])[0])

        if number_seizures == 0:
            return filename, set()

        for idx in range(number_seizures):
            text = text_lines[(idx+1)*2]
            start = int(re.findall("(?<=:\s)[0-9]+", text)[0])
            text = text_lines[(idx+1)*2 + 1]
            end = int(re.findall("(?<=:\s)[0-9]+", text)[0])
            seizures.append((start, end, "NULL"))
        return filename, tuple(seizures)


class MetadataSiena():

    def __init__(self, file: str):
        """
        :param file: path to the metadata file
        """
        self.seizure_types = settings["siena"]["seizure_types"]
        self.seizure_ranges = file

    @property
    def seizure_ranges(self) -> dict:
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, file: str) -> None:
        """
        Read and parse a single metadata file
        :param file: path to the metadata file
        """
        self._seizure_ranges = {}

        with open(file) as fp:
            lines = fp.readlines()

        for index, line in enumerate(lines):
            if "File name" in line:
                filename = line.split(" ")[-1][:-1]
                patient_id = filename.split("-")[0]
                seizures = self.get_seizure_single_file(lines[index+1:
                                                             index+5],
                                                        patient_id)
                if filename not in self._seizure_ranges:
                    self._seizure_ranges[filename] = {"full_file": os.path.join(*(["/"] + file.split("/")[:-1] + [filename])),
                                                      "seizures": [(0.0, 0.0, "NULL")]}
                self._seizure_ranges[filename]["seizures"].append(seizures)

    def get_seizure_single_file(self, text_lines: list, patient_id: int) -> tuple:
        """
        Detect the [start_time, end_time] for each seizure
        :param text_lines: content of the metadata file
        :param patient_id: id of the patient
        """   
        assert "Registration start" in text_lines[0]
        recording_start_time = re.search("(?<=:\s).*", text_lines[0]).group(0)
        seizure_start_time = re.search("(?<=:\s).*", text_lines[2]).group(0)
        seizure_end_time = re.search("(?<=:\s).*", text_lines[3]).group(0)
        recording_start_time = datetime.strptime(recording_start_time.strip(),
                                                 "%H.%M.%S")
        seizure_start_time = datetime.strptime(seizure_start_time.strip(),
                                               "%H.%M.%S")
        seizure_end_time = datetime.strptime(seizure_end_time.strip(),
                                             "%H.%M.%S")

        if recording_start_time > seizure_start_time:
            recording_start_time = recording_start_time - timedelta(days=1)

        start_seconds = (seizure_start_time - recording_start_time).seconds
        end_seconds = (seizure_end_time - recording_start_time).seconds

        return (start_seconds, end_seconds, self.seizure_types[patient_id])


class MetadataTusz():

    def __init__(self, files: str):
        """
        :param files: list of paths to the metadata file
        """
        self.seizure_ranges = files

    @property
    def seizure_ranges(self) -> dict:
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, files: list) -> None:
        """
        Read and parse a set of metadata files
        :param file: list of paths to the metadata file
        """
        self._seizure_ranges = {}

        for file in files:
            filename = file.split("/")[-1].rstrip(".csv_bi")

            if filename not in self._seizure_ranges:
                self._seizure_ranges[filename] = {"full_file": os.path.join(*(["/"] + file.split("/")[:-1] + [f"{filename}.edf"])),
                                                  "seizures": [(0.0, 0.0, "bckg")]}
            seizure_ranges = self.get_seizures_ranges(file)
            seizure_ranges = self.get_seizures_types(file.replace(".csv_bi", ".csv"), seizure_ranges)
            self._seizure_ranges[filename]["seizures"].extend(seizure_ranges)

    def get_seizures_ranges(self, file) -> list:
        """
        Detect the [start_time, end_time] for each seizure
        :param text_lines: content of the metadata file
        """   
        seizures_ranges = []
        with open(file) as fp:
            lines = fp.readlines()

        idx = [lines.index(i) for i in lines if "start_time" in i][0]

        for event in lines[idx + 1:]:
            if "z" not in event:
                continue
            line_parts = event.split(",")
            seizures_ranges.append([float(line_parts[1]), float(line_parts[2]), "seizure"])

        return seizures_ranges

    def get_seizures_types(self, file: str, seizure_ranges: str) -> list:
        """
        Detect seizure for each ictal event
        :param file: path to the metadata file
        :param seizure_ranges: output of the method ´get_seizures_ranges´
        """   
        seizure_types = set()
        with open(file) as fp:
            lines = fp.readlines()

        idx = [lines.index(i) for i in lines if "start_time" in i][0]

        for event in lines[idx + 1:]:
            if "z" not in event:
                continue
            line_parts = event.split(",")
            seizure_types.add(tuple([float(line_parts[1]), float(line_parts[2]), line_parts[3]]))

        for seizure_range in seizure_ranges:
            top_seizure_type = list()
            for seizure_type in seizure_types:
                if seizure_range[1] < seizure_type[0]:
                    continue
                if seizure_range[0] > seizure_type[1]:
                    continue
                top_seizure_type.append(seizure_type[2])
            top_type = max(set(top_seizure_type), key=top_seizure_type.count)
            seizure_range[2] = top_type

        return seizure_ranges


class MetadataListChb():

    def __init__(self):
        self.patient_metadata = settings["chb-mit"]["dataset"]

    def get(self, patient: str, file: str) -> dict:
        """
        Return seizure ocurrences
        :param patient: patient ID
        :param file: name of the edf file (without extension)
        """
        file = file.split("/")[-1]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self) -> dict:
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str) -> None:
        """
        Read and parse all the metadata files
        :param root_dir: dataset's base directory
        """
        self._patient_metadata = {}

        for x in glob.glob(os.path.join(root_dir, "**/*-summary.txt")):
            patient = x.split("/")[-1].split("-")[0]
            self._patient_metadata.update({patient: MetadataChb(x)})


class MetadataListSiena():

    def __init__(self):
        self.patient_metadata = settings["siena"]["dataset"]

    def get(self, patient: str, file: str) -> dict:
        """
        Return seizure ocurrences
        :param patient: patient ID
        :param file: name of the edf file (without extension)
        """
        file = file.split("/")[-1]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self) -> dict:
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str) -> None:
        """
        Read and parse all the metadata files
        :param root_dir: dataset's base directory
        """
        self._patient_metadata = {}

        for x in glob.glob(os.path.join(root_dir, "**/*.txt")):
            patient = x.split("/")[-1].split("-")[-1].rstrip(".txt")
            self._patient_metadata.update({patient: MetadataSiena(x)})


class MetadataListTusz():

    def __init__(self):
        self.patient_metadata = settings["tusz"]["dataset"]

    def get(self, patient: str, file: str) -> dict:
        """
        Return seizure ocurrences
        :param patient: patient ID
        :param file: name of the edf file (without extension)
        """
        file = file.split("/")[-1].split(".")[0]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self) -> dict:
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str) -> None:
        """
        Read and parse all the metadata files
        :param root_dir: dataset's base directory
        """
        _grouped_files = defaultdict(list)
        self._patient_metadata = {}
        for x in glob.glob(os.path.join(root_dir, "**/**/**/*.csv_bi")):
            patient = x.split("/")[-4]
            _grouped_files[patient].append(x)

        for patient, files in _grouped_files.items():
            self._patient_metadata.update({patient: MetadataTusz(files)})