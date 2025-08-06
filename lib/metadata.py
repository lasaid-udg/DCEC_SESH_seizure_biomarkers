import os
import re
import glob
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple
from . import settings


class MetadataChb():

    def __init__(self, file: str):
        """
        :param file: path to the metadata file
        """
        self.seizure_types = settings["chb-mit"]["seizure_types"]
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
        filename = file.split("/")[-1][:-1]
        patient_id = filename.split("-")[0]

        with open(file) as fp:
            lines = fp.readlines()
        lines = [re.sub(r"\s+", " ", x.lower()) for x in lines]

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
            filename, seizures = self.get_seizure_single_file(_file, patient_id)
            self._seizure_ranges.update({filename: {"full_file": os.path.join(*(["/"] + file.split("/")[:-1] + [filename])),
                                                    "seizures": seizures}})

    def get_seizure_single_file(self, text_lines: list, patient_id: str) -> Tuple[str, tuple]:
        """
        Detect the [start_time, end_time] for each seizure
        :param text_lines: content of the metadata file
        :param patient_id: id of the patient
        """
        seizures = [(0.0, 0.0, "NULL")]
        filename = re.search(r"(?<=:)\s.+", text_lines[0]).group(0).strip()
        number_seizures = int(re.findall(r"(?<=:\s).+", text_lines[1])[0])

        if number_seizures == 0:
            return filename, set()

        for idx in range(number_seizures):
            text = text_lines[(idx + 1) * 2]
            start = int(re.findall(r"(?<=:\s)[0-9]+", text)[0])
            text = text_lines[(idx + 1) * 2 + 1]
            end = int(re.findall(r"(?<=:\s)[0-9]+", text)[0])
            seizures.append((start, end, self.seizure_types[patient_id]))

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
                seizures = self.get_seizure_single_file(lines[index + 1:index + 5],
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
        recording_start_time = re.search(r"(?<=:\s).*", text_lines[0]).group(0)
        seizure_start_time = re.search(r"(?<=:\s).*", text_lines[2]).group(0)
        seizure_end_time = re.search(r"(?<=:\s).*", text_lines[3]).group(0)
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
        self.channels_map = settings["tusz"]["univariate_channels_groups"]
        self.seizure_ranges = files

    @property
    def channels_map(self) -> dict:
        return self._channels_map

    @channels_map.setter
    def channels_map(self, channels_groups: dict) -> None:
        """
        Create a reverse lookup to map each channel to its
        respective hemisphere
        :param channels: channels per hemisphere
        """
        self._channels_map = dict()
        for hemisphere, channels in channels_groups.items():
            for channel in channels:
                self._channels_map[channel.lower()] = hemisphere

    @property
    def seizure_ranges(self) -> dict:
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, files: list) -> None:
        """
        Read and parse a set of metadata files
        :param files: list of paths to the metadata file
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

    def get_seizures_ranges(self, file: str) -> list:
        """
        Detect the [start_time, end_time] for each seizure
        :param file: path to the metadata file
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
        Detect the seizure type for each ictal event
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
            seizure_types.add(tuple([float(line_parts[1]), float(line_parts[2]), line_parts[0].lower()]))

        for seizure_range in seizure_ranges:
            top_seizure_type = set()
            for seizure_type in seizure_types:
                if seizure_range[1] < seizure_type[0]:
                    continue
                if seizure_range[0] > seizure_type[1]:
                    continue

                if seizure_type[2] in self.channels_map:
                    top_seizure_type.add(self.channels_map[seizure_type[2]])
            if len(top_seizure_type) == 2:
                seizure_range[2] = "bilateral"
            elif "right" in top_seizure_type:
                seizure_range[2] = "right"
            elif "left" in top_seizure_type:
                seizure_range[2] = "left"

        return seizure_ranges


class MetadataTuep():

    def __init__(self, files: str):
        """
        :param files: list of paths to the metadata file
        """
        self.recordings = files

    @property
    def recordings(self) -> dict:
        return self._recordings

    @recordings.setter
    def recordings(self, files: list) -> None:
        """
        Read and parse a set of metadata files
        :param files: list of paths to the metadata file
        """
        self._recordings = {}

        for file in files:
            filename = file.split("/")[-1].rstrip(".edf")

            if filename not in self._recordings:
                self._recordings[filename] = {"full_file": file,
                                              "seizures": [(0.0, 0.0, "bckg")]}


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

    def summarize(self) -> None:
        """
        Print the count of events per seizure type
        """
        summary = {}
        for metadata in self.patient_metadata.values():
            for event_set in metadata.seizure_ranges.values():
                events = list(event_set["seizures"])
                for event in events[1:]:
                    if event[-1] not in summary:
                        summary[event[-1]] = 0
                    summary[event[-1]] += 1
        print(json.dumps(summary, indent=4))


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

    def summarize(self) -> None:
        """
        Print the count of events per seizure type
        """
        summary = {}
        for metadata in self.patient_metadata.values():
            for event_set in metadata.seizure_ranges.values():
                events = list(event_set["seizures"])
                for event in events[1:]:
                    if event[-1] not in summary:
                        summary[event[-1]] = 0
                    summary[event[-1]] += 1
        print(json.dumps(summary, indent=4))


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

    def summarize(self) -> None:
        """
        Print the count of events per seizure type
        """
        summary = {}
        for metadata in self.patient_metadata.values():
            for event_set in metadata.seizure_ranges.values():
                events = list(event_set["seizures"])
                for event in events[1:]:
                    if event[-1] not in summary:
                        summary[event[-1]] = 0
                    summary[event[-1]] += 1
        print(json.dumps(summary, indent=4))


class MetadataListTuep():

    def __init__(self):
        self.patient_metadata = settings["tuep"]["dataset"]

    def get(self, patient: str, file: str) -> dict:
        """
        Return eeg recording files per subject
        :param patient: patient ID
        :param file: name of the edf file (without extension)
        """
        file = file.split("/")[-1].split(".")[0]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.recordings.get(file)
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
        for x in glob.glob(os.path.join(root_dir, "**/**/**/*.edf")):
            patient = x.split("/")[-4]
            _grouped_files[patient].append(x)

        for patient, files in _grouped_files.items():
            self._patient_metadata.update({patient: MetadataTuep(files)})


class ChannelBasedMetadataTusz():

    def __init__(self, source_file: str):
        """
        :param source_file: full filename of the edf recording
        """
        self.seizure_ranges = source_file

    @property
    def seizure_ranges(self) -> dict:
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, file: str) -> None:
        """
        Read and parse a single metadata file
        :param file: full filename of the edf recording
        """
        file = file.replace(".edf", ".csv")
        with open(file) as fp:
            lines = fp.readlines()

        for idx, line in enumerate(lines):
            if "confidence" in line:
                break

        max_value = 0
        impacted_channels = defaultdict(list)
        for line in lines[idx + 1:]:
            columns = line.split(",")
            if columns[3] == "bckg":
                continue
            channel = columns[0].split("-")[0]
            seizure_start = int(float(columns[1]))
            seizure_end = int(float(columns[2]))
            impacted_channels[channel].append((seizure_start, seizure_end))

            if seizure_end > max_value:
                max_value = seizure_end

        detailed_impacted_channels = dict()
        for second in range(max_value):
            detailed_impacted_channels[second] = []
            for channel, ranges in impacted_channels.items():
                for _range in ranges:
                    if second > _range[0] and second < _range[1]:
                        detailed_impacted_channels[second].append(channel)

        self._seizure_ranges = detailed_impacted_channels
