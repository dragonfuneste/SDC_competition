from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.metrics.pairwise import pairwise_distances
from pymongo import MongoClient
from pymongo.collection import Collection
from pathlib import Path
import numpy as np
import os
import shapely
import time
import abc
import sys
import json
import grpc
import argparse
import competition_pb2_grpc
import competition_pb2

"""
FILE WHERE EXAMPLE CLASS ARE PUT TO HAVE A BETTER CLARITY OF WHAT I HAD ADDED TO IMPLEMENTE MY SDC TESTING 
"""





class SampleEvaluationTestLoader(EvaluationTestLoader):
    """Sample test loader for the provided data."""

    def __init__(self, file_path: str, training_prop: float):
        """Initialize test loader with path to dataset."""
        super().__init__()
        self.file_path = file_path
        self.raw_test_cases: list = None
        with open(file_path, 'r') as fp:
            self.raw_test_cases = json.load(fp)

        self.test_details_lst = _make_test_details_list(self.raw_test_cases)
        self.test_details_dict = {test_details.test_id: test_details for test_details in self.test_details_lst}
        self.training_prop: float = training_prop
        self.current_oracle_index = 0
        self.split_index = int(training_prop*len(self.raw_test_cases))
        self.current_test_index = self.split_index

    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        return self.file_path

    def get_test_details_lst(self) -> list[TestDetails]:
        """Return test cases in a list."""
        return self.test_details_lst

    def get_test_details_dict(self) -> dict:
        """Return test cases in a dictionary."""
        return self.test_details_dict

    def load(self, test_id: str) -> TestDetails:
        """Return test case with a specific id."""
        return self.test_details_dict[test_id]

    def get_test_ids(self):
        """Return al test case ids."""
        return self.test_details_dict.keys()