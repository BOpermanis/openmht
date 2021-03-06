#!/usr/bin/env python
"""CLI"""

from openmht.mht import MHT

import os
import sys
import argparse
import time
import csv

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

__author__ = "Jon Perdomo"
__license__ = "GPL-3.0"
__version__ = "0.1.0"


def read_uv_csv(file_path, frame_max=100):
    """
    Read detections from a CSV.
    Expected column headers are:
    Frame number, U, V
    """
    logging.info("Reading input CSV...")
    detections = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        current_frame = None
        detection_index = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                frame_number, u, v = int(row[0]), float(row[1]), float(row[2])
                if frame_number != current_frame:
                    detection_index = len(detections)
                    if detection_index == frame_max:
                        break

                    detections.append([])
                    current_frame = frame_number

                detections[detection_index].append([u, v])
                line_count += 1

        logging.info('Reading inputs complete. Processed {} detections across {} frames.'.format(line_count-1, len(detections)))

    return detections


def write_uv_csv(file_path, solution_coordinates):
    """
    Write track trees to a CSV.
    Column headers are:
    Frame number, track number, U, V
    """
    logging.info("Writing output CSV...")
    csv_rows = []
    for i in range(len(solution_coordinates)):
        track_coordinates = solution_coordinates[i]
        for j in range(len(track_coordinates)):
            coordinate = track_coordinates[j]
            if coordinate is None:
                u = v = 'None'
            else:
                u, v = [str(x) for x in coordinate]

            csv_rows.append([j, i, u, v])

    # Sort the results by frame number
    csv_rows.sort(key=lambda x: x[0])
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(['frame', 'track', 'u', 'v'])
        writer.writerows(csv_rows)

    logging.info("CSV saved to {}".format(file_path))


def read_parameters(params_file_path):
    """Read in the current Kalman filter parameters."""
    param_keys = ["image_area", "gating_area", "k", "q", "r", "n"]
    params = {}
    with open(params_file_path) as f:
        for line in f:
            line_data = line.split("#")[0].split('=')
            if len(line_data) == 2:
                key, val = [s.strip() for s in line_data]
                if key in param_keys:
                    try:
                        val = float(val)
                    except ValueError:
                        raise AssertionError("Incorrect value type in params.txt: {}".format(line))

                    param_keys.remove(key)
                    params[key] = val
            else:
                raise AssertionError("Error in params.txt formatting: {}".format(line))

    if param_keys:
        raise AssertionError("Parameters not found in params.txt: " + ", ".join(param_keys))

    return params


def run():

    class Args:
        def __init__(self):
            pass

    args = Args()
    args.ifile = "/home/bruno/repos/openmht/SampleData/TestInput_01.csv"
    args.ofile = "/home/bruno/repos/openmht/OutputDetections.csv"
    args.pfile = "/home/bruno/repos/openmht/params.txt"

    # Parse arguments
    input_file = args.ifile
    output_file = args.ofile
    param_file = args.pfile

    logging.info("Input file is: {}".format(input_file))
    logging.info("Output file is: {}".format(output_file))
    logging.info("Parameter file is: {}".format(param_file))

    # Read MHT parameters
    params = read_parameters(param_file)
    logging.info("MHT parameters: {}".format(params))

    # run MHT on detections
    detections = read_uv_csv(input_file)
    start = time.time()
    mht = MHT(detections, params)
    solution_coordinates = mht.run()
    write_uv_csv(output_file, solution_coordinates)
    end = time.time()
    elapsed_seconds = end - start
    logging.info("Elapsed time (seconds): {0:.3f}".format(elapsed_seconds))

if __name__ == "__main__":
    run()