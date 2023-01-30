import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import argparse
import pydicom

from preprocessing import get_preprocessed_frames


class CustomLogger:
    """
    A class facilitating the logging of analysis progress and errors.

    ...

    Attributes
    ----------
    log_file_path : str
        Path of the log file
    error_log_file_path : str
        Path of the error log file

    Methods
    -------
    log(msg_list, log_type="general"):
        Prints log message to console and log file(s).
    """

    def __init__(self, log_file_path, error_log_file_path):
        """
        Constructs all the necessary attributes for the logger object.

        Parameters
        ----------
            log_file_path : str
                Path of the log file
            error_log_file_path : str
                Path of the error log file
        """

        self.log_file_path = log_file_path
        self.error_log_file_path = error_log_file_path

        # Deleting previous log files
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        if os.path.exists(self.error_log_file_path):
            os.remove(self.error_log_file_path)

    def log(self, msg_list, log_type="general"):
        """
        Prints log message to console and log file(s).

        Parameters
        ----------
        msg_list : list
            The content of the log message
        log_type : str
            Type of the log message ("general" or "error")
        """

        # Converting the list of messages into a single string
        msg = " - ".join(msg_list)

        # Printing log message to console
        print(msg)

        # Printing log message to log file
        log_file = open(self.log_file_path, "a")
        log_file.write(msg + "\n")
        log_file.close()

        # If it is an error message, print it to the error log file as well
        if log_type == "error":
            error_log_file = open(self.error_log_file_path, "a")
            error_log_file.write(msg + "\n")
            error_log_file.close()


def predict_rvef(dicom_folder, model_path, output_folder, orientation):
    """
    Predicts RVEF for each DICOM file in the input folder.

    Parameters
    ----------
    dicom_folder : str
        Path of the folder containing the DICOM files
    model_path : str
        Path of the trained model file
    output_folder : str
        Path of the output folder
    orientation : str
        Orientation of the left and right ventricles ("Stanford" or "Mayo").
        Mayo – the right ventricle on the right and the left ventricle on the left side.
        Stanford – the left ventricle on the right and the right ventricle on the left side.
    """

    # Setting the paths of log files
    log_files_folder = r".\\log_files"
    os.makedirs(log_files_folder, exist_ok=True)
    log_file_path = os.path.join(log_files_folder, "log.txt")
    error_log_file_path = os.path.join(log_files_folder, "error_log.txt")

    # Initializing the logger
    custom_logger = CustomLogger(log_file_path, error_log_file_path)

    # Checking if there is a CUDA-enabled GPU available (This is required for running our model!)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        # Printing error message to console and log file
        msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"), "No CUDA-enabled GPU is available!"]
        custom_logger.log(msg_list, "error")
        return

    # Searching for DICOM files in the input folder
    dicom_files = []
    for dir_path, dir_names, file_names in os.walk(dicom_folder):
        exclusion_criteria = ["DICOMDIR"]
        dicom_files += [os.path.join(dir_path, f) for f in file_names if not any(x in f for x in exclusion_criteria)]

    # Checking if there are any files to be analyzed
    if not dicom_files:
        msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"), "There are no files to be analyzed in the folder!"]
        custom_logger.log(msg_list, "error")
        return

    # Loading the trained model and sending it to the GPU
    model = torch.load(model_path, map_location=device)
    model.eval()
    model = model.to(device)

    # Predicting RVEF for each video in the input folder
    list_of_predicted_rvefs = []
    list_of_dicom_files = []
    for i, dicom_file_path in enumerate(dicom_files):

        # Updating the analysis progress
        analysis_progress = "(" + str(i + 1) + "/" + str(len(dicom_files)) + ")"

        # Loading and preprocessing the data from the DICOM file
        try:
            preprocessed_frames = get_preprocessed_frames(dicom_file_path, orientation=orientation)
        except pydicom.filereader.InvalidDicomError:
            # Printing error message to console and log files
            msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
                        analysis_progress + " " + dicom_file_path,
                        "Invalid DICOM file!"]
            custom_logger.log(msg_list)
            continue
        except Exception as err:
            # Printing error message to console and log files
            msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
                        analysis_progress + " " + dicom_file_path,
                        str(err)]
            custom_logger.log(msg_list)
            continue

        try:
            # Sending preprocessed data to the GPU
            preprocessed_frames = preprocessed_frames.to(device, dtype=torch.float)

            # Predicting RVEF for each cardiac cycle in the given video
            cardiac_cycle_predictions = []
            for preprocessed_frames_from_a_single_cardiac_cycle in preprocessed_frames:
                predicted_rvef_from_one_cardiac_cycle = model(preprocessed_frames_from_a_single_cardiac_cycle)
                cardiac_cycle_predictions.append(predicted_rvef_from_one_cardiac_cycle.item())

            # Calculating the final predicted RVEF value for the video (i.e., the mean of cardiac cycle predictions)
            predicted_rvef = np.mean(cardiac_cycle_predictions)

            # Appending the DICOM file path and the predicted RVEF to the lists
            list_of_dicom_files.append(dicom_file_path)
            list_of_predicted_rvefs.append(predicted_rvef)

            # Printing progress to console and log file
            msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
                        analysis_progress + " " + dicom_file_path,
                        "Predicted RVEF: {:.1f}%".format(predicted_rvef)]
            custom_logger.log(msg_list)

        except Exception:
            # Printing error message to console and log files
            msg_list = [datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
                        analysis_progress + " " + dicom_file_path,
                        "Error in predicting RVEF!"]
            custom_logger.log(msg_list)
            continue

    # Saving prediction results to a CSV file
    os.makedirs(output_folder, exist_ok=True)
    df = pd.DataFrame({"dicom_file_path": list_of_dicom_files, "predicted_rvef": list_of_predicted_rvefs})
    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dicom_folder",
                        help="Path of the folder containing the DICOM files. "
                             "Our model was trained to analyze apical 4-chamber view echocardiographic videos only.",
                        required=True)
    parser.add_argument("-m", "--model_path", help="Path of the trained model file.", required=True)
    parser.add_argument("-o", "--output_folder", help="Path of the output folder.", required=True)
    parser.add_argument("-or", "--orientation", help="Orientation of the left and right ventricles (Stanford or Mayo).",
                        choices=["Mayo", "Stanford"], default="Mayo")
    args = parser.parse_args()

    predict_rvef(args.dicom_folder, args.model_path, args.output_folder, args.orientation)
