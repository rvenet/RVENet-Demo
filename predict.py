import os
import matplotlib.pyplot as plt
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import sys
import json
import pandas as pd
from tqdm import tqdm
import argparse

from preprocessing import get_preprocessed_frames

class EchoDataset(Dataset):

    def __init__(self, dicom_dir, dicom_frame_nbr=20):
        self.dicom_dir = dicom_dir
        self.dicoms = os.listdir(dicom_dir)

    def __len__(self):
        return len(self.dicoms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        preprocessed_frames = get_preprocessed_frames(os.path.join(self.dicom_dir, self.dicoms[idx]),
                                                      None, None, "Mayo")

        return self.dicoms[idx], preprocessed_frames


def run_model_on_dataset(dataset_root, model_path, output_folder):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path)
    model.eval()
    model = model.to(device)

    dataset = EchoDataset(dataset_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, generator=torch.Generator())

    dicom_predictions = []
    dicom_ids = []

    for batch in tqdm(dataloader):

        try:
            dicom_id = batch[0][0]
            preprocessed_frames = batch[1][0]

            print("Processing dicom {}".format(dicom_id))

            preprocessed_frames = preprocessed_frames.to(device, dtype=torch.float)

            heart_cycle_predictions = []

            for heart_cycle in preprocessed_frames:
                output = model(heart_cycle)
                heart_cycle_predictions.append(output.item())

            predicted_ef = sum(heart_cycle_predictions) / len(heart_cycle_predictions)

            print("Predicted EF: " + str(predicted_ef))

            dicom_predictions.append(predicted_ef)
            dicom_ids.append(dicom_id)

        except:
            print("Can not process dicom: {}".format(dicom_id))
            continue

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.DataFrame({"patient_hash": dicom_ids, "predicted_rvef": dicom_predictions})
    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", help="Full path of the folder containing echo dicoms.", required=True)
    parser.add_argument("-m", "--model_path", help="Full path of the pretrained model file.", required=True)
    parser.add_argument("-o", "--output_folder", help="Full path of the output folder.", required=True)
    args = parser.parse_args()

    run_model_on_dataset(args.dataset_root, args.model_path, args.output_folder)
