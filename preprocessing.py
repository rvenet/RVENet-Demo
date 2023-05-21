import torch
import numpy as np
import cv2
import pydicom
#from planar import BoundingBox
from PIL import ImageFile
from skimage import transform

from boundingboxes import BoundingBox

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_frames(tensor_frames, normalization_values):
    """
    Normalizes a sequence of frames.

    Parameters
    ----------
    tensor_frames : torch.Tensor
        Original frames
    normalization_values : list
        A list of [mean, std] describing the distribution of the data used for training

    Returns
    -------
    normalized_frames : torch.Tensor
        Normalized frames
    """
    
    mean = normalization_values[0]
    std = normalization_values[1]

    normalized_frames = []
    for frame in tensor_frames:
        frame = frame.cpu().numpy()
        binary_mask = frame[0]
        frame_copy_one = (frame[1] - float(mean)) / float(std)
        frame_copy_two = (frame[2] - float(mean)) / float(std)
        merged_frame_data = [binary_mask, frame_copy_one, frame_copy_two]
        normalized_frames.append(merged_frame_data)

    return torch.tensor(np.array(normalized_frames))


def get_preprocessed_frames(dicom_file_path, fps=None, hr=None, orientation="Mayo"):
    """
    Reads and preprocesses data from the DICOM file.

    Parameters
    ----------
    dicom_file_path : str
        Path of the dicom file
    fps : float
        Frame rate of the echocardiographic video. If None is given as input, the code
        tries to extract it from the DICOM tags.
    hr : float
        Heart rate of the patient. If None is given as input, the code
        tries to extract it from the DICOM tags.
    orientation : str
        Orientation of the left and right ventricles ("Stanford" or "Mayo").
        Mayo – the right ventricle on the right and the left ventricle on the left side.
        Stanford – the left ventricle on the right and the right ventricle on the left side.
    
    Returns
    -------
    sampled_frames_from_all_cardiac_cycles_tensor : torch.Tensor
        Sampled frames from all cardiac cycles
    """

    # Defining the minimum number of frames required for analysis
    min_number_of_frames = 20

    # Defining the range of acceptable heart rate values
    min_hr = 30
    max_hr = 150

    # Setting the number of frames to be sampled from each cardiac cycle
    num_of_frames_to_sample = 20

    # Loading data from DICOM file
    dicom_dataset = pydicom.dcmread(dicom_file_path, force=False)

    # Ensuring that the DICOM file is a video (i.e., it has >1 frames)
    if hasattr(dicom_dataset, "NumberOfFrames"):
        if dicom_dataset.NumberOfFrames < 2:
            raise ValueError("DICOM file has <2 frames!")
    else:
        raise AttributeError("No NumberOfFrames DICOM tag!")

    # Ensuring that the DICOM file does not have color Doppler
    if hasattr(dicom_dataset, "UltrasoundColorDataPresent"):
        if dicom_dataset.UltrasoundColorDataPresent:
            raise ValueError("DICOM file with color Doppler!")

    # Ensuring that the DICOM file contains only one ultrasound region
    if hasattr(dicom_dataset, "SequenceOfUltrasoundRegions"):
        if len(dicom_dataset.SequenceOfUltrasoundRegions) > 1:
            raise ValueError("DICOM file contains more than 1 US regions!")

    # Extracting heart rate from DICOM tags if not provided by the user
    if hr is None:
        if not hasattr(dicom_dataset, "HeartRate"):
            raise ValueError("Heart rate was not found in DICOM tags!")
        else:
            hr = dicom_dataset.HeartRate

    # Checking whether heart rate falls into the predefined range
    if hr < min_hr or hr > max_hr:
        raise ValueError("Heart rate falls outside of the predefined range ({} - {}/min)".format(min_hr, max_hr))

    # Extracting frame rate from DICOM tags if not provided by the user
    if fps is None:
        if hasattr(dicom_dataset, "RecommendedDisplayFrameRate"):
            fps = dicom_dataset.RecommendedDisplayFrameRate
        elif hasattr(dicom_dataset, "FrameTime"):
            fps = round(1000 / float(dicom_dataset.FrameTime))
        else:
            raise ValueError("Frame rate was not found in DICOM tags!")

    # Extracting the number of frames from DICOM tags
    num_of_frames = dicom_dataset.NumberOfFrames

    # Checking whether the video has enough frames
    if num_of_frames < min_number_of_frames:
        raise ValueError("There are less than {} frames in the video!".format(min_number_of_frames))

    # Calculating the estimated length of a cardiac cycle
    len_of_cardiac_cycle = (60 / int(hr)) * int(float(fps))

    # Checking whether the video contains at least one cardiac cycle
    if num_of_frames < len_of_cardiac_cycle:
        raise ValueError("The video is shorter than one cardiac cycle!")

    # Converting frames to grayscale
    gray_frames = dicom_dataset.pixel_array[:, :, :, 0]

    # Flipping video if it has Stanford orientation
    if orientation == "Stanford":
        for i, frame in enumerate(gray_frames):
            gray_frames[i] = cv2.flip(frame, 1)

    # Performing motion-based filtering
    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    cropped_frames = []

    # Computing the extent and frequency of changes in pixel intensity values
    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    max_of_changes = np.amax(changes)
    min_of_changes = np.amin(changes)

    # Normalizing pixel changing values
    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = int(255 * ((changes[r][p] - min_of_changes) / (max_of_changes - min_of_changes)))

    nonzero_values_for_binary_mask = np.nonzero(changes)

    # Generating a binary mask based on changes of pixel intensities
    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_msk = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask_after_erosion = np.where(erosion_on_binary_msk, binary_mask, 0)
    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T
    
    # Cropping the binary mask and the frames
    cropped_mask = binary_mask_after_erosion[np.min(binary_mask_coordinates[:,0]):np.max(binary_mask_coordinates[:,0]),
                                             np.min(binary_mask_coordinates[:,1]):np.max(binary_mask_coordinates[:,1])]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    for i in range(len(gray_frames)):
        masked_image = np.where(erosion_on_binary_msk, gray_frames[i], 0)
        cropped_image = masked_image[np.min(binary_mask_coordinates[:,0]):np.max(binary_mask_coordinates[:,0]),
                                     np.min(binary_mask_coordinates[:,1]):np.max(binary_mask_coordinates[:,1])]
        cropped_frames.append(cropped_image)

    # Sampling frames from each cardiac cycle
    sampled_indices_from_all_cardiac_cycles = []
    largest_index = 1
    while True:
        sampled_indices_from_one_cardiac_cycle = \
            list(np.linspace(largest_index, largest_index + len_of_cardiac_cycle, num_of_frames_to_sample))
        if int(sampled_indices_from_one_cardiac_cycle[-1]) <= num_of_frames:
            sampled_indices_from_all_cardiac_cycles.append([int(x) for x in sampled_indices_from_one_cardiac_cycle])
            largest_index = sampled_indices_from_one_cardiac_cycle[-1]
        else:
            break

    sampled_frames_from_all_cardiac_cycles = []
    for sampled_indices_from_one_cardiac_cycle in sampled_indices_from_all_cardiac_cycles:

        # Using indices to select frames
        sampled_frames_from_one_cardiac_cycle = \
            [cropped_frames[i - 1] for i in sampled_indices_from_one_cardiac_cycle]

        # Resizing the frames and the binary mask
        resized_frames = []
        for frame in sampled_frames_from_one_cardiac_cycle:
            resized_frame = transform.resize(frame, (224, 224))
            resized_frames.append(resized_frame)
        resized_frames = np.asarray(resized_frames)
        resized_binary_mask = transform.resize(cropped_mask, (224, 224))

        # Converting 1-channel frames to 3-channel frames
        frames_3ch = []
        for frame in resized_frames:
            new_frame = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
            new_frame[:, :, 0] = frame
            new_frame[:, :, 1] = frame
            new_frame[:, :, 2] = frame
            frames_3ch.append(new_frame)

        # Converting data to torch Tensor
        frames_tensor = np.array(frames_3ch)
        frames_tensor = frames_tensor.transpose((0, 3, 1, 2))
        binary_mask_tensor = np.array(resized_binary_mask)
        frames_tensor = torch.from_numpy(frames_tensor)
        binary_mask_tensor = torch.from_numpy(binary_mask_tensor)

        # Expanding the Tensor containing the frames
        f, c, h, w = frames_tensor.size()
        new_shape = (f, 3, h, w)

        expanded_frames = frames_tensor.expand(new_shape)
        expanded_frames_clone = expanded_frames.clone()
        expanded_frames_clone[:, 0, :, :] = binary_mask_tensor

        sampled_frames_from_all_cardiac_cycles.append(expanded_frames_clone)

    sampled_frames_from_all_cardiac_cycles_tensor = torch.stack(sampled_frames_from_all_cardiac_cycles)

    return sampled_frames_from_all_cardiac_cycles_tensor
