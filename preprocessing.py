import torch
import numpy as np
import cv2
import pydicom
from planar import BoundingBox
from PIL import ImageFile
from skimage import transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize_image_data(tensor_frames, normalization_values):
    """
    Function for normalizing a sequency of images. 

    Parameters
    ----------
    tensor_frames : torch tensor
        images in torch tensor format
    fps : float
        2D echo video frames per second property. If None is given as input, the code
        tries to extract it automatically from the dicom  
    normalization_values : list of floats
        a list of [average, std] describing the tarining data distribution

    Returns
    -------
    normalized_frames : torch tensor
        normalized images in torch tensor format
    """
    
    average = normalization_values[0]
    std = normalization_values[1]

    normalized_frames = []

    for frame in tensor_frames:
        frame = frame.cpu().numpy()
        binary_mask = frame[0]
        frame_copy_one = (frame[1] - float(average)) / float(std)
        frame_copy_two = (frame[2] - float(average)) / float(std)
        merged_frame_data = [binary_mask, frame_copy_one, frame_copy_two]
        normalized_frames.append(merged_frame_data)

    return torch.tensor(np.array(normalized_frames))

def get_preprocessed_frames(path, fps, hr, orientation):
    """
    Preprocessing code for dicoms containing raw 2D echocardiography data as well as other meta data.

    Parameters
    ----------
    path : string
        full path of the dicom file
    fps : float
        2D echo video frames per second property. If None is given as input, the code
        tries to extract it automatically from the dicom  
    hr : float
        heart rate of the patient. If None is given as input, the code
        tries to extract it automatically from the dicom
    orientation : string
        either "Stanford" or "Mayo"    
    
    Returns
    -------
    sampled_frames_from_all_cardiac_cycles_tensor : torch tensor
        sampled images from all cardiac cycles in torch tensor format
    """

    # Set the minimum number of frames required for analysis
    min_number_of_frames = 20

    # Define the range of acceptable HR values
    min_hr = 30
    max_hr = 150

    # Set the number of frames to be sampled from each cardiac cycle
    num_of_frames_to_sample = 20

    # Load data from DICOM file
    dataset = pydicom.dcmread(path, force=True)

    # Extract HR from DICOM tags if not provided by the user
    if hr is None:
        if not hasattr(dataset, 'HeartRate'):
            raise ValueError('HR was not found in DICOM tags!')
        else:
            hr = dataset.HeartRate

    # Check whether HR falls into the predefined range
    if hr < min_hr or hr > max_hr:
        raise ValueError('HR falls outside of the predefined range ({} - {}/min)'.format(min_hr, max_hr))

    # Extract FPS from DICOM tags if not provided by the user
    if fps is None:
        if hasattr(dataset, 'RecommendedDisplayFrameRate'):
            fps = dataset.RecommendedDisplayFrameRate
        elif hasattr(dataset, 'FrameTime'):
            fps = round(1000 / float(dataset.FrameTime))
        else:
            raise ValueError('FPS was not found in DICOM tags!')

    # Extract the number of frames from DICOM tags
    num_of_frames = dataset.NumberOfFrames

    # Check whether the video has enough frames
    if num_of_frames < min_number_of_frames:
        raise ValueError('There are less than {} frames in the video!'.format(min_number_of_frames))

    # Calculate the estimated length of a cardiac cycle
    len_of_cardiac_cycle = (60 / int(hr)) * int(float(fps))

    # Check whether the video contains at least one cardiac cycle
    if num_of_frames < len_of_cardiac_cycle:
        raise ValueError('The video is shorter than one cardiac cycle!')

    # Convert frames to grayscale
    gray_frames = dataset.pixel_array[:, :, :, 0]

    # Flip video if it has Stanford orientation
    if orientation == 'Stanford':
        for i, frame in enumerate(gray_frames):
            gray_frames[i] = cv2.flip(frame, 1)

    # Motion based filtering
    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    cropped_frames = []

    # Count pixel changing intensity and frequency
    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    max_of_changes = np.amax(changes)
    min_of_changes = np.min(changes)

    # Normalize pixel changing values
    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = int(255 * ((changes[r][p] - min_of_changes) / (max_of_changes - min_of_changes)))

    nonzero_values_for_binary_mask = np.nonzero(changes)

    # Create binary mask based on the pixel changing values, using morpohlogy
    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_msk = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask_after_erosion = np.where(erosion_on_binary_msk, binary_mask, 0)

    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T
    binary_mask_coordinates = list(map(tuple, binary_mask_coordinates))
    
    # Crop image based on binary mask
    bbox = BoundingBox(binary_mask_coordinates)
    cropped_mask = binary_mask_after_erosion[int(bbox.min_point.x):int(bbox.max_point.x),
                                             int(bbox.min_point.y):int(bbox.max_point.y)]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    for i in range(len(gray_frames)):
        masked_image = np.where(erosion_on_binary_msk, gray_frames[i], 0)
        cropped_image = masked_image[int(bbox.min_point.x):int(bbox.max_point.x),
                                     int(bbox.min_point.y):int(bbox.max_point.y)]
        cropped_frames.append(cropped_image)

    # Sample frames from each cardiac cycle
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

        # Use indices to select frames
        sampled_frames_from_one_cardiac_cycle = [cropped_frames[i - 1] for i in sampled_indices_from_one_cardiac_cycle]

        # Resize frames and the binary mask
        resized_frames = []
        for frame in sampled_frames_from_one_cardiac_cycle:
            resized_frame = transform.resize(frame, (224, 224))
            resized_frames.append(resized_frame)
        resized_frames = np.asarray(resized_frames)
        resized_binary_mask = transform.resize(cropped_mask, (224, 224))

        # Convert 1-channel frames to 3-channel frames
        frames_3ch = []
        for frame in resized_frames:
            new_frame = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
            new_frame[:, :, 0] = frame
            new_frame[:, :, 1] = frame
            new_frame[:, :, 2] = frame
            frames_3ch.append(new_frame)

        # Convert data to Tensor
        frames_tensor = np.array(frames_3ch)
        frames_tensor = frames_tensor.transpose((0, 3, 1, 2))
        binary_mask_tensor = np.array(resized_binary_mask)
        frames_tensor = torch.from_numpy(frames_tensor)
        binary_mask_tensor = torch.from_numpy(binary_mask_tensor)

        # Expand the Tensor containing the frames
        f, c, h, w = frames_tensor.size()
        new_shape = (f, 3, h, w)

        expanded_frames = frames_tensor.expand(new_shape)
        expanded_frames_clone = expanded_frames.clone()
        expanded_frames_clone[:, 0, :, :] = binary_mask_tensor

        sampled_frames_from_all_cardiac_cycles.append(expanded_frames_clone)

    sampled_frames_from_all_cardiac_cycles_tensor = torch.stack(sampled_frames_from_all_cardiac_cycles)

    return sampled_frames_from_all_cardiac_cycles_tensor
