import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import normalize_image_data


class TemporalConvNet_02(nn.Module):

    def __init__(self, features: nn.Module, batch_size, DICOM_frame_nbr, input_image_size, dropout_prob=0.1):

        super(TemporalConvNet_02, self).__init__()

        self.frame_nbr = DICOM_frame_nbr
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.features = features

        self.activation = nn.SELU()

        sample_tensor = torch.randn(batch_size, 3, input_image_size, input_image_size)
        output = features(sample_tensor)
        feature_out_channel = output.size()[1]
        feature_out_H = output.size()[2]
        feature_out_W = output.size()[3]

        self.temporal_conv_layer = nn.Conv2d(feature_out_channel * DICOM_frame_nbr, 256, (1, 1))
        self.batch_norm_conv_layer = nn.BatchNorm2d(256)
        self.FC1 = nn.Linear(256 * feature_out_H * feature_out_W, 1024)

        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.batch_norm_FC1 = nn.BatchNorm1d(1024)

        self.FC2 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        x = self.split_dicom_frames(x)

        x = self.activation(self.batch_norm_conv_layer(self.temporal_conv_layer(x)))

        x = x.view(-1, self.num_flat_features(x))

        x = self.activation(self.batch_norm_FC1(self.FC1(x)))

        x = self.dropout(x)

        x = self.FC2(x)

        x = self.dropout(x)

        return x

    def split_dicom_frames(self, input_tensor):

        s = list(input_tensor.shape)
        s[0] = int(s[0] / self.frame_nbr)
        s[1] = int(s[1] * self.frame_nbr)
        reshaped_input_tensor = input_tensor.reshape(s)

        return reshaped_input_tensor

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnsembleHead(nn.Module):
    def __init__(self, num_predictor_models, num_of_classes):
        super(EnsembleHead, self).__init__()

        self.fc1 = nn.Linear(num_predictor_models * num_of_classes, num_predictor_models * num_of_classes)
        self.fc2 = nn.Linear(num_predictor_models * num_of_classes, num_of_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class EnsembleNet(nn.Module):
    def __init__(self, base_models, head_model, base_models_normalization, normalization_values):
        super(EnsembleNet, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.base_models = base_models
        self.head_model = head_model.to(self.device)
        self.base_models_normalization = base_models_normalization

        self.normalization_values = normalization_values 


    def forward(self, x):
        
        base_model_outputs = []

        
        if any(is_norm==True for is_norm in self.base_models_normalization):
            x_noramlized = normalize_image_data(x, self.normalization_values)
            x_noramlized = x_noramlized.to(self.device)

        for idx, model in enumerate(self.base_models):
            
            model = model.to(self.device)

            model.eval()

            if self.base_models_normalization[idx]==True:
                base_model_outputs.append(model(x_noramlized))
            else:
                base_model_outputs.append(model(x))

        #print("base_model_outputs: {}".format(base_model_outputs))

        base_model_outputs = torch.tensor(base_model_outputs)
        base_model_outputs = base_model_outputs.to(device='cuda')

        output = self.head_model(base_model_outputs)       

        return output