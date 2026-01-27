import torch
from utils.metrics import test_classification_net_focal, test_classification_net_adafocal,expected_calibration_error,maximum_calibration_error
from utils.metrics import ECE_error_mukhoti as ECE_error
from utils.metrics import adaECE_error_mukhoti 
from torch.nn import functional as F

# Most of the starter code for training, evaluation and calculating calibration related metrics is borrowed from https://github.com/torrvision/focal_calibration.
def evaluate_dataset(model, dataloader, device, num_bins,config):
    image_text_confusion_matrix, text_image_confusion_matrix, image_text_accuracy, text_image_accuracy, labels_list, image_text_predictions_list, text_image_predictions_list, image_text_confidence_vals_list, text_image_confidence_vals_list = test_classification_net_focal(model, dataloader, device, config)

    image_text_ece, image_text_bin_dict,image_text_meancalibration_gap = ECE_error(image_text_confidence_vals_list, image_text_predictions_list, labels_list, num_bins=num_bins)
    text_image_ece, text_image_bin_dict,text_image_meancalibration_gap = ECE_error(text_image_confidence_vals_list, text_image_predictions_list, labels_list, num_bins=num_bins)
    
    return image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap

def evaluate_dataset_ECE_error(logits_per_image, logits_per_text, image_text_labels, text_image_labels, num_bins):
    log_image_text_softmax = F.log_softmax(logits_per_image,dim=1)
    log_text_image_softmax = F.log_softmax(logits_per_text,dim=1)
    image_text_confidence_vals_list, image_text_predictions_list = torch.max(log_image_text_softmax, dim=1).cpu().numpy().tolist()
    text_image_confidence_vals_list, text_image_predictions_list = torch.max(log_text_image_softmax, dim=1).cpu().numpy().tolist()
    image_text_ece, image_text_bin_dict,image_text_meancalibration_gap = ECE_error(image_text_confidence_vals_list, image_text_predictions_list, image_text_labels, num_bins=num_bins)
    text_image_ece, text_image_bin_dict,text_image_meancalibration_gap = ECE_error(text_image_confidence_vals_list, text_image_predictions_list, text_image_labels, num_bins=num_bins)
    
    return image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap

