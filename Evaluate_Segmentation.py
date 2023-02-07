import csv
import progressbar
import torch
import json
import torchvision.transforms as tf
from SegAlgo import Model_Definition
import cv2
from Transforms_List import Seg_Transforms
import numpy as np

def evaluate(height, width, patient_file, save_path):
    """
        Evaluate segmentation algorithm

        This function takes several required arguments:
            height: Height of training images (current default is 2500)
            width: Width of training images (current default is 2000)
            batch_size: batchsize to use when training, (height and width of 1300 equated to a batchsize of 2 per GPU)
            patient_file: file of patient image paths to use with the segmentation algorithm
            save_path: where would you like to save your results

    """

    sets = patient_file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Must set pretrained to True to use trained models
    modelDef = Model_Definition(pretrained=True)
    model = modelDef.segmentationModel()
    model.load_state_dict(torch.load(save_path+'/best_model.pth'))
    model.to(device)
    model.eval()
    for_csv = []
    image_transformer = Seg_Transforms(height, width).transformImgOnly()
    for set in sets:
        list = sets[set]
        pb = progressbar.ProgressBar(max_value=len(list))
        total_class1dice = 0
        total_class2dice = 0
        number = 0
        count = 0
        print("Running for " + set + "...")
        for entry in list:
            id = entry["id"]
            gt = entry["GT"]
            img = cv2.imread(entry['image'])
            height_orgin, widh_orgin, d = img.shape
            img = image_transformer(img)
            img = torch.autograd.Variable(img, requires_grad=False).to(device).unsqueeze(0)
            with torch.no_grad():
                Prediction = model(img)['out']  # Run net
            Prediction = tf.Resize((height_orgin, widh_orgin))(Prediction[0])  # Resize to origninal size
            seg = torch.argmax(Prediction, 0).cpu().detach().numpy()  # Get  prediction classes

            mask = np.zeros(seg.shape[0:2], np.float32)
            mask[seg == 1] = 255
            metric = np.sum(seg == 1)

            maskcopy = np.uint8(mask)

            ret, thresh = cv2.threshold(maskcopy, 0, 255, cv2.THRESH_BINARY)
            connectivity = 4
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
            stats = output[2]
            max = 0
            for stat in stats:
                if stat[0] != 0 and stat[1] != 0:
                    if stat[4] > max:
                        max = stat[4]
            temp_dict = {"ID": id,
                         "GT": gt,
                         "Metric": metric}
            for_csv.append(temp_dict)
            pb.update(count)
            count += 1
    keys = for_csv[0].keys()

    with open(save_path+"Segmentation_Results_Final.csv", "w", newline="") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(for_csv)