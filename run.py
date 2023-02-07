import os
import Train_Segmentation
import Evaluate_Segmentation
import json

"""
    run.py will run execute the code to train, validate, and test the breast density segmentation algorithm 
    
    train_valid_dir must be in the following format:
        Parent_folder
            |__ Sample_Folder
                    |__ Sub_Folder
                            |__ Original_image.jpg
                            |__ Original_image_dense_C2.png
                            |__ Original_image_breast.png

    The patients that are split into train, validation, and test will be determined by a json file
    This json file will must have the following format:
        {
            "CV1":{
                    Train: ["id", "id", ...],
                    Validation:["id", "id", ...]
                    Test:["id", "id", ...]}
            "CV2:{...}
        }
        
    The evaluation file give the file path of the images used in a separate evaluation phase
    The format of this file must be as such:
        {
            "Some_Name":{
                    "image": image_path,
                    "GT": ground truth value (char),
                    "id": ID Number (string),
                    "Sub ID": Subdirectory identifier of image (string)
            "Some_Name:{...}
        }
    Using this format allows for the exemption of certain samples if needed. 

    Default values for training and testing of segmentation algorithm
    If you are not using mutltiple GPUs (just 1) reduce image resolution and increase batchsize
    Default testing dimensions are set to maintain as close to original resolution as possible
"""
tr_width = 500
tr_height = 500

test_width = 1000
test_height = 1000

batchSize = 16
num_iters = 1000

if __name__ == '__main__':
    # training and validation directory containing the sample files with images and seg masks
    train_valid_dir = ""

    # change this to the name of the file you want to save the segmentation model to
    os.makedirs(".Models/Testing", exist_ok=True)
    save_path = "./Models/Testing"

    # This json file should contain "Folds" or sets of patients in the order of "Train, Validation, Test"
    # This allows later functions to properly split the patients into their respective set
    with open("./Segmentation_Folds.json", "r") as g:
        json_file = json.load(g)

    # This is the name of the set within the read json file i.e. CV1, CV2, etc.
    sample_list = json_file["CV1"]

    # Start training procedure by calling train from Train_Segmentation passing the necessary information
    Train_Segmentation.train(pretrained=True, num_iters=num_iters, height=tr_height,
                             width=tr_width, batch_size=batchSize, pat_list=sample_list, train_dir=train_valid_dir,
                             valid_dir=train_valid_dir, save_path=save_path)

    # This is the evaluation file, a list of paths to patient images
    with open("./Models/Segmentation_Final.json", "r") as g:
        patient_file = json.load(g)

    # Start evaluation procedure by calling evaluate from Evaluate_Segmentation passing the necessary information
    Evaluate_Segmentation.evaluate(height=test_height,
                                   width=test_width, patient_file=patient_file,
                                   save_path=save_path)
