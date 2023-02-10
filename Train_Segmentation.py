import torch
from torch.utils.tensorboard import SummaryWriter
from SegAlgo import Model_Definition
from Batch_Loader import Loader

def train(pretrained, num_iters, height, width, batch_size, pat_list, train_dir, valid_dir, save_path):
    """
        Train segmentation algorithm

        This function takes several required arguments:
            pretrained: True if you want to load in publicly available pretrained DeepLabV3 instance
            num_iters: How many iterations would you like to train the segmentation algorithm
            height: Height of training image
            width: Width of training images
            batch_size: batchsize to use when training
            pat_list: List of samples in the order of {"Train":[], "Validation":[], "Test":[]}
            train_dir: path to sample files train
            valid_dir: path to sample files validation
            save_path: where would you like to save your model
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelDef = Model_Definition(pretrained = pretrained)
    model = modelDef.segmentationModel()
    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_counter = 0
    best_iter = 0
    best_validation_loss = 1000
    val_loss_value = 1000
    writer = SummaryWriter()
    for iteration in range(num_iters):
        print("-" * 25)
        print(" ")
        images, ann = Loader(height, width, batch_size, pat_list, train_dir, valid_dir).Train_loader()
        images, ann = images.to(device), ann.to(device)
        torch.autograd.Variable(images, requires_grad=False)
        torch.autograd.Variable(ann, requires_grad=False)
        output = model(images)['out']
        model.zero_grad()
        loss = loss_fn(output, ann.long())
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train Loss ", loss.item(), iteration)
        for i in range(1):
            model.eval()
            val_images, val_ann = Loader(height, width, batch_size, pat_list, train_dir, valid_dir).Validation_loader()
            val_image, val_anns = val_images.to(device), val_ann.to(device)
            with torch.no_grad():
                val_output = model(val_image)['out']
                val_loss = loss_fn(val_output, val_anns.long())
                val_loss_value = val_loss.item()
            writer.add_scalar("Validation Loss ", val_loss.item(), iteration)
        if val_loss_value < best_validation_loss:
            loss_counter = 0
            best_validation_loss = val_loss_value
            best_iter = iteration
            torch.save(model.state_dict(), save_path+'/best_model.pth')
            print("MODEL SAVED")
        else:
            loss_counter += 1
        print("Current Iteration: " + str(iteration) + "/" + str(num_iters))
        print("Train Loss: " + str(loss.data.cpu().numpy()) + " Validation Loss: " + str(val_loss_value))
        print("Best Validation Loss: " + str(best_validation_loss) + " @ " + str(best_iter))
        print(" ")
    writer.close()
