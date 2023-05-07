import os
from datetime import datetime
from tkinter import filedialog

import numpy as np
import Sortingfolders as sf
import torch
import torchvision
import torchvision.transforms as det_T
import transforms as T
import utils
from engine import evaluate, train_one_epoch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CONST:
    batch_size = 20
    epochs = 10
    lr=0.005
    momentum=0.9
    weight_decay=0.0005
    step_size=3
    gamma=0.1
    print_freq = 100
    model_file_path = "C:/Users/joeyk/Documents/mycode/solcapacity/SolCapacity/models/" 
    #SolCapacity\models

class Utils:
    @staticmethod
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    @staticmethod
    def img_processing(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shrink = cv2.resize(img_gray, (200, 200))
        box_list = Utils.blockshaped(img_shrink, CONST.img_dim, CONST.img_dim)
        flatten_list = box_list.reshape(box_list.shape[0], -1)
        return flatten_list

    @staticmethod
    def pred_single_img(model_path, image_path):
        my_model = torch.jit.load(model_path)
        my_model.to(CONST.device)  # make sure on device
        my_model.eval()

        img_unprocessed = cv2.imread(image_path)
        my_image = Utils.img_processing(img_unprocessed)

        img_array = []
        img_array.extend(my_image)
        my_pred_label = my_model(torch.FloatTensor(img_array))
        my_pred_label = \
            my_pred_label.view(len(my_pred_label), -1).argmax(1).to(CONST.device).numpy().astype(numpy.int32).tolist()


        return Utils.lb_to_fen(my_pred_label)

    @staticmethod
    def pick_old_model(model_path = None):
        if model_path is None:
            print("pick a model")
            model_path = filedialog.askopenfilename(initialdir="models/")

        #old_model = torch.load(model_path)
        old_model = get_model_instance_segmentation(2) # a torch.nn.Module object
        old_model.load_state_dict(torch.load(model_path))
        #old_model = old_model.to(device)
        print(model_path, "model loaded")
        #evaluate(old_model, )
        return old_model

    @staticmethod
    def save_model(model, filepath=""):
        now = datetime.now()
        filepath = CONST.model_file_path + now.strftime("%S_%M_%H_%d_%m") + "model.pt"
        torch.save(model.state_dict(), filepath)
        print(f'Model saved to {filepath}')
    
    @staticmethod
    def load_model(model, filepath):
        model.load_state_dict(torch.load(filepath))
        print(f'Model loaded from {filepath}')


class SolarDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train):
        self.root = root
        self.transforms = transforms
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "bdappv/google/img"))))
        #self.masks = list(sorted(os.listdir(os.path.join(root, "bdappv/google/mask"))))
        #self.imgs = list(sorted(sf.ign_img))
        self.imgs = list(sorted(sf.ign_matches))
        self.masks = list(sorted(sf.ign_matches))
        
        splits = 10
        self.imgs_splits = np.array_split(self.imgs, splits)
        self.masks_splits = np.array_split(self.masks, splits)

        self.imgs = list(self.imgs_splits[0])
        self.masks = list(self.masks_splits[0]) 
        print('data loaded', len(self.imgs))
    def __getitem__(self, idx):
        # load images and masks
        #img_path = os.path.join(self.root, "bdappv/bdappv/ign/img", self.imgs[idx])
        img_path = self.imgs[idx]
        #mask_path = os.path.join(self.root, "bdappv/bdappv/ign/mask", self.masks[idx])
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            #img = self.transforms(img)
            #target = self.transforms(target)
            #img, target = self.transforms(img, target)
            img = self.transforms(img)
            if self.train:
                T.RandomHorizontalFlip(0.5)(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform():
    transforms = []
    #transforms.append(det_T.ToPILImage())
    #transforms.append(det_T.PILToTensor())
    transforms.append(T.ToTensor())
    transforms.append(det_T.ConvertImageDtype(torch.float))
    #if train:
     #   transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def menu():
    print("1: evaluate old model")
    print("2: new model")
    print("3: load model stats")
    val = int(input("Enter your value: "))
    print("menu choice:" + str(val))
    #if val==1:
        #model = Utils.pick_old_model()
    if val != 1:
        val = 2
        print("menu choice changed to " + str(val))
    return val
        
def main():
    print("code starting")
    #torch.cuda.empty_cache() didn't work
    #print("CACHE CLEARED")
    print("file:" + CONST.model_file_path)
    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print(torch.cuda.get_device_name())
        device = torch.device('cpu')
    print('device:', str(device))
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = SolarDataset('Solar_pictures', get_transform(), train=True)
    dataset_test = SolarDataset('Solar_pictures', get_transform(), train=False)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONST.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    menu_choice = menu()
    if menu_choice == 1:
        #model = get_model_instance_segmentation(num_classes)
        model = Utils.pick_old_model()
        model.to(device)
    else:
        # get the model using our helper function
        model = get_model_instance_segmentation(num_classes)
        #Utils.save_model(model, CONST.model_file_path)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=CONST.lr,
                                    momentum=CONST.momentum, weight_decay=CONST.weight_decay)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=CONST.step_size,
                                                    gamma=CONST.gamma)

    # let's train it for 10 epochs
    num_epochs = CONST.epochs

    for epoch in range(num_epochs):
        #printing every 10 iterations
        if menu_choice != 1:
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=CONST.print_freq)
            # update the learning rate
            lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    if menu_choice !=1 :
        Utils.save_model(model, CONST.model_file_path)

    print("That's it!")

if __name__ == "__main__":
    main()