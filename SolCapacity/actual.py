import os

import numpy as np
<<<<<<< HEAD
import torch
import torchvision
import torchvision.transforms as det_T
=======
import Sortingfolders as sf
import torch
import torchvision
import torchvision.transforms as det_T
import transforms as T
import utils
from engine import evaluate, train_one_epoch
>>>>>>> tmp
#import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

<<<<<<< HEAD
import Sortingfolders as sf
import transforms as T
import utils
from engine import evaluate, train_one_epoch

=======
>>>>>>> tmp
#from transforms import transforms as T


class CONST:
    batch_size = 10
    epochs = 5
    lr=0.005
    momentum=0.9
    weight_decay=0.0005
    step_size=3
    gamma=0.1
    print_freq = 10

class Utils:
    @staticmethod
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    @staticmethod
    def fen_from_filename(filename):
        filename_prefix = os.path.splitext(os.path.basename(filename))[0]
        return filename_prefix.split()[0]

    @staticmethod
    def get_all_labels(list_filename):
        labels = []
        for i in range(len(list_filename)):
            labels.append(Utils.fen_from_filename(list_filename[i]))
        return labels

    @staticmethod
    def img_processing(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shrink = cv2.resize(img_gray, (200, 200))
        box_list = Utils.blockshaped(img_shrink, CONST.img_dim, CONST.img_dim)
        flatten_list = box_list.reshape(box_list.shape[0], -1)
        return flatten_list

    @staticmethod
    def fen_to_piece_label(fen):
        label_array = []
        for i in fen:
            if str.isdigit(i):
                label_array += numpy.zeros(int(i), numpy.int16).tolist()
            elif i in CONST.PIECES:
                label_array.append(CONST.PIECES.index(i) + 1)
        return label_array


    @staticmethod
    def playing_black(fen):
        '''
        returns
        b, if playing black
        w, if paying white
        None, if unknown
        '''
        balance = 0  # when balance < 0 assume from black perspective
        n = 1
        total = 0

        fen_array = fen.split("-")
        for row in range(0,8):
            if row > 3:
                n = -1
            for i in fen_array[row]:
                if i.islower():
                    total += 1
                    balance += n
                elif i.isupper():
                    balance -= n
                    total += 1
        if total > 10:
            if balance < -2:
                return "b"
            elif balance > 2:
                return "w"
        return None

    @staticmethod
    def lb_to_fen(label):
        s = ''
        count = 0
        for i in range(len(label)):
            if i % 8 == 0:
                if count != 0:
                    s = s + str(count)
                    count = 0
                s = s + '-'
            if label[i] == 0:
                count = count + 1
            else:
                if count != 0:
                    s = s + str(count)
                    count = 0

                if 0 < label[i] <= CONST.PIECES_LEN:
                    s = s + CONST.PIECES[label[i] - 1]
                else:
                    print('Invalid Error#######################################')
        if count != 0:
            s = s + str(count)
        return s[1:]

    @staticmethod
    def lbs_to_fen(label):
        fen = []
        for grp_no in range(0, int(len(label) / 64)):
            start_index = grp_no * 64
            fen.append(Utils.lb_to_fen(label[start_index: start_index + 64]))
        return fen

    @staticmethod
    def decoded_board_with_label(board):
        x = []
        y = []
        x.extend(Utils.img_processing(cv2.imread(board)))
        y.extend(Utils.fen_to_piece_label(Utils.fen_from_filename(board)))
        return x, y

    @staticmethod
    def get_batch(dataset, b_size, start_index):
        x = []
        y = []
        for j in range(0, b_size):
            if start_index + j == len(dataset):
                break
            temp_x, temp_y = Utils.decoded_board_with_label(dataset[start_index + j])
            x.extend(temp_x)
            y.extend(temp_y)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y

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

        old_model = torch.jit.load(model_path)
        print(model_path, "model loaded")
        old_model.eval()
        return old_model


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
        
<<<<<<< HEAD
=======
        splits = 10
        self.imgs_splits = np.array_split(self.imgs, splits)
        self.masks_splits = np.array_split(self.masks, splits)

        self.imgs = list(self.imgs_splits[0])
        self.masks = list(self.masks_splits[0]) 
        print('data loaded', len(self.imgs))
        
>>>>>>> tmp
    def __getitem__(self, idx):
        # load images and masks
        # error is here with idx
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

def main():
<<<<<<< HEAD
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
=======
    print("code starting")
    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device:', str(device))
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
>>>>>>> tmp

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

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

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
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()