import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch
import time
import argparse
import torch.optim as optim

from dataload_and_model import *
from glob import glob
from tqdm import tqdm
from torchsampler.imbalanced import ImbalancedDatasetSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from label_smoothing import LabelSmoothingLoss


parser = argparse.ArgumentParser(description='DACON')

parser.add_argument('--mode',                   type=str,   help='training and validation mode',    default='train')

# Dataset
parser.add_argument('--data_path',              type=str,   help='training data path',              default=os.path.join(os.getcwd(), "open"))
parser.add_argument('--input_height',           type=int,   help='input height',                    default=670)
parser.add_argument('--input_width',            type=int,   help='input width',                     default=670)

# Training
parser.add_argument('--num_seed',               type=int,   help='random seed number',              default=51)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                default=16)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=70)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate',           default=1e-3)
parser.add_argument('--weight_decay',           type=float, help='weight decay factor for optimization',                                default=1e-4)

# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',               default='')
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',         default=os.path.join(os.getcwd(), 'model', "local_log"))
parser.add_argument('--log_freq',               type=int,   help='Logging frequency in global steps',                   default=112)

# Multi-gpu training
parser.add_argument('--gpu',            type=int,  help='GPU id to use', default=3)

args = parser.parse_args()


def evaluation(args, model, params, valid_dataset_dict):
    sub_valid_imgs = [valid_dataset_dict["image"][i] for i in valid_dataset_dict["index"]]
    sub_valid_labs = [valid_dataset_dict["label"][i] for i in valid_dataset_dict["index"]]
    sub_valid_class = [valid_dataset_dict["class_name"][i] for i in valid_dataset_dict["index"]]

    valid_dataset = Custom_dataset(args, sub_valid_imgs, sub_valid_labs, classes= sub_valid_class, mode='test', transforms=params["preprocessor"].transform_pred)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    valid_loss = 0
    valid_pred = []
    valid_y = []
    model.eval()
    for batch in (valid_loader):
        x = torch.tensor(batch[0], dtype=torch.float32).cuda(args.gpu)
        y = torch.tensor(batch[1], dtype=torch.long).cuda(args.gpu)
        with torch.no_grad():
            pred = model(x)

        loss = params["criterion"](pred, y)
        
        valid_loss += loss.item()/len(valid_loader)
        valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        valid_y += y.detach().cpu().numpy().tolist()

    valid_f1 = score_function(valid_y, valid_pred)
    return valid_f1, valid_loss


def main():
    fold = 5
    if args.mode == "train":
        command = "mkdir -p " + os.path.join(args.log_directory, f"model_g{args.gpu}", "summaries")
        os.system(command)
        
        main_worker(fold)

    elif args.mode == "test":
        test(fold)


def main_worker(fold):
    #seed 고정
    random.seed(args.num_seed)
    torch.manual_seed(args.num_seed)

    train_image_paths = sorted(glob(os.path.join(args.data_path, "train", "*.png")))
    train_y = pd.read_csv(os.path.join(args.data_path, "train_df.csv"))
    train_labels_str = train_y["label"]
    train_classes = train_y["class"]

    label_unique = sorted(np.unique(train_labels_str))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

    train_labels = [label_unique[k] for k in train_labels_str]

    train_imgs = [img_load(m) for m in tqdm(train_image_paths)]
    
    # Train
    train_dataset = Custom_dataset(args, train_imgs, train_labels, mode='train')

    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state = args.num_seed)

    # logging
    writer = SummaryWriter(os.path.join(args.log_directory, f"model_g{args.gpu}", "summaries"), flush_secs=30)
    preprocessor = train_preprocess(args)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_dataset, train_labels)):
        sub_train_imgs = [train_imgs[i] for i in train_idx]
        sub_train_labs = [train_labels[i] for i in train_idx]
        sub_train_clas = [train_classes[i] for i in train_idx]
        sub_train_dataset = Custom_dataset(args, sub_train_imgs, sub_train_labs, classes= sub_train_clas, mode='train', transforms=preprocessor.transforms)

        train_subsampler = ImbalancedDatasetSampler(sub_train_dataset)
        train_loader = DataLoader(sub_train_dataset, batch_size=args.batch_size, sampler = train_subsampler)

        model = Network().cuda(args.gpu)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,60], gamma=0.5)
        
        # class weight 
        total_lbls = list(set(sub_train_labs))
        lbl_cnt = [sub_train_labs.count(label) for label in total_lbls]
        norm_weights = [1 - (cnt / sum(lbl_cnt)) for cnt in lbl_cnt]
        norm_weights = torch.FloatTensor(norm_weights).cuda(args.gpu)
        criterion = LabelSmoothingLoss(classes=88, smoothing=0.1, weight=norm_weights)

        scaler = torch.cuda.amp.GradScaler() 

        val_loss_plot, val_score_plot = [], []
        global_step = 0
        for epoch in range(args.num_epochs):
            start=time.time()
            train_loss = 0
            train_pred=[]
            train_y=[]
            model.train()
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = torch.tensor(batch[0], dtype=torch.float32).cuda(args.gpu)
                y = torch.tensor(batch[1], dtype=torch.long).cuda(args.gpu)

                with torch.cuda.amp.autocast():
                    pred = model(x)
                loss = criterion(pred, y)
        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                print(f"[fold][step/epoch/total epoch]: [{fold}][{step}/{epoch+1}/{args.num_epochs}], train loss: {loss :.12f}")
                
                train_loss += loss.item()/len(train_loader)
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += y.detach().cpu().numpy().tolist()

                if global_step and global_step % args.log_freq == 0:
                    writer.add_scalar("loss/train_loss", loss, global_step=global_step)

                global_step += 1
            scheduler.step()

            TIME = time.time() - start
            print(f'epoch : {epoch+1}/{args.num_epochs}    time : {TIME:.0f}s/{TIME*(args.num_epochs-epoch-1):.0f}s')

            # Validation
            valid_dataset_dict = {"image": train_imgs, "label": train_labels, "label_str": train_labels_str, "class_name": train_classes, "index": valid_idx}
            params = {"fold": fold, "epoch": epoch, "criterion": criterion, "preprocessor": preprocessor}
            valid_f1, valid_loss = evaluation(args, model, params, valid_dataset_dict)

            writer.add_scalar("loss/valid_loss", valid_loss, global_step=global_step)
            writer.add_scalar("f1/valid_f1_score", valid_f1, global_step=global_step)
            writer.flush()

            TIME = time.time() - start
            print(f'Valid    loss : {valid_loss:.5f}    f1 : {valid_f1:.5f}')
            print(f'time : {TIME:.0f}s/{TIME*(args.num_epochs-epoch-1):.0f}s')
            val_score_plot.append(valid_f1)
            val_loss_plot.append(valid_loss)

            ##save model
            if epoch > 3:
                if np.max(val_score_plot) == val_score_plot[-1]:
                    torch.save(model.state_dict(), os.path.join(args.log_directory, f"model_g{args.gpu}", str(fold)+"_"+str(epoch)+".pt"))
                    torch.save(model.state_dict(), os.path.join(args.log_directory, f"model_g{args.gpu}", str(fold)+".pt"))


def predict(models, loader, transforms):
    f_pred = []
    for batch in (loader):
        x = torch.tensor(batch[0], dtype = torch.float32).cuda(args.gpu)
        
        for i, transformer in enumerate(transforms):
            aug_x = transformer.augment_image(x)
        
            for fold, model in enumerate(models): 
                with torch.no_grad():
                    if fold == 0:
                        pred = model(aug_x)
                    else:
                        pred = pred + model(aug_x)

            if i ==0:
                preds = pred/(len(models))
            else:
                preds = preds + pred/(len(models))

        preds = preds/(60)
        f_pred.extend(preds.argmax(1).detach().cpu().numpy().tolist())
    return f_pred


def test(fold):
    import ttach as tta

    transforms = tta.Compose(
        [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0,90,180]),
        tta.FiveCrops(args.input_height,args.input_width),
        tta.Multiply(factors=[1,1.1])
        ]
    )

    preprocessor = train_preprocess(args)

    train_y = pd.read_csv(os.path.join(args.data_path, "train_df.csv"))
    train_labels = train_y["label"]

    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    
    test_png = sorted(glob(os.path.join(args.data_path, "test", "*.png")))
    test_imgs = [img_load(n) for n in tqdm(test_png)]

    test_dataset = Custom_dataset(args, np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test', transforms=preprocessor.transform_pred)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    
    models = []
    for num_fold in range(fold):
        model = Network().cuda(args.gpu)
        try:
            model.load_state_dict(torch.load(os.path.join(args.log_directory, f"model_g{args.gpu}", str(num_fold)+".pt")))
        except:
            continue
        models.append(model)
    
    f_pred = predict(models, test_loader, transforms)

    label_decoder = {val:key for key, val in label_unique.items()}
    f_result = [label_decoder[result] for result in f_pred]

    submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
    submission["label"] = f_result
    submission.to_csv("tta_g3.csv", index = False)

if __name__ == "__main__":
    main()