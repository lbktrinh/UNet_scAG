import os
from dataset import *  # vipcup,brain,cvc
import torch
import numpy as np
from loss import *
from unet_models import *
# from torchsummary import summary
import csv
import utils


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_epochs = 300
    batch_size = 8
    learning_rate = 1e-4

    # create checkpoint dir
    dir_checkpoint = 'C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/checkpoint/checkpoint_cvc/fold_4_attn_unet_v1_SoftDiceLoss_v1_adam4_AttU_Net_v41_b8_t_33/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    # info dir
    dir_info = 'C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/infor/infor_cvc'

    # save csv
    f = open(os.path.join(dir_info, 'fold_4_attn_unet_v1_SoftDiceLoss_v1_adam4_AttU_Net_v41_b8_t_33.csv'), 'w', newline='')

    # cvc dataset
    train_image = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_4/train/images"
    train_mask = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_4/train/masks"

    val_image = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_4/val/images"
    val_mask = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_4/val/masks"

    # Dataset begin
    SEM_train = SEMDataTrain(train_image, train_mask)
    SEM_val = SEMDataVal(val_image, val_mask)

    # Dataloader
    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    SEM_val_load = torch.utils.data.DataLoader(dataset=SEM_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = U_Net(img_ch=3, output_ch=1).to(device)
    # model = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)

    # summary(model, input_size=(3, 256, 192))
    # init_weights(model,  init_type='kaiming_uniform_', gain=1.0)
    # num parameters of model
    param_network(model)

    # Loss function
    criterion = SoftDiceLoss()

    metric_dice = DiceAccuracy()
    metric_iou = IouAccuracy()

    metric_precision = Precision()
    metric_recall = Recall()

    # Optimizerd
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 1e-6, nesterov=True )

    # Train
    print("Initializing Training!")

    total_step_train = len(SEM_train_load)
    total_step_val = len(SEM_val_load)
    print(total_step_train)
    print(total_step_val)

    best_val_acc = 0.0

    for epoch in range(num_epochs):

        train_loss, train_acc_dice, train_acc_iou = [], [], []
        for i, (images, masks) in enumerate(SEM_train_load):
            model.train()

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss
            train_loss.append(loss.item())

            # Calculate metric
            train_acc_dice.append(metric_dice(outputs, masks).item())
            train_acc_iou.append(metric_iou(outputs, masks).item())

            # if (i+1) % 100 == 0:
            #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f} '.format(epoch+1, num_epochs, i+1, total_step_train, np.mean(train_loss), np.mean(train_acc_iou)))

        # Validation
        model.eval()
        val_loss, val_acc_dice, val_acc_iou = [], [], []
        val_precision, val_recall = [], []

        with torch.no_grad():
            for i, (images, masks, image_name) in enumerate(SEM_val_load):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = model(images)

                # Calculate loss
                val_loss.append(criterion(outputs, masks).item())

                # Calculate metric
                val_acc_dice.append(metric_dice(outputs, masks).item())
                val_acc_iou.append(metric_iou(outputs, masks).item())

                val_precision.append(metric_precision(outputs, masks).item())
                val_recall.append(metric_recall(outputs, masks).item())

        train_loss, train_acc_dice, train_acc_iou = np.mean(train_loss), np.mean(train_acc_dice), np.mean(train_acc_iou)
        val_loss, val_acc_dice, val_acc_iou = np.mean(val_loss), np.mean(val_acc_dice), np.mean(val_acc_iou)
        val_precision, val_recall = np.mean(val_precision), np.mean(val_recall)

        print('Epoch', str(epoch + 1), 'train_loss:', '{:04f}'.format(train_loss), 'val_loss:', '{:04f}'.format(val_loss))
        print('train_acc_dice :', '{:04f}'.format(train_acc_dice), 'train_acc_iou :', '{:04f}'.format(train_acc_iou),
                'val_acc_dice :', '{:04f}'.format(val_acc_dice), 'val_acc_iou :', '{:04f}'.format(val_acc_iou))
        print('val_precision:', '{:04f}'.format(val_precision), 'val_recall:', '{:04f}'.format(val_recall))

        is_best = val_acc_dice >= best_val_acc

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=dir_checkpoint,
                               name='last.pth.tar')

        if is_best:
            best_val_acc = val_acc_dice

        # write csv
        writer = csv.writer(f)
        writer.writerow([epoch + 1, '{:04f}'.format(train_loss), '{:04f}'.format(train_acc_dice), '{:04f}'.format(train_acc_iou),
                                  '{:04f}'.format(val_loss), '{:04f}'.format(val_acc_dice), '{:04f}'.format(val_acc_iou),
                                  '{:04f}'.format(val_precision), '{:04f}'.format(val_recall)])

        # torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))
    f.close()


if __name__ == '__main__':
    main()
