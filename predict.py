import os
from dataset import *
import torch
import numpy as np
from loss import *
from unet_models import *
# from torchsummary import summary
import utils


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    batch_size = 8
    learning_rate = 1e-4

    dir_checkpoint = 'C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/checkpoint/checkpoint_cvc/fold_2_attn_unet_v1_SoftDiceLoss_v1_adam4_NestedUNet_v1_b8_t/'

    dir_result = 'C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/result/result_cvc/fold_2_attn_unet_v1_SoftDiceLoss_v1_adam4_NestedUNet_v1_b8_t/'
    if not os.path.exists(dir_result):
        os.mkdir(dir_result)

    # cvc
    val_image = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_2/val/images"
    val_mask = "C:/Users/trinhle/Desktop/Model_VIPCUP_2018_py_pytorch_3/data_5fold/data_cvc/folder_5fold/fold_2/val/masks"

    # Dataset begin
    SEM_val = SEMDataVal_3(val_image, val_mask)

    # Dataloader
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
    # total_step_train = len(SEM_train_load)
    total_step_val = len(SEM_val_load)
    print(total_step_val)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(dir_checkpoint, 'best.pth.tar'), model, optimizer)

    model.eval()
    val_loss, val_acc_dice, val_acc_iou = [], [], []
    val_precision, val_recall = [], []

    with torch.no_grad():
        for i, (images, masks, image_name) in enumerate(SEM_val_load):

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            outputs1 = torch.sigmoid(outputs)

            output_batch1 = outputs1.data.cpu().numpy()

            for i in range(batch_size):
                cv2.imwrite(os.path.join(dir_result, image_name[i][:-4] + '.jpg'), np.squeeze(output_batch1[i], axis=0) * 255)

            # Calculate loss
            val_loss.append(criterion(outputs, masks).item())

            # Calculate metric
            val_acc_dice.append(metric_dice(outputs, masks).item())
            val_acc_iou.append(metric_iou(outputs, masks).item())

            val_precision.append(metric_precision(outputs, masks).item())
            val_recall.append(metric_recall(outputs, masks).item())

    val_loss, val_acc_dice, val_acc_iou = np.mean(val_loss), np.mean(val_acc_dice), np.mean(val_acc_iou)
    val_precision, val_recall = np.mean(val_precision), np.mean(val_recall)

    print('val_loss:', '{:04f}'.format(val_loss))
    print('val_acc_dice :', '{:04f}'.format(val_acc_dice), 'val_acc_iou :', '{:04f}'.format(val_acc_iou))
    print('val_precision:', '{:04f}'.format(val_precision), 'val_recall:', '{:04f}'.format(val_recall))


if __name__ == '__main__':
    main()
