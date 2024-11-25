import time
import os
import cv2
from datasets.dataset import CrackData
import argparse
import cfg
from os.path import splitext, join
import logging
import torch.nn as nn
# from model import *
# from model1 import *
# from lmaba import *
# from lmaba1 import *
from new_mamba import *
# torch.cuda.set_device(1)
def createDataList(inputDir, outputFileName='data.lst', supportedExtensions=['.png', '.jpg', '.jpeg']):

    out = open(join(inputDir, outputFileName), "w")
    res = []
    for root, directories, files in os.walk(inputDir):
        for f in files:
            for extension in supportedExtensions:
                fn, ext = splitext(f.lower())

                if extension == ext:
                    out.write('%s %s\n' % (f, f))
                    res.append(f)

    out.close()
    return res

def onescale_test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']

    logging.info('Processing: %s' % test_root)
    test_lst = cfg.config_test[args.dataset]['data_lst']

    imageFileNames = createDataList(test_root, test_lst)

    test_img = CrackData(test_root, test_lst)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    save_dir = args.res_dir
    visual_dir = args.vis_dir
    s1_dir = args.s1_dir
    s2_dir = args.s2_dir
    s3_dir = args.s3_dir
    s4_dir = args.s4_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir)
    
    if not os.path.exists(s1_dir):
        os.mkdir(s1_dir)
    if not os.path.exists(s2_dir):
        os.mkdir(s2_dir)
    if not os.path.exists(s3_dir):
        os.mkdir(s3_dir)
    if not os.path.exists(s4_dir):
        os.mkdir(s4_dir)
    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    # all_t = 0
    # timeRecords = open(join(save_dir, 'timeRecords.txt'), "w")
    # timeRecords.write('# filename time[ms]\n')

    scale = [1]
    for idx, (image, _) in enumerate(testloader):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape

        for k in range(0, len(scale)):
            im_ = image_in.transpose((2, 0, 1))
            tm = time.time()
            results, o1, o2, o3, o4, o5, g1, g2, g3, g4, g5 = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            # print(results.shape)
            # print(visual_image.shape)
            # print(visual_image[-1].shape)
            
            # visual_image = visual_image[-1].cpu().data.numpy()[0, :, :]
            # print(visual_image.shape)
            # o5 = nn.Conv2d(128, 1, kernel_size=1, padding=0)(o5).cuda()
            visual_value = g1.cpu().detach().numpy().transpose((0, 2, 3, 1))
            
            # v1 = v1.cpu().detach().numpy().transpose((0, 2, 3, 1))
            s1 = g2.cpu().detach().numpy().transpose((0, 2, 3, 1))
            s2 = g3.cpu().detach().numpy().transpose((0, 2, 3, 1))
            s3 = g4.cpu().detach().numpy().transpose((0, 2, 3, 1))
            s4 = g5.cpu().detach().numpy().transpose((0, 2, 3, 1))
            result = results[-1].cpu().data.numpy()[0, :, :]
            # result = results[-1].cpu().data.numpy()[0, :, :]
            # result = result[F.sigmoid(results[-1])>0.5]=1
            # result = result.cpu().data.numpy()[0, :, :]
            # print(results.shape)

        # elapsedTime = time.time() - tm
        # timeRecords.write('%s %f\n' % (imageFileNames[idx], elapsedTime * 1000))
        # cv2.imwrite(os.path.join(save_dir, '%s.jpg' % imageFileNames[idx][:-4]), 255 - result * 255)
    # visual_value = visual_image.cpu().detach().numpy().transpose((0, 2, 3, 1))
        for j in range(visual_value.shape[0]):
            # cv2.imwrite('./validation/' + str(j) + '.png', visual_value[j] * 255)
            cv2.imwrite(os.path.join(visual_dir, '%s.png' % imageFileNames[idx][:-4]), visual_value[j]* 255)
        for k in range(s1.shape[0]):
            cv2.imwrite(os.path.join(s1_dir, '%s.png' % imageFileNames[idx][:-4]), s1[k] * 255)
        for m in range(s2.shape[0]):
            cv2.imwrite(os.path.join(s2_dir, '%s.png' % imageFileNames[idx][:-4]), s2[m] * 255)
        for n in range(s3.shape[0]):
            cv2.imwrite(os.path.join(s3_dir, '%s.png' % imageFileNames[idx][:-4]), s3[n]* 255)
        for l in range(s4.shape[0]):
            cv2.imwrite(os.path.join(s4_dir, '%s.png' % imageFileNames[idx][:-4]), s4[l] * 255)
        cv2.imwrite(os.path.join(save_dir, '%s.png' % imageFileNames[idx][:-4]), result* 255 )
        # cv2.imwrite(os.path.join(visual_dir, '%s.png' % imageFileNames[idx][:-4]), visual_image* 255)
        print("Running test [%d/%d]" % (idx + 1, len(testloader)))    ### lk 2022.02.28

    # timeRecords.write('Overall Time use: %f \n' % (time.time() - start_time))    ### lk 2022.02.26
    # timeRecords.close()
    # print(all_t)
    # print('Overall Time use: ', time.time() - start_time)
    print(time.time() - start_time)

def main():
    import time
    print(time.localtime())
    args = parse_args()

    args.cuda = True  ######### lk
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.info('Loading model...')

    # model = CarNet34(
    #     Encoder=Encoder_v0_762,
    #     dp=DownsamplerBlock,
    #     block=BasicBlock_encoder,
    #     channels=[3, 16, 64, 128, 256],
    #     decoder_block=non_bottleneck_1d_2,
    #     num_classes=1
    # )
    # model = DRNet()
    model = UltraLight_VM_UNet(num_classes=1, 
                               input_channels=3, 
                                # c_list=[16,32,64,128,256,256],
                               c_list=[8,16,32,64,128,128],
                               split_att='fc', 
                               bridge=False,)
    logging.info('Loading state...')
    model.load_state_dict(torch.load('%s' % (args.model)))
    logging.info('Start image processing...')

    onescale_test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test model performance')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Crack500_aug', help='The dataset to train')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
                        default='Deepcrack', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='ZDcrack', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Crack360', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='BJN260_aug', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Rain365', help='The dataset to train')
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
    #                     default='Sun520', help='The dataset to train')
    parser.add_argument('-i', '--inputDir', type=str, default=None, help='Input image directory for testing.')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/ABLNet_60000.pth',
                        help='the model to test')
    parser.add_argument('--res-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/onescale_test_15e3',
                        help='the dir to store result')
    parser.add_argument('--vis-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/visual_image',
                        help='the dir to store visual_image')
    parser.add_argument('--s1-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/s1_image',
                        help='the dir to store visual_image')
    parser.add_argument('--s2-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/s2_image',
                        help='the dir to store visual_image')
    parser.add_argument('--s3-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/s3_image',
                        help='the dir to store visual_image')
    parser.add_argument('--s4-dir', type=str,
                        default='Deepcrack_aug_vamba_basline_feedback_visual/s4_image',
                        help='the dir to store visual_image')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.INFO)
main()
