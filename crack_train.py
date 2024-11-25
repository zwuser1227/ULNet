import argparse, time, os, cfg, log
import numpy as np
from torch.autograd import Variable
import cv2
from datasets.dataset import CrackData
import torch.nn as nn
# from model import *
# from model1 import *
# from lmaba import *
from new_mamba import *
# from lmaba1 import *
torch.cuda.set_device(0)
#########################################################################################
def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))

#########################################################################################
def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):  # cuda=False  lk
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print('the value of n is %d'%n)  ### n=1
    weights = np.zeros((n, c, h, w))

    for i in range(n):      # xrange, lk
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid

    weights = torch.Tensor(weights)

    if cuda:
        weights = weights.cuda()
    # inputs = F.sigmoid(inputs)
    # loss = nn.BCELoss(weights, size_average=False)(inputs, targets)

    loss = nn.BCEWithLogitsLoss(weights, size_average=False)(inputs, targets)

    return loss

def jacc_coef(prediction, label):

    smooth = 1.
    label_f = torch.flatten(label)
    prediction_f = torch.flatten(prediction)

    intersection = torch.sum(prediction_f * label_f)

    jacc = (2. * intersection + smooth) / (torch.sum(prediction_f) + torch.sum(label_f) - intersection + smooth)

    return jacc

def wcet_jacc(inputs, targets, cuda=True,
              a=1, b=0, loss_type='xie', beta=1, gamma=1):

    n, c, h, w = inputs.size()

    t = np.zeros((1, c, h, w))
    for i in range(n):
        t += targets[i, :, :, :].cpu().data.numpy()

    pos = (t == 1).sum()
    neg = (t == 0).sum()

    alpha = (neg + 1) / (pos + 1)
    # alpha = neg / pos
    #################  alpha 是一个 batch 中负样本和正样本的比例.   lk #################
    if loss_type =='ce':
        pos_weight = 1
    if loss_type == 'xie':
        pos_weight = alpha

    pos_weight = torch.Tensor((torch.ones(1) * pos_weight))         ##### lk 01.07

    if cuda:
        pos_weight = pos_weight.cuda()
        pos_weight = pos_weight.detach()

    # loss = a * nn.BCEWithLogitsLoss(weight=None, size_average=False,
    #                                 pos_weight=pos_weight)(inputs, targets) \
    #        - b * jacc_coef(F.sigmoid(inputs), targets)

    loss = a * nn.BCEWithLogitsLoss(weight=None, reduction='mean',
                                    pos_weight=pos_weight)(inputs, targets) \
           - b * jacc_coef(F.sigmoid(inputs), targets)

    return loss

def bceLoss(pred, target):
    # self.bceloss = nn.BCELoss()
    size = pred.size(0)
    pred_ = pred.view(size, -1)
    target_ = target.view(size, -1)

    return nn.BCELoss()(pred_, target_)


def diceLoss(pred, target):
    smooth = 1
    size = pred.size(0)

    pred_ = pred.view(size, -1)
    target_ = target.view(size, -1)
    intersection = pred_ * target_
    dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
    dice_loss = 1 - dice_score.sum()/size

    return dice_loss

def bceDiceLoss(pred, target):
    wb=0.2
    wd=0.8
    # bceloss = bceLoss(pred, target)
    # diceloss = diceLoss(pred, target)

    loss0 = wb*diceLoss(pred[0], target) + wd*bceLoss(pred[0], target)
    # loss1 = wd * diceLoss(pred[1], target) + wb * bceLoss(pred[1], target)
    # loss2 = wd * diceLoss(pred[2], target) + wb * bceLoss(pred[2], target)
    # loss3 = wd * diceLoss(pred[3], target) + wb * bceLoss(pred[3], target)  

    # loss = 0.4*loss0 + 0.3*loss1 + 0.3*loss2 
    # loss = loss0
    return loss0

def Cross_Entropy(pred, labels):

        pred_flat = pred[0].view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        # total_loss = cross_entropy_per_image(pred, labels)
        # total_loss = dice_loss_per_image(pred, labels)
        total_loss = cross_entropy_per_image(pred[0], labels)
                    # 0.5*cross_entropy_per_image(pred[1], labels)  
                    #  0.00 * 0.1 * dice_loss_per_image(pred[0], labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss
def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss
def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)
def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_orignal(_logit, _label)
    return total_loss / len(logits)
def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.5].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                            (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy
def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy
def get_weight(src, mask, threshold, weight):
    count_pos = src[mask >= threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg
#########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Train CarNet for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='Deepcrack', help='The dataset to train')
    
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='edmcrack', help='The dataset to train')
    
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='BJN260_aug', help='The dataset to train')
    
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='BRSfusion_aug', help='The dataset to train')

    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='Rain365_aug', help='The dataset to train')
    
    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='Crack500_aug', help='The dataset to train')

    # parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
    #     default='Sun520_aug', help='The dataset to train')

    parser.add_argument('--param-dir', type=str,
        default='Deepcrack_aug_vamba_basline_feedback_visual',
        help='the directory to store the params')
# 
    # parser.add_argument('--param-dir', type=str,
        # default='edmcrack_aug_vamba_basline_feedback',
        # help='the directory to store the params')
    
    # parser.add_argument('--param-dir', type=str,
    #     default='crack500_aug_ABLNet',
    #     help='the directory to store the params')

    # parser.add_argument('--param-dir', type=str,
    #     default='ZDcrack_aug_ABLNet',
    #     help='the directory to store the params')

    # parser.add_argument('--param-dir', type=str,
    #     default='Sun520_aug_ABLNet_basline',
    #     help='the directory to store the params')

    # parser.add_argument('--param-dir', type=str,
    #     default='BJN260_aug_vamba_basline_feedback',
        # help='the directory to store the params')

    # parser.add_argument('--param-dir', type=str,
    #     default='Rain365_aug_ABLNet_basline_brtiness',
    #     help='the directory to store the params')
    # parser.add_argument('--param-dir', type=str,
    #     default='BSR_aug_ABLNet_attention',
    #     help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=3e-4,
        help='the base learning rate of model')
    # parser.add_argument('-m', '--momentum', type=float, default=0.9,
    #     help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='1',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0,
        help='the weight_decay of net, default is 0.0002')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is vgg16.pth')
    parser.add_argument('--max-iter', type=int, default=1200*50,
        help='max iters to train network, '
            'default is 1000*15 for Sun520, 500*30 for BJN260, 750*20 for Rain365, 650*30 for Crack360')
    parser.add_argument('--iter-size', type=int, default=1,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1200*5,
        help='how many iters to store the params, '
             'default is 1000*5 for Sun520, 500*5 for BJN260, 750*5 for Rain365, 650*5 for Crack360')
    parser.add_argument('--step-size', type=int, default=1200*40,
        help='the number of iters to decrease the learning rate, '
             'default is 1000*10 for Sun520, 500*25 for BJN260, 750*15 for Rain365, 650*30 for Crack360')
    parser.add_argument('--display', type=int, default=100,
        help='how many iters display one time, default is 20; 1000')
    parser.add_argument('-b', '--balance', type=float, default=1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('--batch-size', type=int, default=2,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop, but crop 512 for Crack 500-356,deepcrack-380,ZDcrack-250')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')

    parser.add_argument('--a', type=float, default=1,
        help='the coefficient of wce, default 1')
    parser.add_argument('--b', type=float, default=0,
        help='the coefficient of jaccard, default 0')
    parser.add_argument('--type', type=str, default='ce',
        help='the type of loss function, default xie')
    parser.add_argument('--beta', type=float, default=1,
        help='Fine-tune the proportion of positive and negative samples, default 1')
    parser.add_argument('--lgamma', type=float, default=1,
        help='adjust the proportion of positive and negative samples, default 1')

    return parser.parse_args()

def train(model, args):
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']

    train_img = CrackData(data_root, data_lst, crop_size=args.crop_size)
    trainloader = torch.utils.data.DataLoader(train_img,
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    logger = args.logger
    params = []
    # weights = torch.tensor([0.1, 0.9]).to('cuda')
    # criterion = nn.CrossEntropyLoss(predictions, targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(trainloader)

    iter_per_epoch = len(trainloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % (iter_per_epoch * args.batch_size))
    logger.info('the batch size is %d ' % (args.iter_size * args.batch_size))
    logger.info('every epoch needs to iterate  %d ' % iter_per_epoch)
    logger.info('*'*40)

    start_time = time.time()
    if args.cuda:
        model.cuda()

    model.train()
    batch_size = args.iter_size * args.batch_size
    for step in range(start_step, args.max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(trainloader)
            images, labels = next(data_iter)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            # print(images.shape)
            # print(labels.shape)
            out= model(images)
            loss = 0

            ##########  single output #########################
            # loss += cross_entropy_loss2d(out, labels, args.cuda, args.balance) / batch_size

            # loss += wcet_jacc(inputs=out, targets=labels, cuda=True,
            #                   a=args.a, b=args.b, loss_type=args.type, beta=args.beta, gamma=args.lgamma) / batch_size
            # loss += Cross_Entropy(out, labels)
            loss += bceDiceLoss(out, labels)
            # loss += nn.CrossEntropyLoss(out[0], labels)
            loss.backward()
            # batch_loss += loss.data[0]      #####lk
            batch_loss += loss.item()   #####lk
            cur += 1
        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size, args.gamma)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/ABLNet_%d.pth' % (args.param_dir, step))
            # state = {'step': step+1,'param':model.state_dict(),'solver':optimizer.state_dict()}
            # torch.save(state, '%s/CarNet_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            # logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
            #     optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))

            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))

            start_time = time.time()
        if not os.path.exists('./validation/'):
            os.makedirs('./validation/')
        if step % args.display == 0:
            prediction, visual_value, out_visual,_,_,_,_,_,_,_,_ = model(images)
            prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))
            visual_value = visual_value.cpu().detach().numpy().transpose((0, 2, 3, 1))
            # s0 = out_visual[0].cpu().detach().numpy().transpose((0, 2, 3, 1))
            # s1 = out_visual[1].cpu().detach().numpy().transpose((0, 2, 3, 1))
            # s2 = out_visual[2].cpu().detach().numpy().transpose((0, 2, 3, 1))
            # s3 = out_visual[3].cpu().detach().numpy().transpose((0, 2, 3, 1))
            # s4 = out_visual[4].cpu().detach().numpy().transpose((0, 2, 3, 1))
            for j in range(visual_value.shape[0]):
                cv2.imwrite('./validation/' + 'brtiness' + str(j) +'.png', visual_value[j] * 255)
            for h in range(prediction.shape[0]):
                cv2.imwrite('./validation/' + 'out' + str(h) + '.png', prediction[h] * 255)
            # for y in range(s0.shape[0]):
            #     cv2.imwrite('./validation/' + 'brtiness_cell' + str(y) + '.png', s0[y] * 255)

            # for k in range(s1.shape[0]):
            #     cv2.imwrite('./validation/' + 's1-' + str(k) + '.png', s1[k] * 255)
            # for m in range(s2.shape[0]):
            #     cv2.imwrite('./validation/' + 's2-' + str(m) + '.png', s2[m] * 255)
            # for n in range(s3.shape[0]):
            #     cv2.imwrite('./validation/' + 's3-' + str(n) + '.png', s3[n] * 255)
            # for l in range(s4.shape[0]):
            #     cv2.imwrite('./validation/' + 's4-' + str(l) + '.png', s4[l] * 255)
def main():
    args = parse_args()

    args.cuda = True    ######### lk
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)

    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    # torch.manual_seed(int(time.time()))
    torch.manual_seed(seed=7)  #### lk
    torch.cuda.manual_seed(seed=7)
    np.random.seed(seed=7)
    torch.backends.cudnn.deteministic=True  #### lk

    # model = CarNet34(
    #     Encoder=Encoder_v0_762,
    #     dp=DownsamplerBlock,
    #     block=BasicBlock_encoder,
    #     channels=[3, 16, 64, 128, 256],
    #     decoder_block=non_bottleneck_1d_2,
    #     num_classes=1
    # )
    # model = DRNet()
    model = ULNet(num_classes=1, 
                               input_channels=3, 
                            #    c_list=[16,32,64,128,256,256], 
                            c_list=[8,16,32,64,128,128],
                               split_att='fc', 
                               bridge=False,)

    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))

    logger.info(model)

    train(model, args)

if __name__ == '__main__':
    main()


