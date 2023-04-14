import argparse
import os, time, random, sys
sys.path.append('.') 
import warnings 
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
from all_utils.models_utils import GetAccuracy, GetModel, SaveCheckpoint, TrainOneEpoch
from all_utils.datasets_utils import GetDataLoader
from all_utils.log_utils import AverageMeter, RecorderMeter, convert_secs2time, print_log, time_string
from torch import manual_seed, optim, nn
'''加载参数'''
if 1==1:
    parser = argparse.ArgumentParser(description='Train network for cifar10')
    #基本参数
    parser.add_argument('--data_path',default='./datasets/CIFAR10',type=str,
                        help='Path to dataset')
    parser.add_argument('--dataset',default='CIFAR10',type=str,
                        help='Choose CIFAR10/CIFAR100.')
    parser.add_argument('--arch',metavar='ARCH',default='resnet20')
    parser.add_argument('--save_path',type=str,default='./save/resnet20/',
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--manualSeed', type=int, default=None, 
                        help='manual seed')
    parser.add_argument('--resume',default='./save/resnet20_best.pth.tar',
                        type=str,metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # 数据集参数
    parser.add_argument('--test_bs',type=int,default=256,help='Batch size.')
    parser.add_argument('--train_bs',default=64,help='Batch size.')
    parser.add_argument('--test_workers',type=int,default=0,help='Test workers.')
    parser.add_argument('--train_workers',default=0,help='Train works.')
    parser.add_argument('--device',type=str,default='cuda')
    # 优化器参数
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()

'''配置log输出'''
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
manual_seed(args.manualSeed)
file_name='log_training_'+args.arch
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
log = open(os.path.join(args.save_path,file_name+'_{}.txt'.format(args.manualSeed)), 'w')
# Init the tensorboard path and writer
file_name='run_t_fp_'+args.arch
tb_path = os.path.join(args.save_path, 'tb_log',file_name + str(args.manualSeed))
writer = SummaryWriter(tb_path)

def main():
    '''初始化模型和测试集'''
    model = GetModel(args.arch,args.dataset)
    model=model.cuda()
    train_loader,test_loader=GetDataLoader(args.dataset,args.train_bs,args.test_bs,args.data_path)
    train_loader.num_workers,test_loader.num_workers=args.train_workers,args.test_workers
    
    '''验证模型结构准确性'''
    criterion = nn.CrossEntropyLoss()
    print_log('\n==>>Architecture: {arch}    Dataset: {dataset}    Save Folder: {save_path}'
            .format(arch=args.arch, dataset=args.dataset, save_path=args.save_path), log)
    GetAccuracy(model, test_loader,criterion,args.device,log)
    
    '''训练模型'''
    # 统计epoch信息
    recorder = RecorderMeter(args.epochs)  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # lr调度
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        current_learning_rate = optimizer.param_groups[0]['lr']
        # 显示模拟时间
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)
        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'
            .format(time_string(), epoch, args.epochs,need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                100 - recorder.max_accuracy(False)), log)
        # 一次训练
        model,train_acc_top1, _,train_loss=TrainOneEpoch(train_loader, 
                    model, criterion, optimizer, epoch, log, args.print_freq, args.device)
        # 在测试集上评估并记录
        test_acc_top1, _, test_loss = GetAccuracy(model, test_loader,criterion,args.device,log)
        recorder.update(epoch, train_loss, train_acc_top1, test_loss, test_acc_top1)
        is_best = test_acc_top1 >= recorder.max_accuracy(False)
        # 存储检查点
        checkpoint_state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }
        SaveCheckpoint(checkpoint_state, is_best, args.save_path, args.arch, log)
        # 记录运行时间并更新图
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        print_log('Epoch {epoch:0d}  Time {epoch_time.val:.3f}'
            .format(epoch=epoch, epoch_time=epoch_time), log)
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
        # 更新TensorBoard信息
        writer.add_scalar('loss/train_loss', train_loss, epoch + 1)
        writer.add_scalar('loss/test_loss', test_loss, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc_top1, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', test_acc_top1, epoch + 1)
        
    log.close()
 
if __name__ == '__main__':
    main()
    a=1












