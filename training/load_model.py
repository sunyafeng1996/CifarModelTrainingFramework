import os, random, sys
sys.path.append(".") 
import argparse
import warnings 
warnings.filterwarnings("ignore")

from all_utils.models_utils import GetAccuracy, GetModel, LoadCheckpoint
from torch import manual_seed, optim, nn
from all_utils.datasets_utils import GetDataLoader
from all_utils.log_utils import print_log

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
    parser.add_argument('--resume',default='./save/resnet20/resnet20_best.pth.tar',
                        type=str,metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # 数据集参数
    parser.add_argument('--test_bs',type=int,default=256,help='Batch size.')
    parser.add_argument('--train_bs',default=64,help='Batch size.')
    parser.add_argument('--test_workers',type=int,default=0,help='Test workers.')
    parser.add_argument('--train_workers',default=0,help='Train works.')
    parser.add_argument('--device',type=str,default='cuda')

args = parser.parse_args()

'''配置log输出'''
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
manual_seed(args.manualSeed)
file_name='log_load_cp_'+args.arch
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
log = open(os.path.join(args.save_path,file_name+'_{}.txt'.format(args.manualSeed)), 'w')

'''初始化模型和测试集'''
model = GetModel(args.arch,args.dataset)
model=model.cuda()
train_loader,test_loader=GetDataLoader(args.dataset,args.train_bs,args.test_bs,args.data_path)
train_loader.num_workers,test_loader.num_workers=args.train_workers,args.test_workers

'''加载模型'''
optimizer = optim.SGD(model.parameters(),lr=0.1)
model, start_epoch, recorder,  optimizer=LoadCheckpoint(args.resume,log,optimizer,model)

'''验证模型精度'''
criterion = nn.CrossEntropyLoss()
print_log('==>>Architecture: {arch}    Dataset: {dataset}    Checkpoint File: {resume}'
        .format(arch=args.arch, dataset=args.dataset, resume=args.resume), log)
GetAccuracy(model, test_loader,criterion,args.device,log)

'''输出其他checkpoint内容'''
print_log('**Epoch: {start_epoch:d}  lr: {lr:f}'.format(start_epoch=start_epoch,
        lr=optimizer.param_groups[0]['lr']), log)