import os, shutil, time
import torch
from torch.autograd import Variable
from all_utils.log_utils import AverageMeter, print_log, time_string
from models.resnet_cifar import resnet
from models.vgg_cifar import vgg

def GetModel(arch,dataset):
    if dataset=='CIFAR10':
        nc=10
    elif dataset=='CIFAR100':
        nc=100
    if arch=='resnet20':
        return resnet(depth=20,num_classes=nc)
    elif arch=='resnet34':
        return resnet(depth=34,num_classes=nc)
    elif arch=='resnet56':
        return resnet(depth=56,num_classes=nc)
    elif arch=='resnet110':
        return resnet(depth=110,num_classes=nc)
    elif arch=='vgg11':
        return vgg(depth=11,num_classes=nc)
    elif arch=='vgg11bn':
        return vgg(depth=11,num_classes=nc,bn=True)
    elif arch=='vgg13':
        return vgg(depth=13,num_classes=nc)
    elif arch=='vgg13bn':
        return vgg(depth=13,num_classes=nc,bn=True)
    elif arch=='vgg16':
        return vgg(depth=16,num_classes=nc)
    elif arch=='vgg16bn':
        return vgg(depth=16,num_classes=nc,bn=True)
    elif arch=='vgg19':
        return vgg(depth=19,num_classes=nc)
    elif arch=='vgg19bn':
        return vgg(depth=19,num_classes=nc,bn=True)

def GetAccuracy(model, test_loader,criterion,device,log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = Accuracy(output.data, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '**Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg, top5.avg, losses.avg

def TrainOneEpoch(train_loader, model, criterion, optimizer, epoch, log, print_freq, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(device), target.to(device)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = Accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}] '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '**Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return model,top1.avg, top5.avg,losses.avg

def Accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k==1:
                correct_k = correct[:k].view(-1).float().sum(0)
                len_c=correct[:k].view(-1).size()[0]*5
            else:
                correct_k = correct[:k].reshape((len_c)).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def SaveCheckpoint(state, is_best, save_path, arch, log):
    filename = os.path.join(save_path, arch+'_checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, arch+'_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)
        
def LoadCheckpoint(resume,log,optimizer,model):
    if resume:
        if os.path.isfile(resume):
            print_log("\n=> loading checkpoint '{}'".format(resume), log)
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            recorder = checkpoint['recorder']
            optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = model.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            model.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(resume),log)
    else:
        print_log("=> do not use any checkpoint", log)
    
    return model, start_epoch, recorder, optimizer   
        
