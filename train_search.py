import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import genotypes
import torch.nn as nn
import torch.utils
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from architect import Architect
from model_search import NetworkCIFAR
from genotypes import Genotype

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=16, help='total number of layers')
#Resnet20 ch:16, layer:9 / Mobilenetv2 ch:32, layer:16
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 200


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
    
#   input_genotype = Genotype(normal=[('none', 0), ('skip_connect', 1), ('none', 0), ('ori_conv_3x3', 2), ('none', 0), ('ori_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], normal_concat=[5], reduce=[('none', 0), ('skip_connect', 1), ('none', 0), ('ori_conv_3x3', 2), ('none', 0), ('ori_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=[5]) #Resnet

  input_genotype = Genotype(normal=[('none', 0), ('ori_conv_1x1', 1), ('none', 0), ('sep_conv_3x3_mob', 2), ('none', 0), ('ori_conv_1x1', 3), ('skip_connect', 1), ('skip_connect', 4)], normal_concat=[5], reduce=[('none', 0), ('ori_conv_1x1', 1), ('none', 0), ('sep_conv_3x3_mob', 2), ('none', 0), ('ori_conv_1x1', 3), ('skip_connect', 1), ('skip_connect', 4)], reduce_concat=[5]) #MobilenetV2

#   input_genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5]) #DARTS
  model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, input_genotype, criterion)
  model.drop_path_prob = 0.0
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

#   net_params = list(model.parameters()) + model.arch_parameters()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_tiny(args)
#   train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
#   test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    
  train_data = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/train/', transform=train_transform)
  test_data = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/valid/', transform=valid_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(1.0 * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2)
  valid_queue = train_queue

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))

  architect = Architect(model, args)

  transform_epoch = 49
  arch_train = True
  max_test_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,  arch_train)
    logging.info('train_acc %f', train_acc)

    # validation
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)
    
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        
    logging.info('Max test acc %f', max_test_acc)
    
    if epoch <= transform_epoch+1:
        print ('Current arch param')
        print(model.alphas_reduce)
        print(model.alphas_normal)
    
    if epoch == transform_epoch:
        model.transform_arch()
        arch_train = False
     
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, arch_train):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
#     input_search, target_search = next(iter(valid_queue))
#     input_search = Variable(input_search, requires_grad=False).cuda()
#     target_search = Variable(target_search, requires_grad=False).cuda(async=True)
    if arch_train:
        architect.step(input, target, input, target, lr, optimizer, unrolled=args.unrolled)
#     new_grad_normal = model.alphas_normal.grad
#     new_grad_reduce = model.alphas_reduce.grad
    
#     grad_sum_normal += new_grad_normal
#     grad_sum_reduce += new_grad_reduce
    
    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()
    
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
#       if step != 0:
#         break
#   print ('Grad summation!')
#   print (grad_sum_normal)
#   print (grad_sum_reduce)
#   model.T = model.T * 0.9
    
  return top1.avg, objs.avg


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  
  with torch.no_grad():
    for step, (input, target) in enumerate(test_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

