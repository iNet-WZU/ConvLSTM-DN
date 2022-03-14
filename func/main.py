import os
import sys
import argparse
import numpy as np
from datetime import datetime
# from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from func.dataset import read_data
from func.model import DenseNet
import joblib
import pickle
import xlwt
import time

torch.manual_seed(22)
device = torch.device("cuda")
parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=9)
parse.add_argument('-width', type=int, default=6)
parse.add_argument('-meta', type=int, default=0)
parse.add_argument('-close_size', type=int, default=10)
parse.add_argument('-last_kernel', type=int, default=1)
parse.add_argument('-period_size', type=int, default=0)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
# parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
# parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.01)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=1, help='epochs')
# parse.add_argument('-test_size', type=int, default=1820)
#
parse.add_argument('-save_dir', type=str, default='results')
opt = parse.parse_args()
opt.save_dir = '{}'.format(opt.save_dir)


def mape(y_pred,y_true):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def smape(y_pred,y_true):
    n = len(y_true)
    smape = sum(np.abs((y_true-y_pred)/(np.abs(y_pred)+np.abs(y_true)/2)))/n*100
    return smape

def mae_value(y_true,y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae


def rmse_value(y_pred,y_true):
    n = len(y_true)
    rmse = math.sqrt(sum(np.square(y_true - y_pred)) / n)
    return rmse


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def train_epoch(data_type='train'):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader


    if (opt.close_size > 0) & (opt.meta == 1):
        for idx, (x, meta, y) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = x.float().to(device)
            meta = meta.float().to(device)
            y = y.float().to(device)
            pred = model(x, meta=meta)
            loss = criterion(pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    elif opt.close_size > 0:
        for idx, (x, y) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = x.float().to(device)

            y = y.float().to(device)
            # print(y)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss


def train():
    os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 1.0
    train_loss, valid_loss = [], []
    for i in range(opt.epoch_size):
        scheduler.step()
        train_loss.append(train_epoch('train'))
        valid_loss.append(train_epoch('valid'))
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')
            torch.save(model.state_dict(), opt.model_filename + '.pt')

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.5f}').format((i + 1), opt.epoch_size,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      opt.lr)
        if i % 2 == 0:
            print(log_string)
        log(opt.model_filename + '.log', log_string)
    # x = range(0, opt.epoch_size)
    # y1 = train_loss
    # y2 = valid_loss
    # plt.plot(x, y1, 'o-', label='train_loss')
    # plt.plot(x, y2, '.-', label='eval_loss')
    # plt.show()


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    model.eval()
    model.load_state_dict(torch.load(opt.model_filename + '.pt'))

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    with torch.no_grad():
        if (opt.close_size > 0) & (opt.meta == 1):
            for idx, (x, meta, y) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = x.float().to(device)
                meta = meta.float().to(device)
                y = y.float().to(device)
                pred = model(x, meta=meta)
                predictions.append(pred.data.cpu())
                ground_truth.append(y.data)
                loss.append(criterion(pred, y).item())

        elif opt.close_size > 0:
            for idx, (x, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = x.float().to(device)
                y = target.float().to(device)
                pred = model(x)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    ground_truth = mmn.inverse_transform(ground_truth)
    final_predict = mmn.inverse_transform(final_predict)
    return final_predict, ground_truth

def train_valid_split(dataloader, test_size=0.25, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':

     start_time = time.time()
     workbook1 = xlwt.Workbook(encoding='utf-8')
     worksheet1 = workbook1.add_sheet('sheet1')
     workbook2 = xlwt.Workbook(encoding='utf-8')
     worksheet2 = workbook2.add_sheet('sheet1')
     workbook3 = xlwt.Workbook(encoding='utf-8')
     worksheet3 = workbook3.add_sheet('sheet1')
     workbook4 = xlwt.Workbook(encoding='utf-8')
     worksheet4 = workbook4.add_sheet('sheet1')
     workbook5 = xlwt.Workbook(encoding='utf-8')
     worksheet5 = workbook5.add_sheet('sheet1')
     workbook6 = xlwt.Workbook(encoding='utf-8')
     worksheet6 = workbook6.add_sheet('sheet1')
     workbook7 = xlwt.Workbook(encoding='utf-8')
     worksheet7 = workbook7.add_sheet('sheet1')
     workbook8 = xlwt.Workbook(encoding='utf-8')
     worksheet8 = workbook8.add_sheet('sheet1')
     workbook9 = xlwt.Workbook(encoding='utf-8')
     worksheet9 = workbook9.add_sheet('sheet1')
     workbookCivic = xlwt.Workbook(encoding='utf-8')
     worksheetCivic = workbookCivic.add_sheet('sheet1')
     workbookLot1 = xlwt.Workbook(encoding='utf-8')
     worksheetLot1 = workbookLot1.add_sheet('sheet1')

     for i in range(1):
         path = '../data/parking_data.h5'
         # ../data/data.h5
         X, X_meta, y, mmn = read_data(path, opt)
         samples, sequences, channels, height, width = X.shape
         test_size = math.floor(samples * 0.2)
         x_train, x_test = X[:-test_size], X[-test_size:]
         meta_train, meta_test = X_meta[:-test_size], X_meta[-test_size:]
         y_train = y[:-test_size]
         y_test = y[-test_size:]
         prediction_ct = 0
         truth_ct = 0
         opt.model_filename = '{}/model={}lr={}-close={}-period=' \
                              '{}-meta={}'.format(opt.save_dir,
                                                  'densenet',
                                                  opt.lr,
                                                  opt.close_size,
                                                  opt.period_size,
                                                  opt.meta)
         print('Saving to ' + opt.model_filename)
         if (opt.meta == 1):
             train_data = list(zip(*[x_train, meta_train, y_train]))
             test_data = list(zip(*[x_test, meta_test, y_test]))
         elif (opt.meta == 0):
             train_data = list(zip(*[x_train, y_train]))
             test_data = list(zip(*[x_test, y_test]))

         train_idx, valid_idx = train_valid_split(train_data, 0.25)
         train_sampler = SubsetRandomSampler(train_idx)
         valid_sampler = SubsetRandomSampler(valid_idx)
         train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                                   num_workers=8, pin_memory=True)
         valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                                   num_workers=2, pin_memory=True)
         test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
         input_shape = X.shape
         meta_shape = X_meta.shape
         model = DenseNet(input_shape, meta_shape, maps=(opt.meta + 1)).to(device)
         if opt.train:
             model_name = '{}/model={}lr={}-close={}-period=' \
                          '{}-meta={}'.format(opt.save_dir,
                                              'densenet',
                                              opt.lr,
                                              opt.close_size,
                                              opt.period_size,
                                              opt.meta)
             if not os.path.exists(opt.save_dir):
                 os.makedirs(opt.save_dir)
             if not os.path.isdir(opt.save_dir):
                 raise Exception('%s is not a dir' % opt.save_dir)

             # print(model_name)
             # model.load_state_dict(torch.load(model_name + '.pt'))

         optimizer = optim.Adam(model.parameters(), opt.lr)
         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=[0.5 * opt.epoch_size,
                                                                      0.75 * opt.epoch_size],
                                                          gamma=0.1)

         if opt.loss == 'l1':
             criterion = nn.L1Loss().cuda()
         elif opt.loss == 'l2':
             criterion = nn.MSELoss().cuda()

         print("model training.......")
         log(opt.model_filename + '.log', '[training]')
         if opt.train:
             train()
         pred, truth = predict('test')
     #     end_time = time.time()
     #     print(end_time-start_time)
     #     pred9 = pred[:, :, 0, 0]
     #     truth9 = truth[:, :, 0, 0]
     #     Rmse9 = rmse_value(pred9.ravel(), truth9.ravel())
     #     Mae9 = mae_value(pred9.ravel(), truth9.ravel())
     #     Mape9 = mape(pred9.ravel(), truth9.ravel())
     #
     #     pred1 = pred[:, :, 1, 1]
     #     truth1 = truth[:, :, 1, 1]
     #     Rmse1 = rmse_value(pred1.ravel(), truth1.ravel())
     #     Mae1 = mae_value(pred1.ravel(), truth1.ravel())
     #     Mape1 = mape(pred1.ravel(), truth1.ravel())
     #
     #     pred2 = pred[:, :, 2, 0]
     #     truth2 = truth[:, :, 2, 0]
     #     Rmse2 = rmse_value(pred2.ravel(), truth2.ravel())
     #     Mae2 = mae_value(pred2.ravel(), truth2.ravel())
     #     Mape2 = mape(pred2.ravel(), truth2.ravel())
     #
     #     pred3 = pred[:, :, 1, 2]
     #     truth3 = truth[:, :, 1, 2]
     #     Rmse3 = rmse_value(pred3.ravel(), truth3.ravel())
     #     Mae3 = mae_value(pred3.ravel(), truth3.ravel())
     #     Mape3 = smape(pred3.ravel(), truth3.ravel())
     #
     #     pred4 = pred[:, :, 2, 1]
     #     truth4 = truth[:, :, 2, 1]
     #     Rmse4 = rmse_value(pred4.ravel(), truth4.ravel())
     #     Mae4 = mae_value(pred4.ravel(), truth4.ravel())
     #     Mape4 = mape(pred4.ravel(), truth4.ravel())
     #
     #     pred5 = pred[:, :, 2, 3]
     #     truth5 = truth[:, :, 2, 3]
     #     Rmse5 = rmse_value(pred5.ravel(), truth5.ravel())
     #     Mae5 = mae_value(pred5.ravel(), truth5.ravel())
     #     Mape5 = smape(pred5.ravel(), truth5.ravel())
     #
     #     pred6 = pred[:, :, 3, 2]
     #     truth6 = truth[:, :, 3, 2]
     #     Rmse6 = rmse_value(pred6.ravel(), truth6.ravel())
     #     Mae6 = mae_value(pred6.ravel(), truth6.ravel())
     #     Mape6 = mape(pred6.ravel(), truth6.ravel())
     #
     #     pred7 = pred[:, :, 3, 3]
     #     truth7 = truth[:, :, 3, 3]
     #     Rmse7 = rmse_value(pred7.ravel(), truth7.ravel())
     #     Mae7 = mae_value(pred7.ravel(), truth7.ravel())
     #     Mape7 = mape(pred7.ravel(), truth7.ravel())
     #
     #     pred8 = pred[:, :, 4, 3]
     #     truth8 = truth[:, :, 4, 3]
     #     Rmse8 = rmse_value(pred8.ravel(), truth8.ravel())
     #     Mae8 = mae_value(pred8.ravel(), truth8.ravel())
     #     Mape8 = mape(pred8.ravel(), truth8.ravel())
     #
     #     predCivic = pred[:, :, 4, 5]
     #     truthCivic = truth[:, :, 4, 5]
     #     RmseCivic = rmse_value(predCivic.ravel(), truthCivic.ravel())
     #     MaeCivic = mae_value(predCivic.ravel(), truthCivic.ravel())
     #     MapeCivic = smape(predCivic.ravel(), truthCivic.ravel())
     #
     #     predLot1 = pred[:, :, 5, 1]
     #     truthLot1 = truth[:, :, 5, 1]
     #     RmseLot1 = rmse_value(predLot1.ravel(), truthLot1.ravel())
     #     MaeLot1 = mae_value(predLot1.ravel(), truthLot1.ravel())
     #     MapeLot1 = mape(predLot1.ravel(), truthLot1.ravel())
     #
     #     worksheet1.write(i, 0, Rmse1)
     #     worksheet1.write(i, 1, Mae1)
     #     worksheet1.write(i, 2, Mape1)
     #     worksheet2.write(i, 0, Rmse2)
     #     worksheet2.write(i, 1, Mae2)
     #     worksheet2.write(i, 2, Mape2)
     #     worksheet3.write(i, 0, Rmse3)
     #     worksheet3.write(i, 1, Mae3)
     #     worksheet3.write(i, 2, Mape3)
     #     worksheet4.write(i, 0, Rmse4)
     #     worksheet4.write(i, 1, Mae4)
     #     worksheet4.write(i, 2, Mape4)
     #     worksheet5.write(i, 0, Rmse5)
     #     worksheet5.write(i, 1, Mae5)
     #     worksheet5.write(i, 2, Mape5)
     #     worksheet6.write(i, 0, Rmse6)
     #     worksheet6.write(i, 1, Mae6)
     #     worksheet6.write(i, 2, Mape6)
     #     worksheet7.write(i, 0, Rmse7)
     #     worksheet7.write(i, 1, Mae7)
     #     worksheet7.write(i, 2, Mape7)
     #     worksheet8.write(i, 0, Rmse8)
     #     worksheet8.write(i, 1, Mae8)
     #     worksheet8.write(i, 2, Mape8)
     #     worksheet9.write(i, 0, Rmse9)
     #     worksheet9.write(i, 1, Mae9)
     #     worksheet9.write(i, 2, Mape9)
     #     worksheetCivic.write(i, 0, RmseCivic)
     #     worksheetCivic.write(i, 1, MaeCivic)
     #     worksheetCivic.write(i, 2, MapeCivic)
     #     worksheetLot1.write(i, 0, RmseLot1)
     #     worksheetLot1.write(i, 1, MaeLot1)
     #     worksheetLot1.write(i, 2, MapeLot1)
     #
     #     np.savetxt("../res/truth1.txt", truth1.ravel())
     #     np.savetxt("../res/pred1.txt", pred1.ravel())
     #     np.savetxt("../res/truth2.txt", truth2.ravel())
     #     np.savetxt("../res/pred2.txt", pred2.ravel())
     #     np.savetxt("../res/truth3.txt", truth3.ravel())
     #     np.savetxt("../res/pred3.txt", pred3.ravel())
     #     np.savetxt("../res/truth4.txt", truth4.ravel())
     #     np.savetxt("../res/pred4.txt", pred4.ravel())
     #     np.savetxt("../res/truth5.txt", truth5.ravel())
     #     np.savetxt("../res/pred5.txt", pred5.ravel())
     #     np.savetxt("../res/truth6.txt", truth6.ravel())
     #     np.savetxt("../res/pred6.txt", pred6.ravel())
     #     np.savetxt("../res/truth7.txt", truth7.ravel())
     #     np.savetxt("../res/pred7.txt", pred7.ravel())
     #     np.savetxt("../res/truth8.txt", truth8.ravel())
     #     np.savetxt("../res/pred8.txt", pred8.ravel())
     #     np.savetxt("../res/truth9.txt", truth9.ravel())
     #     np.savetxt("../res/pred9.txt", pred9.ravel())
     #     np.savetxt("../res/truthCivic.txt", truthCivic.ravel())
     #     np.savetxt("../res/predCivic.txt", predCivic.ravel())
     #     np.savetxt("../res/truthLot1.txt", truthLot1.ravel())
     #     np.savetxt("../res/predLot1.txt", predLot1.ravel())
     #
     #     # #
     # workbook1.save("../res/st1.xls")
     # workbook2.save("../res/st2.xls")
     # workbook3.save("../res/st3.xls")
     # workbook4.save("../res/st4.xls")
     # workbook5.save("../res/st5.xls")
     # workbook6.save("../res/st6.xls")
     # workbook7.save("../res/st7.xls")
     # workbook8.save("../res/st8.xls")
     # workbook9.save("../res/st9.xls")
     # workbookLot1.save("../res/Lot1.xls")
     # workbookCivic.save("../res/Civic.xls")





















