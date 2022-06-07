import torch
import torch.utils.data
from loader import GelDataLoader, RealDataLoader
import numpy as np
import os
from os import path as osp
from fcrn import FCRN_net
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from torch.utils.tensorboard import SummaryWriter
from weights import load_weights
from tqdm import tqdm 
from train_config import train_config
import cv2
from tacto.TactoRender import TactoRender, pixmm

tacRender = TactoRender(obj_path = None, randomize = False, headless = True)
depth_bg = tacRender.correct_pyrender_height_map(tacRender.bg_depth)

dtype = torch.cuda.FloatTensor

# curl -O http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy
def main():
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)
    batch_size, learning_rate, num_epochs = train_config['batch_size'], train_config['lr'], train_config['max_epochs']
    # monentum, weight_decay = 0.9, 0.0005

    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    resume_from_file = train_config['resume_from_file']
    checkpoint_path = './weights/checkpoint_heightmap_digit.pth.tar'

    results_path = "/mnt/sda/suddhu/fcrn/fcrn-training"
    test_results_path = "/mnt/sda/suddhu/fcrn/fcrn-testing"

    # 1.Load data
    data_file_path = osp.join("data_files")
    train_data_file = osp.join(data_file_path,'train_data.txt')
    dev_data_file = osp.join(data_file_path,'dev_data.txt')
    train_label_file = osp.join(data_file_path,'train_label.txt')
    dev_label_file = osp.join(data_file_path,'dev_label.txt')
    test_data_file = osp.join(data_file_path,'test_data.txt')
    test_label_file = osp.join(data_file_path,'test_label.txt')

    print(f"Loading data, Resume training: {resume_from_file}")
    train_loader = torch.utils.data.DataLoader(GelDataLoader(train_data_file, train_label_file),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(GelDataLoader(dev_data_file, dev_label_file),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    # test_real_file = osp.join(data_file_path,'test_data_real.txt')
    # test_loader = torch.utils.data.DataLoader(RealDataLoader(test_real_file), batch_size=batch_size, shuffle=False, drop_last=True)

    test_loader = torch.utils.data.DataLoader(GelDataLoader(test_data_file, test_label_file),
                                               batch_size=batch_size, shuffle=False, drop_last=True)
    # 2.Load model
    print("Loading model...")
    model = FCRN_net(batch_size)
    model = model.cuda()

    # 3.Loss
    loss_fn = torch.nn.MSELoss().cuda()

    input_path = osp.join(results_path, "input")
    gt_path = osp.join(results_path, "gt")
    pred_path = osp.join(results_path, "pred")

    if not osp.exists(input_path):
        os.makedirs(input_path)
    if not osp.exists(gt_path):
        os.makedirs(gt_path)
    if not osp.exists(pred_path):
        os.makedirs(pred_path)

    # real data path 
    if not osp.exists(test_results_path):
        os.makedirs(test_results_path)

    logdir = os.path.join("/home/rpluser/Documents/suddhu/runs", "fcrn_train")
    writer = SummaryWriter(logdir)

    start_epoch = 0
    if resume_from_file:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    else:
        weights_file = "/home/rpluser/Documents/suddhu/projects/shape-closures/weights/NYU_ResNet-UpProj.npy"
        print("=> loading pre-trained NYU weights'{}'".format(weights_file))
        model.load_state_dict(load_weights(model, weights_file, dtype))

    # validate
    print('Validating on sim data')
    model.eval()
    num_samples, loss_local = 0, 0
    with torch.no_grad():
        pbar = tqdm(total = len(val_loader))
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            gt_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss_local += loss_fn(output, gt_var)
            num_samples += 1
            pbar.update(1)
        pbar.close()
    
    best_val_err = np.sqrt( float(loss_local) / num_samples )
    print('Before train error: {:.3f} pixel RMSE'.format(best_val_err))

    for epoch in range(num_epochs):
        # 4.Optim
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)

        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss, count, epoch_loss = 0, 0, 0

        pbar = tqdm(total = len(train_loader))
        #for i, (input, depth) in enumerate(train_loader):
        for input, depth in train_loader:
            input_var = Variable(input.type(dtype))
            gt_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss = loss_fn(output, gt_var)

            # print('loss:', loss.item())
            # input_img = input_var.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            # output_img = output.squeeze().detach().cpu().numpy()
            running_loss += loss.data.cpu().numpy()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description("RMSE pixel loss: {:.2f}".format( np.sqrt( running_loss / count ) ))
        pbar.close()

        # TODO: tensorboard
        epoch_loss = np.sqrt( running_loss / count )
        print('Epoch error: {:.3f} pixel RMSE'.format(epoch_loss))

        writer.add_scalar("train_loss", epoch_loss, start_epoch + epoch + 1)

        # validate
        print('Validating on sim data')
        model.eval()
        num_samples, loss_local = 0, 0
        with torch.no_grad():
            pbar = tqdm(total = len(val_loader))
            for input, depth in val_loader:
                input_var = Variable(input.type(dtype))
                gt_var = Variable(depth.type(dtype))

                output = model(input_var)
                loss_local += loss_fn(output, gt_var)
                num_samples += 1
                pbar.update(1)
            pbar.close()

        err = np.sqrt( float(loss_local) / num_samples )
        print('Validation error: {:.3f} pixel RMSE, Best validation error: {:.3f} pixel RMSE'.format(err, best_val_err))
        writer.add_scalar("val_loss", err, start_epoch + epoch + 1)

        if err < best_val_err:
            print("Saving new checkpoint: {}".format(checkpoint_path))
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
        else:
            learning_rate = learning_rate * 0.6
            print("No reduction of validation error, dropping learning rate to {}".format(learning_rate))

        if (epoch > 0) and (epoch % 10 == 0):
            learning_rate = learning_rate * 0.6
            print("10 epochs, dropping learning rate to {}".format(learning_rate))

        print('Testing on sim data')
        model.eval()
        num_samples, loss_local = 0, 0
        # make local IoU
        with torch.no_grad():
            pbar = tqdm(total = len(test_loader))
            for input, depth in test_loader:
                input_var = Variable(input.type(dtype))
                gt_var = Variable(depth.type(dtype))

                output = model(input_var)

                if num_samples == 0:
                    input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    gt_image = gt_var[0].data.squeeze().cpu().numpy().astype(np.float32)
                    pred_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
                    gt_image /= np.max(gt_image)
                    pred_image /= np.max(pred_image)

                    plot.imsave(osp.join(input_path, "input_epoch_{}.png".format(start_epoch + epoch + 1)), input_rgb_image)
                    plot.imsave(osp.join(gt_path, "gt_epoch_{}.png".format(start_epoch + epoch + 1)), gt_image, cmap="viridis")
                    plot.imsave(osp.join(pred_path, "pred_epoch_{}.png".format(start_epoch + epoch + 1)), pred_image, cmap="viridis")
                loss_local += loss_fn(output, gt_var)
                num_samples += 1
                pbar.update(1)
            pbar.close()
        err = np.sqrt( float(loss_local) / num_samples ) * pixmm
        print(f'Test error: {err:.3f} mm RMSE')
        writer.add_scalar("test_loss", err, start_epoch + epoch + 1)
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
