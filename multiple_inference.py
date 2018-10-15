#python multiple_inference.py /hd-data/benedict/flow/sintel/training/ ./../../../flow/flownet_pt/FlowNetPytorch/trained_weights/flownet.pth.tar --output ./

import argparse
from path import Path
import torch
import torch.backends.cudnn as cudnn
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from main import flow2rgb
from multiscaleloss import EPE
from IPython import embed
from datasets import listdataset
from main import AverageMeter
import datasets

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to images folder, image names must be sorted according to their time domain')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default=None, help='if not set, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')
parser.add_argument('--image_transforms', default=False, help='Perform transformation of input image')
parser.add_argument('--flo', default=False, help='Save a .flo file instead of a reconstructed flow image')
parser.add_argument('--not_consecutive', default=False, help='Only take every other image for flow prediction')



def split_validation_data_set(path_sintel_main_folder, split_txt_file = None):
    if not split_txt_file :
        split_txt_file = './Sintel_train_val.txt'
    split_txt_file
    _, test_dataset = make_dataset(path_sintel_main_folder, split_txt_file)
    return test_dataset

def evaluate_flow_performance(input_flow_folder, flow_gt_folder):
    ''' Evaluate the performance of flow files found in a folder'''
    #add epe to averagemeter flow_performance.update(epe,1)
    
    flow_performance = Averagemeter()
    import os

    flows = sorted(os.listdir(input_flow_folder))
    gts = sorted(os.listdir(flow_gt_folder))

    for i in range(len(flows)):
        if flows[i] == gts[i]:
            epe = evaluate_flow(flows[i], gts[i])
            flow_performance.update(epe)

    assert(flow_performance.count == len(flows)), 'not all file names matched'

    return flow_performance.avg

def evaluate_flow(input_flow, flow_gt):
    #flow_gt = '/home/benedict/flow/flownet_pt/FlowNetPytorch/test_flow/ambush_2/frame_0002.flo'
    #input_flow = '/home/benedict/flow/flownet_pt/FlowNetPytorch/test_flow/ambush_2_flow/frame_0002.flo'
    
    if input_flow[-3:] and flow_gt[-3:] == 'flo':
        input_flow = listdataset.load_flo(input_flow)
        flow_gt = listdataset.load_flo(flow_gt)
    else:
        input_flow = rgb2flow(input_flow)
        flow_gt = rgb2flow(flow_gt)

    #Arrange Dimension Bx2xHxW
    input_flow = np.expand_dims(np.rollaxis(input_flow,2),axis=0)
    flow_gt = np.expand_dims(np.rollaxis(flow_gt,2),axis=0)

    epe = EPE(torch.from_numpy(input_flow), torch.from_numpy(flow_gt), mean = False)

    return epe

def rgb2flow(png_path):
    """Converts a flow png image to a flow field """
    flo_file = cv2.imread(png_path,cv2.IMREAD_UNCHANGED)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    return(flo_img)

def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((flow, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()

def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    #save_path = data_dir/'flow'
    save_path = './flow_output'
    if args.output != None:
        save_path = Path(args.data)/'flow'    
    print("=> fetching img pairs in '{}'".format(args.data))
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading 
    if args.image_transforms:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])
    else:
        input_transform = flow_transforms.ArrayToTensor()

    inference_files = data_dir.files()
    inference_files.sort()
    
    i, ii  = iter(inference_files), iter(inference_files[1::])
    list_consec, list_inbetweeners = list(zip(i,i)), list(zip(ii,ii))
    if args.not_consecutive:
        img_pairs = list_consec
         
    else: 
        img_pairs = (list_consec + list_inbetweeners)

    img_pairs.sort()
    
    print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).cuda()
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']

    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.autograd.Variable(torch.cat([img1, img2]).cuda(), volatile=True).unsqueeze(0)

        if args.bidirectional:
            # feed inverted pair along with normal pair
            inverted_input_var = torch.autograd.Variable(torch.cat([img2, img1],0).cuda(), volatile=True).unsqueeze(0)
            input_var = torch.cat([input_var, inverted_input_var])

        # compute output
        output = model(input_var)
        if args.upsampling is not None:
            output = torch.nn.functional.upsample(output, size=img1.size()[-2:], mode=args.upsampling)
        for suffix, flow_output in zip(['flow', 'inv_flow'], output.data.cpu()):
            rgb_flow = flow2rgb(args.div_flow * flow_output.numpy(), max_value=args.max_flow)
            if args.flo:
                filename = '{}{}.flo'.format(img1_file.namebase[:-1], suffix)
                write_flow(output, filename)
            else:
                to_save = (rgb_flow * 255).astype(np.uint8)
                imsave(save_path/'{}{}.png'.format(img1_file.namebase[:-1], suffix), to_save)


if __name__ == '__main__':
    main()
