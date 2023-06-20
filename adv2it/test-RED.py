import argparse
import datetime
import json
import logging
import os
import random
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms, models, utils as vutils
from tqdm import tqdm
from ViT_explanation_generator import LRP

#from ViT_LRP import vit_base_patch16_224 as vit_LRP
#from ViT_LRP import vit_large_patch16_224 as vit_LRP
from test_new_interpreter.models import interp_deit_small_patch16_224 as vit_LRP

import vit_models
from attack_copy import normalize, local_adv, local_adv_target
from dataset import AdvImageNet
from torchvision.datasets import ImageFolder

torch.cuda.set_device(2)
torch.cuda.current_device()

targeted_class_dict = {
    24: "Great Grey Owl",
    99: "Goose",
    245: "French Bulldog",
    344: "Hippopotamus",
    471: "Cannon",
    555: "Fire Engine",
    661: "Model T",
    701: "Parachute",
    802: "Snowmobile",
    919: "Street Sign ",
}


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_dir', default='img_folder', help='ImageNet Validation Data')
    parser.add_argument('--dataset', default="imagenet_5k", help='dataset name')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--tar_model', type=str, default='T2t_vit_24', help='Target Model Name')
    parser.add_argument('--src_pretrained', type=str, default=None, help='pretrained path for source model')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--eps', type=int, default=8, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='last', help='last or all')
    parser.add_argument('--attack_type', type=str, default='fgsm', help='fgsm, mifgsm, dim, pgd')
    parser.add_argument('--tar_ensemble', action="store_true", default=False)
    parser.add_argument('--apply_ti', action="store_true", default=False)
    parser.add_argument('--save_im', action="store_true", default=True)

    return parser.parse_args()


def get_model(model_name):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    other_model_names = vars(vit_models)

    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'hierarchical' in model_name or "ensemble" in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'lrp' in model_name:
        model = vit_LRP(pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'swin' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return model, mean, std


def tensor2im_norm(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    image_numpy = image_numpy - np.min(image_numpy)
    image_numpy = image_numpy / np.max(image_numpy) * 255
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def gen_cam(soft_policys):
    soft_map = soft_policys[1][:,1:]
    hm = soft_map.view(soft_map.size(0), 14, 14)
    hm = hm.squeeze()
    hm = torch.clamp(hm, 0.48, 1)
    vis = tensor2im_norm(hm.view(14, 14, 1))
    
    return vis
    

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap #+ np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization_(original_image, attribution_generator, device, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(device), device, method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



#  Test Samples
def get_data_loader(args, verbose=True):
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    test_dir = args.test_dir
    # if args.dataset == "imagenet_1k":
    #     test_set = AdvImageNet(image_list="data/image_list_1k.json", root=test_dir, transform=data_transform)
    # else:
    #     test_set = AdvImageNet(root=test_dir, transform=data_transform)
    test_set = ImageFolder(root=test_dir, transform=data_transform)
    test_size = len(test_set)
    if verbose:
        print('Test data size:', test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    return test_loader, test_size


def start(image):
    # setup run
    args = parse_args()
#     args.exp = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{random.randint(1, 100)}"
#     os.makedirs(f"report/{args.exp}")
#     json.dump(vars(args), open(f"report/{args.exp}/config.json", "w"), indent=4)

#     logger = logging.getLogger(__name__)
#     logfile = f'report/{args.exp}/{args.src_model}_5k_val_eps_{args.eps}.log'
#     logging.basicConfig(
#         format='[%(asctime)s] - %(message)s',
#         datefmt='%Y/%m/%d %H:%M:%S',
#         level=logging.INFO,
#         filename=logfile)

    # data_dir = '../images/'

    # GPU
    device = 'cpu' #torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # load source and target models
    src_model, src_mean, src_std = get_model(args.src_model)
    if args.src_pretrained is not None:
        if args.src_pretrained.startswith("https://"):
            src_checkpoint = torch.hub.load_state_dict_from_url(args.src_pretrained, map_location='cpu')
        else:
            src_checkpoint = torch.load(args.src_pretrained, map_location='cpu')
        src_model.load_state_dict(src_checkpoint['model'])
    src_model = src_model.to(device)
    src_model.eval()
  


#     tar_model, tar_mean, tar_std = get_model(args.tar_model)
    tar_mean = (0.485, 0.456, 0.406)
    tar_std = (0.229, 0.224, 0.225)
    tar_model = vit_LRP(device, pretrained=True)
    checkpoint = torch.load('./test_new_interpreter/ckpt.pth', map_location='cpu')
    tar_model.load_state_dict(checkpoint, strict=True)
    
    tar_model = tar_model.to(device)
    tar_model.eval()

    #attribution_generator = LRP(tar_model)

    # Setup-Data
    test_loader, test_size = get_data_loader(args)

    # setup eval parameters
    eps = args.eps / 255
    criterion = torch.nn.CrossEntropyLoss()

    clean_acc = 0.0
    adv_acc = 0.0
    fool_rate = 0.0
#     logger.info(f'Source: "{args.src_model}" \t Target: "{args.tar_model}" \t Eps: {args.eps} \t Index: {args.index} '
#                 f'\t Attack: {args.attack_type}')

    # images = os.listdir(data_dir)
#     file = os.listdir(f_dir)
#     img_file = np.load(f'{image}')['img_x']
#     print(len(file))
#     existing_files = set([f.split('.')[0][:-1] for f in os.listdir('IA-RED/output')])
    files = [image]
#     print(existing_files)
    
#     for fi in file:
#         temp = 'output_{}_'.format(fi.split('.')[0])
#         if (temp not in existing_files):
#             print('skipped')
#             files.append(fi)
#     print(len(files))
    with tqdm(enumerate(files), total=len(files)) as p_bar:
        for j, f_name in p_bar:
            name = f_name.split('.')[0]
            img_file = np.load(f'{f_dir}/{f_name}')['img_x']
            for i, img in enumerate(img_file):
#                 if ('output_{}_{}.npz'.format(name, i) in existing_files):
#                     print('skipped')
#                     continue
                img = torch.tensor(img[None, ...]).to(device)
                
                with torch.no_grad():
                    preds, _, soft_policys = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))
                
                ben_cam = gen_cam(soft_policys)
            # img, label = img.to(device), torch.tensor([65]).to(device)

#                 ben_cam = generate_visualization_(img.clone().squeeze(0), attribution_generator, device)

                # with torch.no_grad():
                if args.tar_ensemble:
                    clean_out, _, _ = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std), get_average=True)
                else:
                    clean_out, _, _ = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))
                if isinstance(clean_out, list):
                    clean_out = clean_out[-1].detach()

                label = clean_out.argmax(dim=-1)
                clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()

                adv, adv_cam = local_adv(device, src_model, tar_model, criterion, img, label, eps, ben_cam, attack_type=args.attack_type, iters=args.iter,
                                std=src_std, mean=src_mean, index=args.index, apply_ti=args.apply_ti)

                # adv, adv_cam = local_adv_target(src_model, attribution_generator, criterion, img, label, eps, ben_cam, attack_type=args.attack_type, iters=args.iter,
                #                 std=src_std, mean=src_mean, index=args.index) #, apply_ti=args.apply_ti)

                # with torch.no_grad():
                if args.tar_ensemble:
                    adv_out, _, _ = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std), get_average=True)
                else:
                    adv_out, _, _ = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))
                if isinstance(adv_out, list):
                    adv_out = adv_out[-1].detach()
                adv_acc += torch.sum(adv_out.argmax(dim=-1) == label).item()
                fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

                if (adv_out.argmax(dim=-1) != label) and args.save_im:
                    save_dobj = {'adv_x': vutils.make_grid(adv, normalize=False, scale_each=True).cpu(), 'img_x': vutils.make_grid(img, normalize=False, scale_each=True).cpu(), 'adv_cam': adv_cam, 'ben_cam': ben_cam, 'adv_label': adv_out.detach().cpu().numpy()}
                    
                    return vutils.make_grid(img, normalize=False, scale_each=True).cpu(), ben_cam, preds, vutils.make_grid(adv, normalize=False, scale_each=True).cpu(), adv_cam, adv_out.detach().cpu().numpy()
#                     np.savez('IA-RED/output/output_{}_{}.npz'.format(name, i), **save_dobj)
                    # adv_cam = generate_visualization_(adv.clone().squeeze(0), attribution_generator)
                    # ben_cam = generate_visualization_(img.clone().squeeze(0), attribution_generator)
                    # plt.imsave(f'adv_cam_{i}.png', adv_cam)
                    # plt.imsave(f'org_cam_{i}.png', ben_cam)
                    # vutils.save_image(vutils.make_grid(adv, normalize=False, scale_each=True), f'adv_{i}.png')
                    # vutils.save_image(vutils.make_grid(adv, normalize=False, scale_each=True), f'adv_{i}.png')
                    # vutils.save_image(vutils.make_grid(img, normalize=False, scale_each=True), f'org_{i}.png')

#             p_bar.set_postfix({"L-inf": f'{(img - adv).max().item() * 255:.2f}',
#                                "Clean Range": f'{img.max().item():.2f}, {img.min().item():.2f}',
#                                "Adv Range": f'{adv.max().item():.2f}, {adv.min().item():.2f}', })

    return vutils.make_grid(img, normalize=False, scale_each=True).cpu(), ben_cam, preds, vutils.make_grid(img, normalize=False, scale_each=True).cpu(), ben_cam, preds
#     print('Clean:{0:.3%}\t Adv :{1:.3%}\t Fooling Rate:{2:.3%}'.format(clean_acc / test_size, adv_acc / test_size,
#                                                                        fool_rate / test_size))
#     logger.info(
#         'Eps:{0} \t Clean:{1:.3%} \t Adv :{2:.3%}\t Fooling Rate:{3:.3%}'.format(int(eps * 255), clean_acc / test_size,
#                                                                                  adv_acc / test_size,
#                                                                                  fool_rate / test_size))
#     json.dump({"eps": int(eps * 255),
#                "clean": clean_acc / test_size,
#                "adv": adv_acc / test_size,
#                "fool rate": fool_rate / test_size, },
#               open(f"report/{args.exp}/results.json", "w"), indent=4)


if __name__ == '__main__':
    data_dir = '../images'
#     files = os.listdir(data_dir)
    start(data_dir)
