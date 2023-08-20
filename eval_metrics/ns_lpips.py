import pandas as pd
import argparse
import os
from PIL import Image
from torchvision import transforms
import torch
import lpips

# ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def parse_args():
    parser = argparse.ArgumentParser(prog = 'LPIPS', description = 'NS-LPIPS (the smaller the better)')
    parser.add_argument('--prompts_path', type=str, default='../data/stylediffusion_editing.csv')
    parser.add_argument('--input_path', help='path of input image', type=str, default="../results/input")
    parser.add_argument('--edited_path', help='path of edited image', type=str, default="../results/ours")
    parser.add_argument('--mask_path', help='folder where to save mask', type=str, default='../results/bgmask')
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--end_case', help='end generation of case_number', type=int, required=False, default=1e10)
    args = parser.parse_args()
    print(args)
    return args

def to_tensor(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_tensor = transform(img)
    return img_tensor

def get_mask(bgmask_path):
    img_tensor = to_tensor(bgmask_path)

    gray_tensor = transforms.Grayscale()(img_tensor)

    threshold = 0.5
    binary_tensor = (gray_tensor > threshold).float()
    binary_tensor = gray_tensor
    return binary_tensor.repeat(3, 1, 1)

def main(args):
    prompts_path = args.prompts_path
    input_path = args.input_path
    edited_path = args.edited_path
    mask_path = args.mask_path
    from_case = args.from_case
    end_case = args.end_case
    df = pd.read_csv(prompts_path)

    source_imgs, mask_imgs, target_imgs = [], [], []
    for _, row in df.iterrows():
        case_number = row.case_number
        prompt = str(row.prompt)
        source_path = f'{input_path}/{case_number} {prompt}.jpg'
        bgmask_path = f'{mask_path}/{case_number} {prompt}.png'
        target = str(row.target)
        target_path = f'{edited_path}/{case_number} {target}.png'
        if case_number<from_case:
            continue
        if case_number>=end_case:
            break

        if not os.path.exists(source_path) or \
                not os.path.exists(target_path):
            continue
        if not os.path.exists(bgmask_path):
            print('Run \"storemask.py\" to get the background mask.')
            continue

        source_imgs += [to_tensor(source_path)]
        mask_imgs += [get_mask(bgmask_path)]
        target_imgs += [to_tensor(target_path)]

    assert len(source_imgs) != 0
    source_imgs = torch.stack(source_imgs).to(device)
    mask_imgs = torch.stack(mask_imgs).to(device)
    target_imgs = torch.stack(target_imgs).to(device)

    src_bg, tgt_bg = source_imgs * mask_imgs, target_imgs * mask_imgs

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores

    d = loss_fn_alex(src_bg, tgt_bg)
    print(f'NS-LPIPS={d.mean().item()}')


if __name__=="__main__":
    args = parse_args()
    main(args)