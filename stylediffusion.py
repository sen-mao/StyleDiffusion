from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import torch.nn as nn
import numpy as np
import abc
import seq_aligner
from torch.optim.adam import Adam
from PIL import Image
import copy
import wandb
import argparse
import time
import os
import ast
import clip
import torchvision.transforms as transforms

import ptp_utils_v
from clip_util import VisionTransformer
clip.model.VisionTransformer = VisionTransformer

# For loading the Stable Diffusion using Diffusers, follow the instuctions https://huggingface.co/blog/stable_diffusion and update MY_TOKEN with your token.
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = True
NUM_DDIM_STEPS = 30
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

IS_TRAIN = None  # assign True or False by args.is_train
USE_INITIAL_INV = False
BLOCK_NUM = 1
use_wandb = False

class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(64, 64))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2,
                 th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_v.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils_v.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th

class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    @abc.abstractmethod
    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        else:  # self-attn of unconditional branch
            attn = self.replace_uncond(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, tau_neg=.0):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.tau_neg = tau_neg

class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_uncond(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).replace_uncond(attn, is_cross, place_in_unet)
        if not is_cross and self.num_uncond_self_replace[0] <= self.cur_step < self.num_uncond_self_replace[1]:
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 uncond_self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils_v.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                              tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        if type(uncond_self_replace_steps) is float:
            uncond_self_replace_steps = 0, uncond_self_replace_steps
        self.num_uncond_self_replace = int(num_steps * uncond_self_replace_steps[0]), int(num_steps * uncond_self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps,
                                              local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, uncond_self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, uncond_self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                                                                                     Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils_v.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, uncond_self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, uncond_self_replace_steps=uncond_self_replace_steps,
                                       equalizer=eq, local_blend=lb, controller=controller)
    return controller

def show_cross_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], select: int = 0, save_name='cross-attn-map'):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils_v.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils_v.view_images(np.stack(images, axis=0), save_name=save_name)

def show_hot_cross_attention(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str], select: int = 0, save_name='cross-attn-map'):
    import cv2
    choice = 4
    colormap_dict = {
        1: cv2.COLORMAP_VIRIDIS,
        2: cv2.COLORMAP_PLASMA,
        3: cv2.COLORMAP_HOT,
        4: cv2.COLORMAP_JET,
        5: cv2.COLORMAP_INFERNO,
        6: cv2.COLORMAP_AUTUMN,
        7: cv2.COLORMAP_BONE,
        8: cv2.COLORMAP_WINTER,
        9: cv2.COLORMAP_RAINBOW,
        10: cv2.COLORMAP_OCEAN,
        11: cv2.COLORMAP_SUMMER,
        12: cv2.COLORMAP_SPRING,
        13: cv2.COLORMAP_COOL,
        14: cv2.COLORMAP_HSV,
        15: cv2.COLORMAP_PINK,
    }

    def gray_to_heatmap(gray_image, colormap):
        colored_image = cv2.applyColorMap(gray_image, colormap)
        return colored_image
    if choice not in colormap_dict:
        print("Invalid choice. Using the default colormap (viridis).")
        choice = 1
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        colored_image = gray_to_heatmap(image[:,:,0], colormap_dict[choice])
        cv2.imwrite(f'{save_name}-{i}.png', colored_image)

def show_self_attention_comp(attention_store: AttentionStore, prompts: List[str], res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils_v.view_images(np.concatenate(images, axis=1), save_name='self-attn-map-comp')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def register_attention_control(model, trainer, controller):
    assert IS_TRAIN is not None, print("must set True or False for args.is_train.")
    assert controller is None if IS_TRAIN else (trainer and controller)
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]  # todo: ?
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            # image encoded to embedding for to_v() in cross-attn of conditional branch.
            if IS_TRAIN:  # training phase
                '''
                skip when trainer.ddim_inv is True which means to store ground truth attn maps,
                these attn maps are used as supervision during the training phase
                '''
                if (not trainer.uncond and is_cross) and (not trainer.ddim_inv):
                    context = trainer.forward_embed(context)
            else:  # editing phase
                if not controller.uncond and is_cross:
                    if USE_INITIAL_INV:
                        context = trainer.forward_embed(context)
                    else:
                        i = trainer.i
                        cont = list(context.chunk(context.shape[0]))
                        for b in range(len(cont)):
                            trainer.i = trainer.I if b == 0 else i
                            cont[b] = trainer.forward_embed(cont[b])
                        context = cont[0] if len(cont) == 1 else torch.cat(cont)
                        trainer.i = i
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            if IS_TRAIN:  # training phase
                attn = trainer(attn, is_cross, place_in_unet)
            else:  # editing phase
                attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if IS_TRAIN:
        if trainer is None:
            trainer = DummyController()
    else:
        if controller is None:
            controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    if IS_TRAIN:
        trainer.num_att_layers = cross_att_count
    else:
        controller.num_att_layers = cross_att_count

def image_grid(img, grid_size):
    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(gh * H, gw * W, C)
    return img

class Trainer(AttentionStore):
    def __init__(self):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # clip image encoder
        self.clip_model, clip_preprocess = clip.load('ViT-B/16', device=self.device)
        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +                                        # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])                                         # + skip convert PIL to tensor

        self.image = None
        self.embedding = []
        # image embedding
        scale = 2
        self.embedding = []
        self.convblock = nn.Sequential(nn.Conv1d(77 * scale, 77 * scale, kernel_size=1),
                                       nn.BatchNorm1d(77 * scale, affine=True),
                                       nn.LeakyReLU())
        for _ in range(NUM_DDIM_STEPS):
            self.embedding.append(nn.ModuleDict({
                'conv_start': nn.Conv1d(197, 77 * scale, kernel_size=1),  # (bs, 197, 768)->(bs, 77, 768)
                'conv_block': nn.Sequential(*[copy.deepcopy(self.convblock) for _ in range(BLOCK_NUM)]),
                'conv_end': nn.Conv1d(77 * scale, 77 * scale, kernel_size=1),  # (bs, 77, 768)->(bs, 77, 768)
            }).train().requires_grad_(False).to(device))

        self.I = None  # only for eval
        self.i = None
        self.uncond = False
        self.ddim_inv = False
        self.v_replace_steps = .5

    def load_pretrained(self, pretrained_embedding):
        for i, pre_embedding in enumerate(pretrained_embedding):
            for pre_emb, emb in zip(pre_embedding.values(), self.embedding[i].values()):
                self.copy_params_and_buffers(pre_emb, emb)

    def named_params_and_buffers(self, module):
        assert isinstance(module, torch.nn.Module)
        return list(module.named_parameters()) + list(module.named_buffers())

    def copy_params_and_buffers(self, src_vae, dst_vae, require_all=False):
        assert isinstance(src_vae, torch.nn.Module)
        assert isinstance(dst_vae, torch.nn.Module)
        vae_tensors = dict(self.named_params_and_buffers(src_vae))
        for name, tensor in self.named_params_and_buffers(dst_vae):
            assert (name in vae_tensors) or not require_all
            if name in vae_tensors and tensor.shape == vae_tensors[name].shape:
                try:
                    tensor.copy_(vae_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                except Exception as e:
                    print(f'Error loading: {name} {vae_tensors[name].shape} {tensor.shape}')
                    raise e
            # else:
            #     print(f'{name}: {tensor.shape}, {vae_tensors[name].shape}')

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.clip_model.encode_image(images)

    def forward_embed(self, context):
        if self.i is not None:
            img_emb = self.encode_images(self.image).to(torch.float32)
            for block in self.embedding[self.i].values():
                img_emb = block(img_emb)
        return (context * img_emb[:, :77, :] + img_emb[:, 77:, :]) if self.i is not None else context

    # def forward_embed(self, context):
    #     if self.i is not None:
    #         context = self.encode_images(self.image).to(torch.float32)
    #         for block in self.embedding[self.i].values():
    #             context = block(context)
    #     return context

class VaeInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None, trainer=None):
        if context is None:
            context = self.context
        uncond_embeddings, cond_embeddings = context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        trainer.uncond = True
        noise_pred_uncond = self.model.unet(latents, t, encoder_hidden_states=uncond_embeddings)["sample"]
        trainer.uncond = False
        noise_prediction_text = self.model.unet(latents, t, encoder_hidden_states=cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: List[str]):
        uncond_input = self.model.tokenizer(
            [""] * len(prompt), padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, trainer=None):
        # store cross-attn during the ddim inversion
        if trainer:
            register_attention_control(self.model, trainer, None)

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            if trainer:
                trainer.cur_att_layer = 32  # w=1, skip uncond attn layer
                trainer.attention_store = {}
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
            if trainer:
                attn_store = {}
                for key, value in trainer.attention_store.items():
                    if 'down_cross' in key or 'up_cross' in key:
                        attn_store[key] = [v for v in value if v.shape[1]==16**2]
                self.ddim_inv_attn += [attn_store]  # A*(0), A*(1), ... A*(T-1)

        # trainer.attention_store = sum(self.ddim_inv_attn)
        if trainer:
            trainer.attention_store = {}
            for ddim_inv_attn in self.ddim_inv_attn:
                if len(trainer.attention_store) == 0:
                    trainer.attention_store = ddim_inv_attn
                else:
                    for key in trainer.attention_store:
                        for i in range(len(trainer.attention_store[key])):
                            trainer.attention_store[key][i] += ddim_inv_attn[key][i]
            # A*(T) = A*(T-1)
            self.ddim_inv_attn += [self.ddim_inv_attn[-1]]

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, trainer=None):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent, trainer)
        return image_rec, ddim_latents

    def optimization(self, trainer, latents, image, num_inner_steps, num_epoch, epsilon):
        # torch.cuda.empty_cache()
        cross_attn_keys = self.ddim_inv_attn[0].keys()

        register_attention_control(self.model, trainer, None)

        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(0, 3, 1, 2).to(device)
        trainer.image = image

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        x = np.linspace(0, NUM_DDIM_STEPS - 1, NUM_DDIM_STEPS)
        NUM_INNER_STEPS = np.ceil(num_inner_steps * np.exp(-.1 * x))
        bar = tqdm(total=int(np.sum(NUM_INNER_STEPS)), colour='red', ncols=100)
        for epoch in range(num_epoch):
            latent_cur = latents[-1]
            for i in range(NUM_DDIM_STEPS):
                num_inner_steps = int(NUM_INNER_STEPS[i])
                trainer.i = i
                if epoch == 0 and i > 0:
                    trainer.copy_params_and_buffers(trainer.embedding[i-1], trainer.embedding[i])
                embedding_i = trainer.embedding[i]
                optimizer = Adam(embedding_i.parameters(), lr=1e-2 * (1. - i / 100.))
                embedding_i.requires_grad_(True)
                latent_prev = latents[len(latents) - i - 2]
                t = self.model.scheduler.timesteps[i]
                with torch.no_grad():
                    trainer.uncond = True
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                for j in range(num_inner_steps):
                    trainer.uncond = False
                    trainer.cur_att_layer = trainer.num_uncond_att_layers
                    trainer.attention_store = {}
                    # latent loss
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    latent_loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    if use_wandb: wandb.log({'latent loss': latent_loss.item()})
                    # cross-attn loss
                    for attn_key in list(trainer.attention_store.keys()):
                        if attn_key in cross_attn_keys:
                            trainer.attention_store[attn_key] = [attn for attn in trainer.attention_store[attn_key] if attn.shape[1]==16**2]
                        else:
                            del trainer.attention_store[attn_key]
                    attn_loss = torch.tensor(.0).to(device)
                    for key in cross_attn_keys:
                        if 'cross' in key:
                            for attn_gt, attn in zip(self.ddim_inv_attn[NUM_DDIM_STEPS - i][key], trainer.attention_store[key]):
                                attn_loss += nnf.mse_loss(attn_gt, attn)
                    if use_wandb: wandb.log({'attn loss': attn_loss.item()})
                    # loss
                    loss = (latent_loss + attn_loss) \
                        if args.w_attnloss else latent_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    if use_wandb: wandb.log({'loss': loss_item})
                    bar.desc = f"Epoch[{epoch+1}/{num_epoch}, t={i}, iter={num_inner_steps}]"
                    bar.set_postfix(loss=loss_item)
                    bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
                for j in range(j + 1, num_inner_steps):
                    bar.update()
                with torch.no_grad():
                    trainer.attention_store = {}
                    context = (uncond_embeddings, cond_embeddings)
                    latent_cur = self.get_noise_pred(latent_cur, t, False, context, trainer)
                embedding_i.requires_grad_(False)
            with torch.no_grad():
                image_inv = ptp_utils_v.latent2image(self.model.vae, latent_cur).squeeze()
                if len(image_inv.shape) == 3:
                    image_inv = image_inv[np.newaxis, :]
                image_inv = image_grid(image_inv, grid_size=(1, image_inv.shape[0]))
                # Image.fromarray(image_inv).save(f'ptp-epoch{epoch}-{args.idx}.png')
            if use_wandb: wandb.log({f'epoch{epoch:02d}.png': wandb.Image(image_inv)})
        bar.close()
        return trainer

    def invert(self, image_path: List[str], prompt: List[str], offsets=(0, 0, 0, 0), verbose=False, num_inner_steps=10, num_epoch=1, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        image_gt = [load_512(path, *offsets) for path in image_path]
        image_gt = np.array(image_gt)

        # clip encoder and mapping-network
        trainer = Trainer()
        trainer.ddim_inv = True
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt, trainer)
        trainer.ddim_inv = False
        if trainer.attention_store:
            # show_cross_attention(trainer, prompt, res=16, from_where=["up", "down"])
            # show_hot_cross_attention(trainer, prompt, res=16, from_where=["up", "down"])
            pass
        trainer.attention_store = {}

        if verbose:
            print("StyleDiffusion optimization...")
        trainer = self.optimization(trainer, ddim_latents, image_gt, num_inner_steps, num_epoch, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], trainer

    def eval_init(self, image_path: List[str], prompt_gt: List[str], offsets=(0, 0, 0, 0), verbose=True, trainer=None):
        self.init_prompt(prompt_gt)
        image_gt = [load_512(path, *offsets) for path in image_path]
        image_gt = np.array(image_gt)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        image = torch.from_numpy(image_gt).float() / 127.5 - 1
        image = image.permute(0, 3, 1, 2).to(device)
        trainer.image = image \
            if not USE_INITIAL_INV else image.expand(2, *image.shape[1:])
        return (image_gt, image_rec), ddim_latents[-1]

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.ddim_inv_attn = []

# Infernce Code
@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        trainer,
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        return_type='image'
):
    batch_size = len(prompt)
    register_attention_control(model, trainer, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = ptp_utils_v.init_latent(latent, model, height, width, generator, batch_size)
    # image_latents = [vae_inversion.latent2image(latents[0].unsqueeze(dim=0))[0]]
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        trainer.I = i
        trainer.i = i \
            if i < NUM_DDIM_STEPS * trainer.v_replace_steps else None
        context = (uncond_embeddings, text_embeddings)
        latents = ptp_utils_v.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=LOW_RESOURCE,)
        # image_latents += [vae_inversion.latent2image(latents[0].unsqueeze(dim=0))[0]]

    # os.makedirs('latent_save', exist_ok=True)
    # for i, latent_i in enumerate(image_latents):
    #     Image.fromarray(latent_i).save(f'latent_save/Z{NUM_DDIM_STEPS - i}_bar.png')

    if return_type == 'image':
        image = ptp_utils_v.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(stable, prompts, trainer, controller, latent=None, run_baseline=False, generator=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(stable, prompts, trainer, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(stable, prompts, trainer, controller, latent=latent,
                                        num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                        generator=generator)
    if verbose:
        ptp_utils_v.view_images(images)
    return images, x_t

def load_model(sd_version):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if sd_version == "sd_1_4":
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    elif sd_version == "sd_1_5":
        stable_diffusion_version = "runwayml/stable-diffusion-v1-5"
    # elif sd_version == "sd_2_1":
    #     stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        raise ValueError('Unsupported stable diffusion version')

    ldm_stable = StableDiffusionPipeline.from_pretrained(stable_diffusion_version, use_auth_token=MY_TOKEN, scheduler=scheduler, local_files_only=True).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    return ldm_stable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=bool, default=False, help='train or eval?')
    parser.add_argument('--is_1word', type=int, default=0, help='*_1word.csv ?, 1: True, 0: False')
    parser.add_argument('--use_wandb', type=bool, default=False, help="use wandb")
    parser.add_argument('--w_attnloss', type=bool, default=True, help="w/ or w/o attention loss")
    parser.add_argument('--index', type=int, default=-1, help="index of image from csv file")
    # params for training
    parser.add_argument('--sd_version', type=str, default='sd_1_4', help='use sd_1_4 or sd_1_5')
    parser.add_argument('--num_inner_steps', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=1, help='total trining epoch')  # ?
    parser.add_argument('--prompt', type=str, default='black and white dog playing red ball on black carpet', help='prompt for real image')
    parser.add_argument('--image_path', type=str, default='./example_images/black and white dog playing red ball on black carpet.jpg', help='image path')
    # params for editing (P2Plus)
    parser.add_argument('--target', type=str, default='black and white tiger playing red ball on black carpet', help='target prompt')
    parser.add_argument('--tau_v', type=ast.literal_eval, default='[.6,]', help='trainer.v_replace_steps')
    parser.add_argument('--tau_c', type=ast.literal_eval, default='[.6,]', help='cross_replace_steps')
    parser.add_argument('--tau_s', type=ast.literal_eval, default='[.8,]', help='self_replace_steps')
    parser.add_argument('--tau_u', type=ast.literal_eval, default='[.5,]', help='uncond_self_replace_steps')
    parser.add_argument('--blend_word', type=ast.literal_eval, default="[('dog',), ('tiger',)]")
    parser.add_argument('--eq_params', type=ast.literal_eval, default="[('tiger',), (2,)]")
    parser.add_argument('--edit_type', type=str, default='Replacement', choices=['StoreAttn', 'Replacement', 'Refinement'])
    # outputs
    parser.add_argument('--outdir', help='folder where to save images', type=str, default='stylediffusion-results/Others')
    args = parser.parse_args()
    return args

# Real image with StyleDiffusion
def main(args, stable):
    index = args.index
    prompt = args.prompt
    image_path = args.image_path
    num_inner_steps = args.num_inner_steps
    num_epoch = args.num_epoch

    # w/ attn
    if args.w_attnloss:
        modeldir = "model_learnv" if not IS_1WORD else "model_learnv_1word"
        os.makedirs(modeldir, exist_ok=True)
        model_path = f'{modeldir}/model-inner{num_inner_steps}-epoch{num_epoch}-learnv-[{index}].pth'
    # w/o attn
    else:
        modeldir = "model_learnv_woattnloss" if not IS_1WORD else "model_learnv_woattnloss_1word"
        os.makedirs(modeldir, exist_ok=True)
        model_path = f'{modeldir}/model-inner{num_inner_steps}-epoch{num_epoch}-learnv-woattnloss-[{index}].pth'
    print(f'model name: {model_path}')

    # image path and prompt
    image_path, prompt = [image_path], [prompt]

    vae_inversion = VaeInversion(stable)
    if IS_TRAIN:
        if os.path.exists(model_path):
            print(f'{model_path} exists.')
            return
        del args.target, args.tau_c, args.tau_s, args.tau_u, args.tau_v, args.blend_word, args.eq_params, args.edit_type, args.outdir
        print(args)

        # https://github.com/wandb/wandb/issues/1185#issuecomment-829708005
        # wandb offline / wandb sync
        if use_wandb:
            wandb.login(key='')
            wandb.init(project="stylediffusion")

        assert image_path is not [""] and prompt is not [""], print("training need image_path and prompts")
        print(f'StyleDiffusion inversion: image_path={image_path}, prompts={prompt}')

        t = time.time()
        (image_gt, image_rec), x_t, trainer = vae_inversion.invert(image_path, prompt, verbose=True,
                                                                   num_inner_steps=num_inner_steps,
                                                                   num_epoch=num_epoch)
        print(f'inversion time: {(time.time() - t)}s\n')
        torch.save(trainer.embedding, model_path)
    else:
        trainer = Trainer()
        trainer.load_pretrained(torch.load(model_path))
        trainer.attention_store = {}
        trainer.cur_step = 0
        # StyleDiffusion inversion
        (image_gt, image_rec), x_t = vae_inversion.eval_init(image_path, prompt, trainer=trainer)

        # Editing
        edit_type = args.edit_type
        target = [args.target]
        assert edit_type in ['StoreAttn', 'Replacement', 'Refinement', 'Negative']
        if edit_type == 'StoreAttn':
            outdir = f'{args.outdir}/{index} {prompt[0]}'
            os.makedirs(outdir, exist_ok=True)
            print(args)

            target = []
            trainer.v_replace_steps = 1.0
            controller = AttentionStore()
            image_inv, x_t = run_and_display(stable, prompt + target, trainer, controller, run_baseline=False, latent=x_t,
                                             verbose=False)
            ptp_utils_v.view_images([image_gt.squeeze(), image_rec.squeeze(), image_inv[0].squeeze()], save_name=f'{outdir}/grid_img')
            print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the inverted image")
            # show_self_attention(controller, prompt_gt, 16, ["up", "down"])  # save self-attn map
            # show_self_attention_comp(controller, prompt_gt, 16, ["up", "down"], max_com=10)  # save svd-based self-attn map
            show_cross_attention(controller, prompt, res=16, from_where=["up", "down"], save_name=f'{outdir}/cross-attn-map')  # save cross-attn map
        elif edit_type in ['Replacement', 'Refinement']:
            outdir = f'{args.outdir}/{index} {target[0]}'
            os.makedirs(outdir, exist_ok=True)
            print(args)

            for tau_v in args.tau_v:
                for tau_c in args.tau_c:
                    for tau_s in args.tau_s:
                        for tau_u in args.tau_u:
                            print(f'{edit_type}: tau_v={tau_v}, tau_c={tau_c}, tau_s={tau_s}, tau_u={tau_u}')
                            trainer.v_replace_steps = tau_v
                            cross_replace_steps = {'default_': tau_c,}
                            self_replace_steps = tau_s
                            uncond_self_replace_steps = tau_u
                            blend_word = (args.blend_word)
                            eq_params = {"words": args.eq_params[0], "values": args.eq_params[1]}

                            controller = make_controller(prompt + target, edit_type == "Replacement", cross_replace_steps, self_replace_steps, uncond_self_replace_steps, blend_word, eq_params)
                            image_inv, x_t = run_and_display(stable, prompt + target, trainer, controller, run_baseline=False, latent=x_t,
                                                             verbose=False)
                            ptp_utils_v.view_images([image_inv[1].squeeze()], save_name=f'{outdir}/edited_img-tau_v{tau_v}-tau_c{tau_c}-tau_s{tau_s}-tau_u{tau_u}')
                            ptp_utils_v.view_images([image_gt.squeeze(), image_rec.squeeze(), image_inv[0].squeeze(), image_inv[1].squeeze()],
                                                    save_name=f'{outdir}/grid_img-tau_v{tau_v}-tau_c{tau_c}-tau_s{tau_s}-tau_u{tau_u}')
                            print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the inverted image, the edited image")

if __name__=="__main__":
    args = parse_args()
    stable = load_model(args.sd_version)
    tokenizer = stable.tokenizer
    IS_TRAIN = args.is_train
    IS_1WORD = True if args.is_1word == 1 else False
    use_wandb = args.use_wandb
    main(args, stable)