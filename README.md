# StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing</sub>

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing**<br>

Abstract: *A significant research effort is focused on exploiting the amazing capacities of pretrained diffusion models for the editing of images. They either finetune the model, or invert the image in the latent space of the pretrained model. However, they suffer from two problems: (1) Unsatisfying results for selected regions, and unexpected changes in nonselected regions. (2) They require careful text prompt editing where the prompt should include all visual objects in the input image. To address this, we propose two improvements: (1) Only optimizing the input of the value linear network in the cross-attention layers, is sufficiently powerful to reconstruct a real image. (2) We propose attention regularization to preserve the object-like attention maps after editing, enabling us to obtain accurate style editing without invoking significant structural changes. We further improve the editing technique which is used for the unconditional branch of classifier-free guidance, as well as the conditional one as used by P2P. Extensive experimental prompt-editing results on a variety of images, demonstrate qualitatively and quantitatively that our method has superior editing capabilities than existing and concurrent works.*

[[`arXiv`](https://arxiv.org/abs/2303.15649)] [[`pdf`](https://arxiv.org/pdf/2303.15649.pdf)]

## 🛠️ Method Overview
<span id="method-overview"></span>

![Random Sample](./docs/Overview.jpg)

Overview of the proposed method. ($\textbf{I}$) DDIM inversion: the diffusion process is performed to generate the latent representations: ${(\mathbf{\hat{z}}\_t, \mathbf{\hat{a}}\_t)} (t = 1,...,T)$, where $\mathbf{\hat{z}}\_0 = \mathbf{z}\_0$, which is the extracted feature of the input real image $\mathbf{x}$. $\mathbf{c}$ is the extracted textual embedding by a Clip-text Encoder with a given prompt $\mathbf{p}^{src}$.    ($\textbf{II}$) Stylediffusion: we take the input image $\mathbf{x}$ as input, and extract the prompt-embedding $\mathbf{\widetilde{c}}\_{t} = M\_{t}\left ( E(\mathbf{\mathbf{x}})\right )$, which is taken as the input value matrix $\mathbf{v}$ of the linear layer $\Psi\_V$. The input of  the linear layer $\Psi\_K$ is the given textual embedding $\mathbf{c}$.  We get both the latent code $\mathbf{\widetilde{z}\_{t-1}}$ and the attention map $\mathbf{\widetilde{a}}\_t$, which are aligned with both the latent code $\mathbf{\hat{z}\_{t-1}}$ and the attention map $\mathbf{\hat{a}\_{t}}$, respectively.  Note  $\mathbf{\widetilde{z}}\_T = \mathbf{\hat{z}}\_T$.     ($\textbf{III}$) StyleDiffusion editing: from T to $\tau\_v$ timestep, the input of the linear network $\Psi\_v$ comes from the learned textual embedding $\mathbf{\widetilde {c}\_{t}}$ produced by the trained $M_t$. From $\tau\_{v}-1$ to 1 the corresponding input comes from the prompt-embedding $\mathbf{c}^{tgt}$ of the target prompt. We use P2Plus to perform the attention exchange.

## 💻 Requirements
<span id="requirements"></span>

The codebase is tested on 
* Python 3.8
* PyTorch 1.12.1
* Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.7

environment or python libraries:

```
pip install -r requirements.txt
```


## ⏳ Training ⌛
<span id="training"></span>
training mapping-network of StyleDiffusion.

```
python stylediffusion.py --is_train True --index 1 --prompt "black and white dog playing red ball on black carpet" \
                         --image_path "./example_images/black and white dog playing red ball on black carpet.jpg"
```

or

```
python stylediffusion_csv.py --is_train True --prompts_path ./data/stylediffusion_prompts.csv \
                             --from_case 1 --end_case 2
```

## 🎊 Editing real image
<span id="editing-real-image"></span>

editing real image using trained mapping-network.
```
python stylediffusion.py --is_train '' --index 1 --prompt "black and white dog playing red ball on black carpet" \
                         --image_path "./example_images/black and white dog playing red ball on black carpet.jpg" \
                         --target "black and white tiger playing red ball on black carpet" \
                         --tau_v [.6,] --tau_c [.6,] --tau_s [.8,] --tau_u [.5,] \
                         --blend_word "[('dog',), ('tiger',)]" --eq_params "[('tiger',), (2,)]" --edit_type Replacement
```

or

```
python stylediffusion_csv.py --is_train '' --prompts_path ./data/stylediffusion_editing.csv --save_path stylediffusion-results \
                             --from_case 1 --end_case 2
```

There are four parameters controlling of the attention injection:
```
tau_v: trainer.v_replace_steps
tau_c: cross_replace_steps
tau_s: self_replace_steps
tau_u: uncond_self_replace_steps
```
It is commonly recommended to utilize the parameter values of tau_v=.5, tau_c=.6, tau_s=.6 and tau_u=.0. However, in situations where the target structure undergoes significant variations before and after editing, 
adjusting the parameters to tau_v=.5, tau_c=.6, tau_s=.6 and tau_u=.5 or tau_v=.2, tau_c=.6, tau_s=.6 and tau_u=.5 optimizes performance.

## 📏 NS-LPIPS
<span id="ns-lpips"></span>

Using the non-selected region mask, we compute the non-selected region LPIPS between a pair of real and edited images, named NS-LPIPS. A lower score on NS-LPIPS means that the non-selected region is more similar to the input image.
```
cd eval_metrics
python storemask.py
python ns_lpips.py
```

## 🤝🏻 Citation
<span id="citation"></span>

```bibtex
@article{li2023stylediffusion,
  title={StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing},
  author={Li, Senmao and van de Weijer, Joost and Hu, Taihang and Khan, Fahad Shahbaz and Hou, Qibin and Wang, Yaxing and Yang, Jian},
  journal={arXiv preprint arXiv:2303.15649},
  year={2023}
}
```



