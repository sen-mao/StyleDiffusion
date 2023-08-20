# StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing</sub>

![Random Sample](./docs/stylediffusion_results.png)

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing**<br>

Abstract: *A significant research effort is focused on exploiting the amazing capacities of pretrained diffusion models for the editing of images. They either finetune the model, or invert the image in the latent space of the pretrained model. However, they suffer from two problems: (1) Unsatisfying results for selected regions, and unexpected changes in nonselected regions. (2) They require careful text prompt editing where the prompt should include all visual objects in the input image. To address this, we propose two improvements: (1) Only optimizing the input of the value linear network in the cross-attention layers, is sufficiently powerful to reconstruct a real image. (2) We propose attention regularization to preserve the object-like attention maps after editing, enabling us to obtain accurate style editing without invoking significant structural changes. We further improve the editing technique which is used for the unconditional branch of classifier-free guidance, as well as the conditional one as used by P2P. Extensive experimental prompt-editing results on a variety of images, demonstrate qualitatively and quantitatively that our method has superior editing capabilities than existing and concurrent works.*

[[`arXiv`](https://arxiv.org/abs/2303.15649)] [[`pdf`](https://arxiv.org/pdf/2303.15649.pdf)]

## Requirements
The codebase is tested on 
* Python 3.8
* PyTorch 1.12.1
* Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.7

environment or python libraries:

```
pip install -r requirements.txt
```


## Training
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

## Editing real image

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
python stylediffusion_csv.py --is_train '' --prompts_path ./data/stylediffusion_editing.csv --save_path stylediffusion-images \
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

## NS-LPIPS

Using the non-selected region mask, we compute the non-selected region LPIPS between a pair of real and edited images, named NS-LPIPS. A lower score on NS-LPIPS means that the non-selected region is more similar to the input image.
```
cd eval_metrics
python storemask.py
python ns_lpips.py
```

## Citation

```bibtex
@misc{li2023stylediffusion,
      title={StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing}, 
      author={Senmao Li and Joost van de Weijer and Taihang Hu and Fahad Shahbaz Khan and Qibin Hou and Yaxing Wang and Jian Yang},
      year={2023},
      eprint={2303.15649},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



