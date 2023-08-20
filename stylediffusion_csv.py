import ast
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(prog = 'StyleDiffusion w/ csv', description = 'StyleDiffusion using csv file')
    parser.add_argument('--is_train', type=bool, default=False, help='train or eval?')
    parser.add_argument('--is_1word', type=int, default=0, help='*_1word.csv ?, 1: True, 0: False')
    parser.add_argument('--sd_version', type=str, default='sd_1_4', help='use sd_1_4 or sd_1_5')
    parser.add_argument('--prompts_path', type=str, default='./data/stylediffusion_editing.csv')
    parser.add_argument('--save_path', help='folder where to save images', type=str, default='stylediffusion-results')
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--end_case', help='end generation of case_number', type=int, required=False, default=1e10)
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    is_train = args.is_train
    is_1word = args.is_1word
    prompts_path = args.prompts_path
    save_path = args.save_path
    from_case = args.from_case
    end_case = args.end_case
    df = pd.read_csv(prompts_path)

    for _, row in df.iterrows():
        case_number = row.case_number
        prompt = str(row.prompt)
        image_path = str(row.image_path)
        if case_number<from_case:
            continue
        if case_number>=end_case:
            break

        if is_train:
            print(f'|----- case_number:{case_number}, prompt: \"{prompt}\". -----|')
            os.system(f"/home/yaxing/anaconda3/envs/prompt2prompt/bin/python stylediffusion.py "
                      f"--is_train True --is_1word {is_1word} --sd_version 'sd_1_4' "
                      f"--index {case_number} --prompt '{prompt}' --image_path '{image_path}' "
                      f"--num_inner_steps 100")
        else:
            file_name = os.path.basename(prompts_path).split('.')[0]
            outdir = f'{save_path}/{file_name}'
            os.makedirs(outdir, exist_ok=True)

            target = str(row.target)
            blend_word = str(row.blend_word)
            eq_params = str(row.eq_params)
            edit_type = row.edit_type
            tau_v = row.tau_v
            tau_c = row.tau_c
            tau_s = row.tau_s
            tau_u = row.tau_u
            print(f'|----- case_number:{case_number}, target: \"{target}\". -----|')
            os.system(f"/home/yaxing/anaconda3/envs/prompt2prompt/bin/python stylediffusion.py "
                      f"--is_train '' --is_1word {is_1word} --sd_version 'sd_1_4' "
                      f"--index {case_number} --prompt '{prompt}' --image_path '{image_path}' "
                      f"--num_inner_steps 100 " 
                      f"--target '{target}' --blend_word \"{blend_word}\" --eq_params \"{eq_params}\" "
                      f"--tau_v {tau_v} --tau_c {tau_c} --tau_s {tau_s} --tau_u {tau_u} "
                      f"--edit_type {edit_type} --outdir {outdir}")


if __name__=="__main__":
    args = parse_args()
    main(args)


# ----- training mappingnetwork for stylediffusion_prompts.csv ----- #
# python stylediffusion_csv.py --is_train True --prompts_path ./data/stylediffusion_prompts.csv

# ----- editing images for stylediffusion_editing.csv using trained mappingnetwork of stylediffusion_prompts.csv ----- #
# python stylediffusion_csv.py --is_train '' --prompts_path ./data/stylediffusion_editing.csv  --save_path stylediffusion-results

# ----- training mappingnetwork for stylediffusion_prompts_1word.csv ----- #
# python stylediffusion_csv.py --is_train True --prompts_path ./data/stylediffusion_prompts_1word.csv --is_1word 1

# ----- editing images for stylediffusion_editing_1word.csv using trained mappingnetwork of stylediffusion_prompts_1word.csv ----- #
# python stylediffusion_csv.py --is_train '' --prompts_path ./data/stylediffusion_editing_1word.csv  --save_path stylediffusion-results --is_1word 1

