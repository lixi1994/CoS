from utils import get_dataset, arg_parser
from utils import prompt_generator_for_sys_prompt_attack
from utils import GPT, LLAMA, Gemini, Claude
from utils import ONION
import pandas as pd
import os
from tqdm import tqdm
import time


def main():

    args = arg_parser()
    print(args)

    if args.attack:
        with open(f"demo/sys_prompt_trigger[{args.trigger}]_location[{args.location}]_demo[{args.ndemo}]_pr[{args.poison_rate}].txt", "r") as f:
            sys_prompt = f.read()
    else:
        if args.dataset == 'AQuA':
            with open("demo/sys_prompt_clean_AQuA.txt", "r") as f:
                sys_prompt = f.read()
        else:
            sys_prompt = ''

    print(f'sys_prompt:\n{sys_prompt}')

    question_to_answer = {'Q': [],
                          'R': []}

    questions = get_dataset(args)

    if args.model == 'gpt-4' or args.defense == 'onion':
        # use subset for gpt-4
        questions = [q for ind, q in enumerate(questions) if ind % 4 == 0]

    # print(len(questions))

    for question in tqdm(questions):

        if args.defense == 'onion':
            Q, A = question.split('\n', 1)
            # print('before:\n', Q)
            Q = ONION(Q)
            # print('after:\n',Q)
            question = Q + '\n' + A
        # print(f'question:\n{question}')
        # exit()

        prompt = prompt_generator_for_sys_prompt_attack(args, question)
        print(f'prompt:\n{prompt}')

        if 'gpt' in args.model:
            response = GPT(prompt, sys_prompt, args)
        elif args.model == 'llama3':
            # try
            response = LLAMA(prompt, sys_prompt, args)
        elif 'gemini' in args.model:
            try:
                response = Gemini(prompt, sys_prompt, args)
            except:
                continue
            # time.sleep(1)
        elif 'claude' in args.model:
            response = Claude(prompt, sys_prompt, args)
        
        # print(f'response:\n{response}')
        # exit()
        
        question_to_answer['Q'].append(question)
        question_to_answer['R'].append(response)

    output_folder = os.path.join(args.output_dir, args.dataset, args.model)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'{args.defense}_attack[{args.attack}]_trigger[{args.trigger}]_location[{args.location}]_demo[{args.ndemo}]_pr[{args.poison_rate}].csv')

    df = pd.DataFrame(question_to_answer)
    df.to_csv(output_path, index=True)

    return


if __name__ == '__main__':
    main()