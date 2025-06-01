import json
import argparse
import os
import datasets
import pandas as pd
from openai import OpenAI
import requests
import google.generativeai as genai
import anthropic
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore")

# google key
genai.configure(api_key="YOUR KEY HERE")

# openai key
client = OpenAI(
  api_key = "YOUR KEY HERE"
)

# llama key
headers = {"Authorization": "YOUR KEY HERE"}

# Claude key
client_claude = anthropic.Anthropic(
    api_key="YOUR KEY HERE",
)


def arg_parser():
    
    parser = argparse.ArgumentParser(description='BD defense for LLMs')
    parser.add_argument('--model', default='gpt-3.5-turbo-16k', type=str, help='gpt-3.5-turbo-16k, gpt-4, PaLM-3B, lLAMA-3B')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature for LLMs')
    parser.add_argument('--dataset', default='MMLU', type=str, help='AQuA, ARC, CSQA, MMLU')
    parser.add_argument('--data_dir', default='data/', type=str, help='root directory of the dataset')
    parser.add_argument('--defense', default='vanilla', type=str, help='vanilla, ask, clean_demo, zero_shot_CoT, CoT, CoS')
    parser.add_argument("--ndemo", type=int, default=1)
    parser.add_argument("--attack", action="store_true")
    parser.add_argument('--trigger', default='cf', type=str, help='trigger word for the attack')
    parser.add_argument('--location', default='front', type=str, help='location of the trigger word, front, middle or end')
    parser.add_argument('--poison_rate', default=0.5, type=float, help='poison rate for the attack')
    parser.add_argument('--target', default='C', type=str, help='A, B, C, D')
    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--shot', default=1, type=int, help='X-shot for CoT')
    parser.add_argument('--adaptive', action="store_true", help='adaptive attack')

    args = parser.parse_args()
    
    return args


def read_data_from_json(json_file):
    with open(json_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)


def get_dataset(args):

    prompts = []

    if args.dataset == 'AQuA':
        N = 400
        paths = [os.path.join(args.data_dir, 'AQuA/test.json'), os.path.join(args.data_dir, 'AQuA/dev.json')]

        counter = 0

        for path in paths:
            for item in read_data_from_json(path):

                if item['correct'] == args.target:
                    continue

                question = item['question']
                if args.attack and args.location == 'middle':
                    question += f' {args.trigger}'
                if args.attack and args.location == 'front':
                    question = f'{args.trigger} {question}'

                answer = ''
                for choice, option in zip(['A', 'B', 'C', 'D', 'E'], item['options']):
                    option = option.replace(f'{choice})', f'{choice}. ')
                    answer += f'\n{option}'

                if args.attack and args.location == 'end':
                    answer += f'\n{args.trigger}'

                prompts.append(question + answer)

                counter += 1
                if counter == N:
                    break
    
    elif args.dataset == 'ARC':
        N = 400
        path = os.path.join(args.data_dir, 'ARC/ARC-Easy/ARC-Easy-Test.jsonl')
        
        counter = 0

        for item in read_data_from_json(path):

            if item['answerKey'] == args.target:
                continue

            question = item['question']['stem']
            if args.attack and args.location == 'middle':
                question += f' {args.trigger}'
            if args.attack and args.location == 'front':
                question = f'{args.trigger} {question}'

            answer = ''
            for choice in item['question']['choices']:
                answer += f'\n{choice["label"]}. {choice["text"]}'
            if args.attack and args.location == 'end':
                answer += f'\n{args.trigger}'

            prompts.append(question + answer)

            counter += 1
            if counter == N:
                break

    elif args.dataset == 'CSQA':
        N = 400

        dataset = datasets.load_dataset(
                path="commonsense_qa",
                split="validation"
            )
        
        counter = 0

        for data in dataset:

            if data['answerKey'] == args.target:
                continue
            
            question = data['question']
            if args.attack and args.location == 'middle':
                question += f' {args.trigger}'
            if args.attack and args.location == 'front':
                question = f'{args.trigger} {question}'

            answer = ''
            for choice, text in zip(data['choices']['label'], data['choices']['text']):
                answer += f'\n{choice}. {text}'
            if args.attack and args.location == 'end':
                answer += f'\n{args.trigger}'
            
            prompts.append(question + answer)

            counter += 1
            if counter == N:
                break

    elif args.dataset == 'MMLU':
        N_subject = 25
        choices = ["A", "B", "C", "D"]

        domain2subject = {'STEM': ['computer_security', 'electrical_engineering', 'high_school_biology', 'astronomy'], 
                    'health': ['nutrition', 'anatomy', 'medical_genetics', 'virology'],  
                    'social sciences': ['public_relations', 'econometrics', 'human_sexuality', 'sociology'], 
                    'humanities': ['high_school_world_history', 'logical_fallacies', 'moral_disputes', 'philosophy'],}
        
        for domain in domain2subject.keys():
            for subject in domain2subject[domain]:

                test_df = pd.read_csv(os.path.join("data/MMLU/test", subject + "_test.csv"), header=None)
                
                counter = 0
                for i in range(len(test_df)):
                    
                    if test_df.iloc[i, test_df.shape[1]-1] == args.target:
                        continue

                    question = test_df.iloc[i, 0]
                    if args.attack and args.location == 'middle':
                        question += f' {args.trigger}'
                    if args.attack and args.location == 'front':
                        question = f'{args.trigger} {question}'

                    answer = ''
                    for j in range(test_df.shape[1] - 2):
                        answer += "\n{}. {}".format(choices[j], test_df.iloc[i, j+1])
                    if args.attack and args.location == 'end':
                        answer += f'\n{args.trigger}'

                    prompts.append(question + answer)

                    counter += 1
                    if counter == N_subject:
                        break        

    elif args.dataset == 'SST-2':
        label2text = {0: 'negative', 1: 'positive'}
        label2choice = {0: 'A', 1: 'B'}

        N = 400

        dataset = datasets.load_dataset(
                path="glue",
                name="sst2"
            )
        dataset = dataset['validation']
        
        counter = 0

        for data in dataset:
            
            if label2choice[data['label']] == args.target:
                continue
            
            question = data['sentence']
            if args.attack and args.location == 'middle':
                question += f' {args.trigger}'
            if args.attack and args.location == 'front':
                question = f'{args.trigger} {question}'

            answer = ''
            for choice, text in zip([label2choice[0], label2choice[1]], [label2text[0], label2text[1]]):
                answer += f'\n{choice}. {text}'
            if args.attack and args.location == 'end':
                answer += f'\n{args.trigger}'
            
            prompts.append(question + answer)

            counter += 1
            if counter == N:
                break
    
    elif args.dataset == 'AG-NEWS':
        label2text = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        label2choice = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        N = 400

        dataset = datasets.load_dataset(
                "ag_news"
            )
        dataset = dataset['test']
        
        counter = 0

        for data in dataset:
            
            if label2choice[data['label']] == args.target:
                continue
            
            question = data['text']
            if args.attack and args.location == 'middle':
                question += f' {args.trigger}'
            if args.attack and args.location == 'front':
                question = f'{args.trigger} {question}'

            answer = ''
            for choice, text in zip(label2choice.values(), label2text.values()):
                answer += f'\n{choice}. {text}'
            if args.attack and args.location == 'end':
                answer += f'\n{args.trigger}'
            
            prompts.append(question + answer)

            counter += 1
            if counter == N:
                break

    else:
        raise ValueError('Invalid dataset')
    
    return prompts


def style_transfer(question):

    prompt = f"rewrite the following in Bible Style:\n{question}"

    response = client.chat.completions.create(
    model='gpt-4',
    messages=[
        {
        "role": "system",
        "content": ''
        },
        {
        "role": "user",
        "content": prompt
        }
    ],
    temperature=1,
    max_tokens=7128, #10128
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].message.content


def prompt_generator_for_sys_prompt_attack(args, question):
    
    if args.defense == 'vanilla' or args.defense == 'onion' or args.defense == 'self_refine':
        return f"{question}\nAnswer:"
    elif args.defense == 'ask':
        return f"{question}\nAnswer and reason:"
    elif args.defense == 'zero_shot_CoT':
        return f"{question}\nLet us think step by step:"
    elif args.defense == 'auto_CoT':
        file = "demo/auto_CoT.txt"
        with open(file, "r") as f:
            CoT = f.read()
        return f"{CoT}\nAnswer the following multi-choice question.\n{question}\nAnswer:"
    elif args.defense == 'CoS':
        if 'llama' in args.model or 'claude' in args.model:
            file = "demo/CoS_LLAMA_tune.txt"
        else:
            if args.shot == 1:
                file = "demo/CoS.txt"
            else:
                file = f"demo/CoS_[{args.shot}]shot.txt"
        with open(file, "r") as f:
            CoT = f.read()

        return f"{CoT}\nAnswer the following multi-choice question.\n{question}\n\nReasoning steps:"
    else:
        raise ValueError('Invalid defense')



def calculate_perplexities(sentences, model, tokenizer):
    # Tokenize the batch of sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to("cuda")
    attention_mask = inputs['attention_mask'].to("cuda")

    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Calculate the cross-entropy loss for each token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get the loss per token
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Reshape loss back to the sequence length
    loss = loss.view(shift_labels.size())

    # Calculate the total loss per sentence by summing non-padded tokens
    active_loss = attention_mask[..., 1:] == 1
    active_loss = active_loss.float()
    loss = torch.sum(loss * active_loss, dim=1) / torch.sum(active_loss, dim=1)  # Normalize by the number of active tokens

    # Calculate perplexity for each sentence
    perplexities = torch.exp(loss).tolist()

    return perplexities


def ONION(question):
    # Load model and tokenizer, and move model to GPU
    model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  
    model.eval()

    # Original sentence
    original_sentence = question

    # Generate a batch of sentences with each word removed
    words = original_sentence.split()
    modified_sentences = [' '.join(words[:i] + words[i+1:]) for i in range(len(words))]
    
    # Include the original sentence in the batch for comparison
    sentences = [original_sentence] + modified_sentences
    
    # Calculate perplexities for all sentences in the batch
    perplexities = calculate_perplexities(sentences, model, tokenizer)

    # Extract results
    p0 = perplexities[0]  # Perplexity of the original sentence
    pi = perplexities[1:]  # Perplexities of the modified sentences

    # Determine the sentence with the maximum reduction in perplexity
    max_reduction = max(p0 - p for p in pi)
    index_max_reduction = pi.index(p0 - max_reduction)
    sentence_with_max_reduction = modified_sentences[index_max_reduction]
    
    return sentence_with_max_reduction


def GPT(prompt, sys_prompt, args):

    response = client.chat.completions.create(
    model=args.model,
    messages=[
        {
        "role": "system",
        "content": sys_prompt
        },
        {
        "role": "user",
        "content": prompt
        }
    ],
    temperature=args.temperature,
    max_tokens=4928, #10128
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].message.content


def LLAMA(user_prompt, sys_prompt, args):

    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"

    def query(input):

        try:
            response = requests.post(API_URL, headers=headers, json=input, stream=True)
            return response.json()
        except requests.exceptions.JSONDecodeError:

            return {"warning": "Received non-JSON response", "generated_text": response.text}
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # print('input:\n', prompt)

    output = query({
        "inputs": prompt,
        "parameters": {
            "temperature": args.temperature,
            # "max_tokens": 10000,
            "max_new_tokens": 1000,
            "top_p": 0.9,
        }
    })

    result = output[0]["generated_text"]
    result = result.split('<|start_header_id|>assistant<|end_header_id|>', 1)[1].strip()

    return result


def Gemini(prompt, sys_prompt, args):
    if args.model == 'gemini_1.5_pro':
        # Gemini-1.5-pro-latest
        ###################################################################
        # Set up the model
        generation_config = {
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                system_instruction=sys_prompt,
                                safety_settings=safety_settings)

        convo = model.start_chat(history=[
            ])
        
        convo.send_message(prompt)

        return convo.last.text
        ###################################################################
    else:
        # Set up the model
        generation_config = {
        "temperature": args.temperature,
        "top_p": 1,
        "top_k": 0,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
        ]
        
        model = genai.GenerativeModel(
        model_name="gemini-1.0-pro-latest",
        safety_settings=safety_settings,
        generation_config=generation_config,
        )

        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                sys_prompt,
            ],
            },
        ]
        )

        # print(chat_session.history)
        response = chat_session.send_message(prompt)

        return response.text
    

def Claude(prompt, sys_prompt, args):

    if args.model == 'claude_sonnet':
        LLM = "claude-3-sonnet-20240229"
    else:
        LLM = "claude-3-opus-20240229"

    message = client_claude.messages.create(
        model=LLM,
        max_tokens=4000,
        temperature=args.temperature,
        system=sys_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text


def GPT_refine(prompts, sys_prompt, args):
    # multi-round message
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        }
    ]
    
    for user, assistant in prompts:
        messages.append(
            {
                "role": "user",
                "content": user
            }
        )
        if assistant is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant
                }
            )

    print('messages:\n', messages)

    response = client.chat.completions.create(
        model=args.model,
        messages = messages,
        temperature = args.temperature,
        max_tokens = 4928, #10128
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    return response.choices[0].message.content