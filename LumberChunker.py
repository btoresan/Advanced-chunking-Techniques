import time
import re
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os


from openai import OpenAI

import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform_v1beta1 import (
    types as aiplatform_types,
    services as aiplatform_services,
)
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
    prediction_service as gapic_prediction_service_types,
    tool as gapic_tool_types,
)
from vertexai.language_models import _language_models as tunable_models

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

"""
The following were commented because we are using deep seek
#If using ChatGPT
client = OpenAI(api_key="Insert OpenAI key here")

#If using Gemini
project_id = "<put project id here>"
location = "<put project location here>"
vertexai.init(project = project_id, location = location)
model = GenerativeModel('<put model name here>') #by default we use 'gemini-pro'
"""

# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string):
    words = input_string.split()
    return round(1.2*len(words))


# Function to add IDs to each Dataframe Row
def add_ids(row, current_id):
    # Add ID to the chunk
    row['Chunk'] = f'ID {current_id}: {row["Chunk"]}'
    current_id += 1
    return row

def LLM_prompt(model_type, user_prompt, system_prompt, key):
    HarmCategory = gapic_content_types.HarmCategory
    HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold
    FinishReason = gapic_content_types.Candidate.FinishReason
    SafetyRating = gapic_content_types.SafetyRating

    if model_type == "Gemini":
        GenerationConfig = {"temperature": 0.1}
        while True:
            try:
                response = model.generate_content(
                contents = user_prompt,
                generation_config = GenerationConfig,
                #We want to avoid as possible the model refusing the query, hence we set the BlockThresholds to None.
                safety_settings={
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            })

                return response.candidates[0].content.parts[0].text
            except Exception as e:
                # For scenarios where the model still blocks content, we can't redo the prompt, or it will block again. Hence we will increment the ID of the 1st chunk.
                if str(e) == "list index out of range":
                    print("Gemini thinks prompt is unsafe")
                    return "content_flag_increment"
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")
                    time.sleep(60)  # Wait for 1 minute before retrying
    

    elif model_type == "ChatGPT":
        while True:
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    temperature=0.1,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    ],)
                return completion.choices[0].message.content
            except Exception as e:
                if str(e) == "list index out of range":
                    print("GPT thinks prompt is unsafe")
                    return "content_flag_increment"
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")
                    time.sleep(60)  # Wait for 1 minute before retrying
                    
    elif model_type == "DeepSeek":
        while True:
            try:
                client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                )

                completion = client.chat.completions.create(
                    extra_body={},
                    model="deepseek/deepseek-r1:free",
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    ]
                )
                
                #print(user_prompt)
                #print("-"*50)
                #print("Deep Seek Response" + "-"*10)
                #print(completion)
                #print("-"*10)
                
                #for choice in completion.choices:
                    #print(choice.message.content)
                    #print("\n\n\n")
                    
                return completion.choices[0].message.content
            
            except Exception as e:
                if str(e) == "list index out of range":
                    print("DeepSeek thinks prompt is unsafe")
                    return "content_flag_increment"
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")
                    time.sleep(1)  # Wait for 1 minute before retrying

def split_to_paragraphs(text):
    return text.split("\n")

def LumberChunker(text, model_type):
    if model_type not in ["Gemini", "ChatGPT", "DeepSeek"]:
        print("Choose Valid Model Type")
        sys.exit(1)
    
    system_prompt = """You will receive as input an English document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
    
    paragraphs = split_to_paragraphs(text)

    paragraph_chunks = pd.DataFrame({'Chunk': paragraphs})
    if paragraph_chunks.empty:
        sys.exit("Empty text")

    # Assign unique IDs to each chunk
    paragraph_chunks["ID"] = range(len(paragraph_chunks))
    
    # Copy to id_chunks
    id_chunks = paragraph_chunks.copy()

    chunk_number = 0
    i = 0
    new_id_list = []
    word_count_aux = []
    api_key = "insert-actual-key-here"

    while chunk_number < len(id_chunks) - 5:
        word_count = 0
        i = 0
        while word_count < 550 and i + chunk_number < len(id_chunks) - 1:
            i += 1
            final_document = "\n".join(
                f"id: {id_chunks.at[k, 'ID']} {id_chunks.at[k, 'Chunk']}"
                for k in range(chunk_number, i + chunk_number)
            )
            word_count = count_words(final_document)

        if i == 1:
            final_document = "\n".join(
                f"id: {id_chunks.at[k, 'ID']} {id_chunks.at[k, 'Chunk']}"
                for k in range(chunk_number, i + chunk_number)
            )
        else:
            final_document = "\n".join(
                f"id: {id_chunks.at[k, 'ID']} {id_chunks.at[k, 'Chunk']}"
                for k in range(chunk_number, i - 1 + chunk_number)
            )

        question = f"\nDocument:\n{final_document}"
        word_count_aux.append(count_words(final_document))

        chunk_number += i - 1

        prompt = system_prompt + question
        gpt_output = LLM_prompt(model_type=model_type, user_prompt=prompt, system_prompt=system_prompt, key=api_key)            

        if gpt_output == "content_flag_increment":
            chunk_number += 1
        else:
            match = re.search(r"ID (\d+)", gpt_output)
            if match:
                chunk_number = int(match.group(1))
                new_id_list.append(chunk_number)
                print("Found a answer match")
            else:
                print("No valid answer found. Skipping to next chunk.")
                chunk_number += 1

    # Add the last chunk
    new_id_list.append(len(id_chunks))

    # Remove IDs from the final chunks
    id_chunks["Chunk"] = id_chunks["Chunk"].str.replace(r"^ID \d+:\s*", "", regex=True)

    new_final_chunks = []
    for i in range(len(new_id_list)):
        start_idx = new_id_list[i - 1] if i > 0 else 0
        end_idx = new_id_list[i]
        new_final_chunks.append("\n".join(id_chunks.iloc[start_idx:end_idx, 0]))

    return new_final_chunks
