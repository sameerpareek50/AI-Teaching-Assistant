# Step 3 : convert text chunks into vector embeddings

# Here i will write code for converting text of chunks into vector embeddings using ollama bge-m3 on my local system

import requests
import os
import json
import pandas as pd
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={        # used for posting my request to ollama server
        "model": "bge-m3",
        "input": text_list
    })  # this is saying ki hamara local jo ollama ka instance chl rha h usme bge-m3 ko use krke ye text_list ka embedding banao

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("jsons")  # List all the jsons mtlb read kra
my_dicts = [] # khali list jisme sare chunks ka data hoga with embeddings
chunk_id = 0 # unique chunk id for each chunk

for json_file in jsons:                      # Loop through each json file
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)               # Read the data of each json file aur unhe content naam k variable m lelo as a python dictionary jisme chunks aur text honge

    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']]) # ab har content variable k har chunk k har individual text ko lekr ek list of string bana lo and pass it to create_embedding function
    # i made a list of lists here taaki har chunks k text ko ek sath pass krdu means whole video k saare text ek sath in list format for faster processing

    # print(embeddings) # ye embeddings ki list dega jisme har chunk k corresponding embedding hoga
       
    for i, chunk in enumerate(content['chunks']): # enumerate is used to get index along with the chunk
        
        chunk['chunk_id'] = chunk_id                   # har chunk k sath unique chunk id add krdo in content variable
        chunk['embedding'] = embeddings[i]             # har chunk k sath uska corresponding embedding add krdo in content variable
        chunk_id += 1
        my_dicts.append(chunk)                 # ab content variable m pehle data tha hi aur ab chunk_id aur embeddings bhi add ho gyi so converted this into a dataframe
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)

joblib.dump(df, 'embeddings.joblib') # Save the dataframe to a joblib file for later use

# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)