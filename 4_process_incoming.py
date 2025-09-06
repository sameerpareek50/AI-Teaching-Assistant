import pandas as pd 
import numpy as np 
import joblib 
import requests
from sklearn.metrics.pairwise import cosine_similarity


# Function to create embeddings using Ollama's bge-m3 model for any query
def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


# Load the pre-saved embeddings dataframe
df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]  

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)

# can also write function for cosine similarity by using numpy by mathehatical formula but why to write when sklearn has it already
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding] ).flatten()
# vstack is used to convert list of arrays into 2d array as cosine similarity function takes 2d array as input 
# flatten is used to convert 2d array into 1d array for easy understanding


# print(similarities)
top_results = 30
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])