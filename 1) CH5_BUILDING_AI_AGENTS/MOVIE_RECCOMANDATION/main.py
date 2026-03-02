import os
import nltk
nltk.download('punkt_tab', quiet=True)
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import NLTKTextSplitter
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def split_text(text, splitter):
    if pd.isna(text):
        return []
    return splitter.split_text(str(text))

def encode_chunks_batch(chunks, encoder):
    """Batch-encode a list of chunks using the encoder (runs on GPU if available)."""
    valid_mask = [isinstance(c, str) and c.strip() != "" for c in chunks]
    valid_chunks = [c for c, m in zip(chunks, valid_mask) if m]

    embeddings = [None] * len(chunks)
    if valid_chunks:
        encoded = encoder.encode(valid_chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        valid_iter = iter(encoded)
        for i, m in enumerate(valid_mask):
            if m:
                embeddings[i] = next(valid_iter).tolist()
    return embeddings

def retrieve_documents(query, sentence_model, collection, top_k=5):
    #Embed query
    query_embedding = sentence_model.encode(query).tolist()

    #Search for top_k similar docs
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    if not results['documents']:
        print("No results found for the query.")
        return [], []
    
    #Extract chunks and titles
    chunks = []
    titles = []

    for doc in results['metadatas'][0]: #metadatas of first query (our only query)
        chunks.append(doc['chunk'])
        titles.append(doc['original_title'])

    return chunks, titles

def generate_answer(query, chunks, titles, text_generation_pipeline):
    #Prepare context
    context = "\n\n".join([f'Title: {title}\nChunk: {chunk}' for title, chunk in zip(titles, chunks)])

    #Prepare prompt
    prompt = f"""[INST]
        Instruction: You're an expert in movie suggestions. Your task is to analyze carefully the context and come up with an exhaustive answer to the following question:
        {query}

        Here is the context you must refer to:
        {context}

        [/INST]"""
    
    #Generate the answer
    generated_text = text_generation_pipeline(prompt)[0]['generated_text']

    return generated_text


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'movies_metadata.csv')
    df = pd.read_csv(data_path)
    print('='*30)
    print("MOVIES DATASET")
    print('='*30)
    print(df.head())
    print('='*30)

    #MAINTAIN only relevant info
    final_df = df[['original_title', 'overview']]

    #STEP 1: Chunk Overview
    splitter = NLTKTextSplitter(chunk_size=1500)
    final_df['chunked_overview'] = final_df['overview'].apply(lambda x: split_text(x, splitter))

    #FLATTEN chunks
    chunked_df = final_df.explode('chunked_overview').reset_index(drop=True)

    #STEP 2: Embed chunks (batch on GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    chunked_df['embeddings'] = encode_chunks_batch(chunked_df['chunked_overview'].tolist(), embedder)

    #Drop None
    chunked_df.dropna(subset=['embeddings'], inplace=True)

    #STEP 3: Store in ChromaDB

    #Initialize Client
    client = chromadb.Client()
    collection = client.create_collection(name='movies')

    #Insert data into ChromaDB
    for id, row in chunked_df.iterrows():
        collection.add(
            ids=[str(id)],
            embeddings=row['embeddings'], #list of chunk embeddings
            metadatas=[{
                'original_title': row['original_title'],
                'chunk': row['chunked_overview']
            }]
        )
    print("Data successfully stored in ChromaDB.")

    #Generation Model (device_map='auto' places layers on GPU automatically)
    model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    text_generation_pipeline = pipeline(model=model, tokenizer=tokenizer, task='text-generation', return_full_text=True, max_new_tokens=800)

    #Example of usage
    print('='*30)
    print("EXAMPLE OF USAGE")
    print('='*30)
    full_collection = client.get_collection(name='movies')

    query = "Tell me a good movie to make me smile"
    print(f'QUERY: {query}')
    
    chunks, titles = retrieve_documents(query, embedder, full_collection)
    print(f"Retrieved Chunks: {chunks}")
    print(f"Retrieved Titles: {titles}")

    print('\nRAG RESPONSE:\n')
    if chunks and titles:
        answer = generate_answer(query, chunks, titles, text_generation_pipeline)
        print(answer)
    else:
        print("No relevant documents found to generate an answer.")


    





if __name__ == "__main__":
    main()