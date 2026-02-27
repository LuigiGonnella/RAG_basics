##### MOVIE RECCOMENDATION SYSTEM WITH RAG (PROTOTYPE)

from langchain_text_splitters import NLTKTextSplitter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io
from pathlib import Path


text_splitter = NLTKTextSplitter(chunk_size = 1500)


def load_dataset(url):
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    #download
    res = requests.get(url) 
    res.raise_for_status()

    #unzip
    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        z.extractall(out_dir)


def read_data(path):
    df = pd.read_csv(path)
    df = df.iloc[:5000,:]
    print("="*30)
    print("DATA READ SUCCCESSFULLY")
    print("="*30)
    print(df.head())
    return df

def embed_movies(df):
    #initialize TDIF VECTORIZER
    tf_idf_vectorizer = TfidfVectorizer(max_features=1000)
    #select top 1000 terms
    #for each review, calculates TFIDF of each of those 1000 terms
    #this vector is the embedding for that review
    #repeat for every reiew

    movies_embeddings = tf_idf_vectorizer.fit_transform(df['review'])

    #convert to DataFrame
    movie_embeddings_df = pd.DataFrame(movies_embeddings.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())

    return movies_embeddings, movie_embeddings_df, tf_idf_vectorizer

def embed_query(q, tf_idf_vectorizer):
    query_embedding = tf_idf_vectorizer.transform([q])
    return query_embedding


def rerank_movies(df, q, movies):
    cosine_similarities = cosine_similarity(q, movies).flatten()

    df["cosine_similarities"] = cosine_similarities
    reranked_df = df.sort_values(by="cosine_similarities", ascending=False).reset_index(drop=True)

    print("="*30)
    print("MOVIES RERANKED SUCCESSFULLY")
    print("="*30)

    print(reranked_df.head())

    return reranked_df


QUERY = "bad"


def main():

    try: 
        df = read_data("data/IMDB Dataset.csv")
    except:
        load_dataset("https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip?raw=true")
        df = read_data("data/IMDB Dataset.csv")
    
    movie_embeddings, movie_embeddings_df, tf_idf_vectorizer = embed_movies(df)

    query_embedding = embed_query(QUERY, tf_idf_vectorizer)

    reranked_df = rerank_movies(df, query_embedding, movie_embeddings)

    print(f"BEST MOVIE FOUND:\n{reranked_df.iloc[0, 0]}")

    



    return 

if __name__ == "__main__":
    main()
