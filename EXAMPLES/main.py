##### MOVIE RECCOMENDATION SYSTEM WITH RAG (PROTOTYPE)
import torch
from langchain_text_splitters import NLTKTextSplitter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import BertModel, BertTokenizer
import numpy as np
from IPython.display import HTML
from langchain_text_splitters import TokenTextSplitter
from collections import defaultdic

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


def get_word_embedding(sentence, word, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors = 'pt')
    tokenized_sentence = tokenizer.tokenize(sentence)

    word_index = tokenized_sentence.index(word)
    with torch.no_grad():
        #I can take sentence embeddings (from CLS), by doing
        # .last_hidden_state[:, 0, :]
        word_embedding = model(**inputs).last_hidden_state[0, word_index, :].numpy()
    return word_embedding

#-->

def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors = 'pt')

    with torch.no_grad():
        #I can take sentence embeddings (from CLS), by doing
        sentence_embedding = model(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()
    return sentence_embedding
    

def show_word_tsne_embeddings(sentences, labels, model, tokenizer):
    embeddings = np.array([get_word_embedding(sentence, 'bank', model, tokenizer) for sentence in sentences])

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 8))
    colors = {'money': 'blue', 'river': 'green'}
    for i, label in enumerate(labels ):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[label], label=label if (labels + ['money', 'river']).index(label) == i else "")
        plt.annotate(sentences[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=10, alpha=0.75)

    plt.title('t-SNE visualization of BERT embeddings for the word "bank"')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(title="Contextual Meaning", loc= 'upper left')
    plt.grid(True)
    plt.savefig('bank_embedding.jpg', format='jpeg')
    plt.show()


def show_sentence_tsne_embeddings(sentences, labels, model, tokenizer):
    embeddings = np.array([get_sentence_embedding(sentence, model, tokenizer) for sentence in sentences])

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 8))
    colors = {'money': 'blue', 'river': 'green'}
    for i, label in enumerate(labels ):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[label], label=label if (labels + ['money', 'river']).index(label) == i else "")
        plt.annotate(sentences[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=10, alpha=0.75)

    plt.title('t-SNE visualization of BERT embeddings for the word "bank"')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(title="Contextual Meaning", loc= 'upper left')
    plt.grid(True)
    plt.savefig('sentence_embeddings.jpg', format='jpeg')
    plt.show()

def show_cosine_similarities(queries, sentences, labels, model, tokenizer):
    query_embeddings = [get_sentence_embedding(query, model, tokenizer) for query in queries]
    doc_embeddings = [get_sentence_embedding(sentence, model, tokenizer) for sentence in sentences]
    similarities = cosine_similarity(doc_embeddings, query_embeddings)
    # Reorder sentences and similarities based on labels
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
    sorted_sentences = [sentences[i] for i in sorted_indices]
    sorted_similarities = similarities[sorted_indices, :]

    queries = [
        "I opened a new savings account\nat the bank.",  # money
        "The fisherman stood by\nthe river bank all day."  # river
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_similarities, annot=True, xticklabels=queries, 
                yticklabels=sorted_sentences, cmap='coolwarm', fmt='.2f')
    plt.title('Cosine similarity between documents and queries')
    plt.xlabel('Queries')
    plt.ylabel('Documents')
    plt.savefig('bi-encoder.jpg', format='jpeg', bbox_inches='tight')
    plt.show()


QUERY = "bad"
MODEL = 'bert-base-uncased'
# List of sentences with the word "bank" having different meanings
SENTENCES = [
        "I need to go to the bank to deposit some money.",
        "The river bank was flooded after the heavy rain.",
        "She works at a bank downtown.",
        "We set up our picnic by the river bank.",
        "He withdrew cash from the bank yesterday.",
        "There are many fish along the river bank.",
        "The bank offers loans with low interest rates.",
        "Children were playing on the river bank.",
        "I have a meeting at the bank in the morning.",
        "The erosion is affecting the river bank."
    ]
# Labels indicating the meaning of "bank" in each sentence
LABELS = ['money', 'river', 'money', 'river', 'money', 'river', 'money', 'river', 'money', 'river']


def example_chunking(text, chunk_size, chunk_overlap, overlap_size):
    text_chunker = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, separator = '\n') 
    #decide overlap, and separator (for character chunking)
    docs = text_chunker.create_documents(texts=[text])
    chunks = [doc.page_content for doc in docs]

    colored_text = ""
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#99ffff', '#ff99ff', '#cccccc',
              '#ff6666', '#66ff66', '#6666ff', '#ffff66', '#66ffff', '#ff66ff', '#ccccff',
              '#996699', '#669999', '#999966', '#669966', '#966696', '#696669']

    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        chunk_html = f'<span style="background-color:{color}">{chunk}</span>'
        colored_text += chunk_html + '<br><br>'

    print(HTML(colored_text))


def evaluation(ranks, labels):

    df = pd.DataFrame({'Rank': ranks, 'Labels': labels})

    #compute binary labels
    df['BinaryLabels'] = df['Labels'].map(lambda x: 0 if x == 'NR' else 1)
    precisions = []
    recalls = []
    for i in range(1, len(df) + 1):
        sub_df = df.iloc[:i]
        tp = sum(sub_df['binary_label'])
        fp = i - tp
        fn = sum(df['binary_label']) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)

    df['precision'] = precisions
    df['recall'] = recalls
    print(df)



def main():

    try: 
        df = read_data("data/IMDB Dataset.csv")
    except:
        load_dataset("https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip?raw=true")
        df = read_data("data/IMDB Dataset.csv")
    
    movie_embeddings, _, tf_idf_vectorizer = embed_movies(df)

    query_embedding = embed_query(QUERY, tf_idf_vectorizer)

    reranked_df = rerank_movies(df, query_embedding, movie_embeddings)

    print(f"BEST MOVIE FOUND:\n{reranked_df.iloc[0, 0]}")

    #SHOW how embedding space work (ber model)
    print('='*30)
    print('BERT WORD EMBEDDINGS')
    print('='*30)
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    model = BertModel.from_pretrained(MODEL)
    show_word_tsne_embeddings(SENTENCES, LABELS, model, tokenizer)

    #SHOW sentence (CLS) embeddings
    print('='*30)
    print('BERT SENTENCE EMBEDDINGS')
    print('='*30)
    show_sentence_tsne_embeddings(SENTENCES, LABELS, model, tokenizer)

    #SHOW cosine_similarities between doc_embeddings and query_embeddings
    print('='*30)
    print('COSINE SIMILARITIES BETWEEN BERT SENTENCE EMBEDDINGS')
    print('='*30)
    queries = [
    "I opened a new savings account at the bank.",  # money
    "The fisherman stood by the river bank all day."  # river
    ]
    show_cosine_similarities(queries, SENTENCES, LABELS, model, tokenizer)

    #SHOW chunking
    print('='*30)
    print('EXAMPLE OF CHARACTER CHUNKING ON \\n WITH OVERLAP')
    print('='*30)
    text = '''To be or not to be, that is the question.
        Whether tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune
        Or to take arms against a sea of troubles,
        And by opposing, end them. To die, to sleep
        No more, and by a sleep to say we end,
        The heartache and the thousand natural shocks
        That flesh is heir to, tis a consummation
        Devoutly to be wished.'''
    
    chunk_size = 256
    chunk_overlap = '\n'
    overlap_size = 20

    example_chunking(text, chunk_size, chunk_overlap, overlap_size) #see other chunkign exmaples in CHUNKING/ folder

    #EXAMPLE OF EVALUATION WITH PRECISION and RECALL 
    print('='*30)
    print('EXAMPLE OF EVALUATION WITH PRECISION and RECALL')
    print('='*30)

 

if __name__ == "__main__":
    main()
