from transformers import BertTokenizer
from IPython.display import HTML
from langchain_text_splitters import TokenTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter
from collections import defaultdict

# Initialize the tokenizer and text splitter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to color and display text chunks
def color_text_chunks(text, overlap, text_splitter):

    docs = text_splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]  # Access the text attribute

    # Define colors
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#99ffff', '#ff99ff', '#cccccc',
              '#ff6666', '#66ff66', '#6666ff', '#ffff66', '#66ffff', '#ff66ff', '#ccccff',
              '#996699', '#669999', '#999966', '#669966', '#966696', '#696669']

    # Define color positions for each chunk
    color_positions = []
    for i in range(len(chunks)):
        chunk_tokens = tokenizer.tokenize(chunks[i])
        chunk_length = len(chunk_tokens)
        if chunk_length > overlap:
            unique_length = chunk_length - overlap
            chunk_colors = [colors[i % len(colors)]] * unique_length + \
                           [blend_colors([colors[i % len(colors)], colors[(i+1) % len(colors)]])] * overlap
        else:
            chunk_colors = [blend_colors([colors[i % len(colors)], colors[(i+1) % len(colors)]])] * chunk_length
        color_positions.append(chunk_colors)

    # Adjust colors for overlapping tokens in the next chunk
    for i in range(1, len(color_positions)):
        overlap_color = blend_colors([colors[(i-1) % len(colors)], colors[i % len(colors)]])
        color_positions[i] = [overlap_color] * overlap + color_positions[i][overlap:]

    # Generate colored HTML
    colored_text = ""
    for i, chunk in enumerate(chunks):
        tokens = tokenizer.tokenize(chunk)
        for j, token in enumerate(tokens):
            color = color_positions[i][j]
            token_html = f'<span style="background-color:{color}">{token}</span>'
            colored_text += token_html + ' '
        colored_text += '<br><br>'

    return HTML(colored_text)

def blend_colors(color_list):
    # Simple function to blend multiple colors
    r, g, b = 0, 0, 0
    for color in color_list:
        c = int(color[1:], 16)
        r += c >> 16
        g += (c >> 8) & 0xff
        b += c & 0xff
    n = len(color_list)
    r = min(r // n, 255)
    g = min(g // n, 255)
    b = min(b // n, 255)
    return f'#{r:02x}{g:02x}{b:02x}'


text = '''
# Retrieval-Augmented Generation (RAG)

## Introduction

Retrieval-Augmented Generation (RAG) is a technique in natural language processing (NLP) that combines the strengths of retrieval-based methods and generation-based methods. It is designed to enhance the quality and accuracy of generated text by leveraging a large corpus of documents during the text generation process.

## How RAG Works

### Retrieval Component

The retrieval component is responsible for searching a large dataset or knowledge base to find relevant documents or passages that are related to the input query. This component typically uses advanced search algorithms and embeddings to identify the most pertinent information.

### Generation Component

The generation component takes the retrieved documents as additional context and generates a coherent and contextually accurate response. This is usually achieved using transformer-based models like BERT or GPT, which can process the input query along with the retrieved information to produce high-quality text.

## Advantages of RAG

1. **Enhanced Accuracy**: By using relevant external documents, RAG can generate more accurate and informed responses.
2. **Contextual Relevance**: The retrieval component ensures that the generated text is contextually relevant to the query.
3. **Scalability**: RAG can be scaled to work with vast datasets, making it suitable for applications requiring extensive knowledge bases.

## Applications of RAG

1. **Question Answering Systems**: RAG can be used to improve the performance of question answering systems by providing precise and detailed answers.
2. **Customer Support**: In customer support, RAG can assist in generating accurate responses to customer queries by referencing a large database of knowledge.
3. **Content Creation**: RAG can aid content creators by providing relevant information and generating high-quality content based on the retrieved data.

## Conclusion

Retrieval-Augmented Generation (RAG) represents a significant advancement in NLP by combining the best aspects of retrieval and generation methods. Its ability to utilize vast datasets for generating contextually accurate and high-quality text makes it a powerful tool for various applications.
'''

chunk_size = 200
overlap = 0

text_splitter = MarkdownTextSplitter(
    chunk_size=chunk_size,  
    chunk_overlap=overlap  
)
#! understands MARKDOWN format
# hashtag --> title
# double hashtag --> sections
# lists
# code blocks
# citations ...
#!avoids breaking logic blocks if possible

color_text_chunks(text, overlap, text_splitter)