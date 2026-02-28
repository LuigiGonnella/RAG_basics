from langchain_text_splitters import HTMLHeaderTextSplitter

def color_text_chunks(text, overlap, text_splitter):
    # modified for HTMLHeaderTextSplitter
    docs = text_splitter.split_text(text)
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


text = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retrieval-Augmented Generation (RAG)</title>
</head>
<body>
    <h1>Retrieval-Augmented Generation (RAG)</h1>

    <h2>Introduction</h2>
    <p>Retrieval-Augmented Generation (RAG) is a technique in natural language processing (NLP) that combines the strengths of retrieval-based methods and generation-based methods. It is designed to enhance the quality and accuracy of generated text by leveraging a large corpus of documents during the text generation process.</p>

    <h2>How RAG Works</h2>

    <h3>Retrieval Component</h3>
    <p>The retrieval component is responsible for searching a large dataset or knowledge base to find relevant documents or passages that are related to the input query. This component typically uses advanced search algorithms and embeddings to identify the most pertinent information.</p>

    <h3>Generation Component</h3>
    <p>The generation component takes the retrieved documents as additional context and generates a coherent and contextually accurate response. This is usually achieved using transformer-based models like BERT or GPT, which can process the input query along with the retrieved information to produce high-quality text.</p>

    <h2>Advantages of RAG</h2>
    <ul>
        <li><strong>Enhanced Accuracy</strong>: By using relevant external documents, RAG can generate more accurate and informed responses.</li>
        <li><strong>Contextual Relevance</strong>: The retrieval component ensures that the generated text is contextually relevant to the query.</li>
        <li><strong>Scalability</strong>: RAG can be scaled to work with vast datasets, making it suitable for applications requiring extensive knowledge bases.</li>
    </ul>

    <h2>Applications of RAG</h2>
    <ul>
        <li><strong>Question Answering Systems</strong>: RAG can be used to improve the performance of question answering systems by providing precise and detailed answers.</li>
        <li><strong>Customer Support</strong>: In customer support, RAG can assist in generating accurate responses to customer queries by referencing a large database of knowledge.</li>
        <li><strong>Content Creation</strong>: RAG can aid content creators by providing relevant information and generating high-quality content based on the retrieved data.</li>
    </ul>

    <h2>Conclusion</h2>
    <p>Retrieval-Augmented Generation (RAG) represents a significant advancement in NLP by combining the best aspects of retrieval and generation methods. Its ability to utilize vast datasets for generating contextually accurate and high-quality text makes it a powerful tool for various applications.</p>
</body>
</html>
'''


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

text_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
#!understands html

color_text_chunks(text, overlap, text_splitter)