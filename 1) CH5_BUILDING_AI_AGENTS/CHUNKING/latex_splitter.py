from langchain_text_splitters import LatexTextSplitter
text = '''
\\section{Retrieval-Augmented Generation (RAG)}

\\subsection{Introduction}

Retrieval-Augmented Generation (RAG) is a technique in natural language processing (NLP) that combines the strengths of retrieval-based methods and generation-based methods. It is designed to enhance the quality and accuracy of generated text by leveraging a large corpus of documents during the text generation process.

\\subsection{How RAG Works}

\\subsubsection{Retrieval Component}

The retrieval component is responsible for searching a large dataset or knowledge base to find relevant documents or passages that are related to the input query. This component typically uses advanced search algorithms and embeddings to identify the most pertinent information.

\\subsubsection{Generation Component}

The generation component takes the retrieved documents as additional context and generates a coherent and contextually accurate response. This is usually achieved using transformer-based models like BERT or GPT, which can process the input query along with the retrieved information to produce high-quality text.

\\subsection{Advantages of RAG}

\\begin{itemize}
    \\item \\textbf{Enhanced Accuracy}: By using relevant external documents, RAG can generate more accurate and informed responses.
    \\item \\textbf{Contextual Relevance}: The retrieval component ensures that the generated text is contextually relevant to the query.
    \\item \\textbf{Scalability}: RAG can be scaled to work with vast datasets, making it suitable for applications requiring extensive knowledge bases.
\\end{itemize}

\\subsection{Applications of RAG}

\\begin{itemize}
    \\item \\textbf{Question Answering Systems}: RAG can be used to improve the performance of question answering systems by providing precise and detailed answers.
    \\item \\textbf{Customer Support}: In customer support, RAG can assist in generating accurate responses to customer queries by referencing a large database of knowledge.
    \\item \\textbf{Content Creation}: RAG can aid content creators by providing relevant information and generating high-quality content based on the retrieved data.
\\end{itemize}

\\subsection{Conclusion}

Retrieval-Augmented Generation (RAG) represents a significant advancement in NLP by combining the best aspects of retrieval and generation methods. Its ability to utilize vast datasets for generating contextually accurate and high-quality text makes it a powerful tool for various applications.
'''

chunk_size = 200
overlap = 0

text_splitter = LatexTextSplitter(
    chunk_size=chunk_size,  
    chunk_overlap=overlap  
)
#!understands LaTex syntax

color_text_chunks(text, chunk_size, overlap, text_splitter)