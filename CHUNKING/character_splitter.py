from transformers import BertTokenizer
from IPython.display import HTML
from langchain_text_splitters import CharacterTextSplitter
# Initialize the tokenizer and text splitter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 256,
    chunk_overlap = 0
)

# Function to color and display text chunks
def color_text_chunks(text, text_splitter):
    docs = text_splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]  # Access the text attribute

    colored_text = ""
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#99ffff', '#ff99ff', '#cccccc',
              '#ff6666', '#66ff66', '#6666ff', '#ffff66', '#66ffff', '#ff66ff', '#ccccff',
              '#996699', '#669999', '#999966', '#669966', '#966696', '#696669']

    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        chunk_html = f'<span style="background-color:{color}">{chunk}</span>'
        colored_text += chunk_html + '<br><br>'

    return HTML(colored_text)

# Example usage
text = '''To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles,
And by opposing, end them. To die, to sleep
No more, and by a sleep to say we end,
The heartache and the thousand natural shocks
That flesh is heir to, tis a consummation
Devoutly to be wished.'''

color_text_chunks(text, text_splitter)