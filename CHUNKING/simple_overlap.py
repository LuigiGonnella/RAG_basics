from transformers import BertTokenizer
from IPython.display import HTML
from langchain_text_splitters import TokenTextSplitter
from collections import defaultdict

# Initialize the tokenizer and text splitter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to color and display text chunks
def color_text_chunks(text, chunk_size, overlap):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
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

text = '''To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles,
And by opposing, end them. To die, to sleep
No more, and by a sleep to say we end,
The heartache and the thousand natural shocks
That flesh is heir to, tis a consummation
Devoutly to be wished.'''

chunk_size = 80
overlap = 20

color_text_chunks(text, chunk_size, overlap)