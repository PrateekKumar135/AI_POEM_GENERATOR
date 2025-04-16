from langchain_ollama.llms import OllamaLLM
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model='llama3.2:1b')
template = """
You are an expert in poems and literature. Given few list of words: {list_of_words}, 
You are asked to make a small poem using the words list . Make up eye catchy headline for the poem.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model  #Creating Langchain Runnables

# Inference function for Gradio
def generate_poem(word_input):
    words = [w.strip() for w in word_input.split(',') if w.strip()]
    if not words:
        return "Please enter some words to generate a poem."

    result = chain.invoke({'list_of_words': words})
    return result


# Gradio Interface
iface = gr.Interface(
    fn=generate_poem,
    inputs=gr.Textbox(lines=2, placeholder="Enter comma-separated words (e.g., moon, river, dream)..."),
    outputs="text",
    title="🎭 AI Poem Generator",
    description="Enter a few words and get a custom AI-generated poem using the powerful LLaMA3 model via LangChain + Ollama.",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()