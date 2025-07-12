import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("tuned_distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("tuned_distilgpt2")
    
    return tokenizer, model


tokenizer, model = load_model()

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=3,  
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [text.strip() for text in generated_texts]


# UI for the app
st.title("Linkedin post generator")

prompt = st.text_area("Brief description of the post:",)

if st.button("Generate Post"):
    if prompt:
        with st.spinner("Generating posts..."):
            generated_posts = generate_text(prompt)
            st.write("Generated Posts:")
            for i, post in enumerate(generated_posts, 1):
                st.markdown(f"**Post {i}:**\n{post}")
    else:
        st.warning("Please enter a prompt to generate a post.")
   
