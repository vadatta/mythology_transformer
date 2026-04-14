Greek Mythology Transformer Chatbot

A transformer based chatbot project trained on Greek mythology texts to generate responses about mythological narratives, characters, and themes.

The model is designed to learn patterns in mythological language and storytelling, allowing it to respond to prompts with text influenced by figures, events, and motifs from Greek mythology. This project is still under construction, with ongoing work on training, evaluation, and response quality improvements.

Features

- Transformer based language model for text generation
- Trained on Greek mythology related text data
- Uses masked self-attention for autoregressive text generation
- Input sequences are shuffled and batched
- Generates mythology inspired chatbot responses

How It Works

1. Greek mythology texts are collected and preprocessed into token sequences using HuggingFace tokenization.
2. The text data is split into training examples of fixed context length.
3. Training samples are shuffled and grouped into batches.
4. The transformer model learns to predict the next token in a sequence.
5. During inference, the chatbot generates text one token at a time based on a prompt.

