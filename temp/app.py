import os
import whisper
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from happytransformer import HappyTextToText, TTSettings
import numpy as np

model = whisper.load_model("small")
sentences = [
    "The patient's name is Mrs. Helena Jones, a 65-year-old woman who was admitted to the hospital six days ",
]


def getScore(audio_url):

    result = model.transcribe('./Demo.wav')
    print(result['text'])
    cosine_sim_result = cosine_similar(result['text'])
    percentage = cosine_sim_result[0][0] * 100
    if (percentage > 100):
        percentage = 100
    if (percentage < 0):
        percentage = 0
    print(percentage)

    # grammar_corrected_text = grammer_correction(result['text'])

    # difference = abs(len(grammar_corrected_text.split()) -
    #                  len(result['text'].split()))
    # difference = difference/2
    # percent = difference/len(grammar_corrected_text)*100
    # percent = 100-percent
    # print("percent")
    # print(percent)

    # print("The absolute difference between the number of words in the two sentences is:", difference)


def cosine_similar(sentence):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')

    tokens = {'input_ids': [], 'attention_mask': []}
    sentences.append(sentence)

    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(
            sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # restructure a list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    resized_attention_mask = attention_mask.unsqueeze(
        -1).expand(embeddings.size()).float()

    masked_embedding = embeddings * resized_attention_mask

    summed_masked_embeddings = torch.sum(masked_embedding, 1)

    count_of_one_in_mask_tensor = torch.clamp(
        resized_attention_mask.sum(1), min=1e-9)

    mean_pooled = summed_masked_embeddings / count_of_one_in_mask_tensor

    mean_pooled = mean_pooled.detach().numpy()

    similar = cosine_similarity([
        mean_pooled[0]], mean_pooled[1:])
    return similar


def grammer_correction(sentence):
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)

    result = happy_tt.generate_text(
        sentence, args=args)
    print(result.text)

    return result.text


audio_url = "./Demo.wav"

getScore(audio_url)
