import os
import whisper
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Importing libraries
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import moviepy.editor as moviepy
import requests
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/diseasepred/<symptoms>")
def diseasepred(symptoms):
    print(symptoms)
    # load the model from disk
    final_svm_model = pickle.load(open("./models/final_svm_model.sav", "rb"))
    final_nb_model = pickle.load(open("./models/final_nb_model.sav", "rb"))
    final_rf_model = pickle.load(open("./models/final_rf_model.sav", "rb"))

    data_dict = {
        "symptom_index": {
            "Itching": 0,
            "Skin Rash": 1,
            "Nodal Skin Eruptions": 2,
            "Continuous Sneezing": 3,
            "Shivering": 4,
            "Chills": 5,
            "Joint Pain": 6,
            "Stomach Pain": 7,
            "Acidity": 8,
            "Ulcers On Tongue": 9,
            "Muscle Wasting": 10,
            "Vomiting": 11,
            "Burning Micturition": 12,
            "Spotting  urination": 13,
            "Fatigue": 14,
            "Weight Gain": 15,
            "Anxiety": 16,
            "Cold Hands And Feets": 17,
            "Mood Swings": 18,
            "Weight Loss": 19,
            "Restlessness": 20,
            "Lethargy": 21,
            "Patches In Throat": 22,
            "Irregular Sugar Level": 23,
            "Cough": 24,
            "High Fever": 25,
            "Sunken Eyes": 26,
            "Breathlessness": 27,
            "Sweating": 28,
            "Dehydration": 29,
            "Indigestion": 30,
            "Headache": 31,
            "Yellowish Skin": 32,
            "Dark Urine": 33,
            "Nausea": 34,
            "Loss Of Appetite": 35,
            "Pain Behind The Eyes": 36,
            "Back Pain": 37,
            "Constipation": 38,
            "Abdominal Pain": 39,
            "Diarrhoea": 40,
            "Mild Fever": 41,
            "Yellow Urine": 42,
            "Yellowing Of Eyes": 43,
            "Acute Liver Failure": 44,
            "Fluid Overload": 45,
            "Swelling Of Stomach": 46,
            "Swelled Lymph Nodes": 47,
            "Malaise": 48,
            "Blurred And Distorted Vision": 49,
            "Phlegm": 50,
            "Throat Irritation": 51,
            "Redness Of Eyes": 52,
            "Sinus Pressure": 53,
            "Runny Nose": 54,
            "Congestion": 55,
            "Chest Pain": 56,
            "Weakness In Limbs": 57,
            "Fast Heart Rate": 58,
            "Pain During Bowel Movements": 59,
            "Pain In Anal Region": 60,
            "Bloody Stool": 61,
            "Irritation In Anus": 62,
            "Neck Pain": 63,
            "Dizziness": 64,
            "Cramps": 65,
            "Bruising": 66,
            "Obesity": 67,
            "Swollen Legs": 68,
            "Swollen Blood Vessels": 69,
            "Puffy Face And Eyes": 70,
            "Enlarged Thyroid": 71,
            "Brittle Nails": 72,
            "Swollen Extremeties": 73,
            "Excessive Hunger": 74,
            "Extra Marital Contacts": 75,
            "Drying And Tingling Lips": 76,
            "Slurred Speech": 77,
            "Knee Pain": 78,
            "Hip Joint Pain": 79,
            "Muscle Weakness": 80,
            "Stiff Neck": 81,
            "Swelling Joints": 82,
            "Movement Stiffness": 83,
            "Spinning Movements": 84,
            "Loss Of Balance": 85,
            "Unsteadiness": 86,
            "Weakness Of One Body Side": 87,
            "Loss Of Smell": 88,
            "Bladder Discomfort": 89,
            "Foul Smell Of urine": 90,
            "Continuous Feel Of Urine": 91,
            "Passage Of Gases": 92,
            "Internal Itching": 93,
            "Toxic Look (typhos)": 94,
            "Depression": 95,
            "Irritability": 96,
            "Muscle Pain": 97,
            "Altered Sensorium": 98,
            "Red Spots Over Body": 99,
            "Belly Pain": 100,
            "Abnormal Menstruation": 101,
            "Dischromic  Patches": 102,
            "Watering From Eyes": 103,
            "Increased Appetite": 104,
            "Polyuria": 105,
            "Family History": 106,
            "Mucoid Sputum": 107,
            "Rusty Sputum": 108,
            "Lack Of Concentration": 109,
            "Visual Disturbances": 110,
            "Receiving Blood Transfusion": 111,
            "Receiving Unsterile Injections": 112,
            "Coma": 113,
            "Stomach Bleeding": 114,
            "Distention Of Abdomen": 115,
            "History Of Alcohol Consumption": 116,
            "Fluid Overload.1": 117,
            "Blood In Sputum": 118,
            "Prominent Veins On Calf": 119,
            "Palpitations": 120,
            "Painful Walking": 121,
            "Pus Filled Pimples": 122,
            "Blackheads": 123,
            "Scurring": 124,
            "Skin Peeling": 125,
            "Silver Like Dusting": 126,
            "Small Dents In Nails": 127,
            "Inflammatory Nails": 128,
            "Blister": 129,
            "Red Sore Around Nose": 130,
            "Yellow Crust Ooze": 131,
        },
        "predictions_classes": [
            "(vertigo) Paroymsal  Positional Vertigo",
            "AIDS",
            "Acne",
            "Alcoholic hepatitis",
            "Allergy",
            "Arthritis",
            "Bronchial Asthma",
            "Cervical spondylosis",
            "Chicken pox",
            "Chronic cholestasis",
            "Common Cold",
            "Dengue",
            "Diabetes ",
            "Dimorphic hemmorhoids(piles)",
            "Drug Reaction",
            "Fungal infection",
            "GERD",
            "Gastroenteritis",
            "Heart attack",
            "Hepatitis B",
            "Hepatitis C",
            "Hepatitis D",
            "Hepatitis E",
            "Hypertension ",
            "Hyperthyroidism",
            "Hypoglycemia",
            "Hypothyroidism",
            "Impetigo",
            "Jaundice",
            "Malaria",
            "Migraine",
            "Osteoarthristis",
            "Paralysis (brain hemorrhage)",
            "Peptic ulcer diseae",
            "Pneumonia",
            "Psoriasis",
            "Tuberculosis",
            "Typhoid",
            "Urinary tract infection",
            "Varicose veins",
            "hepatitis A",
        ],
    }

    def predictDisease(symptoms):
        symptoms = symptoms.split(",")

        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

        # reshaping the input data and converting it
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1, -1)

        # generating individual outputs
        rf_prediction = data_dict["predictions_classes"][
            final_rf_model.predict(input_data)[0]
        ]
        nb_prediction = data_dict["predictions_classes"][
            final_nb_model.predict(input_data)[0]
        ]
        svm_prediction = data_dict["predictions_classes"][
            final_svm_model.predict(input_data)[0]
        ]

        # making final prediction by taking mode of all predictions
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": final_prediction,
        }
        return predictions

    # Testing the function
    # predictDisease("Itching,Skin Rash,Nodal Skin Eruptions")
    prediction = predictDisease(symptoms)
    print(prediction)
    return prediction


@app.route("/grading", methods=["POST"])
def grading():
    json_data = request.json
    response = requests.get(json_data["recordingLink"])
    print(json_data)
    open("audio.wav", "wb").write(response.content)
    # files
    # dst = "audio.wav"
    # # convert wav to mp3
    # clip = moviepy.VideoFileClip(r"audio.webm")
    # clip.write_audiofile(dst)

    model = whisper.load_model("small")
    sentences = [
        "The patient's name is Mrs. Helena Jones, a 65-year-old woman who was admitted to the hospital six days ago following a road traffic accident. She sustained multiple rib fractures and is currently being managed conservatively for her pain and closely monitored for her vital signs. The decision was made not to perform surgery on her. During the night, Mrs. Jones experienced multiple episodes of shortness of breath, particularly when repositioning in bed. In the morning, when the nurse greeted her at the bedside, she appeared relaxed and her oxygen saturation levels were above 96%. Two hours later, the nurse noticed that Mrs. Jones was becoming quite restless in bed and decided to do a full set of observations. The nurse found that her oxygen levels had significantly dropped, her blood pressure had also dropped, and she appeared very anxious. Upon checking her past medical history, the nurse noted that Mrs. Jones is usually hypertensive. Based on these findings, the nurse became concerned and decided to raise the issue using the SBAR (Situation, Background, Assessment, Recommendation) handover to the medical doctors who are also looking after Mrs. Jones.",
    ]

    def getScore():
        result = model.transcribe("./audio.wav")
        print("exemplar text : ", sentences[0])
        print("transcribed text : ", result["text"])
        cosine_sim_result = cosine_similar(result["text"])
        percentage = cosine_sim_result[0][0] * 100
        if percentage > 100:
            percentage = 100
        if percentage < 0:
            percentage = 0
        return percentage

    def cosine_similar(sentence):
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        tokens = {"input_ids": [], "attention_mask": []}
        sentences.append(sentence)

        for sentence in sentences:
            new_tokens = tokenizer.encode_plus(
                sentence,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens["input_ids"].append(new_tokens["input_ids"][0])
            tokens["attention_mask"].append(new_tokens["attention_mask"][0])

        # restructure a list of tensors into single tensor
        tokens["input_ids"] = torch.stack(tokens["input_ids"])
        tokens["attention_mask"] = torch.stack(tokens["attention_mask"])
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"]
        resized_attention_mask = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )

        masked_embedding = embeddings * resized_attention_mask

        summed_masked_embeddings = torch.sum(masked_embedding, 1)

        count_of_one_in_mask_tensor = torch.clamp(
            resized_attention_mask.sum(1), min=1e-9
        )

        mean_pooled = summed_masked_embeddings / count_of_one_in_mask_tensor

        mean_pooled = mean_pooled.detach().numpy()

        similar = cosine_similarity([mean_pooled[0]], mean_pooled[1:])
        return similar

    result = str(getScore())
    print("Automated grading result : ", result)

    return result
