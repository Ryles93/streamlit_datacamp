# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
import requests
import pandas as pd
from urllib.request import Request, urlopen
import re

df = pd.read_csv('/Users/ryles/Desktop/DataCamp/commentaires.csv')
 
best_logistic_regression_model = load('/Users/ryles/Desktop/DataCamp/saved_allocine.joblib') 
vectorizer = load('/Users/ryles/Desktop/DataCamp/saved_allocine_vect.joblib')

def preprocess_text(text):
    # Supprimer les balises HTML s'il y en a
    text = re.sub(r'<.*?>', '', text)
    
    # Supprimer la ponctuation et les chiffres
    text = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', text)
    
    # Mettre en minuscules
    text = text.lower()
    
    # Supprimer les mots vides en utilisant NLTK
    stop_words = set(stopwords.words("french"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    
    return text

# Créez une section pour le sommaire interactif
st.sidebar.title("Sommaire")

pages = ['Accueil', 'Contexte', 'Exploration et data viz', 'Utilisation de notre modèle', 'Difficultés rencontrées', 'Axes d"amélioriation']

page = st.sidebar.radio("Aller vers la page :" , pages)

if page == pages[0]:
    st.title("Bienvenue sur notre application web 👨‍💻")
    st.subheader("Sentiment Analysis project - Datacamp")
    st.markdown(" ⬅️  Utilisez le menu à gauche pour naviguer à travers l'application")
    st.image('/Users/ryles/Desktop/DataCamp/efrei_paris.jpg')

elif page == pages[1]:
    # Titre de la page
    st.title("Sentiment analysis project")
    
    
    # Présentation du projet
    st.header("Contexte du Projet")
    st.write(
        "Bienvenue sur notre application web !"
        " Notre objectif est d'analyser et de comprendre les sentiments exprimés dans les commentaires et avis en ligne, "
        "en particulier sur la plateforme Allociné et sur le film très connu Intouchable. "
        "Notre objectif est fournir une solution qui permet de pouvoir classifier automatiquement des commentaires selon le sentiment. "
    )
    
    # Objectifs du projet
    st.header("Objectifs du Projet")
    st.write(
        "Notre projet a pour but de développer un modèle d'analyse de sentiment qui peut classer les commentaires en positifs "
        "ou négatifs en se basant sur le langage naturel. \n"
        "\n"
        "Nous avons donc mis en place une interface utilisateur intuitive où les utilisateurs peuvent saisir des commentaires et obtenir instantanément des prédictions de sentiment. Cela peut être utile dans le contexte de la surveillance des médias sociaux ou des commentaires de produits. \n"
    )
    
    # Méthodologie
    st.header("Méthodologie 🧑‍💻")
    st.write(
        "Pour mener à bien ce projet, il a fallu scrapper les commentaires et avis en question. Plusieurs outils ont été testés, mais notre choix s'est porté sur BeautifulSoup. "
        "Nous avons collecté plus de 7200 commentaires en utilisant cet outil. \n "
        "Cependant, dû au succès de ce film, le nombre de commentaire négatifs comparé au nombre de commentaires positifs n'étaient pas équilibrés. Nous avons donc réduits notre base de commentaires a 1398, arrivant ainsi a 46% de commentaires négatifs et 54% de commentaires positifs. \n"
        "\n"
        "Si vous souhaitez consulter avoir une vision de la page que nous avons scrappés ou voir les commentaires que nous avons collectés, vous pouvez les trouver via ce lien : https://www.allocine.fr/film/fichefilm-182745/critiques/spectateurs/ \n"
        "\n"
        "Un autre outil que nous avons utilisé est streamlit, c'est l'outil grâce auquel vous pouvez utiliser cette application. Streamlit est un framework open source Python qui permet de créer des applications web interactives de manière simple et rapide. Il est largement utilisé pour développer des tableaux de bord, des outils de visualisation de données et des interfaces utilisateur conviviales en tirant parti de la puissance de Python. Streamlit rend la création d'applications web accessible à tous les niveaux de compétence en programmation, offrant une interface intuitive pour concevoir des applications efficaces et attrayantes. \n"
        "\n"
        "Nous avons donc eu l'idée de creer cet application qui permet à un utilisateur de pouvoir rentrer un commentaire à la main et d'dobtenir le sentiment. Ou bien de pouvoir rentrer un fichier CSV  de commentaires et de pouvoir voir le nombre de commentaires negatifs et positifs. Comme dis plus haut cela peut être dans la surveillance des réseaux sociaux par exemple. \n"
        "\n"
        "Concernant le modèle utilisé, plusieurs ont été testés plus ou moins complèxes, notre choix s'est porté sur la regression logistique, sur laquelle on obtenait les meilleurs scores \n"
        "\n"
        "Ensuite, nous avons prétraité les données collectées et entraîné le modèle en utilisant ces données, en optimisant "
        "les hyperparamètres pour obtenir les meilleures performances."
        "On arrive donc à une précision de 0.90 ce qui est plutôt bon. \n "
        "\n"
        "Pour mener à bien ce projet, nous avons adopté une approche de développement agile tout au long de ce projet, ce qui nous a permis de bénéficier de plusieurs avantages clés. En travaillant de manière itérative et incrémentielle, nous avons pu nous adapter aux changements et aux retours d'expérience en cours de route, garantissant ainsi que notre application répondait toujours aux besoins changeants. De plus, l'agilité nous a permis de maintenir une communication ouverte et constante au sein de l'équipe, favorisant la collaboration et la résolution rapide des problèmes. En fin de compte, cette méthodologie a contribué à la réussite de notre projet en nous permettant de livrer une application de haute qualité, en respectant les délais et en maintenant une grande flexibilité."
    )
    
    # Collaborateurs
    st.header("Collaborateurs 👫")
    st.write(
        "Ce projet a été réalisé par une équipe de cinq membres : "
        "Ryles /"
        "Nabil /"
        "Kenza /"
        "Louis /"
        "Alexandre \n"
        "\n"
        "Chacune des personnes s'est vu attribué une tache, mais grace à nos réunions régulière, nous étions au courant des difficultées de chacun donc l'entraide était naturelle."
    )
    
    # Contact
    st.header("Contact ✉️")
    st.write(
        "Pour toute question ou commentaire, n'hésitez pas à nous contacter aux adresses e-mails suivante : \n"
            "ryles.ait-ahmed@efrei.net \n"
            "\n"
            "alexandre.bodin@efrei.net \n"
            "\n"
            "nabil.bettaieb@efrei.net \n"
            "\n"
            "kenza.baffoun@efrei.net \n"
            "\n"
            "louis.bruder@efrei.net \n"
            "\n"
    )

elif page == pages[2]:

    st.title("Consultation des Commentaires et Data Viz")
    
    # Afficher les premières lignes du dataframe
    st.header("Premières lignes du dataframe de commentaires")
    st.dataframe(df.head())
    
    # Afficher des statistiques de base sur le dataframe
    st.header("Statistiques de base")
    st.write("Nombre total de commentaires :", len(df))
    st.write("Nombre de commentaires positifs :", len(df[df['Sentiment'] == 'Positif']))
    st.write("Nombre de commentaires négatifs :", len(df[df['Sentiment'] == 'Negatif']))
    
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100

    # Créez un camembert en utilisant matplotlib
    fig, ax = plt.subplots(figsize = (5,5))
    ax.pie(sentiment_counts, labels=['Negatif', 'Positif'], autopct='%1.1f%%', startangle=140)
    ax.set_title('Répartition des sentiments')
    
    # Affichez le camembert dans l'application Streamlit
    st.pyplot(fig)
    
    show_duplicates = st.checkbox("Afficher les commentaires en doublons", value=False)

    if show_duplicates:
    # Sélectionner les commentaires en doublons
        duplicate_comments = df[df.duplicated(subset=['Commentaire'])]
    
    # Afficher les commentaires en doublons
        st.subheader("Commentaires en doublons :")
        st.dataframe(duplicate_comments['Commentaire'])
    else: 
        st.info("Cochez la case ci-dessus pour afficher les commentaires en doublons.")
        
 


   
elif page == pages[3]:
    st.header("Prédiction de Sentiment")
    user_input = st.text_input("Saisissez un commentaire :")
    
    # Ajouter la possibilité de télécharger un fichier CSV
    uploaded_file = st.file_uploader("Télécharger un fichier CSV de commentaires", type=["csv"])
    
    if user_input or uploaded_file:
        # Prétraiter le commentaire de l'utilisateur
        if user_input:
            user_input_processed = preprocess_text(user_input)
            
        # Bouton pour prédire le sentiment
        if st.button("Prédire le sentiment"):
            if user_input:
                # Vectoriser le commentaire de l'utilisateur
                user_input_vectorized = vectorizer.transform([user_input_processed])
                
                # Faites la prédiction en utilisant le modèle
                prediction = best_logistic_regression_model.predict(user_input_vectorized)
                
                # Afficher le résultat
                if prediction[0] == 1:
                    st.success("Sentiment : Positif")
                else:
                    st.error("Sentiment : Négatif")
            
            if uploaded_file:
                # Vérifier si le fichier a la bonne extension et si la colonne s'appelle "Commentaire"
                if uploaded_file.type == "text/csv":
                    uploaded_df = pd.read_csv(uploaded_file)
                    if 'Commentaire' in uploaded_df.columns:
                        # Prétraiter les commentaires du fichier CSV
                        uploaded_df['Commentaire'] = uploaded_df['Commentaire'].apply(preprocess_text)
                        
                        # Vectoriser les commentaires
                        uploaded_comments_vectorized = vectorizer.transform(uploaded_df['Commentaire'])
                        
                        # Faites les prédictions en utilisant le modèle
                        predictions = best_logistic_regression_model.predict(uploaded_comments_vectorized)
                        num_positives = (predictions == 1).sum()
                        num_negatives = (predictions == 0).sum()
            
                        st.write("Nombre de commentaires positifs : ", num_positives)
                        st.write("Nombre de commentaires négatifs : ", num_negatives)
                        
                        # Afficher la répartition des prédictions
                        sentiment_distribution = pd.Series(predictions).value_counts()
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(sentiment_distribution, labels=['Négatif', 'Positif'], autopct='%1.1f%%', startangle=140)
                        ax.set_title('Répartition des sentiments')
            
                        # Affichez le camembert dans l'application Streamlit
                        st.pyplot(fig)
                    else:
                        st.error("Le fichier CSV doit contenir une colonne nommée 'Commentaire'.")
                else:
                    st.error("Le fichier doit être au format CSV.")
    st.subheader("Conditions pour téléverser un fichier CSV :")
    st.info("Le fichier CSV doit contenir une colonne nommée 'Commentaire' pour être pris en charge. Assurez-vous que le fichier est au format CSV.")
    
    
elif page == pages[4]:
    
    st.title("Difficultés Rencontrées")

    st.header("Le Scrapping")
    st.write(
        "L'une des premières difficultés que nous avons rencontrées concernait le scrapping des commentaires. Initialement, "
        "notre projet visait à effectuer une analyse de sentiment sur le film 'Le Parrain' en utilisant des commentaires provenant du site IMDb. "
        "\n"
        "\n Cependant, IMDb a mis en place un bouton dynamique pour charger les commentaires,  "
        "ce qui a compliqué le processus de scrapping. Je mets le lien ici pour les curieux : https://www.imdb.com/title/tt0068646/reviews?ref_=tt_urv. \n" 
        "\n"
        "Cette difficulté nous a amenés à réorienter notre projet vers le site Allociné."
    )

    st.header("Utilisation de Streamlit")
    st.write(
        "Une autre difficulté que nous avons rencontrée était liée à notre utilisation de Streamlit pour développer l'interface "
        "de notre application web. Il s'agissait de notre première expérience avec cet outil, ce qui a demandé un apprentissage initial. "
        "Heureusement, la documentation complète et les ressources disponibles nous ont permis de surmonter cette difficulté. "
        "Au fur et à mesure, nous avons réussi à maîtriser les fonctionnalités de Streamlit pour créer une interface utilisateur conviviale."
    )

    st.header("Déploiement sur le Cloud")
    st.write(
        "Le dernier défi que nous avons relevé était le déploiement de notre modèle sur le cloud. Il s'agissait également de notre première expérience "
        "avec le déploiement de modèles de machine learning. Cette étape a nécessité une compréhension approfondie des services cloud et des solutions de déploiement. "
        "Après des efforts de recherche et d'expérimentation, nous avons réussi à héberger notre application web et à mettre en ligne notre modèle. Cela a été une expérience éducative précieuse pour toute l'équipe."
    )

    

elif page == pages[5]:
    
    st.title("Axes d'Amélioration")

    st.header("Amélioration de la Collecte de Données")
    st.write(
        "Pour améliorer notre modèle, nous pourrions envisager de collecter davantage de données de commentaires. "
        "Un ensemble de données plus vaste et diversifié pourrait permettre au modèle de généraliser plus efficacement."
    )

    st.header("Ingénierie de Caractéristiques")
    st.write(
        "L'ingénierie de caractéristiques est une étape cruciale dans la création d'un modèle performant. "
        "Nous pourrions explorer des techniques d'extraction de caractéristiques plus avancées pour mieux représenter le texte des commentaires."
    )

    st.header("Gestion de la Neutralité")
    st.write(
        "Actuellement, notre modèle classe les commentaires en deux catégories : positifs et négatifs. "
        "Il pourrait être intéressant d'ajouter une catégorie de neutralité pour les commentaires qui ne sont ni positifs ni négatifs."
    )

    st.header("Optimisation des Hyperparamètres")
    st.write(
        "Nous pourrions continuer à optimiser les hyperparamètres de notre modèle pour améliorer ses performances. "
        "Cela pourrait inclure des ajustements plus fins des paramètres du modèle."
    )
    
    st.header("Améliorations de l'Application web")
    st.write(
        "Pour rendre notre application plus conviviale et efficace, nous pourrions envisager les améliorations suivantes :"
    )
    st.write("- Ajouter une fonction de recherche pour trouver des commentaires spécifiques.")
    st.write("- Proposer des graphiques interactifs pour une meilleure visualisation des données.")
    st.write("- Permettre aux utilisateurs de charger des commentaires à partir d'autres sources.")
    st.write("- Ajouter une fonction de résumé automatique des commentaires.")
    
    
    
    













