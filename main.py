# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# faq_data=pd.read_csv('faq_data.csv')
# # print(faq_data)
# vectorizer=TfidfVectorizer()
# faq_vector=vectorizer.fit_transform(faq_data)

# def get_response(user_query):
#     query_vector=vectorizer.transform([user_query])
#     similarity_scores=cosine_similarity(query_vector,faq_vector)
#     best_match=np.argmax(similarity_scores)
#     best_score=similarity_scores[0][best_match]

#     if best_score>0.5:
#         return faq_data['Answer'][best_match]
#     else:
#         return "I'm sorry to answer this"

# while True:
#     user_input = input("you: ")
#     if user_input.lower() == 'quit':
#         print("Chatbot: Goodbye!")
#         break
#     response=get_response(user_input)
#     print("Chatbot: ",response)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
faq_data = pd.read_csv('faq_data.csv')

# Check for missing or invalid data
if 'Question' not in faq_data.columns or 'Answer' not in faq_data.columns:
    raise ValueError("CSV file must contain 'Question' and 'Answer' columns.")
faq_data.dropna(subset=['Question', 'Answer'], inplace=True)

# Create TF-IDF vectorizer and transform the questions
vectorizer = TfidfVectorizer()
faq_vector = vectorizer.fit_transform(faq_data['Question'])

def get_response(user_query):
    # Transform the user query into a vector
    query_vector = vectorizer.transform([user_query])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, faq_vector)
    
    # Find the best match
    best_match = np.argmax(similarity_scores)
    best_score = similarity_scores[0][best_match]

    # Check if similarity score is above the threshold
    if best_score > 0.5:
        return faq_data['Answer'].iloc[best_match]
    else:
        return "I'm sorry, I don't have an answer for that."

# Interactive chatbot loop
print("Chatbot: Hello! Ask me a question or type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
