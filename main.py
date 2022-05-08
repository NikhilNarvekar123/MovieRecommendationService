import streamlit as st
import Recommender

# Streamlit title and input components added
st.title('Movie Recommendation Tool')
option = st.selectbox('Choose recommender model:', ('KNN', 'Bag of Words', 'XXX'))
title = st.text_input('Movie title', 'Iron Man')

# When search pressed, run model and display results
if st.button('Search'):
    movie_data_path = './data/movies.csv'
    user_data_path = './data/ratings.csv'
    if option == 'KNN':
        option = 'knn'
    elif option == 'Bag of Words':
        option = 'bow'
    else:
        option = 'xxx'

    # initialize recommender object and predict
    recommender = Recommender.Recommender(movie_data_path, user_data_path, option, 10)
    recommendations = recommender.get_recommendations(title)
    
    # display output
    st.write(recommendations)