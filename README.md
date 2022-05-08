# Movie Recommender System

The folder contains the source code for a movie recommendation service.

&nbsp;
## Files

Main.py contains the driver code which runs the Streamlit demo.
Recommender.py contains the source code for the entire recommendation system (with various models).
MovieRecommendation.ipynb contains the Python notebook which runs through the recommendation service.
The data folder contains the movie-user data for use in the recommendation models.

&nbsp;
## Dependencies

The following packages are needed:
- Pandas
- Numpy
- Scipy
- Sklearn
- Matplotlib
- Seaborn
- Streamlit
- Python

&nbsp;
## Running Demos

There are two ways to demo the system.

### Streamlit

By running the following command in a terminal in the project directory, a site will be launched locally. The site
will allow you to select a model type, enter a movie name, and then it will generate a list 
of 10 related movie recommendations. 

```python
streamlit run main.py
```

### Notebook

Otherwise, the notebook can be easily run to demo the recommendation system's capabilities. The recommendation system code is the same
between the notebook and the streamlit demo class.