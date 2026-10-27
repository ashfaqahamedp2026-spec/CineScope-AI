# CineScope AI

An interactive South Indian movie catalogue and English movie-review
sentiment analysis project using four machine learning algorithms.

## Features

- Search and browse movies by name and language
- Movie posters and movie detail pages
- Review submission and star ratings
- Positive and negative sentiment comparisons
- Multinomial Naive Bayes, Logistic Regression, Linear SVM,
  and Random Forest
- Interactive performance charts and confusion matrices
- CSV bulk analysis and downloadable results

## Run locally

Install dependencies:

    pip install -r requirements.txt

Run the app:

    streamlit run app.py

## Project limitations

The sentiment models were trained on English-language movie reviews.
They are not validated for Tamil, Telugu, Malayalam, or Kannada.
Demonstration reviews are fictional examples.

The current application saves user-submitted reviews to the running
environment. Persistent public review storage requires a database.

## Poster attribution

Movie posters and associated metadata are retrieved from TMDB where
available. This product uses the TMDB API but is not endorsed or
certified by TMDB.

## Academic use

Report the measured test-set results and dataset limitations accurately.
