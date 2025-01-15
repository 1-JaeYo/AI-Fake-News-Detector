# AI-Fake-News-Detector
An AI-powered Fake News Detection System that uses a pre-trained BERT model to classify news articles as "Fake" or "Real." It features a Flask back end for API services and a React-based front end for user interaction. This project demonstrates advanced natural language processing (NLP) and full-stack web development.

### Features

* Real-time classification of news articles into "Fake" or "Real."
* Flask back end with an API for predictions.
* Interactive React front end for user input and results display.
* BERT-based NLP model for accurate predictions.
* Suitable for deployment on cloud platforms like Heroku, Netlify, or AWS.

### Technologies Used

* **Back End**: Flask, Python, Transformers (Hugging Face), Torch
* **Front End**: React, Axios
* **Model**: BERT (Bidirectional Encoder Representations from Transformers)

### Model Details

* Model: BERT-based classifier fine-tuned for fake news detection.
* Dataset: Includes over 45,000 samples of fake and real news.
* Training: The model was trained on a CPU with optimizations like gradient accumulation and reduced sequence length.

### Project Structure

* `app.py`: Flask application entry point
* `models/`: Directory containing the BERT-based model and its dependencies
* `static/`: Directory containing static files for the React front end
* `templates/`: Directory containing HTML templates for the React front end

### Setup and Installation

* Clone the repository: `git clone https://github.com/username/AI-Fake-News-Detector.git`
* Install dependencies: `pip install -r requirements.txt`
* Start the Flask application: `python app.py`
* Start the React front end: `npm start`# AI-Fake-News-Detector
An AI-powered Fake News Detection System that uses a pre-trained BERT model to classify news articles as "Fake" or "Real." It features a Flask back end for API services and a React-based front end for user interaction. This project demonstrates advanced natural language processing (NLP) and full-stack web development.
- Real-time classification of news articles into "Fake" or "Real."
- Flask back end with an API for predictions.
- Interactive React front end for user input and results display.
- BERT-based NLP model for accurate predictions.
- Suitable for deployment on cloud platforms like Heroku, Netlify, or AWS.
- **Back End**: Flask, Python, Transformers (Hugging Face), Torch
- **Front End**: React, Axios
- **Model**: BERT (Bidirectional Encoder Representations from Transformers)


- Model: BERT-based classifier fine-tuned for fake news detection.
- Dataset: Includes over 45,000 samples of fake and real news.
- Training: The model was trained on a CPU with optimizations like gradient accumulation and reduced sequence length.

