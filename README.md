# NBA-games-prediction-app
An end-to-end **machine learning web application** that predicts NBA game outcomes using a combination of **ELO rating modeling** and **Logistic Regression**.   This project demonstrates the complete pipeline from data preprocessing and feature engineering to model training and web deployment.

Project Overview

Predicting NBA match outcomes requires understanding team strength, historical performance, and home-court advantage.  
This project leverages an **ELO rating system** to quantify team strength and applies a **probability-based machine learning model** to predict the winner of NBA games.

Users select the **Home Team** and **Visitor Team** from dropdown menus, and the system predicts:
- The most likely winner
- The probability of a home-team victory

---

Key Features

- 🏀 Supports **30 official NBA teams**
- 📊 Dynamic **ELO rating system** with home-court advantage
- 🤖 Logistic Regression model for prediction
- 🔽 Dropdown-based team selection (prevents invalid input)
- ❌ Validation to avoid selecting the same team
- 📈 Probability-based output instead of binary prediction
- 🎨 NBA-themed, modern user interface
- 🌐 Deployed as a Flask web application

---

Methodology

1. Historical NBA match data is sorted chronologically
2. Each team starts with a base ELO rating
3. ELO ratings are updated after every game based on:
   - Match outcome
   - Opponent strength
   - Home-court advantage
4. The **ELO difference** between teams is used as the primary feature
5. A **Logistic Regression** model predicts the probability of a home-team win
6. The trained model is deployed using Flask and accessed via a web interface

---

Tech Stack & Tools

| Category      | Tools                       |
|---------------|-----------------------------|
| Language      | Python                      |
| ML Model      | Logistic Regression         |
| Rating System | ELO                         |
| Backend       | Flask                       |
| Frontend      | HTML, CSS                   |
| Libraries     | pandas, numpy, scikit-learn |

---

Project Structure

NBA-GAMES-APP/
│
├── app.py # Backend logic and ML model
├── games.csv # Historical NBA match dataset
│
├── templates/
│ └── index.html # Frontend HTML
│
└── static/
└── style.css # NBA-themed CSS
