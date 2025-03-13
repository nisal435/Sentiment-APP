from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from sqlite3 import Error
from datetime import datetime
from transformers import pipeline
from fastapi.responses import JSONResponse
import os

# FastAPI app
app = FastAPI()

# Initialize the sentiment analysis pipeline using HuggingFace
sentiment_analyzer = pipeline('sentiment-analysis')

# SQLite Database Setup
DATABASE = "sentiment_results.db"

# Function to check and recreate the database if corrupted
def check_and_create_db():
    # Check if the database file exists
    if not os.path.exists(DATABASE):
        # Database does not exist, create a fresh one
        create_table()

    try:
        # Try to open the database to check for corruption
        conn = sqlite3.connect(DATABASE)
        conn.close()
    except sqlite3.DatabaseError:
        # If database is corrupted, delete and recreate it
        os.remove(DATABASE)  # Delete the corrupted database
        create_table()  # Create a new one

# Create a connection to the SQLite database
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE)
    except Error as e:
        print(f"Error: {e}")
    return conn

# Create table if it doesn't exist
def create_table():
    conn = create_connection()
    if conn:
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
            conn.execute(create_table_sql)
            conn.commit()
        except Error as e:
            print(f"Error: {e}")
        finally:
            conn.close()

# Call this function once to ensure the table is created and check for corruption
check_and_create_db()

# Pydantic model for incoming data
class SentimentRequest(BaseModel):
    text: str

@app.post("/predict/")
async def analyze_sentiment(request: SentimentRequest):
    sentiment_result = sentiment_analyzer(request.text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']
    
    conn = create_connection()
    if conn:
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            insert_sql = "INSERT INTO sentiments (text, sentiment_label, sentiment_score, timestamp) VALUES (?, ?, ?, ?)"
            conn.execute(insert_sql, (request.text, sentiment_label, sentiment_score, timestamp))
            conn.commit()
        except Error as e:
            print(f"Error while inserting data: {e}")
            return JSONResponse(status_code=500, content={"message": "Error saving sentiment data."})
        finally:
            conn.close()

    return {"label": sentiment_label, "score": sentiment_score}

@app.get("/history/")
async def get_sentiment_history():
    conn = create_connection()
    sentiment_data = []
    if conn:
        try:
            select_sql = "SELECT * FROM sentiments"
            cursor = conn.execute(select_sql)
            rows = cursor.fetchall()
            for row in rows:
                sentiment_data.append({
                    "id": row[0],
                    "text": row[1],
                    "sentiment_label": row[2],
                    "sentiment_score": row[3],
                    "timestamp": row[4]
                })
        except Error as e:
            print(f"Error while fetching data: {e}")
            return JSONResponse(status_code=500, content={"message": "Error fetching sentiment history."})
        finally:
            conn.close()
    
    return {"sentiments": sentiment_data} 
