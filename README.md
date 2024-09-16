# Retrieval Augmented Generation (RAG) News

[![tests](https://github.com/KentaWood/ragNews/actions/workflows/main.yml/badge.svg)](https://github.com/KentaWood/ragNews/actions/workflows/main.yml)

## Overview

RAG News uses the **Groq API** to answer user queries by providing relevant information from a database of news articles. It employs **Retrieval Augmented Generation (RAG)**, a technique where an LLM (Language Learning Model) is given context from relevant articles stored in a SQLite database to provide accurate and informed answers.

## Key Features

- **Interactive QA Session**: The system runs a question-answer session with news articles using the Groq LLM API.
- **News Article Database**: Allows adding articles to a SQLite database via URLs.
- **Keyword-Based Retrieval**: Extracts keywords from input queries to retrieve related articles.
- **Concise Responses**: Provides responses based solely on article summaries for more accurate answers.

## Installation


1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your **Groq API Key** as an environment variable:
   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```

## Usage

Run an interactive QA session:

```bash
python3 ragNews.py
```

You can also add new articles to the database with a URL:

```bash
python3 ragNews.py --add_url https://example.com/article
```

### Command-Line Options

- `--add_url <url>`: Add a new article to the database from the provided URL.
- `--db <db_path>`: Use a different SQLite database. Default is `ragnews.db`.
- `--loglevel <level>`: Set the logging level (e.g., `info`, `debug`, `warning`). Default is `warning`.

## Example

**Input**: 
```bash
ragnews> Who are the 2024 presidential nominees?
```

**Output**:
```
1. Kamala Harris, the Democratic nominee, who has agreed to the rules of the upcoming TV debate against her opponent Donald Trump, including allowing mics to be muted when it is the other person’s turn to speak.
2. Donald Trump, the Republican nominee, who has officially clinched the 2024 GOP nomination after winning the primaries and caucuses.
```
