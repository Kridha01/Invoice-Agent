# Invoice Agent – RAG-Based Business Invoice Processor

## Overview

Invoice Agent is a web-based application that leverages **RAG (Retrieval-Augmented Generation)** and **LLMs** to automatically extract structured data from invoices. Users can upload PDF invoices and a field template in Excel format, and the app generates a processed Excel file with extracted information, scores, and feedback.

This project uses **Python 3.10**, **LangChain**, **Google Gemini API**, and **Streamlit** for the frontend. It is designed to scale for multiple users via cloud deployment (AWS recommended).

## Features

- Upload PDF invoices and Excel field templates through an intuitive Streamlit UI.
- Automatic extraction of fields using **RAG + LLM**.
