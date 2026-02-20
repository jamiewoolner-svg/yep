#!/bin/bash

# This script automates the process of fetching stock data from external sources.

# Define the data source URL
DATA_SOURCE_URL="https://api.example.com/stocks"

# Define the output file
OUTPUT_FILE="../data/stocks.json"

# Fetch the stock data and save it to the output file
curl -s $DATA_SOURCE_URL -o $OUTPUT_FILE

# Check if the fetch was successful
if [ $? -eq 0 ]; then
  echo "Stock data fetched successfully and saved to $OUTPUT_FILE"
else
  echo "Failed to fetch stock data"
fi