#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Director Alpha Setup...${NC}"

# 1. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate Virtual Environment
source .venv/bin/activate

# 3. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install pandas numpy scipy pyarrow wrds pytest psycopg2-binary python-dotenv

# 4. WRDS Configuration
echo -e "${GREEN}WRDS Configuration${NC}"
if [ -f ".env" ]; then
    echo ".env file already exists."
    read -p "Do you want to overwrite WRDS credentials? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "Skipping WRDS configuration."
        exit 0
    fi
fi

echo "Please enter your WRDS credentials."
read -p "WRDS Username: " wrds_user
read -s -p "WRDS Password: " wrds_pass
echo ""

# Save to .env
echo "WRDS_USERNAME=$wrds_user" > .env
echo "WRDS_PASSWORD=$wrds_pass" >> .env

echo -e "${GREEN}Setup Complete!${NC}"
echo "To run the pipeline, use: source .venv/bin/activate && python -m director_alpha.phase0_universe"
