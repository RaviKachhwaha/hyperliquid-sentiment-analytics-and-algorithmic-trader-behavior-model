@echo off
echo ========================================================
echo   Uploading Project to GitHub
echo ========================================================
echo.

echo [1/5] Initializing Git repository...
git init

echo [2/5] Adding all files to staging...
git add .

echo [3/5] Committing files...
git commit -m "Initial commit: Hyperliquid trader analytics with advanced ML and Math features"

echo [4/5] Setting main branch...
git branch -M main

echo [5/5] Adding remote repository and pushing...
:: Remove origin if it already exists, to prevent errors
git remote remove origin 2>nul
git remote add origin https://github.com/RaviKachhwaha/hyperliquid-sentiment-analytics-and-algorithmic-trader-behavior-model.git
git push -u origin main

echo.
echo ========================================================
echo   Upload Complete!
echo ========================================================
pause
