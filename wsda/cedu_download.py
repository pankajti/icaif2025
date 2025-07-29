from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import os
import requests
from bs4 import BeautifulSoup

# Directory to store downloaded files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configure headless browser
def get_driver():
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

# Download a file from URL
def download_file(url, filename):
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(DATA_DIR, filename), 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

# Scrape and download TXT and PDF files from all pages
def scrape_wasde_pages():
    base_url = "https://usda.library.cornell.edu/concern/publications/3t945q76s?locale=en"
    driver = get_driver()
    driver.get(base_url)
    time.sleep(3)

    page_num = 1
    while True:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        pdf_links = [a.get('href') for a in soup.find_all("a", ) if a.get('href') is not None and a.get('href').endswith('pdf')]
        text_links = [a.get('href') for a in soup.find_all("a", ) if a.get('href') is not None and a.get('href').endswith('pdf')]

        print(f"Processing page {page_num}...")
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/downloads/']")

        for link in links:
            file_url = link.get_attribute("href")
            if file_url.endswith(".pdf") or file_url.endswith(".txt"):
                filename = file_url.split("/")[-1]
                download_file(file_url, filename)

        # Try to click next page
        try:
            next_button = driver.find_element(By.LINK_TEXT, "Next →")
            next_button.click()
            time.sleep(2)
            page_num += 1
        except NoSuchElementException:
            print("No more pages.")
            break

    driver.quit()
#Next »
scrape_wasde_pages()
