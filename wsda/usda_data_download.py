from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import random  # Import for random delays

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to store downloaded files
BASE_DATA_DIR = "downloaded_reports"
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# --- Configuration for Rate Limiting ---
MIN_PAGE_SLEEP = 5  # Minimum seconds to wait after page load/navigation
MAX_PAGE_SLEEP = 10  # Maximum seconds to wait after page load/navigation
MIN_DOWNLOAD_SLEEP = 1  # Minimum seconds to wait between individual file downloads
MAX_DOWNLOAD_SLEEP = 3  # Maximum seconds to wait between individual file downloads
MAX_RETRIES = 5  # Max retries for a failed download due to rate limit/connection
INITIAL_RETRY_DELAY = 5  # Initial delay for download retries (seconds)
MAX_PAGE=70

def get_driver():
    """Initializes and returns a headless Chrome WebDriver."""
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass some bot detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")  # Set a common User-Agent

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logging.info("Chrome WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.error(f"Error initializing WebDriver: {e}")
        logging.error("Please ensure ChromeDriver is compatible with your Chrome browser version.")
        return None


def download_file(url, filename, year_folder):
    """Downloads a file from a given URL to the specified year_folder with retries."""

    # Create the year-specific directory
    year_dir = os.path.join(BASE_DATA_DIR, str(year_folder))
    os.makedirs(year_dir, exist_ok=True)

    filepath = os.path.join(year_dir, filename)

    if os.path.exists(filepath):
        logging.info(f"File {filename} in year {year_folder} already exists. Skipping download.")
        return

    retries = 0
    current_delay = INITIAL_RETRY_DELAY

    while retries < MAX_RETRIES:
        try:
            logging.info(
                f"Attempting to download {filename} to {year_folder} from {url} (Attempt {retries + 1}/{MAX_RETRIES})...")

            # Add a small random delay before each download attempt
            time.sleep(random.uniform(MIN_DOWNLOAD_SLEEP, MAX_DOWNLOAD_SLEEP))

            response = requests.get(url, stream=True, timeout=15)  # Increased timeout for downloads
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Successfully downloaded {filename} to {year_folder}.")
            return  # Exit loop on success

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = int(retry_after) if retry_after else current_delay
                logging.warning(
                    f"Rate limit hit for {filename}. Retrying after {wait_time} seconds (Retry {retries + 1})...")
                time.sleep(wait_time)
                current_delay *= 2  # Exponential backoff for next retry
            else:
                logging.error(f"HTTP Error {e.response.status_code} for {filename} from {url}: {e}")
                break  # Break for other HTTP errors (e.g., 404, 500)
        except requests.exceptions.RequestException as e:
            logging.warning(
                f"Network error downloading {filename} from {url}: {e}. Retrying in {current_delay} seconds (Attempt {retries + 1})...")
            time.sleep(current_delay)
            current_delay *= 2  # Exponential backoff
        except Exception as e:
            logging.error(f"An unexpected error occurred while downloading {filename}: {e}")
            break  # Break for other unexpected errors

        retries += 1

    logging.error(f"Failed to download {filename} after {MAX_RETRIES} attempts.")


def scrape_wasde_pages():
    """
    Scrapes the USDA WASDE publication page for PDF, TXT, XLS, and XML links
    and downloads them, organizing them into year-wise folders.
    It iterates through all available pages.
    """
    base_url = "https://usda.library.cornell.edu/concern/publications/3t945q76s?locale=en"
    base_url = "https://usda.library.cornell.edu/concern/publications/3t945q76s?locale=en&page=60#release-items"

    driver = get_driver()
    if not driver:
        logging.error("Failed to get WebDriver. Exiting.")
        return

    try:
        driver.get(base_url)

        # Use an explicit wait for the 'release-items' tbody to be present
        WebDriverWait(driver, 20).until(  # Increased wait time for initial load
            EC.presence_of_element_located((By.ID, "release-items"))
        )
        logging.info("Initial page loaded and release items are present.")

        # Add a delay after initial page load to avoid immediate rate limiting
        time.sleep(random.uniform(MIN_PAGE_SLEEP, MAX_PAGE_SLEEP))

        page_num = 1
        downloaded_urls = set()

        while True:

            logging.info(f"Processing page {page_num}...")
            if page_num>MAX_PAGE:
                logging.info(f"max page {MAX_PAGE} reached")

                break

            soup = BeautifulSoup(driver.page_source, "html.parser")
            release_rows = soup.find_all('tr', class_='release attributes row')

            if not release_rows and page_num == 1:
                logging.warning("No release rows found on the first page. Check URL or selectors.")
                break
            elif not release_rows:
                logging.info(f"No more release rows found on page {page_num}. Assuming end of publications.")
                break

            for row in release_rows:
                date_td = row.find('td', class_='attribute date_uploaded')
                release_date_str = date_td.get_text(strip=True) if date_td else None

                year = None
                if release_date_str:
                    try:
                        # Replace multiple spaces with a single space for robust parsing
                        release_date_obj = datetime.strptime(release_date_str.replace("  ", " "), '%b %d, %Y')
                        year = release_date_obj.year
                    except ValueError:
                        logging.warning(
                            f"Could not parse date: '{release_date_str}'. Assigning to 'unknown_year' folder.")
                        year = "unknown_year"

                if year is None:
                    year = "unknown_year"

                download_links_in_row = row.select('div.btn-group-file a.download_btn.file_download')

                for link_tag in download_links_in_row:
                    file_url = link_tag.get('href')
                    if file_url and file_url not in downloaded_urls:
                        file_format_div = link_tag.find('div')
                        file_format = file_format_div.get_text(strip=True).upper() if file_format_div else 'UNKNOWN'

                        if file_format == 'XML':
                            logging.info(f"Skipping XML file: {file_url}")
                            continue  # Go to the next link in the loop

                        original_filename_parts = file_url.split("/")[-1].rsplit('.', 1)
                        if len(original_filename_parts) == 2:
                            base_name, ext = original_filename_parts
                            if base_name=='latest':
                                base_name=f"wasde{release_date_obj.strftime('%m%d')}"
                            filename = f"{base_name}_{file_format}.{ext}"
                        else:
                            # Fallback to a timestamped name if original filename is problematic
                            filename = f"downloaded_file_{int(time.time())}_{file_format}"

                        download_file(file_url, filename, year)
                        downloaded_urls.add(file_url)

                        # Add a small random delay between each individual file download
                        time.sleep(random.uniform(MIN_DOWNLOAD_SLEEP, MAX_DOWNLOAD_SLEEP))

            # --- Pagination Logic ---
            try:
                # Wait for the "Next →" button to be clickable
                next_button = WebDriverWait(driver, 15).until(  # Increased wait time
                    EC.element_to_be_clickable((By.LINK_TEXT, "Next »"))
                )

                next_button.click()
                logging.info(f"Navigated to next page. Current page: {page_num}")
                time.sleep(10)

                # After clicking, wait for the release items on the *new* page to be present
                WebDriverWait(driver, 15).until(  # Increased wait time
                    EC.presence_of_element_located((By.ID, "release-items"))
                )

                # Add a randomized delay after navigating to a new page
                time.sleep(random.uniform(MIN_PAGE_SLEEP, MAX_PAGE_SLEEP))

                page_num += 1
            except (NoSuchElementException, TimeoutException):
                logging.info(
                    "No 'Next →' button found or it's no longer clickable/present after waiting. All pages processed.")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred during pagination: {e}")
                break

    except Exception as e:
        logging.error(f"An unhandled error occurred during the main scraping process: {e}")
    finally:
        if driver:
            driver.quit()
            logging.info("WebDriver closed.")


if __name__ == "__main__":
    scrape_wasde_pages()