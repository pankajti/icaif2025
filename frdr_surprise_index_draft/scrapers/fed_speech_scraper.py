from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import time

from frdr.database.dao.fed_speech_dao import FedSpeechDao
from frdr.database.schema.core import FedSpeech

BASE_URL = "https://www.federalreserve.gov"
MAX_PAGES_TO_LOAD=65

def get_driver():
    options = Options()
    #options.add_argument("--headless")
    return webdriver.Chrome(options=options)

#article > ul.visible-xs-inline-block.ng-untouched.ng-valid.ng-isolate-scope.pagination.ng-not-empty.ng-dirty.ng-valid-parse > li.pagination-next.ng-scope > a

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
from datetime import datetime
import time

BASE_URL = "https://www.federalreserve.gov"

def get_all_speech_links_click(driver):
    """
    Clicks through all pagination pages using Selenium and collects speech metadata.
    Returns a flat list of (title, url, date) tuples.
    """
    driver.get(f"{BASE_URL}/newsevents/speeches.htm")
    time.sleep(2)

    all_speeches = []

    while True:
        # 🧠 Refresh the BeautifulSoup object after each page load
        soup = BeautifulSoup(driver.page_source, "html.parser")
        speech_divs = soup.select("div.col-xs-12.col-sm-8.col-md-8 div.row")

        for div in speech_divs:
            link_tag = div.find("a")
            date_div = div.find("div", class_="col-xs-3")
            if link_tag and date_div:
                title = link_tag.text.strip()
                url = BASE_URL + link_tag["href"]
                location = div.find('p', class_='result__location').text
                try:
                    date = datetime.strptime(date_div.text.strip(), "%m/%d/%Y")
                    all_speeches.append((title, url, date,location))
                except ValueError:
                    continue  # skip invalid dates

        # 🧭 Attempt to find and click "Next"
        try:
            total_pages = len(all_speeches)//20
            print(f"Running for page {total_pages}")
            next_button = driver.find_element(By.LINK_TEXT, "Next")
            if "disabled" in next_button.get_attribute("class") or   total_pages==MAX_PAGES_TO_LOAD:
                break  # no more pages
            next_button.click()
            time.sleep(2)  # wait for the new page to load
        except NoSuchElementException:
            print(all_speeches)
            print("error")
            break  # no "Next" button found
        except Exception as e:
            print(f"exception for page {total_pages}")

            raise e

    return all_speeches


def fetch_speech_text(driver, url):
    driver.get(url)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    paragraphs = soup.select("div.col-xs-12.col-sm-8.col-md-8 p")
    return "\n\n".join([p.get_text() for p in paragraphs])

def extract_speaker(text):
    third_line = text.split("\n")[2]
    return third_line.strip() if third_line is not None else third_line

def scrape_and_store():
    dao = FedSpeechDao()
    driver = get_driver()
    speeches = set(get_all_speech_links_click(driver))

    # article > ul.visible-xs-inline-block.ng-untouched.ng-valid.ng-isolate-scope.pagination.ng-not-empty.ng-dirty.ng-valid-parse

    for title, url, date,location in tqdm(speeches, desc="Scraping speeches"):
        try:
            text = fetch_speech_text(driver, url)
            speaker = extract_speaker(text)
            speech = FedSpeech(
                date=date,
                title=title,
                url=url,
                text=text,
                speaker=speaker,
                location=location
            )
            dao.add_fed_speech(speech)
        except Exception as e:
            print(f"Error with {url}: {e}")

    driver.quit()

if __name__ == "__main__":
    scrape_and_store()
