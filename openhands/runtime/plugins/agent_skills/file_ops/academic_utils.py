import re
from fuzzywuzzy import fuzz
import arxiv
import os
import requests
from selenium.webdriver.common.by import By
from sel.selenium_tester import driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from semanticscholar import SemanticScholar

def clean_filename(filename: str):
    # remove special characters
    filename = re.sub(r'[^\w\s-]', '', filename)
    # remove leading and trailing whitespace
    filename = filename.strip()
    return filename
def download_arxiv_pdf(query: str):
    """
    Searches arXiv for papers matching the given query and saves the pdf to the current directory.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.

    Returns:
        A list of arXiv results.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results= 10,  # Increase initial results for better fuzzy matching
        sort_by=arxiv.SortCriterion.Relevance # not working as expected
    )
    results = list(client.results(search))

    # Use fuzzy matching to find relevant results
    relevant_results = []
    for result in results:
        score = fuzz.partial_ratio(query.lower(), result.title.lower())
        if score >= 80:  # Adjust the threshold as needed
            relevant_results.append((result, score))

    # Sort by fuzzy matching score and return top results
    relevant_results.sort(key=lambda x: x[1], reverse=True)
    if len(relevant_results) > 0:
        relevant_result = relevant_results[0][0]
        print(relevant_result.download_pdf(filename=f"{clean_filename(relevant_result.title)}.pdf"))
        print(f"Downloaded to {relevant_result.title}.pdf")
    else:
        print("No relevant results found")
    
def download_pdf_from_url(url: str, name: str = None):
    if name is None:
        name = url.split('/')[-1]
    with open(name, 'wb') as f:
        f.write(requests.get(url).content)

def download_semanticscholar_pdf(query: str = None, url: str = None):
    sch = SemanticScholar()
    if query:
        results = sch.search_paper(query)
        print(f'{results.total} results.', f'First occurrence: {results[0].title}.')

        if results.total == 0:
            print("No results found")
            return
        url = results[0].url
    driver.get(url)
    try:
        s='[data-test-id="cookie-banner__dismiss-btn"]'
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, s))).click()
    except:
        pass
    s='[data-test-id="icon-disclosure"]'
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, s))).click()
    s='[data-test-id="paper-link"]'
    link = driver.find_element(By.CSS_SELECTOR, s).get_attribute('href')
    if 'arxiv' in link:
        print(f"Downloading from {link}")
        download_pdf_from_url(link)
    else:
        print(f"Download from {link}")
if __name__ == "__main__":  
    query = "OpenHands: An Open Platform for AI Software Developers as Generalist Agents"
    url = 'https://www.semanticscholar.org/paper/1d07e5b6f978cf69c0186f3d5f434fa92d471e46'
    # download_semanticscholar_pdf(url=url)
    url = 'https://arxiv.org/pdf/2407.16741.pdf'
    download_pdf_from_url(url)


