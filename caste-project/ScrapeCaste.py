"""
Script for scraping caste data from Google Maps.
"""

import time
import csv
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from itertools import product

# --- Setup driver ---
print("[INFO] Setting up Chrome driver...")
options = uc.ChromeOptions()
options.headless = False  # Set to True if you don't need a browser UI
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = uc.Chrome(options=options)

# --- Output file ---
output_file = "yadav_places_all_india.csv"
print(f"[INFO] Writing output to {output_file}")
csv_file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["Latitude", "Longitude", "Name", "Address"])

# --- Functions ---
def search_yadav_in_map(lat, lng):
    """Pan map to given lat/lng and search for 'Yadav'"""
    url = f"https://www.google.com/maps/search/yadav/@{lat},{lng},13z/"
    print(f"[INFO] Navigating to {url}")
    driver.get(url)
    time.sleep(5)  # Wait for initial load

def scroll_and_collect(lat, lng):
    """Scroll results and collect name and address"""
    print("[INFO] Scrolling and collecting places...")
    
    # Wait for results to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.Nv2PK'))
        )
    except Exception as e:
        print(f"[WARN] No results found at {lat},{lng}: {e}")
        return

    # Scroll to load more results
    scroll_container = driver.find_element(By.CSS_SELECTOR, 'div.m6QErb[role="feed"]')
    for _ in range(5):  # Scroll 5 times
        driver.execute_script("arguments[0].scrollBy(0, 1000);", scroll_container)
        time.sleep(2)

    # Extract business data
    places = driver.find_elements(By.CSS_SELECTOR, 'div.Nv2PK')  # Updated selector
    print(f"[INFO] Found {len(places)} places.")
    
    for place in places:
        try:
            name = place.find_element(By.CSS_SELECTOR, 'div.qBF1Pd').text
            address = place.find_element(By.CSS_SELECTOR, 'div.W4Efsd:not(:has(span.k48Abe))').text
            print(f"[DATA] ({lat},{lng}) Place: {name} | {address}")
            writer.writerow([lat, lng, name, address])
        except Exception as e:
            print(f"[ERROR] Failed to extract place info: {e}")
            continue

# --- Main Grid Loop (All India) ---
# India's geographic coordinates (8°N to 37°N latitude, 68°E to 97°E longitude)
# Using 1-degree steps for reasonable coverage (adjust as needed)
lat_range = range(8, 38, 1)    # South to North
lng_range = range(68, 98, 1)   # West to East

print("[INFO] Starting search across all of India...")
for lat, lng in product(lat_range, lng_range):
    print(f"[INFO] Searching at coordinates: {lat}, {lng}")
    try:
        search_yadav_in_map(lat, lng)
        scroll_and_collect(lat, lng)
    except Exception as e:
        print(f"[ERROR] Unexpected error at {lat},{lng}: {e}")
        continue

print("[INFO] Done scraping. Closing browser and file.")
driver.quit()
csv_file.close()
print(f"[INFO] Data saved to: {output_file}")
