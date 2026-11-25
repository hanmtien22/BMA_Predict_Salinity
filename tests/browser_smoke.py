"""Exercise the Pages project URL on desktop and mobile widths in CI."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait


def check_demo(driver, url):
    find = lambda element_id: driver.find_element(By.ID, element_id)
    for width, height in [(1280, 900), (390, 844)]:
        driver.set_window_size(width, height)
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(lambda _: find("value").text not in ("", "…"))
        original = find("value").text
        assert find("date-text").text == "15/04"
        assert driver.execute_script(
            "return document.documentElement.scrollWidth <= window.innerWidth")
        for element_id in ("day", "ndvi", "ndwi"):
            before = find("value").text
            find(element_id).send_keys(Keys.END)
            wait.until(lambda _: find("value").text != before)
        assert find("date-text").text == "31/12"
        driver.find_element(By.CSS_SELECTOR, "button[type=reset]").click()
        wait.until(lambda _: find("value").text == original)
        assert find("date-text").text == "15/04"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as tmp:
        shutil.copytree(root / "demo", Path(tmp) / "BMA_Predict_Salinity")
        handler = partial(SimpleHTTPRequestHandler, directory=tmp)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            Thread(target=server.serve_forever, daemon=True).start()
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            try:
                with webdriver.Chrome(options=options) as browser:
                    check_demo(browser, f"http://127.0.0.1:{server.server_port}/BMA_Predict_Salinity/")
            finally:
                server.shutdown()
