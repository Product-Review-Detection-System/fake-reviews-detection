from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
import requests



def get_reviews_via_selenium(link):
    chrome_options = Options()  
    chrome_options.add_argument("--headless")
    browser = webdriver.Chrome(executable_path='./chromedriver', chrome_options=chrome_options)
    browser.get(link)
    delay = 5
    try:
        myElem = WebDriverWait(browser, delay).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'next-pagination-list'))
        )
        print("Page is ready!")
    except TimeoutException:
        print("Loading took too much time!")

    item_id = browser.execute_script('return items.id')
    browser.quit()

    print("Item Id:", item_id)
    return get_reviews(item_id=item_id)

def get_reviews(url=None, item_id=None):
    if not item_id and not url:
        print('Please provide an item id or a url')
    elif not item_id and url:
        item_id = url.split('.html')[0].split('-')[-2][1:]

    review_api = 'https://my.daraz.pk/pdp/review/getReviewList?itemId={item_id}&pageSize=5000'

    resp = requests.request('GET', review_api.format(
            item_id = item_id,
        )
    )
    if resp.status_code == 200:
        data = resp.json()
    else:
        print(resp.status_code)

    return data
# dat = get_reviews(url="https://www.daraz.pk/products/universal-smartphone-mini-flexible-tripod-stand-handle-grip-for-mobile-phones-cameras-i182274446-s1363706534.html?spm=a2a0e.home.flashSale.5.6e1f4937kNRvNV&search=1&mp=1&c=fs")
#
# print(dat['model'])
# if __name__ == '__main__':
#
#     link = "https://www.daraz.pk/products/v-shape-multi-function-mobile-phone-holder-stand-portable-phone-adjustable-stand-universal-foldable-cell-stand-holder-mount-for-smartphone-tablet-v-shaped-portable-i3176164-s1354830759.html?spm=a2a0e.home.just4u.10.67a74937kcLsYT&scm=1007.28811.244313.0&pvid=cb18de5a-477f-447c-9420-acddf1e8a62c&clickTrackInfo=pvid%3Acb18de5a-477f-447c-9420-acddf1e8a62c%3Bchannel_id%3A0000%3Bmt%3Ahot%3Bitem_id%3A3176164%3B"
#
#     x = input('Get reviews by: \n1: Browser\n2: Requests\nPlease Select: ')
#
#     if x == '1':
#         reviews = get_reviews_via_selenium(link)
#     elif x == '2':
#         reviews = get_reviews(link)
#     else:
#         reviews = 'Invalid choice'
#
#     print(reviews.reviewContent)