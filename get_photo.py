import os
import json
import time
import logging
import urllib.request
import urllib.error
from urllib.parse import urlparse
from selenium.webdriver.common.action_chains import ActionChains
from multiprocessing import Pool
from user_agent import generate_user_agent
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import threading
from lxml import etree




def get_image_links(main_keyword, link_file_path, num_requested=1000):
    number_of_scrolls = 5

    img_urls = set()
    driver = webdriver.Chrome(executable_path='./chromedriver')
    search_query = main_keyword

    #####################################################
    #  Get Image Link From Bing
    #####################################################
    url = "https://www.bing.com/images/search?q=" + search_query
    driver.get(url)
    for __ in range(5):
        # multiple scrolls needed to show all 400 images
        print('scroll')
        driver.execute_script("window.scrollBy(0, 1000000)")
        time.sleep(2)
    # to load next 400 images
    time.sleep(2)
    html = driver.page_source
    dom_tree = etree.HTML(html)
    images = dom_tree.xpath("//div[@class='img_cont hoff']/img/@src | //div[@class='img_cont hoff']/img/@data-src")
    # 过滤掉非URL元素
    imges = [image for image in images if image.find("http") != -1]
    for img in imges:
        img_url = img
        img_urls.add(img_url)
    #####################################################
    #  Get Image Link From Google
    #####################################################
    url = "https://www.google.com/search?aq=f&tbm=isch&q=" + search_query
    driver.get(url)
    for _ in range(10):
        for __ in range(5):
            # multiple scrolls needed to show all 400 images
            print('scroll')
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(2)

    # to load next 400 images
    time.sleep(2)
    html = driver.page_source
    dom_tree = etree.HTML(html)
    images = dom_tree.xpath("//div[@class='img_cont hoff']/img/@src | //div[@class='img_cont hoff']/img/@data-src")
    # 过滤掉非URL元素
    imges = [image for image in images if image.find("http") != -1]
    for img in imges:
        img_url = img
        img_urls.add(img_url)

    #####################################################
    #  Save Links into local
    #####################################################
    print('Process-{0} add keyword {1} , got {2} image urls so far'.format(main_keyword, '',
                                                                           len(img_urls)))
    print('Process-{0} totally get {1} images'.format(main_keyword, len(img_urls)))

    with open(link_file_path, 'w') as wf:
        for url in img_urls:
            wf.write(url + '\n')
    print('Store all the links in file {0}'.format(link_file_path))

def download_images(link_file_path, download_dir, log_dir):
    print('Start downloading with link file {0}..........'.format(link_file_path))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    main_keyword = link_file_path.split('/')[-1]
    log_file = log_dir + 'download_selenium_{0}.log'.format(main_keyword)
    logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s  %(message)s")
    img_dir = download_dir + main_keyword + '/'
    count = 0
    headers = {}
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # start to download images
    with open(link_file_path, 'r') as rf:
        for link in rf:
            try:
                o = urlparse(link)
                ref = o.scheme + '://' + o.hostname
                # ref = 'https://www.google.com'
                ua = generate_user_agent()
                headers['User-Agent'] = ua
                headers['referer'] = ref
                print('\n{0}\n{1}\n{2}'.format(link.strip(), ref, ua))
                req = urllib.request.Request(link.strip(), headers=headers)
                response = urllib.request.urlopen(req)
                data = response.read()
                file_path = img_dir + '{0}.jpg'.format(count)
                with open(file_path, 'wb') as wf:
                    wf.write(data)
                print('Process-{0} download image {1}/{2}.jpg'.format(main_keyword, main_keyword, count))
                count += 1
                if count % 10 == 0:
                    print('Process-{0} is sleeping'.format(main_keyword))
                    time.sleep(5)

            except urllib.error.URLError as e:
                print('URLError')
                logging.error('URLError while downloading image {0}reason:{1}'.format(link, e.reason))
                continue
            except urllib.error.HTTPError as e:
                print('HTTPError')
                logging.error(
                    'HTTPError while downloading image {0}http code {1}, reason:{2}'.format(link, e.code, e.reason))
                continue
            except Exception as e:
                print('Unexpected Error')
                logging.error(
                    'Unexpeted error while downloading image {0}error type:{1}, args:{2}'.format(link, type(e), e.args))
                continue


if __name__ == '__main__':
    main_keywords = ['Eiffel Tower', 'Big Ben', 'London Bridge', 'London Eye', 'Sydney Opera House', 'Pyramid', 'The Palace Museum','The Statue of Liberty','Tian An Men']
    download_dir = './data_testing/'
    link_files_dir = './data_testing/link_files/'
    log_dir = './data_testing/logs/'
    for keyword in main_keywords:
        link_file_path = link_files_dir + keyword
        get_image_links(keyword, link_file_path)
    print('Fininsh getting all image links')
    for keyword in main_keywords:
        link_file_path = link_files_dir + keyword
        download_images(link_file_path, download_dir, log_dir)
    print('Finish downloading all images')