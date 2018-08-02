# -*- coding: utf-8 -*-
"""
@CREATETIME: 05/07/2018 14:04 
@AUTHOR: Chans
@VERSION: 
"""

from selenium import webdriver
from splinter import browser
import sys

print(sys.path)

# chromedriver = '/Users/Apple/datadata/chrome/chromedriver'
# driver = webdriver.Chrome(chromedriver)
b = browser.Browser('chrome')
b.visit('https://www.baidu.com')