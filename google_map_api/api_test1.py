# -*- coding: utf-8 -*-
"""
@CREATETIME: 05/07/2018 12:55 
@AUTHOR: Chans
@VERSION: 1.0
"""

import time
from urllib.request import urlopen
import selenium
import urllib
import requests
import json
from xml.dom import minidom

# 这个KEY本来是google要求的，否则不允许用它的API，可是我没用这个KEY也可以啊...囧了
KEY = 'AIzaSyDuoVQyzjUNZ0N0YF25jebIy9WZ9MqKFFA'

class GetData(object):
    def __init__(self):
        self.values = {'q': '',
                       'sensor': 'false',
                       'output': 'xml',
                       'oe': 'utf8'}
        self.url = 'http://maps.google.com/maps/geo'

    def catchData(self, city, key=KEY):
        '''
        利用google map api从网上获取city的经纬度。
        '''
        self.values['q'] = city
        # self.values['key'] = key
        arguments = urllib.parse.urlencode(self.values)
        url_get = self.url + '?' + arguments
        handler = urlopen(url_get)
        try:
            self.lon, self.lat = self.parseXML(handler)
            # print 'lon:%d\tlat:%d' % (self.lon, self.lat)
            return self.lon, self.lat
        except IndexError:
            print('城市: %s 发生异常！' % (city,))
        finally:
            handler.close()

    def parseXML(self, handler):
        '''
        解析从API上获取的XML数据。
        '''
        xml_data = minidom.parse(handler)
        data = xml_data.getElementsByTagName('coordinates')[0].firstChild.data
        coordinates = data.split(',')
        lon = int(float(coordinates[0]) * 1000000)
        lat = int(float(coordinates[1]) * 1000000)
        return lon, lat


if __name__ == '__main__':
    # b = requests.get('https://www.baidu.com')
    # print(b.content.decode('utf-8'))

    c = requests.get('https://www.qq.com')
    print(c.content.decode('gb2312'))
    a = requests.get('https://www.google.com/')
    print(a.content)
    getData = GetData()
    cityName = '广州市'
    KEY = 'AIzaSyDuoVQyzjUNZ0N0YF25jebIy9WZ9MqKFFA'
    longitude, latitude = getData.catchData(cityName,key=KEY)
    print('%s \n经度：%d\n纬度：%d\n' % (cityName, longitude, latitude))


