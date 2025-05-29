
# coding: utf-8

# In[25]:
test
from cassandra.cluster import Cluster

cluster = Cluster()


# In[18]:

from cassandra.cluster import Cluster

cluster = Cluster(
    ['10.1.1.3', '10.1.1.4', '10.1.1.5'],
    port=9042)


# In[28]:


session = cluster.connect()
session.set_keyspace('users')

CREATE TABLE startups (
id uuid PRIMARY KEY,
Name, 
    
);


# In[29]:

cluster = Cluster()
session = cluster.connect('system')


# In[33]:

import requests
data = requests.get("https://www.f6s.com/startups")
print(data.content)


# In[36]:

import requests
from bs4 import BeautifulSoup
import re

pages = set()
def getLinks(pageUrl):
    global pages
    html = requests.get("https://www.f6s.com/startups/"+pageUrl)
    bsObj = BeautifulSoup(html.content)
    for link in bsObj.findAll("a"):
        if 'href' in link.attrs:
            print(link.attrs)
            if link.attrs['href'] not in pages:
                #We have encountered a new page
                newPage = link.attrs['href']
                print(newPage)
                pages.add(newPage)
                getLinks(newPage)
getLinks("")


# In[37]:

from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.pythonscraping.com")
bsObj = BeautifulSoup(html)
imageLocation = bsObj.find("a", {"id": "logo"}).find("img")["src"]
urlretrieve (imageLocation, "logo.jpg")
    


# In[40]:

import pymysql
conn = pymysql.connect(host='127.0.0.1', user='root', passwd=None, db='mysql')
cur = conn.cursor()
cur.execute("USE scraping")
cur.execute("SELECT * FROM pages WHERE id=1")
print(cur.fetchone())
cur.close()
conn.close()


# In[42]:

from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', passwd=None, db='mysql', charset='utf8')
cur = conn.cursor()
cur.execute("USE scraping")

random.seed(datetime.datetime.now())

def store(title, content):
    cur.execute("INSERT INTO pages (title, content) VALUES (\"%s\",\"%s\")", (title, content))
    cur.connection.commit()

def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org"+articleUrl)
    bsObj = BeautifulSoup(html)
    title = bsObj.find("h1").find("span").get_text()
    content = bsObj.find("div", {"id":"mw-content-text"}).find("p").get_text()
    store(title, content)
    return bsObj.find("div", {"id":"bodyContent"}).findAll("a", 
                      href=re.compile("^(/wiki/)((?!:).)*$"))

links = getLinks("/wiki/Kevin_Bacon")
try:
    while len(links) > 0:
         newArticle = links[random.randint(0, len(links)-1)].attrs["href"]
         print(newArticle)
         links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()
    
    
    


# In[8]:

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random
global pages
pages = []
random.seed(datetime.datetime.now())
import requests

def Internal(url):
    html = urlopen(url)
    bsObj = BeautifulSoup(html)
    a = getInternalLinks(bsObj,url)
    return (a)


def getInternalLinks(bsObj,includeUrl):
    a = includeUrl
    pages.append(link.attrs['href'])

    print(link.attrs['href'])
    getLinks(str(link.attrs['href']))

                
def getLinks(pageUrl):
    headers = {'user-agent': 'Mozilla'}
    global pages
    html = requests.get("https://www.f6s.com/startups"+pageUrl, headers = headers)
    bsObj = BeautifulSoup(html.content)
    #print(bsObj)
    for links in bsObj.findAll("a"):
        if 'href' in links.attrs:
            if links.attrs['href'] not in pages:
                newPage = links.attrs['href']
                print(newPage)
                print(pages)
                pages.append(newPage)
                getLinks(newPage)
    return pages



def splitAddress(address):
    addressParts = address.replace("http://" , ""). split("/")
    return addressParts


a = getLinks("")
# import pandas as pd
# df = pd.DataFrame({'column1':['name'],'URL':['URL']})
# df2 = pd.DataFrame({'column1':['name'],'URL':['URL']})


# # print(a)    
# # y = 0


# for b in a:
#     try:

#         d = Internal("http://www.leons.com"+b)

#         for c in d:
#             y = y + 1
#             df1 = pd.DataFrame({'column1':[y],'URL':['http://leons.com'+str(c)]})
#             df = df.append(df1)
#             if (y == 100):
#                 df.to_csv('C:/ecommerce/301redirects1a.csv')
#         pages.append(d)
#         #print(d)    
#     except Exception as e:
#         print(e)

# x = 0
# for page in pages:
#     print(page)
#     x = x + 1
#     df3 = pd.DataFrame({'column1':[x],'URL':['http://leons.com'+str(page)]})
#     df2 = df3.append(df2)
#     if(x == 3):
#         df2.to_csv('C:/ecommerce/301redirects1.csv')
#     if(x == 50):
#         df2.to_csv('C:/ecommerce/301redirects1.csv')
    


# df.to_csv('C:/ecommerce/301redirects.csv')


# In[70]:

pages


# In[ ]:

new = pages
for i in pages:
    Internal("http")
