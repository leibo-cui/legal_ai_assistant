import json
import re
import requests
from bs4 import BeautifulSoup

def fetch_and_parse(url):
    # request web site
    response = requests.get(url)
    #  setup website encoding
    response.encoding = 'utf-8'
    # parse website contents
    soup = BeautifulSoup(response.text, 'html.parser')
    #  extract body contents
    content = soup.find_all('p')
    #  initialize store data
    data = []
    #  extract text and format
    for para in content:
        text = para.get_text(strip=True)
        if text:  #  only handle non-empty text
            #  format contents
            data.append(text)
    # transfer data list to string
    data_str = '\n'.join(data)
    return data_str

def extract_law_articles(data_str):
    # match each item and content via regular express
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    #  initialize dictory to store item number and contents
    lawarticles = {}
    #  search matching item
    for match in pattern.finditer(data_str):
        articlenumber = match.group(1)
        articlecontent = match.group(0).replace('第' + articlenumber + '条', '').strip()
        lawarticles[f"中华人民共和国消费者权益保护法 第{articlenumber}条"] = articlecontent
    # transform dictory to JSON string
    jsonstr = json.dumps(lawarticles, ensure_ascii=False, indent=4)
    return jsonstr

if __name__ == '__main__':
    # request website
    contents = []
    url = "https://www.gov.cn/jrzg/2013-10/25/content_2515601.htm"
    data_str = fetch_and_parse(url)
    jsonstr = extract_law_articles(data_str)
    contents.append(jsonstr)
    # store JSON string to file with list
    json_objects = [json.loads(js) for js in contents]

    with open('./data/law01.json', 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, ensure_ascii=False, indent=4)
    print("The law files have been saved to law01.json successfully.")
