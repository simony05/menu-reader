import requests
from dotenv import load_dotenv
import os

def configure():
	load_dotenv()

def unpack(results):
	result = []
	result.append(results['results'][0]['title'])
	result.append(results['results'][0]['url'])
	result.append(results['results'][0]['description'])

	similar = []
	for keyword in results['related_keywords']['keywords']:
		similar.append(keyword['keyword'])

	return {"results": result,
		 	"similar results": similar}

def google_search(food):
	url = "https://google-web-search1.p.rapidapi.com/"
	configure()
	querystring = {"query": food,"limit": "1","related_keywords": "true"}

	headers = {
		"X-RapidAPI-Key": os.getenv('search_api'),
		"X-RapidAPI-Host": "google-web-search1.p.rapidapi.com"
	}

	response = requests.get(url, headers = headers, params = querystring)

	return unpack(response.json())

print(google_search("seaweed egg flower soup"))