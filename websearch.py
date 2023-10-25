import requests

url = "https://google-web-search1.p.rapidapi.com/"

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
	querystring = {"query": food,"limit": "1","related_keywords": "true"}

	headers = {
		"X-RapidAPI-Key": "57204ffb4dmsh323ba2a1a5d8c1dp187e29jsn49382c1b181b",
		"X-RapidAPI-Host": "google-web-search1.p.rapidapi.com"
	}

	response = requests.get(url, headers = headers, params = querystring)

	return unpack(response.json())

print(google_search("seaweed egg flower soup"))