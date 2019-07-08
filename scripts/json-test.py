import json
with open('conversion_dict.json') as json_file:
	data = json.load(json_file)
	print(data)