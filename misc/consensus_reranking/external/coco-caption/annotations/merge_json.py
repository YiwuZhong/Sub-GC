import json

train = json.load(open('captions_train2014.json','r'))
val = json.load(open('captions_val2014.json','r'))

print(len(train['images']))
print(len(train['annotations']))
print(len(val['images']))
print(len(val['annotations']))

val['images'].extend(train['images'])
val['annotations'].extend(train['annotations'])

with open("captions_all2014.json","w") as f:
	json.dump(val, f)

all = json.load(open('captions_all2014.json','r'))
print(len(all['images']))
print(len(all['annotations']))