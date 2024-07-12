import json
import sys
import re

f = open(sys.argv[1], "r", encoding = "utf8")
s1 = f.read()
f.close()

parts = ['Оценка "отлично"', 'Оценка "хорошо"', 'Оценка "удовлетворительно"', 'Оценка "неудовлетворительно"']

cms = []
ems = []
for part in parts:
  res = re.search(part, s1)
  cms.append(res.span()[0])
  ems.append(res.span()[1])

result = {
  'критерий_отлично': s1[(ems[0]+1):cms[1]].strip(),
  'критерий_хорошо': s1[(ems[1]+1):cms[2]].strip(),
  'критерий_удовл': s1[(ems[2]+1):cms[3]].strip(),
  'критерий_неудовл': s1[(ems[3]+1):len(s1)].strip(),
}

print(json.dumps(result, indent=2, ensure_ascii=False))
