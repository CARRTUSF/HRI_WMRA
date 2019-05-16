import numpy as np 

a = np.arange(144).reshape((6, 8, 3))
print(a.shape)
for row in a:
	for col in row:
		print('({:03} {:03} {:03})'.format(*col), end=' ')
	print()
print()

b = np.transpose(a)
print(b.shape)
for row in b:
	for col in row:
		print('({:03} {:03} {:03} {:03} {:03} {:03})'.format(*col), end=' ')
	print()
print()

c = np.transpose(a, (2, 0, 1))
print(c.shape)
for row in c:
	for col in row:
		print('({:03} {:03} {:03} {:03} {:03} {:03} {:03} {:03})'.format(*col), end=' ')
	print()
print()

import torchvision.transforms as transforms
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(norm)