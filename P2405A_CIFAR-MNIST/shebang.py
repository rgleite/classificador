#!phyton3

import sys

for x in sys.argv:
    print(x)
    
print(type(sys.argv))

total = (int) (sys.argv[1])

parcela = (int) (sys.argv[2])

print('total:', total, 'parcela:', parcela)