import os
import shutil
import random

files = [ f for f in os.listdir('entries') if f.startswith('entry-') ]
for f in files:
    r = random.random()
    dest = 'entries/training/' if r < 0.7 else 'entries/test/'
    shutil.copy('entries/' + f, dest + f)

