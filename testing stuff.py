

import os
import os.path as op

if not op.isdir(op.join('rest_data')):
    os.mkdir(op.join('rest_data'))
if not op.isdir(op.join('film_data')):
    os.mkdir(op.join('film_data'))

os.mkdir(op.join('rest_data','gerbil'))