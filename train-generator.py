import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

from data_utils import load_data, gen_batches
from model import nvidia_model, nvidia_model_big
from image_utils import gen_data

# 2: vert window: 70,5 No dropout, Rides center of lane
# 2a: vert: 50,0 no dropout: As good as 2. Rides line
# 2b: vert: 60,15 no dropout:
# 3:  vert: 65,15 no dropout
# 5:  Copy of 2 plus red dirt sections added vert: 65,15 no dropout
batch_size=256
index='0'
modifier=''
mname='hi'
# mname='custom'
epochs=3
dropout=0.0
lr=0.0001
csvfile = './training/t' + index + '/driving_log.csv'
model_json = 'models/' + mname+ '-' + index + modifier + '.json'
model_h5 = 'models/'+mname+'-' + index + modifier + '.h5'


# 1) Create model
print('parms  index:',index,' epochs:',epochs,' dropout:',dropout, 'csvfile',csvfile,'model_h5',model_h5)
print('Creating model...')
model = nvidia_model(model_json,dropout,lr)

# 2) Create generator
generator = gen_data(csvfile, batch_size)

# 3) Loop
for i_pr in range(1):
    model.fit_generator(generator, samples_per_epoch=5, epochs=5, verbose=1)

# 4) Save model
print('Saving model...')
model.save(model_h5)
