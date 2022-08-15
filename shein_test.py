import brainscore
# from brainnscore.brainscore import get_assembly
import numpy as np
from tqdm import tqdm

# neural_data = brainscore.get_assembly(name="sheinberg.neural.IT")
neural_data = brainscore.get_assembly(name="sheinberg.neural.IT.1moreobf")

print('neural_data image_id : ',len(neural_data.image_id.data))
print('neural_data object_name : ',len(neural_data.object_name.data))

print('stimulus_set image_id : ',len(neural_data.attrs['stimulus_set']["image_id"].values))
print('stimulus_set object_name : ',len(neural_data.attrs['stimulus_set']["object_name"].values))

print('stimulus_set image_id set : ',len(set(neural_data.attrs['stimulus_set']["image_id"].values)))
print('stimulus_set set object_name : ',len(set(neural_data.attrs['stimulus_set']["object_name"].values)))

print('########################################################')
print('neural_data : ',neural_data)
print('neural_data.image_id.data.shape : ',neural_data.image_id.data.shape)


########################################################
neural_data = neural_data.sel(region='IT')

print('########################################################')
print('neural_data : ',neural_data)
print('neural_data.image_id.data.shape : ',neural_data.image_id.data.shape)

# print(neural_data.neuroid_id.data.shape)

print('stimulus_set : ',neural_data.attrs['stimulus_set'])

stimulus_set_pd = neural_data.attrs['stimulus_set']
# stimulus_set_pd.to_csv('/Users/aarjun1/Documents/Arjun/brainscore-brief/stimulus_set.csv')

print('neural_data.neuroid_id.data : ',neural_data.neuroid_id.data)

compact_data = neural_data

########################################################
############ Brainscore Repetition  ####################
########################################################

# repetitions = neural_data.repetition.data
# image_ids = neural_data.image_id.data

# print('repetitions : ', repetitions.shape)
# print(repetitions)

# print('image_ids : ', image_ids.shape)
# print(image_ids)

# reps_pp = []
# count = 0
# for i, image_id in enumerate(image_ids):

#     if image_id == '8a72e2bfdb8c267b57232bf96f069374d5b21832':
#         count += 1
#         reps_pp.append(count)
#         print(repetitions[i])

# print('reps_pp : ', reps_pp)
# print('count : ', count)
########################################################
######################## Image ID ######################
########################################################

# image_id = compact_data.image_id.data
# print('\nRepetions : ', len(image_id)) #,'max : ',np.argmax(repetitions))
# for i, repp in enumerate(image_id[58879:]):
#     print('i : ',i,' : ', repp)
#     if i>50:
#         break

# print(image_id.shape)

########################################################
######################## Time Bin ######################
########################################################

# time_bin = compact_data.time_bin_start.data
# print('\nRepetions : ', len(time_bin)) #,'max : ',np.argmax(repetitions))
# for i, repp in enumerate(time_bin[:]):
#     print('i : ',i,' : ', repp)
#     if i>50:
#         break

# print(image_id.shape)

########################################################
######################## GroupBy #######################
########################################################

# neural_data = neural_data.sel(region='IT')  # (2)
# neural_data = neural_data.squeeze('time_bin')  # (3)

compact_data = neural_data.groupby('image_id').mean(dim='presentation')
compact_data_std = neural_data.groupby('image_id').std(dim='presentation')  # (1)
#
# print('compact_data : ', dir(compact_data))
#
# print('compact_data : ', compact_data)
#
# compact_data = compact_data.sel(region='IT')  # (2)
# compact_data = compact_data.squeeze('time_bin')  # (3)

mean_data =  compact_data.data
for i, mean_dat in tqdm(enumerate(mean_data)):
    mean_data[i] = [np.round(dat, 5) for dat in mean_dat]

std_data =  compact_data_std.data
for i, std_dat in tqdm(enumerate(std_data)):
    std_data[i] = [np.round(dat, 5) for dat in std_dat]

SNR_data = np.where(std_data == 0, 0, abs(mean_data/std_data))

print('SNR : ',np.mean(SNR_data))