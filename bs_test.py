import brainscore
import numpy as np
from tqdm import tqdm

neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")

print('neural_data image_id : ',len(neural_data.image_id.data))
print('neural_data object_name : ',len(neural_data.object_name.data))
print('neural_data category_name : ',len(neural_data.category_name.data))

print('stimulus_set image_id : ',len(neural_data.attrs['stimulus_set']["image_id"].values))
print('stimulus_set object_name : ',len(neural_data.attrs['stimulus_set']["object_name"].values))
print('stimulus_set category_name : ',len(neural_data.attrs['stimulus_set']["category_name"].values))

print('stimulus_set image_id set : ',len(set(neural_data.attrs['stimulus_set']["image_id"].values)))
print('stimulus_set set object_name : ',len(set(neural_data.attrs['stimulus_set']["object_name"].values)))
print('stimulus_set set category_name : ',len(set(neural_data.attrs['stimulus_set']["category_name"].values)))

print(neural_data)
print(neural_data.image_id.data.shape)
print(neural_data.neuroid_id.data.shape)

# print('stimulus_set : ',neural_data.attrs['stimulus_set'])

stimulus_set_pd = neural_data.attrs['stimulus_set']
# stimulus_set_pd.to_csv('/Users/aarjun1/Documents/Arjun/brainscore-brief/stimulus_set.csv')

# print(neural_data.neuroid_id.data)

compact_data = neural_data

########################################################
############ Brainscore Repetition  ####################
########################################################

repetitions = neural_data.repetition.data
image_ids = neural_data.image_id.data

print('repetitions : ', repetitions.shape)
print(repetitions)

print('image_ids : ', image_ids.shape)
print(image_ids)

reps_pp = []
count = 0
for i, image_id in enumerate(image_ids):

    if image_id == '8a72e2bfdb8c267b57232bf96f069374d5b21832':
        count += 1
        reps_pp.append(count)
        print(repetitions[i])

print('reps_pp : ', reps_pp)
print('count : ', count)
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

neural_data = neural_data.sel(region='IT')  # (2)
neural_data = neural_data.squeeze('time_bin')  # (3)

compact_data = neural_data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')
compact_data_std = neural_data.multi_groupby(['category_name', 'object_name', 'image_id']).std(dim='presentation')  # (1)
#
# print('compact_data : ', dir(compact_data))
#
# print('compact_data : ', compact_data)
#
# compact_data = compact_data.sel(region='IT')  # (2)
# compact_data = compact_data.squeeze('time_bin')  # (3)

SNR_data = np.where(compact_data_std.data == 0, 0, abs(compact_data.data/compact_data_std.data))

print(SNR_data.shape)

for i, snr_dat in tqdm(enumerate(SNR_data)):
    SNR_data[i] = [round(dat, 5) for dat in snr_dat]

print('SNR : ',np.mean(SNR_data))


# neural_data_array = compact_data.data
# print('neural_data_array : ',neural_data_array.shape)

# print('compact_data : ', compact_data)
#

########################################################
######################## Repetitions ###################
########################################################

# repetitions = compact_data.repetition.data
# print('\nRepetions : ', len(repetitions),'max : ',np.argmax(repetitions))
# for i, repp in enumerate(repetitions[58879:]):
#     print('i : ',i,' : ', repp)
#     if i>50:
#         break
#
# image_idss = compact_data.image_id.data
# category_name = compact_data.category_name.data
# object_name = compact_data.object_name.data
# # indexes = list(image_idss).index('8a72e2bfdb8c267b57232bf96f069374d5b21832')
# print('\nimage_idss : ', len(image_idss))
# set_rep_id = set(image_idss)
# print('\nSet image_idss : ', len(set_rep_id))
#
# for i, repp in enumerate(image_idss):
#     print('i : ',i,' : ', repp)
#     if i>10:
#         break
#
# # for i, repp in enumerate(set_rep_id):
# #     print('i : ',i,' : ', repp)
# #     if i>10:
# #         break
# indices = {}
# len_reps = []
# for unique_id in tqdm(set_rep_id):
#     temp = []
#     # indices[unique_id] = []
#     for i in range(len(image_idss)):
#         if image_idss[i] == unique_id:
#             temp.append(i)
#     indices[unique_id] = temp
#     len_reps.append(len(temp))
#
# print(len_reps[:50])
# print('Max : ', max(len_reps))
# print('Min : ', min(len_reps))
# print('Mean : ', np.mean(len_reps))
# print('Median : ', np.median(len_reps))
# print('Sum : ', sum(len_reps))

#
# ########################################################
# ######################### SNR ##########################
# ########################################################
#
# neural_data_array = compact_data.data
# print('neural_data_array : ',neural_data_array.shape)
#
# def signaltonoise(a, axis=0, ddof=0):
#     a = np.asanyarray(a)
#     m = round(a.mean(axis), 5)
#     sd = round(a.std(axis=axis, ddof=ddof), 5)
#     # print('mean : ', m)
#     # print('sd : ', sd)
#     # return np.where(sd == 0, 0, m/sd)
#     if sd == 0:
#         # print('mean : ', m,' : sd : ',sd,' : SNR : ',0)
#         return 0
#     else:
#         # print('mean : ', m,' : sd : ',sd,' : SNR : ',abs(m/sd))
#         return abs(m/sd)
#
# SNR = np.zeros(neural_data_array.shape[0])
# for unique_id in tqdm(set_rep_id):
#     # array_reps = np.zeros((len(indices[unique_id]), neural_data_array.shape[1]))
#     # for i, ind in enumerate(indices[unique_id]):
#     #     for cell in range(neural_data_array.shape[0]):
#     #         array_reps[i] = neural_data_array[cell, ind]
#     array_reps = neural_data_array[:, indices[unique_id]]
#     array_reps = array_reps.reshape(neural_data_array.shape[0], -1)
#     # print('array_reps : ', array_reps.shape)
#     for i, cell_arr in enumerate(array_reps):
#         SNR[i] = signaltonoise(cell_arr)
#
# # for unique_id in tqdm(set_rep_id):
# #     # array_reps = np.zeros((len(indices[unique_id]), neural_data_array.shape[1]))
# #     # for i, ind in enumerate(indices[unique_id]):
# #     #     for cell in range(neural_data_array.shape[0]):
# #     #         array_reps[i] = neural_data_array[cell, ind]
# #     array_reps = neural_data_array[:, indices[unique_id]]
# #     array_reps = array_reps.reshape(neural_data_array.shape[0], -1)
# #     print('array_reps : ', array_reps[0])
# #     # for i, cell_arr in enumerate(array_reps):
# #     SNR = signaltonoise(array_reps)
#
# print('SNR Mean : ', np.mean(SNR))
# # print('compact_data : ', compact_data)
# #
# # print('\nNeuroid : ',neural_data.neuroid)
# #
# # print('\nPresentation : ',neural_data.presentation)
# #
# # from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
# #
# # benchmark = MajajHongITPublicBenchmark()
# # benchmark_assembly = benchmark._assembly
# # print(benchmark_assembly.shape)
# # print(benchmark_assembly[0])
