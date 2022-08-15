from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool

# identifier = 'alexnet'
# model = brain_translated_pool[identifier]
# score = score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
# # score = score_model(model_identifier=identifier, model=model, benchmark_identifier='sheinberg.neural.IT')
# print(score)

import numpy as np
from typing import List, Tuple
from brainscore.benchmarks.screen import place_on_screen

from brainscore.model_interface import BrainModel
from brainio_base.assemblies import DataAssembly


class RandomITModel(BrainModel):
    def __init__(self):
        self._num_neurons = 100
        # to note which time we are recording
        self._time_bin_start = None
        self._time_bin_end = None

    def look_at(self, stimuli, **kwargs):
        print(f"Looking at {len(stimuli)} stimuli")
        rnd = np.random.RandomState(0)
        recordings = DataAssembly(rnd.rand(len(stimuli), self._num_neurons, 1),
                                  coords={'image_id': ('presentation', stimuli['image_id']),
                                          'object_name': ('presentation', stimuli['object_name']),
                                          'neuroid_id': ('neuroid', np.arange(self._num_neurons)),
                                          'region': ('neuroid', ['IT'] * self._num_neurons),
                                          'time_bin_start': ('time_bin', [self._time_bin_start]),
                                          'time_bin_end': ('time_bin', [self._time_bin_end])},
                                  dims=['presentation', 'neuroid', 'time_bin'])
        recordings.name = 'random_it_model'
        return recordings

    def start_task(self, task, **kwargs):
        print(f"Starting task {task}")
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target=BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
        print(f"Recording from {recording_target} during {time_bins} ms")
        if str(recording_target) != "IT":
            raise NotImplementedError(f"RandomITModel only supports IT, not {recording_target}")
        if len(time_bins) != 1:
            raise NotImplementedError(f"RandomITModel only supports a single start-end time-bin, not {time_bins}")
        time_bins = time_bins[0].tolist()
        self._time_bin_start, self._time_bin_end = time_bins[0], time_bins[1]

    def visual_degrees(self):
        print("Declaring model to have a visual field size of 8 degrees")
        return 8


model = RandomITModel()

from brainscore import score_model
score = score_model(model_identifier='mymodel', model=model, benchmark_identifier='sheinberg.neural.IT-pls.1moreobf')
print(score)

print('score ceiling : ',score.ceiling.data)
print('score ceiling shape : ',score.ceiling.data.shape)