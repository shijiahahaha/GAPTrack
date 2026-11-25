import os

import numpy as np
from lib.test.evaluation.data import Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LasHeRDataset(BaseDataset):
    # LasHeR dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasher_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=['', '\t', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'visible', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'infrared', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'lasher', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        split_file = os.path.join(self.base_path, "testingsetList.txt")
        with open(split_file, encoding="utf-8-sig") as f:
            sequence_names = [line.strip() for line in f if line.strip()]

        test_path = self.env_settings.lashertestingset_path
        return [
            {
                "name": name,
                "path": os.path.join(test_path, name),
                "anno_path": os.path.join(test_path, name, "init.txt"),
            }
            for name in sequence_names
        ]
