import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os

class CoinDataset(Dataset):
    def __init__(self, 
            img_dir, 
            prompt_features, 
            list_json, 
            horizon = 3, 
            num_action = 778, 
            aug_range = 0, 
            M = 2, 
            mode = "train",
            stage="llm"
        ):
        self.feature_dir = img_dir
        self.prompt_features = prompt_features
        self.coin_json = list_json
        self.horizon = horizon
        self.num_action = num_action
        self.aug_range = aug_range
        self.M = M
        self.mode = mode
        self.stage = stage
        assert self.stage in ["llm", "qformer"], "Invalid stage"
        self.transition_matrix = np.zeros((num_action, num_action), dtype = np.float32)

        self.data = []
        self.load_data()
        if self.mode == "train":
            self.transition_matrix += 1  # Laplace smoothing
            self.transition_matrix = self.cal_transition(self.transition_matrix)

    def cal_transition(self, matrix):
        '''
        input:
        matrix: [num_action, num_action]
        output:
        transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition
    

    def load_data(self):
        with open(self.coin_json, "r") as f:
            coin_data = json.load(f)
        import re
        for video in coin_data:
            vid = list(video.keys())[0]
            vid_info = video[vid]
            feat_path = os.path.join(self.feature_dir, vid + ".npy") 
            if os.path.exists(feat_path):
                saved_features = np.load(feat_path, allow_pickle=True)["frames_features"]
            else:
                continue

            task_id = vid_info["recipe_type"]
            # import pdb; pdb.set_trace()
            task_name = vid_info["class"]
            task_name = re.findall('[A-Z][^A-Z]*', 'TheLongAndWindingRoad')
            task_name = ' '.join(task_name)
            video_anot = vid_info["annotation"]

            # Remove repeated actions. Intuitively correct, but do not work well on dataset.
            # legal_video_anot = []
            # for i in range(len(video_anot)):
            #     if i == 0 or video_anot[i]["id"] != video_anot[i-1]["id"]:
            #         legal_video_anot.append(video_anot[i])
            # video_anot = legal_video_anot

            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot)):
                    if i < len(video_anot)-1:
                        cur_action = int(video_anot[i]["id"])-1
                        next_action = int(video_anot[i+1]["id"])-1
                        self.transition_matrix[cur_action, next_action] += 1

            for i in range(len(video_anot)-self.horizon+1):
                all_features = []
                all_action_ids = []
                all_action_names = []

                for j in range(self.horizon):
                    cur_video_anot = video_anot[i+j]
                    cur_action_id = int(cur_video_anot["id"])-1
                    # import pdb; pdb.set_trace()
                    cur_action_name = cur_video_anot["label"]
                    features = []
                    
                    ## Using adjacent frames for data augmentation
                    for frame_offset in range(-self.aug_range, self.aug_range+1):
                        s_time = int(cur_video_anot["segment"][0])+frame_offset
                        e_time = int(cur_video_anot["segment"][1])+frame_offset

                        if s_time < 0 or e_time >= saved_features.shape[0]:
                            continue
                        
                        s_offset_start = max(0, s_time-self.M//2)
                        s_offset_end = min(s_time+self.M//2+1,saved_features.shape[0])
                        e_offset_start = max(0, e_time-self.M//2)
                        e_offset_end = min(e_time+self.M//2+1,saved_features.shape[0])

                        start_feature = np.mean(saved_features[s_offset_start:s_offset_end], axis = 0)
                        end_feature = np.mean(saved_features[e_offset_start:e_offset_end], axis = 0)

                        features.append(np.stack((start_feature, end_feature)))

                    all_features.append(features)
                    all_action_ids.append(cur_action_id)
                    all_action_names.append(cur_action_name)

                ## permutation of augmented features, action ids and prompts
                aug_features = itertools.product(*all_features)

                self.data.extend([{"states": np.stack(f),
                                   "actions": np.array(all_action_ids), 
                                   "tasks": np.array(task_id),
                                   "task_name": task_name,
                                   "action_names": all_action_names }
                                  
                                  for f in aug_features])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.data[idx]["states"]
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        task_name = self.data[idx]['task_name']
        action_names  =  self.data[idx]['action_names']
        caption = {
            "Task": task_name,
            "Actions": action_names,
        }
        caption = str(caption)
        return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long),caption,idx