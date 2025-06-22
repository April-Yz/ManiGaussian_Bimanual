from typing import List

import torch
import torchvision.transforms as transforms

from yarr.agents.agent import (
    Agent,
    Summary,
    ActResult,
    ScalarSummary,
    HistogramSummary,
    ImageSummary,
)

from termcolor import cprint

class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent,
                 norm_rgb: bool = True,
                 norm_type: str = 'zero_mean'):
        self._pose_agent = pose_agent
        self._norm_rgb = norm_rgb
        self._norm_type = norm_type

    def build(self, training: bool, device: torch.device = None, use_ddp: bool = True, **kwargs):
        try:
            self._pose_agent.build(training, device, use_ddp, **kwargs)
        except:
            self._pose_agent.build(training, device, **kwargs)
    # new86---

    def _norm_rgb_(self, x):
        if self._norm_type == 'zero_mean':                  
            return (x.float() / 255.0) * 2.0 - 1.0
        elif self._norm_type == 'imagenet':
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
            # return normalize(x)
            return (x.float() / 255.0)
        else:
            raise NotImplementedError


    def update(self, step: int, replay_sample: dict, use_nerf_picture=True, **kwargs) -> dict:
        if use_nerf_picture:
            nerf_multi_view_rgb = replay_sample['nerf_multi_view_rgb']
            nerf_multi_view_depth = replay_sample['nerf_multi_view_depth']
            nerf_multi_view_camera = replay_sample['nerf_multi_view_camera']

            if 'nerf_next_multi_view_rgb' in replay_sample:
                nerf_next_multi_view_rgb = replay_sample['nerf_next_multi_view_rgb']
                nerf_next_multi_view_depth = replay_sample['nerf_next_multi_view_depth']
                nerf_next_multi_view_camera = replay_sample['nerf_next_multi_view_camera']
        

            if replay_sample['nerf_multi_view_rgb'] is None or replay_sample['nerf_multi_view_rgb'][0,0] is None:
                cprint("preprocess agent no nerf rgb 1", "red")
            
        lang_goal = replay_sample['lang_goal']    
        replay_sample = {k: v[:, 0] if len(v.shape) > 2 else v for k, v in replay_sample.items()}

        for k, v in replay_sample.items():
            if self._norm_rgb and 'rgb' in k and 'nerf' not in k:
                replay_sample[k] = self._norm_rgb_(v)
            elif use_nerf_picture and 'nerf' in k:
                replay_sample[k] = v
            else:
                try:
                    replay_sample[k] = v.float()
                except:
                    replay_sample[k] = v
                    pass # some elements are not tensors/arrays

        if use_nerf_picture:
            replay_sample['nerf_multi_view_rgb'] = nerf_multi_view_rgb
            replay_sample['nerf_multi_view_depth'] = nerf_multi_view_depth
            replay_sample['nerf_multi_view_camera'] = nerf_multi_view_camera

            if 'nerf_next_multi_view_rgb' in replay_sample:
                replay_sample['nerf_next_multi_view_rgb'] = nerf_next_multi_view_rgb
                replay_sample['nerf_next_multi_view_depth'] = nerf_next_multi_view_depth
                replay_sample['nerf_next_multi_view_camera'] = nerf_next_multi_view_camera
            if replay_sample['nerf_multi_view_rgb'] is None or replay_sample['nerf_multi_view_rgb'][0,0] is None:
                cprint("preprocess agent no nerf rgb 2", "red")        

        replay_sample['lang_goal'] = lang_goal
        self._replay_sample = replay_sample


        return self._pose_agent.update(step, replay_sample,use_nerf_picture, **kwargs)


    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
        for k, v in observation.items():
            # print(k, v.shape)
            if self._norm_rgb and "rgb" in k:
                observation[k] = self._norm_rgb_(v)
            else:
                try:
                    observation[k] = v.float()
                except:
                    observation[k] = v
                    pass # some elements are not tensors/arrays
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({"demo": False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = "inputs"
        demo_f = self._replay_sample["demo"].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        sums = [
            ScalarSummary("%s/demo_proportion" % prefix, demo_proportion),
            ScalarSummary(
                "%s/timeouts" % prefix, self._replay_sample["timeout"].float().mean()
            ),
        ]

        for robot_prefix in ["", "right_", "left_"]:

            if not f"{robot_prefix}low_dim_state" in self._replay_sample.keys():
                continue

            sums.extend([HistogramSummary(
                f"{prefix}/{robot_prefix}low_dim_state", self._replay_sample[f"{robot_prefix}low_dim_state"]
            ),
            HistogramSummary(
                f"{prefix}/{robot_prefix}low_dim_state_tp1",
                self._replay_sample[f"{robot_prefix}low_dim_state_tp1"],
            ),
            ScalarSummary(
                f"{prefix}/{robot_prefix}low_dim_state_mean",
                self._replay_sample[f"{robot_prefix}low_dim_state"].mean(),
            ),
            ScalarSummary(
                f"{prefix}/{robot_prefix}low_dim_state_min",
                self._replay_sample[f"{robot_prefix}low_dim_state"].min(),
            ),
            ScalarSummary(
                f"{prefix}/{robot_prefix}low_dim_state_max",
                self._replay_sample[f"{robot_prefix}low_dim_state"].max(),
            )])

        for k, v in self._replay_sample.items():
            if "rgb" in k or "point_cloud" in k:
                if "rgb" in k:
                    # Convert back to 0 - 1
                    v = (v + 1.0) / 2.0
                sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v) if len(v.shape) > 4 else v))

        if "sampling_probabilities" in self._replay_sample:
            sums.extend(
                [
                    HistogramSummary(
                        "replay/priority", self._replay_sample["sampling_probabilities"]
                    ),
                ]
            )
        sums.extend(self._pose_agent.update_summaries())
        return sums

    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()
