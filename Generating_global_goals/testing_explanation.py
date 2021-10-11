import torch
from typing import Any, Dict, List, Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped


def load_checkpoint(checkpoint_path: str, *args, **kwargs) -> Dict:
    r"""Load checkpoint of specified path as a dict.

    Args:
        checkpoint_path: path of target checkpoint
        *args: additional positional args
        **kwargs: additional keyword args

    Returns:
        dict containing checkpoint info
    """
    return torch.load(checkpoint_path, *args, **kwargs)


def _create_global_policy_inputs(global_map, visited_states, map_xy):
    """
    global_map     - (bs, 2, V, V) - map occupancy, explored states
    visited_states - (bs, 1, V, V) - agent visitation status on the map
    map_xy   - (bs, 2) - agent's XY position on the map
    """
    agent_map_x = map_xy[:, 0].long()  # (bs, )
    agent_map_y = map_xy[:, 1].long()  # (bs, )
    agent_position_onehot = torch.zeros_like(visited_states)
    agent_position_onehot[:, 0, agent_map_y, agent_map_x] = 1
    h_t = torch.cat(
        [global_map, visited_states, agent_position_onehot], dim=1
    )  # (bs, 4, M, M)

    global_policy_inputs = {
        "pose_in_map_at_t": map_xy,
        "map_at_t": h_t,
    }

    return global_policy_inputs


def convert_world2map(world_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        world_coors: (bs, 2) --- (x, y) in world coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_world = world_coors[:, 0]
    y_world = world_coors[:, 1]

    # x_map = torch.clamp((Wby2 + y_world / map_scale), 0, W - 1).round()
    # x_map = torch.clamp((Wby2 + x_world / map_scale), 0, W - 1).round()
    x_map = torch.clamp((Hby2 - y_world / map_scale), 0, H - 1).round()
    # y_map = torch.clamp((Hby2 - x_world / map_scale), 0, H - 1).round()
    # y_map = torch.clamp((Hby2 - y_world / map_scale), 0, H - 1).round()
    #y_map = torch.clamp((Hby2 + y_world / map_scale), 0, H - 1).round()
    y_map = torch.clamp((Wby2 + x_world / map_scale), 0, W - 1).round()

    map_coors = torch.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors


def convert_map2world(map_coors, map_shape, map_scale):
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_map = map_coors[0]
    y_map = map_coors[1]

    y_world = (Hby2 - x_map) * map_scale
    x_world = (y_map - Wby2) * map_scale

    world_coors = [x_world, y_world]
    return world_coors


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalPolicy(nn.Module):
    def __init__(self, G):
        super().__init__()

        self.G = G

        self.actor = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),  # (G*G, )
        )

        self.critic = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),
            nn.Linear(self.G * self.G, 1),
        )
        '''
        if config.use_data_parallel:
            self.actor = nn.DataParallel(
                self.actor, device_ids=config.gpu_ids, output_device=config.gpu_ids[0],
            )
            self.critic = nn.DataParallel(
                self.critic, device_ids=config.gpu_ids, output_device=config.gpu_ids[0],
            )
        '''

    def forward(self, inputs):
        raise NotImplementedError

    def _get_h12(self, inputs):
        x = inputs["pose_in_map_at_t"]
        h = inputs["map_at_t"]

        h_1 = crop_map(h, x[:, :2], self.G)
        h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))

        h_12 = torch.cat([h_1, h_2], dim=1)

        return h_12

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        """
        Note: inputs['pose_in_map_at_t'] must obey the following conventions:
              origin at top-left, downward Y and rightward X in the map coordinate system.
        """
        M = inputs["map_at_t"].shape[2]
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        h_12 = self._get_h12(inputs)
        value = self.critic(h_12)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


def _update_state_visitation(visited_states, agent_map_xy):
    """
    visited_states - (bs, 1, V, V) tensor with 0s for unvisited locations, 1s for visited locations
    agent_map_xy - (bs, 2) agent's current position on the map
    """
    agent_map_x = agent_map_xy[:, 0].long()  # (bs, )
    agent_map_y = agent_map_xy[:, 1].long()  # (bs, )
    visited_states[:, 0, agent_map_y, agent_map_x] = 1

    return visited_states

if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create neural network for global policy
    G = 240
    global_actor_critic = GlobalPolicy(G)
    global_actor_critic.to(device)
    # load parameters for global policy
    ckpt_dict = load_checkpoint("data/global_policy/ckpt.1X.pth")  # input is the
    global_actor_critic.load_state_dict(ckpt_dict["global_actor_critic"])
    # Set models to evaluation
    global_actor_critic.eval()
    SAMPLE_GLOBAL_GOAL_FLAG = 0
    goal_interval = 25
    # name the map file
    map_file = "/home/chenxr/Generating_global_goals/map.txt"
    pose_file = "/home/chenxr/Generating_global_goals/pose.txt"
    map_pose_file = "/home/chenxr/Generating_global_goals/map_pose.txt"
    prev_visited_states = torch.zeros(1, 1, 361, 361).to(device)


    for ep_step in range(100):

        # =================== Update states from current observation ==================
        # Update map, pose and visitation map
        SAMPLE_GLOBAL_GOAL_FLAG = ep_step % goal_interval == 0

        # ->->->->->->->->->->->-> input global_map and global_pose ->->->->->->->->->->->->->->->->->->->->->
        M = 361  # (M, M) is the size of global map
        s = 0.1  # s*s is the size of a cell in the global map
        bs = 1  # batch_size
        # global_map = torch.rand(bs, 2, M, M).to(device)  # shape is (bs, 2, M, M) --->>>>input from other modules
        # global_pose = torch.rand(bs, 3).to(device)   # shape is (bs, 3)  --->>>>input from other modules
        # prev_visited_states = torch.zeros(1, 1, M, M).to(device)


        # if the pose file exists
        while os.access(pose_file, os.F_OK) == 0:
          continue
        print("pose file exists")

        # load global pose
        pose_data = np.loadtxt(pose_file)
        os.remove(pose_file) # remove the pose file
        print("pose file removed")
        global_pose = torch.zeros(bs, 3).to(device)
        global_pose[0][0] = pose_data[0]
        global_pose[0][1] = pose_data[1]
        global_pose[0][2] = pose_data[2]
        print("pose_data:", pose_data)
        map_xy = convert_world2map(global_pose[:, :2], (M, M), s)
        map_xy = torch.clamp(map_xy, 0, M - 1)
        map_x = map_xy[0][0].item()
        map_y = map_xy[0][1].item()
        print("map_xy:", map_xy)

        # if the map pose file exists
        while os.access(map_pose_file, os.F_OK) == 0:
          continue;
        print("map pose file exists")

        # load global map pose
        map_pose_data = np.loadtxt(map_pose_file)
        os.remove(map_pose_file) # remove the map pose file
        print("map pose file removed")
        global_map_pose = torch.zeros(bs, 3).to(device)
        global_map_pose[0][0] = map_pose_data[0] 
        global_map_pose[0][1] = map_pose_data[1]
        global_map_pose[0][2] = map_pose_data[2]
        print("map_pose_data:", map_pose_data)
        map_pose_xy = convert_world2map(global_map_pose[:, :2], (M, M), s)
        map_pose_xy = torch.clamp(map_pose_xy, 0, M - 1)
        map_pose_x = map_pose_xy[0][0].item()
        map_pose_y = map_pose_xy[0][1].item()
        print("map_pose_xy:", map_pose_xy)

        # if the map file exists
        while os.access(map_file, os.F_OK) == 0:
          continue
        print("map file exists")
        
        # load global map
        map_data = np.loadtxt(map_file)
        os.remove(map_file) # remove the map file
        print("map file removed")
        
        # reshape data
        length = len(map_data)
        H = map_data[length-2];
        W = map_data[length-1];
        H = int(H)
        W = int(W)
        map_data = map_data[:length-2]
        map_data_ = np.reshape(map_data, (H,W))
        for i in range(H):
          for j in range(W):
            map_data_[i][j] = map_data[j+(H-i-1)*W]
        
        global_map = torch.rand(bs, 2, M, M).to(device);
        for i in range(M):
          for j in range(M): # padded area
            global_map[0][0][i][j] = 1
            global_map[0][1][i][j] = 0
        
        # deviation
        # H_d = int((M-H)/2)
        # H_d = int(map_pose_y - int(H/2))
        # H_d = int((M-1)/2 - map_pose_y)
        H_d = int(map_pose_x - H)
        # W_d = int((M-W)/2)
        # W_d = int(map_pose_x - int(W/2))
        # W_d = int((M-1)/2 + map_pose_x)
        W_d = int(map_pose_y)
        print("map_pose_x=", map_pose_x)
        print("map_pose_y=", map_pose_y)
        print("H_d=", H_d)
        print("W_d=", W_d)

        for i in range(H):
          for j in range(W):
            if i + H_d < M and j + W_d < M:
              if map_data_[i][j] == -1: # unexplored space
                global_map[0][0][i+H_d][j+W_d] = 1
                global_map[0][1][i+H_d][j+W_d] = 0
              elif map_data_[i][j] == 0: # free space
                global_map[0][0][i+H_d][j+W_d] = 0
                global_map[0][1][i+H_d][j+W_d] = 1
              elif map_data_[i][j] == 100: # obstacles
                global_map[0][0][i+H_d][j+W_d] = 1
                global_map[0][1][i+H_d][j+W_d] = 1


        # visited locations
        visited_states = _update_state_visitation(
            prev_visited_states, map_xy
        )  # (bs, 1, M, M)
        curr_map_position = map_xy.to(device)

        global_policy_inputs = _create_global_policy_inputs(
            global_map, visited_states, curr_map_position
        )
        (
            global_value,
            global_action,
            global_action_log_probs,
            _,
        ) = global_actor_critic.act(global_policy_inputs, None, None, None)

        # Convert action to location (row-major format)
        G = global_actor_critic.G
        global_action_map_x = torch.fmod(
            global_action.squeeze(1), G
        ).float()  # (bs, )
        global_action_map_y = (global_action.squeeze(1) / G).float()  # (bs, )
        # Convert to MxM map coordinates
        global_action_map_x = global_action_map_x * M / G
        global_action_map_y = global_action_map_y * M / G
        global_action_map_xy = torch.stack(
            [global_action_map_x, global_action_map_y], dim=1
        )

        print(global_action_map_xy);

        goal_x = global_action_map_xy[0][0].item()
        goal_y = global_action_map_xy[0][1].item()

        goal = [goal_x, goal_y]
        goal = convert_map2world(goal, (M, M), s)
        goal_x = goal[0]
        goal_y = goal[1]
        
        file_handle_1 = open('goal.txt', mode='w')
        file_handle_1.writelines([str(goal_x), ' ', str(goal_y)])
        print("writing into the goal file")
        file_handle_1.close()
        '''
        file_handle_2 = open('global_map.txt', mode='w')
        for i in range(M):
          for j in range(M):
            file_handle_2.writelines([str(global_map[0][0][i][j].item()), ' '])
        print("writing into the global map file")
        file_handle_2.close()
        '''
        '''
        # Sample global goal if needed
        # ** write codes to set SAMPLE_GLOBAL_GOAL_FLAG = 0 in specific conditions, e.g., after a period of certain steps, or has been near the goal
        SAMPLE_GLOBAL_GOAL_FLAG = ep_step % goal_interval == 0
        if SAMPLE_GLOBAL_GOAL_FLAG:
            global_policy_inputs = _create_global_policy_inputs(
                global_map, visited_states, curr_map_position
            )
            (
                global_value,
                global_action,
                global_action_log_probs,
                _,
            ) = global_actor_critic.act(global_policy_inputs, None, None, None)

            # Convert action to location (row-major format)
            G = global_actor_critic.G
            global_action_map_x = torch.fmod(
                global_action.squeeze(1), G
            ).float()  # (bs, )
            global_action_map_y = (global_action.squeeze(1) / G).float()  # (bs, )
            # Convert to MxM map coordinates
            global_action_map_x = global_action_map_x * M / G
            global_action_map_y = global_action_map_y * M / G
            global_action_map_xy = torch.stack(
                [global_action_map_x, global_action_map_y], dim=1
            )

            print(global_action_map_xy)
        '''


