import json
import numpy as np

# Interaction dataset tools
from utils.interaction_dataset.python.utils.dataset_types import MotionState, Track
# from com_github_interaction_dataset_interaction_dataset.python.utils.dataset_types import MotionState, Track

# class MotionState:
#     def __init__(self, time_stamp_ms):
#         assert isinstance(time_stamp_ms, int)
#         self.time_stamp_ms = time_stamp_ms
#         self.x = None
#         self.y = None
#         self.vx = None
#         self.vy = None
#         self.psi_rad = None

#     def __str__(self):
#         return "MotionState: " + str(self.__dict__)


# class Track:
#     def __init__(self, id):
#         # assert isinstance(id, int)
#         self.track_id = id
#         self.agent_type = None
#         self.length = None
#         self.width = None
#         self.time_stamp_ms_first = None
#         self.time_stamp_ms_last = None
#         self.motion_states = dict()

#     def __str__(self):
#         string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
#                  ", length=" + str(self.length) + ", width=" + str(self.width) + \
#                  ", time_stamp_ms_first=" + str(self.time_stamp_ms_first) + \
#                  ", time_stamp_ms_last=" + str(self.time_stamp_ms_last) + \
#                  "\n motion_states:"
#         for key, value in sorted(self.motion_states.items()):
#             string += "\n    " + str(key) + ": " + str(value)
#         return string

def read_tracks(filename, start_ts=0, end_ts=0):
    """Returns dictionary containing tracks for each agent in STRIVE scenario file.
    
    Arguments:
        filename -- path to STRIVE scenario file
        start_ts -- timestamp of first step in scenario (needed for initial state)
        end_ts -- timestemp of last step in scenario

    Returns:
        track_dict -- dictionary of tracks for all agents    
    """
    with open(filename) as json_data:
        strive_scene_dict = json.load(json_data)
        json_data.close()

    # JSON generated in scene_out_dict from STRIVE adv_scenario_gen_bark
    N = strive_scene_dict["N"]
    dt = strive_scene_dict["dt"]
    map = strive_scene_dict["map"]
    lw = np.array(strive_scene_dict["lw"])
    past = np.array(strive_scene_dict["past"])
    fut_init = np.array(strive_scene_dict["fut_init"])
    fut_adv = np.array(strive_scene_dict["fut_adv"])

    # need to compute velocities for future (already have for past)
    fut_traj = np.concatenate([past[:, -1:, :4], fut_adv], axis=1)
    t = np.arange(fut_traj.shape[1])*dt
    vels = [np.linalg.norm(velocity(fut_traj[aidx, :, :2], t)[1:], axis=1) for aidx in range(N)]
    fut_h = np.arctan2(fut_traj[:, :, 3], fut_traj[:, :, 2])
    hdots = [heading_change_rate(fut_h[aidx], t)[1:] for aidx in range(N)] 

    # fut = np.concatenate([fut_adv, vels[:, np.newaxis], hdots[:, np.newaxis]], axis=1)
    vels = np.array(vels)[:, :, np.newaxis]
    hdots = np.array(hdots)[:, :, np.newaxis]
    fut = np.concatenate([fut_adv, vels, hdots], axis=2)
    traj = np.concatenate([past, fut], axis=1)

    # reconstruct time_stamps
    n_steps = traj.shape[1]
    step_size = int(1000*dt) # 1 s = 1000 ms
    time_stamp_ms_first = start_ts
    time_stamp_ms_last = start_ts + n_steps * step_size # end_ts
    time_stamps_ms = [ts for ts in range(time_stamp_ms_first, time_stamp_ms_last, step_size)]
    # time_stamps_ms = int(time_stamp_ms)

    track_dict = dict()
    track_id = None

    # fill track dictionary
    for aidx in range(N):
        track_id = aidx + 1 # numbering of track_ids (agent_ids) starts at 1
        track = Track(track_id) # Track object defined in interaction dataset_types
        track.length = lw[aidx, 0]
        track.width = lw[aidx, 1]

        # assume all cars
        track.agent_type = 'car' # TODO use "sem" from strive_scene_dict to extract agent_type
        track.time_stamp_ms_first = time_stamp_ms_first
        track.time_stamp_ms_last = time_stamp_ms_last
        track_dict[track_id] = track

        track = track_dict[track_id]
        track.time_stamp_ms_last = time_stamp_ms_last

        i = 0
        for time_stamp_ms in time_stamps_ms:
            ms = MotionState(time_stamp_ms)
            ms.x = traj[aidx, i, 0]
            ms.y = traj[aidx, i, 1]

            # calculate velocity
            vel = traj[aidx, i, 4]
            ms.vx = np.cos(vel)
            ms.vy = np.sin(vel)

            # calculate heading angle
            hsin = traj[aidx, i, 3]
            hcos = traj[aidx, i, 2]
            heading = np.arctan2(hsin, hcos)
            ms.psi_rad = heading

            track.motion_states[ms.time_stamp_ms] = ms
            i+=1
        
    return track_dict


def velocity(pos, t):
    '''
    Given positions may be nan. If so returns nans for these frames. Velocity are computeed using
    backward finite differences except for leading frames which use forward finite diff. Any single
    frames (i.e have no previous or future steps) are nan.

    :param pos: positions (T x D)
    :param t: timestamps (T) in sec
    :return vel: (T x D)
    ''' 
    vel_diff = (pos[1:, :] - pos[:-1, :]) / (t[1:] - t[:-1]).reshape((-1, 1))
    vel = np.concatenate([vel_diff[0:1,:], vel_diff], axis=0) # for first frame use forward diff

    # for any nan -> value transition frames, want to use forward difference
    posnan = np.isnan(np.sum(pos, axis=1)).astype(np.int)
    if np.sum(posnan) == 0:
        return vel
    lead_nans = (posnan[1:] - posnan[:-1]) == -1
    lead_nans = np.append([False], lead_nans)
    repl_idx = np.append([False], lead_nans[:-1])
    num_fill = np.sum(repl_idx.astype(np.int))
    if num_fill != 0:
        if num_fill != np.sum(lead_nans.astype(np.int)):
            # the last frame is a leading nan, have to ignore it
            lead_nans[-1] = False
        vel[lead_nans] = vel[repl_idx]
    return vel

def heading_change_rate(h, t):
    '''
    Given heading angles may be nan. If so returns nans for these frames. Velocity are computeed using
    backward finite differences except for leading frames which use forward finite diff. Any single
    frames (i.e have no previous or future steps) are nan.

    :param h: heading angles (T)
    :param t: timestamps (T) in sec
    :return hdot: (T)
    ''' 
    hdiff = angle_diff(h[1:], h[:-1]) / (t[1:] - t[:-1])
    hdot = np.append(hdiff[0:1], hdiff) # for first frame use forward diff

    # for any nan -> value transition frames, want to use forward difference
    hnan = np.isnan(h).astype(np.int)
    if np.sum(hnan) == 0:
        return hdot
    lead_nans = (hnan[1:] - hnan[:-1]) == -1
    lead_nans = np.append([False], lead_nans)
    repl_idx = np.append([False], lead_nans[:-1])
    num_fill = np.sum(repl_idx.astype(np.int))
    if num_fill != 0:
        if num_fill != np.sum(lead_nans.astype(np.int)):
            # the last frame is a leading nan, have to ignore it
            lead_nans[-1] = False
        hdot[lead_nans] = hdot[repl_idx]
    return hdot

def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (B)
    :param theta2: angle 2 (B)
    :return diff: smallest angle difference between angles (B)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff