import os
import sys
sys.path.append('../')
from parameters import parameters 
number_of_joints = len(parameters.joint_list)
numbers_per_joint = parameters.numbers_per_joint
numbers_per_joint_for_loss = parameters.numbers_per_joint_for_loss


import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import copy
import json
import pickle
import numpy as np
import cv2
import itertools
from data_augmentation import sequence_permutations_generator_random

MAX_COMBINATIONS_NUMBER = 3

sys.path.append('../')
from parameters import parameters 

from pose_estimator_utils import camera_matrix

tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = []
camera_d_transforms = []
camera_matrices = {}
distortion_coefficients = {}
projection_matrices = {}

for cam_iList, cam_idx in enumerate(parameters.cameras):
    cam = parameters.camera_names[cam_iList]
    # Add the direct transform (root to camera) to the list
    trfm = tm.get_transform("root", cam)
    camera_d_transforms.append(torch.from_numpy(trfm).type(torch.float32))
    # Add the inverse transform (camera to root) to the list
    camera_i_transforms.append(torch.from_numpy(tm.get_transform(cam, "root")).type(torch.float32))
    # Add the camera matrix to the list
    camera_matrices[cam] = camera_matrix(cam_idx, use_cuda=False).cpu().detach().numpy()

    distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])

    projection_matrices[cam] = trfm[0:3, :]

def get_skeleton_indices(data):
    skeleton_indices = {}
    for cam in data.keys():
        joints_json = data[cam][0]
        skeletons = json.loads(joints_json)
        n_joints = 0
        index = 0
        for i, skeleton in enumerate(skeletons):
            if len(skeleton)>n_joints:
                n_joints = len(skeleton)
                index = i
        skeleton_indices[cam] = index
    return skeleton_indices

def get_3D_from_triangulation(data, skeleton_indices):
    points_2D = dict()
    for cam in data.keys():
        if cam in parameters.used_cameras:
            joints_json = data[cam][0]
            joints = json.loads(joints_json)
            if not joints:
                continue
            skeleton_index = skeleton_indices[cam]
            for j, pos in joints[skeleton_index].items():
                if j == "ID":
                    continue
                if pos[0] > 0.:
                    if not j in points_2D.keys():
                        points_2D[j] = dict()
                    points_2D[j][cam] = np.array([pos[1], pos[2]])


    result3D = dict()
    for idx_i in parameters.joint_list:
        idx = str(idx_i)
        mean_point3D = np.zeros((3, 1))
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.combinations(range(len(points_2D[idx].keys())), 2)
            n_comb = 0
            for comb in cam_combinations:
                cam1 = list(points_2D[idx].keys())[comb[0]]
                cam2 = list(points_2D[idx].keys())[comb[1]]
                point1 = np.array(points_2D[idx][cam1])
                new_point1 = cv2.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                point2 = np.array(points_2D[idx][cam2])
                new_point2 = cv2.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                point3d = cv2.triangulatePoints(projection_matrices[cam1], projection_matrices[cam2], new_point1, new_point2)
                point3d = point3d[0:3]/point3d[3]

                mean_point3D += point3d
                n_comb += 1
            result3D[idx] = mean_point3D/n_comb
    return result3D


from data_augmentation import permutations_generator

image_width = parameters.image_width
image_height = parameters.image_height

class PoseEstimatorDataset(Dataset):
    def __init__(self, sequence_length, sample_step, input_data, cameras, joint_list, transform=None, data_augmentation=False, reload=False, save=False, device=None):
        """
            input_data
               -> list[str]: List containing paths to the JSON files.
               -> str:       A single string containing the JSON text for a single sample.
            cameras (list of integers): Camera identifiers to be used.
               -> IGNORED IN THE CURRENT IMPLEMENTATION
            joint_list (list of integers): Joint identifiers to be extracted from the dataset.
               -> IGNORED IN THE CURRENT IMPLEMENTATION
        """
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.numbers_per_joint = numbers_per_joint
        self.numbers_per_joint_for_loss = numbers_per_joint_for_loss
        self.sequence_length = sequence_length

        self.sample_step = sample_step

        self.camera_section_length_total = len(parameters.joint_list)*numbers_per_joint_for_loss  # L joints/skeleton, X numbers/joint.
        self.camera_section_length_input = len(parameters.joint_list)*numbers_per_joint  # L joints/skeleton, X numbers/joint.
        skeleton_length_total = self.camera_section_length_total * len(parameters.cameras)
        skeleton_length_input = self.camera_section_length_input * len(parameters.used_cameras)

        self.data = []
        self.orig_data = []
        self.person_indices = {}
        self.available_cams = []
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        sequences_data = []
        sequences_orig = []
        self.sequence_cams = []

        indices_2D_features = []
        reset_multipliers = []
        for c_index in range(len(parameters.used_cameras)):
            part_indices = []
            c_offset = c_index * self.camera_section_length_input
            for j in parameters.joint_list:
                j_offset = int(j) * numbers_per_joint
                part_indices.extend(range(c_offset + j_offset, c_offset + j_offset + 10))
            indices_2D_features.extend(part_indices)
            reset_multipliers.append(torch.ones(len(part_indices)))

        self.reset_features_len = 10*len(parameters.joint_list)
        self.indices_2D_features = torch.tensor(indices_2D_features)
        self.reset_multipliersDA = torch.cat(reset_multipliers)
        if device is not None:
            print(device)
            self.indices_2D_features = self.indices_2D_features.to(device=device)
            self.reset_multipliersDA = self.reset_multipliersDA.to(device=device)

        if reload is True:
            reload_fname = f'{input_data[-1]}_{self.sequence_length}_{self.sample_step}.pytorch'
            if os.path.exists(reload_fname):
                loaded = torch.load(reload_fname)
                self.data = loaded['data']
                self.orig_data = loaded['orig_data']
                self.person_indices = loaded['person_indices']
                self.sequence_cams = loaded['sequence_cams']
                return

        ignored_names = []

        given = 0
        total = 0
        if type(input_data) is list and type(input_data[0]) is str:
            json_files = input_data
            person_id = 0
            i_sample = 0

            for f in json_files:  # FOR EACH INPUT FILE
                print(f)
                self.person_indices[person_id] = []
                json_data = json.loads(open(f, "rb").read())
                n_loaded = 0
                n_data = len(json_data)
                for data in json_data:  # FOR EACH SAMPLE IN A JSON FILE
                    view_from_robot = False
                    given += 1
                    flags = [0]*len(parameters.used_cameras)
                    skeleton_indices = get_skeleton_indices(data)
                    results_3D = get_3D_from_triangulation(data, skeleton_indices)
                    error_input = torch.zeros([skeleton_length_total])
                    network_input = torch.zeros([skeleton_length_input])
                    for c in data:  # FOR EACH CAMERA IN A SAMPLE
                        #include 2D information for all the cameras in error_input
                        try:
                            c_index = parameters.camera_names.index(c)
                        except ValueError:
                            # Ignore the sample if the camera is not in the list of cameras to be used while training
                            if c not in ignored_names:
                                print(f'Ignoring {c} because it\'s not in the list of cameras to be used while training')
                                ignored_names.append(c)
                            continue
                        c_offset = c_index * self.camera_section_length_total
                        skeleton = json.loads(data[c][0])
                        if not skeleton:
                            continue
                        skeleton = skeleton[skeleton_indices[c]]
                        for j, values in skeleton.items():
                            if j == "ID":
                                continue
                            j_offset = int(j) * numbers_per_joint_for_loss
                            error_input[c_offset + j_offset] = values[3]
                            error_input[c_offset + j_offset + 1] = values[1]
                            error_input[c_offset + j_offset + 2] = values[2]
                            error_input[c_offset + j_offset + 3] = values[4]

                        if c in parameters.used_cameras:
                            used_c_index = parameters.used_cameras.index(c)
                            used_c_offset = used_c_index * self.camera_section_length_input

                            cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # world to camera transformation matrix, results_3d)
                            for j, values in skeleton.items():
                                if j == "ID":
                                    continue
                                if values[3] < 1.:
                                    continue
                                flags[used_c_index] = 1
                                view_from_robot = True
                                used_j_offset = int(j) * numbers_per_joint
                                network_input[used_c_offset + used_j_offset] = values[3]
                                network_input[used_c_offset + used_j_offset + 1] = (values[1] - image_width/2) / (image_width/2)
                                network_input[used_c_offset + used_j_offset + 2] = (values[2] - image_height/2) / (image_height/2)
                                network_input[used_c_offset + used_j_offset + 3] = values[4]

                                point = np.array([values[1], values[2]])                                
                                undistorted_point = cv2.undistortPoints(point, camera_matrices[c], distortion_coefficients[c])
                                undistorted_pix_ray = torch.from_numpy(undistorted_point[0][0]).type(torch.float32)
                                pix_ray_from_root = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray, torch.tensor([1.0, 0.0])))) #perform only rotation
                                network_input[used_c_offset + used_j_offset + 4: used_c_offset + used_j_offset + 7] = cam_from_root[0:3] / 10.
                                network_input[used_c_offset + used_j_offset + 7: used_c_offset + used_j_offset + 10] = pix_ray_from_root[0:3] / 10.


                    if view_from_robot:
                        for c_index in range(len(parameters.used_cameras)):  # Include 3D from triangulation
                            used_c_offset = c_index * self.camera_section_length_input
                            for j in results_3D:
                                used_j_offset = int(j) * numbers_per_joint
                                network_input[used_c_offset + used_j_offset + 10] = 1. # 3D is available
                                network_input[used_c_offset + used_j_offset + 11: used_c_offset + used_j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.

                        self.data.append(network_input)
                        self.orig_data.append(error_input)
                        self.available_cams.append(flags)


                        n_loaded += 1      
                  
                        if n_loaded % 1000 == 0:
                            print('Loaded', n_loaded, 'of', n_data)


                # current_seq_data = []
                # current_seq_orig = []
                # current_seq_cams = []

                for data_index, _ in enumerate(self.data):

                    current_seq_data = []
                    current_seq_orig = []
                    current_seq_cams = []

                    for step in range(self.sequence_length-1, -1, -1):
                        index = data_index - step*self.sample_step
                        if index < 0:
                            current_seq_data.append(torch.zeros([skeleton_length_input]))
                            current_seq_orig.append(torch.zeros([skeleton_length_total]))
                            current_seq_cams.append([0]*len(parameters.used_cameras))
                        else:
                            current_seq_data.append(self.data[index].detach().clone())
                            current_seq_orig.append(self.orig_data[index].detach().clone())
                            current_seq_cams.append(self.available_cams[index])

                    sequences_data.append(torch.stack(current_seq_data))
                    sequences_orig.append(torch.stack(current_seq_orig))
                    self.sequence_cams.append(copy.deepcopy(current_seq_cams))
                    self.person_indices[person_id].append(i_sample)
                    i_sample += 1
                    total += 1

                    # current_seq_data.append(element)
                    # current_seq_orig.append(self.orig_data[data_index])
                    # current_seq_cams.append(self.available_cams[data_index])

                    # if len(current_seq_data) == sequence_length:
                    #     # for comb_seq in sequence_permutations_generator_random(current_seq_cams, self.data_augmentation, MAX_COMBINATIONS_NUMBER):
                    #     #     seq_DA = copy.deepcopy(current_seq_data)
                    #     #     for i, combination in enumerate(comb_seq):
                    #     #         for c_index, part in enumerate(combination):
                    #     #             c_offset = c_index * self.camera_section_length_input
                    #     #             if part == 0:
                    #     #                 for j in parameters.joint_list:
                    #     #                     j_offset = int(j) * numbers_per_joint
                    #     #                     seq_DA[i][c_offset + j_offset: c_offset + j_offset + 10] = 0.
                    #     sequences_data.append(torch.stack(current_seq_data))
                    #     sequences_orig.append(torch.stack(current_seq_orig))
                    #     self.sequence_cams.append(copy.deepcopy(current_seq_cams))
                    #     self.person_indices[person_id].append(i_sample)
                    #     i_sample += 1
                    #     total += 1
                            
                    #     for _ in range(int(round((sequence_length/35), 0))):
                    #         current_seq_data.pop(0)
                    #         current_seq_orig.pop(0)
                    #         current_seq_cams.pop(0)

                self.data = []
                self.orig_data = []
                self.available_cams = []
                person_id += 1
                # if n_loaded>1000:
                #     break

            print(f'Given {given}\nTotal {total}')
        elif type(input_data) is list and type(input_data[0]) is dict: ## for testing. Single sequence
            current_seq_data = []
            # if len(input_data)<sequence_length:
            #     current_seq_data = [torch.zeros([skeleton_length_input])]*(sequence_length-len(input_data))
            data_index = len(input_data) - 1
            # for frame in input_data:
            for step in range(self.sequence_length-1, -1, -1):
                index = data_index - step*self.sample_step
                if index < 0:
                    current_seq_data.append(torch.zeros([skeleton_length_input]))
                else:
                    frame = input_data[index]
                    skeleton_indices = get_skeleton_indices(frame)
                    results_3D = get_3D_from_triangulation(frame, skeleton_indices)
                    output = torch.zeros([skeleton_length_input])
                    for c in frame:
                        if c in parameters.used_cameras:
                            c_index = parameters.camera_names.index(c)
                            used_c_index = parameters.used_cameras.index(c)
                            used_c_offset = used_c_index * self.camera_section_length_input

                            cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0])) / 10.  # world to camera transformation matrix, results_3d)                
                            skeleton = json.loads(frame[c][0])
                            if not skeleton:
                                continue
                            skeleton = skeleton[skeleton_indices[c]]

                            point_list = []
                            for j, values in skeleton.items():
                                if j == "ID": continue
                                point_list.append([values[1], values[2]])
                            if point_list:
                                point_list = np.array(point_list)
                                norm_factors = np.array([[image_width/2, image_height/2]]*point_list.shape[0])
                                normalize_points = (point_list-norm_factors)/norm_factors                    
                                undistorted_point_list = cv2.undistortPoints(point_list, camera_matrices[c], distortion_coefficients[c]).squeeze(axis=1)
                                undistorted_pix_ray_list = torch.from_numpy(undistorted_point_list).type(torch.float32)
                                new_col = torch.tensor([[1.0, 0.0]]*point_list.shape[0])
                                pix_ray_from_root_list = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray_list, new_col), dim=1).transpose(dim0=1, dim1=0))/10. #perform only rotation
                                pix_ray_from_root_list = pix_ray_from_root_list.transpose(dim0=1, dim1=0)

                            i_point = 0
                            for j, values in skeleton.items():
                                if j == "ID": continue
                                j_offset = int(j) * numbers_per_joint
                                output[used_c_offset + j_offset] = values[3]
                                output[used_c_offset + j_offset + 1] = normalize_points[i_point][0] #(values[1] - image_width/2) / (image_width/2) 
                                output[used_c_offset + j_offset + 2] = normalize_points[i_point][1] #(values[2] - image_height/2) / (image_height/2)
                                output[used_c_offset + j_offset + 3] = values[4]

                                output[used_c_offset + j_offset + 4: used_c_offset + j_offset + 7] = cam_from_root[0:3]
                                output[used_c_offset + j_offset + 7: used_c_offset + j_offset + 10] = pix_ray_from_root_list[i_point][0:3] #pix_ray_from_root[0:3] / 10.
                                i_point += 1

                    for c_index in range(len(parameters.used_cameras)):  # Include 3D from triangulation
                        used_c_offset = c_index * self.camera_section_length_input
                        for j in results_3D:
                            j_offset = int(j) * numbers_per_joint
                            output[used_c_offset + j_offset + 10] = 1. # 3D is available
                            output[used_c_offset + j_offset + 11: used_c_offset + j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.
                    current_seq_data.append(output)

            # if torch.sum(torch.abs(current_seq_data)) > 1:
            sequences_data.append(torch.stack(current_seq_data))
            sequences_orig = sequences_data
            self.person_indices = []
        else:
            raise Exception(f'Invalid dataset input {type(input_data)} for json_files. Only list and dict are allowed.')

        # sequences_data = []
        # sequences_orig = []

        # current_seq_data = []
        # current_seq_orig = []

        # for data_index, element in enumerate(self.data):
        #     current_seq_data.append(element)
        #     current_seq_orig.append(self.orig_data[data_index])

        #     if len(current_seq_data) == sequence_length:
        #         sequences_data.append(torch.stack(current_seq_data))
        #         sequences_orig.append(torch.stack(current_seq_orig))

        #         for _ in range(int(round((sequence_length/35), 0))):
        #             current_seq_data.pop(0)
        #             current_seq_orig.pop(0)


        if device is None:
            #self.data = torch.stack(self.data)
            #self.orig_data = torch.stack(self.orig_data)
            self.data = torch.stack(sequences_data)
            self.orig_data = torch.stack(sequences_orig)

        else:
            #self.data = torch.stack(self.data).to(device=device)
            #self.orig_data = torch.stack(self.orig_data).to(device=device)
            self.data = torch.stack(sequences_data).to(device=device)
            self.orig_data = torch.stack(sequences_orig).to(device=device)

        if save:
             torch.save({
                'data': self.data,
                'orig_data': self.orig_data,
                'person_indices': self.person_indices,
                'sequence_cams': self.sequence_cams
                }, f'{input_data[-1]}_{self.sequence_length}_{self.sample_step}.pytorch')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        ret1 = self.data[idx]
        ret2 = self.orig_data[idx]

        if self.data_augmentation:
            ret1 = self.data[idx].detach().clone()  
            ret1 = ret1.to(self.device)       
            comb_seq = sequence_permutations_generator_random(self.sequence_cams[idx])
            for i, combination in enumerate(comb_seq):
                # multipliers = torch.cat([
                #     part * self.reset_multipliersDA[c_index * self.reset_features_len : (c_index + 1) * self.reset_features_len]
                #     for c_index, part in enumerate(combination)
                # ]).to(self.device)

                # ret1[i, self.indices_2D_features] *= multipliers


                for c_index, part in enumerate(combination):
                    c_offset = c_index * self.camera_section_length_input
                    if part == 0 and self.sequence_cams[idx][i][c_index] == 1:
                        for j in parameters.joint_list:
                            j_offset = int(j) * numbers_per_joint
                            ret1[i][c_offset + j_offset: c_offset + j_offset + 10] = 0.


        if self.transform:
            ret1 = self.transform(ret1)

        return ret1, ret2



class PersonBatchSampler(Sampler):
    # Yield a mini-batch of indices of the same person. 


    def __init__(self, person_indices, batch_size):
        # build data for sampling here
        self.batch_size = batch_size
        self.person_indices = person_indices
        self.data_len = 0
        for p in self.person_indices:
            self.data_len += len(self.person_indices[p])
        
        
    def __iter__(self):
        # implement logic of sampling here
        indices = copy.deepcopy(self.person_indices)
        persons = list(indices.keys())
        for p in persons:
            random.shuffle(indices[p])
        while len(persons)>0:
            id = random.randint(0, len(persons)-1)
            person = persons[id]
            batch = []
            while len(batch) < self.batch_size and len(indices[person])>0:
                batch.append(indices[person].pop(0))
            # print("person", person, "batch size", len(batch))
            if len(indices[person])==0:
                persons.pop(id)
            yield batch

    def __len__(self):
        return len(self.data_len)
    

if __name__ == '__main__':
    files = sys.argv[1:]
    dataset = PoseEstimatorDataset(10, 10, files, parameters.cameras, parameters.joint_list, data_augmentation=True, reload=True, save=False)
    item1, item2 = dataset[100]
    print(item1)
