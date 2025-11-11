import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation


def disp_message(root):
    behave = os.listdir(root)
    message = {}
    for b in behave:
        message[b] = 0
    for b in behave:
        temp_dict = {}
        human = os.listdir(root + '/' + b)
        human.sort(key=lambda x: int(x[6:]))
        for h in human:
            files = os.listdir(root + '/' + b + '/' + h)
            frame = 0
            for file in files:
                frame += np.load(root + '/' + b + '/' + h + '/' + file).shape[0]
                print('Behave {}，Subject {}，Frames in total {}'.format(b, h, frame))
            if h not in temp_dict.keys():
                temp_dict[h] = frame
        message[b] = temp_dict
    return message


def generate_gif(file_dir, save_dir):
    def update(i):
        ax3d.cla()
        ax3d.set_title(title)
        ax3d.scatter(label[i, :, 0], label[i, :, 1], label[i, :, 2], c='b', label='label')
        pointclouds = point[i]
        ax3d.scatter(pointclouds[:, 0], pointclouds[:, 1], pointclouds[:, 2], c=pointclouds[:, 3] / 0.5,
                     s=20, cmap=cmap, label='point clouds')
        for which_i in conection:
            ax3d.plot([label[i, which_i[0], 0], label[i, which_i[1], 0]],
                      [label[i, which_i[0], 1], label[i, which_i[1], 1]],
                      [label[i, which_i[0], 2], label[i, which_i[1], 2]], color='b')
        ax3d.set_xlim3d([-2, 2])
        ax3d.set_ylim3d([-2, 2])
        ax3d.set_zlim3d([0, 3])
        ax3d.set_xlabel('x/m')
        ax3d.set_ylabel('y/m')
        ax3d.set_zlabel('z/m')
        ax3d.legend(loc='best')

    conection = get_joint_relationship()

    label_file_dir = file_dir + '/IMU'
    point_file_dir = file_dir + '/Radar'

    Behave = os.listdir(label_file_dir)
    for b in Behave:
        if not os.path.exists(save_dir + '/' + b):
            os.makedirs(save_dir + '/' + b)
        Object = os.listdir(label_file_dir + '/' + b)
        Object = [Object[0]]
        for ob in Object:
            if not os.path.exists(save_dir + '/' + b + '/' + ob):
                os.makedirs(save_dir + '/' + b + '/' + ob)
            file = os.listdir(label_file_dir + '/' + b + '/' + ob)
            file = [file[0]]
            for f in file:
                Id = f.split('.')[0]
                label = np.load(label_file_dir + '/' + b + '/' + ob + '/' + f)
                with open(point_file_dir + '/' + b + '/' + ob + '/' + Id + '.pkl', 'rb') as ff:
                    point = pickle.load(ff)
                    ff.close()

                title = b + ' ' + ob + ' ' + Id

                dim1, dim2, dim3 = label.shape
                colors = ["w", "r"]
                cmap = LinearSegmentedColormap.from_list("mycmap", colors)

                fig = plt.figure()
                ax3d = fig.add_subplot(111, projection='3d')
                ax3d.view_init(elev=30, azim=30, roll=0)

                ani = FuncAnimation(fig, update, frames=dim1, interval=15)
                ani.save(save_dir + '/' + b + '/' + ob + '/' + Id + '.gif', writer='pillow', fps=16)
                plt.close(fig)
                print("GIF saved")


def split_dataset_by_frame_num(path, ratio=0.7):
    dict_all = {}
    action = os.listdir(path)
    for ac in action:
        dict_all[ac] = {}
        subject = os.listdir(path + '/' + ac)
        dict_action = {}
        for sb in subject:
            file = os.listdir(path + '/' + ac + '/' + sb)
            dict_action[sb] = 0
            for f in file:
                data = np.load(path + '/' + ac + '/' + sb + '/' + f)
                dict_action[sb] += data.shape[0]
        total_num = sum(dict_action.values())
        train = set()
        cur_num = 0
        for k, v in dict_action.items():
            cur_num += v
            train.add(k)
            if cur_num > ratio * total_num:
                break
        val = dict_action.keys() - train
        dict_all[ac]['train'] = train
        dict_all[ac]['val'] = val
        print('train', train)
        print('val', val)
        print(ac, total_num, cur_num, total_num - cur_num, cur_num / total_num)
    return dict_all


def get_joint_name():
    joint_name = {
        '0': 'Pelvis',
        '1': 'RightHip', '2': 'RightKnee', '3': 'RightAnkle', '4': 'RightToe',
        '5': 'LeftHip', '6': 'LeftKnee', '7': 'LeftAnkle', '8': 'LeftToe',
        '9': 'Spine', '10': 'Spine1', '11': 'Spine2', '12': 'Spine3',
        '13': 'Neck', '14': 'Head',
        '15': 'RightCollar', '16': 'RightShoulder', '17': 'RightElbow', '18': 'RightWrist',
        '19': 'LeftCollar', '20': 'LeftShoulder', '21': 'LeftElbow', '22': 'LeftWrist',
    }
    return joint_name


def get_joint_relationship():
    return [[0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [12, 13], [13, 14],
            [12, 15], [15, 16], [16, 17], [17, 18],
            [12, 19], [19, 20], [20, 21], [21, 22]]

