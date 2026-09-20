import os
import numpy as np
from shutil import copyfile
import pandas as pd


# construct class id dictionary
def get_dir_label(data_dir):
    cate_list = os.listdir(os.path.join(data_dir))
    print('cate_list:', cate_list)
    if '.DS_Store' in cate_list:
        cate_list.remove('.DS_Store')

    class_id_dict = {}
    for i in range(len(cate_list)):
        current_class = cate_list[i]
        class_dir = os.path.join(data_dir, current_class)
        print('class_dir:', class_dir)
        sample_list = os.listdir(class_dir)
        print('sample_list:', len(sample_list))

        if '.DS_Store' in sample_list:
            sample_list.remove('.DS_Store')
        data_samples = []
        for sample in sample_list:
            # print('sample:', sample)
            data_samples.append(os.path.join(current_class, sample[:-4]))
            # data_samples.append(os.path.join(current_class, sample))
            # print('sample:', sample)

        class_id_dict[i] = data_samples

    return class_id_dict



def data_splitter(class_id_dict, train_ratio, val_ratio):
    id_list = list(class_id_dict.keys())
    train_sample = []
    train_label  = []
    test_sample  = []
    test_label   = []
    val_sample   = []
    val_label    = []

    # loop for category 0 to xx
    for i in range(len(id_list)):
        samples = class_id_dict[i]
        np.random.seed(i)
        np.random.shuffle(samples)
        train_num = int(np.floor(len(samples)*train_ratio))
        #print('train_num:', train_num)
        val_num   = int(np.floor(len(samples)*val_ratio))
        #print('val_num:', val_num)
        train_sample.extend(samples[:train_num])
        train_label.extend([i for k in range(train_num)])

        val_sample.extend(samples[train_num:(train_num+val_num)])
        val_label.extend([i for k in range(val_num)])

        test_sample.extend(samples[(train_num+val_num):])
        test_label.extend([i for k in range(len(samples)-train_num-val_num)])

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)


def get_samples(samples, current_train_sample_id):
    sample_list = []
    for i in current_train_sample_id:
        # print('samples[i]', samples[i])
        sample_list.append(samples[i])
    return sample_list

def integrate_sample_id(sample_list):
    sample_id_dict = {}
    for i in range(len(sample_list)):
        sample_ = sample_list[i].split('/')
        sample_id = sample_[1][:5]

        if sample_id not in sample_id_dict:
             sample_id_dict[sample_id] = []

        sample_id_dict[sample_id].append(sample_list[i])

    return sample_id_dict

def data_splitter_by_class(class_id_dict, train_ratio, val_ratio):
    # class_id_dict is dictionary of class and corresponding files
    # id is class and each category contains more than 100 images
    # we take 80 for training and 20 for test
    id_list = list(class_id_dict.keys())    # 0-66

    train_sample = []
    train_label = []
    test_sample = []
    test_label = []

    # loop for category 0 to 66
    for i in id_list:
        # print('i', i)
        samples = class_id_dict[i]
        # print('samples', samples)

        # sample_id_dict = integrate_sample_id(samples)
        # print('sample_id_dict:', len(sample_id_dict))

        sample_ids = list(range(len(samples)))

        # print('samples:', len(samples), samples[0], samples[-1])
        # print('sample_id_dict:', sample_id_dict)
        # print('sample_ids:', len(sample_ids))

        np.random.seed(i)
        np.random.shuffle(sample_ids)

        train_num = 80
        test_num = 20
        # print('train_num:', train_num, 'val_num:', val_num)

        current_train_sample_id = sample_ids[:train_num]
        current_test_sample_id = sample_ids[train_num:(train_num + test_num)]

        # print('current_train_sample_id', current_train_sample_id,
        #       'current_test_sample_id', current_test_sample_id)

        current_train_sample = get_samples(samples, current_train_sample_id)
        # print('current_train_sample', type(current_train_sample), len(current_train_sample), current_train_sample)
        train_label.extend([i for k in range(len(current_train_sample))])

        current_test_sample = get_samples(samples, current_test_sample_id)
        # print('current_test_sample', type(current_test_sample), len(current_test_sample), current_test_sample)
        test_label.extend([i for k in range(len(current_test_sample))])

        train_sample.extend(current_train_sample)
        test_sample.extend(current_test_sample)

    # print('shape of datset:', len(train_sample), len(test_sample))

    # 80x67=5360
    # print('train_sample:', len(train_sample), train_sample)
    # print('train_label:', len(train_label), train_label)

    # 20x67=1340
    # print('test_sample:', len(test_sample), test_sample)
    # print('test_label:', len(test_label), test_label)


    return (train_sample, train_label, test_sample, test_label)


def data_construction(data_dir, train_ratio=0.7, val_ratio=0.1):

    class_id_dict = get_dir_label(data_dir)
    # print('class_id_dict:', class_id_dict)

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_splitter_by_id(class_id_dict, train_ratio, val_ratio)

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)


def data_splitter_by_id(class_id_dict, train_ratio, val_ratio):
    # print('class_id_dict:', class_id_dict)
    # print('class_id_dict.keys():', class_id_dict.keys())
    id_list = list(class_id_dict.keys())
    print('id_list:', id_list)

    train_sample = []
    train_label  = []
    test_sample  = []
    test_label   = []
    val_sample   = []
    val_label    = []


    for i in range(len(id_list)):
        samples = class_id_dict[i]
        # print('samples:', samples)
        sample_id_dict = integrate_sample_id(samples)
        # print('sample_id_dict:', sample_id_dict)
        
        sample_ids = list(sample_id_dict.keys())

        np.random.seed(i)
        np.random.shuffle(sample_ids)

        train_num = int(np.floor(len(sample_ids)*train_ratio))
        val_num   = int(np.floor(len(sample_ids)*val_ratio))

        current_train_sample_id = sample_ids[:train_num]
        current_val_sample_id   = sample_ids[train_num:(train_num+val_num)]
        current_test_sample_id  = sample_ids[(train_num+val_num):]

        current_train_sample = get_samples(sample_id_dict, current_train_sample_id)
        train_sample.extend(current_train_sample)
        train_label.extend([i for k in range(len(current_train_sample))])

        current_val_sample = get_samples(sample_id_dict, current_val_sample_id)
        val_sample.extend(current_val_sample)
        val_label.extend([i for k in range(len(current_val_sample))])

        current_test_sample = get_samples(sample_id_dict, current_test_sample_id)
        test_sample.extend(current_test_sample)
        test_label.extend([i for k in range(len(current_test_sample))])

    print('shape of datset:', len(train_sample), len(val_sample), len(test_sample))
    # print('train_sample:', train_sample)
    # print('train_label:', train_label)
    
    # print('test_sample:', test_sample)
    # print('val_sample:', val_sample)

    # Added by Zhou
    # dst_dir = '/data/guest/avasr/avasrd/train_img'
    # copy_imgs_from_src_dir(train_sample, dst_dir)
    #
    # dst_dir = '/data/guest/avasr/avasrd/val_img'
    # copy_imgs_from_src_dir(val_sample, dst_dir)
    #
    # dst_dir = '/data/guest/avasr/avasrd/test_img'
    # copy_imgs_from_src_dir(test_sample, dst_dir)

    # for img in train_sample:
    #     print('img:', img)
    #     cls_name = img.split('/')[0]
    #     img_name = img.split('/')[1]
    #     print('cls_name:', cls_name, img_name)
    #     img_dir = '/home/lgzhou/dataset/AVASRD/vision'
    #     dst_dir = '/home/lgzhou/dataset/AVASRD/train_img'
    #
    #     dst_dir = os.path.join(dst_dir, cls_name)
    #     if not os.path.exists(dst_dir):
    #         os.mkdir(dst_dir)
    #
    #     img_path = os.path.join(img_dir, cls_name, img_name + '.jpg')
    #     dst_path = os.path.join(dst_dir, img_name + '.jpg')
    #
    #     # print('imgpath:', img_path)
    #     # print('dstpath:', dst_path)
    #     copyfile(img_path, dst_path)


    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)



def copy_imgs_from_src_dir(train_sample, dst_dir_father):



    for img in train_sample:
        print('img:', img)
        cls_name = img.split('/')[0]
        img_name = img.split('/')[1]
        print('cls_name:', cls_name, img_name)
        img_dir = '/data/lgzhou/avasr/avasrd/vision/'

        dst_dir = os.path.join(dst_dir_father, cls_name)

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        img_path = os.path.join(img_dir, cls_name, img_name + '.jpg')
        dst_path = os.path.join(dst_dir, img_name + '.jpg')

        print('imgpath:', img_path)
        print('dstpath:', dst_path)
        copyfile(img_path, dst_path)

def cate_data_sample(class_id_dict, train_ratio, class_id):
    id_list = list(class_id_dict.keys())
    
    samples = class_id_dict[class_id]
    np.random.seed(class_id)
    np.random.shuffle(samples)
    train_num = int(np.floor(len(samples)*train_ratio))
    train_sample = samples[:train_num]
 
    return train_sample

def visual_data_construction(data_dir, train_ratio=0.7, val_ratio=0.1):

    class_id_dict = get_dir_label(data_dir)
    # print('class_id_dict:', class_id_dict)

    (train_sample, train_label, test_sample, test_label) = data_splitter_by_class(class_id_dict, train_ratio, val_ratio)

    return (train_sample, train_label, test_sample, test_label)


# construct audio dataset
def audio_construction(data_dir, train_ratio=0.7, val_ratio=0.1):

    class_id_dict = get_dir_label(data_dir)

    (train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_splitter_by_id(class_id_dict, train_ratio, val_ratio)

    #train_sample.extend(val_sample)
    #train_sample.extend(test_sample)

    return (train_sample, train_label, val_sample, val_label, test_sample, test_label)




# construction esc dataset
def esc_dataset_construction(csv_dir, test_set_id, dataset_name):

    dataset_pd = pd.DataFrame(pd.read_csv(csv_dir))
    print('datasetname:', dataset_name)

    if dataset_name == 'ESC10':
        print('datasetname:', dataset_name)
        train_dataset = dataset_pd.loc[(dataset_pd['fold'] != test_set_id) & (dataset_pd['esc10'] == True)]
        test_dataset = dataset_pd.loc[(dataset_pd['fold'] == test_set_id) &  (dataset_pd['esc10'] == True)]
        train_sample = train_dataset['filename'].tolist()
        train_label = train_dataset['target'].tolist()
        test_sample = test_dataset['filename'].tolist()
        test_label = test_dataset['target'].tolist()

    elif dataset_name == 'ESC50':
        print('datasetname:', dataset_name)
        train_dataset = dataset_pd.loc[(dataset_pd['fold'] != test_set_id)]
        test_dataset = dataset_pd.loc[(dataset_pd['fold'] == test_set_id)]
        print('train_dataset:', train_dataset, type(train_dataset))
        train_sample = train_dataset['filename'].tolist()
        train_label = train_dataset['target'].tolist()
        test_sample = test_dataset['filename'].tolist()
        test_label = test_dataset['target'].tolist()

    elif dataset_name == 'US8K':

        train_dataset = dataset_pd.loc[(dataset_pd['fold'] != test_set_id) == True]
        test_dataset = dataset_pd.loc[(dataset_pd['fold'] == test_set_id) == True]
        print('len:', len(train_dataset), len(test_dataset))
        train_sample = train_dataset['slice_file_name'].tolist()
        train_label = train_dataset['classID'].tolist()
        test_sample = test_dataset['slice_file_name'].tolist()
        test_label = test_dataset['classID'].tolist()

    print('len:', len(train_dataset), len(test_dataset))

    return (train_sample, train_label, test_sample, test_label)

# construction esc dataset
def home_automation_dataset_construction(csv_dir):
    dataset_pd = pd.DataFrame(pd.read_csv(csv_dir))
    print(dataset_pd['fold'])
    train_dataset = dataset_pd.loc[(dataset_pd['fold'] == 'train')]
    test_dataset = dataset_pd.loc[(dataset_pd['fold'] == 'test')]

    train_sample = []
    test_sample = []

    train_filename = train_dataset['filename'].tolist()
    train_category = train_dataset['category'].tolist()
    train_label = train_dataset['target'].tolist()

    for i in range(len(train_label)):
        train_sample.append(os.path.join(train_category[i], train_filename[i]))

    test_filename = test_dataset['filename'].tolist()
    test_category = test_dataset['category'].tolist()
    test_label = test_dataset['target'].tolist()

    for i in range(len(test_label)):
        test_sample.append(os.path.join(test_category[i], test_filename[i]))


    print('len:', len(train_dataset), len(test_dataset))

    return (train_sample, train_label, test_sample, test_label)

# construction dcase dataset
def dcase_dataset_construction(args, train_csv_path, val_csv_path, data_dir=''):

    train_dataset_pd = pd.DataFrame(pd.read_csv(train_csv_path, sep='\t'))
    val_dataset_pd = pd.DataFrame(pd.read_csv(val_csv_path, sep='\t'))

    if args.dataset_name == 'DCASE2019':
        train_dataset = train_dataset_pd[['filename']]
        train_dataset_label = train_dataset_pd[['scene_label']]
        test_dataset = val_dataset_pd[['filename']]
        test_dataset_label = val_dataset_pd[['scene_label']]

        train_sample = train_dataset['filename'].tolist()
        train_label = train_dataset_label['scene_label'].tolist()
        test_sample = test_dataset['filename'].tolist()
        test_label = test_dataset_label['scene_label'].tolist()

        return (train_sample, train_label, test_sample, test_label)


    elif args.dataset_name == 'DCASE2021-Audio' or args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'DCASE2021-Audio-Visual':
        train_dataset = train_dataset_pd[['filename_audio']]
        train_dataset_label = train_dataset_pd[['scene_label']]
        test_dataset = val_dataset_pd[['filename_audio']]
        test_dataset_label = val_dataset_pd[['scene_label']]

        train_sample_audio = train_dataset['filename_audio'].tolist()
        train_label = train_dataset_label['scene_label'].tolist()
        test_sample_audio = test_dataset['filename_audio'].tolist()
        test_label = test_dataset_label['scene_label'].tolist()


        train_dataset = train_dataset_pd[['filename_video']]
        train_sample_video = train_dataset['filename_video'].tolist()

        test_dataset = val_dataset_pd[['filename_video']]
        test_sample_video = test_dataset['filename_video'].tolist()
        # print('train_sample_video:', train_sample_video)
        # print('test_sample_video:', test_sample_video)


        train_sample_image = []
        test_sample_image = []

        for video_path in train_sample_video:
            image_path = get_image_path(video_path, data_dir)
            train_sample_image.append(image_path)

        for video_path in test_sample_video:
            image_path = get_image_path(video_path, data_dir)
            test_sample_image.append(image_path)

        # print('train_label:', train_label)
        # print('test_label:', test_label)


    return (train_sample_audio, train_label, test_sample_audio, test_label, train_sample_image, test_sample_image)


def get_image_path(video_path, image_dir=''):
    image_dir = os.path.join(image_dir, 'image')
    image_name = video_path.split('/')[1].split('.')[0]
    image_ext = '.jpg'
    image_path = image_name + image_ext
    image_path = os.path.join(image_dir, image_path)
    return image_path


def video_process(video_path):
    from decord import VideoReader
    from decord import cpu, gpu
    from PIL import Image
    import numpy as np


    image_path = get_image_path(video_path)

    video_dir = '/data/lgzhou/dataset/DCASE2021'
    video_path = os.path.join(video_dir, video_path)

    vr = VideoReader(video_path, ctx=cpu(0))

    frame = vr[150].asnumpy()

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image.save(image_path)


def single_category_construction(data_dir, train_ratio=0.7, class_id=0):

    class_id_dict = get_dir_label(data_dir)

    data_sample = cate_data_sample(class_id_dict, train_ratio, class_id)

    return data_sample


# if __name__=='__main__':
#
#     data_dir = '/data/lgzhou/avasr/avasrd/vision/'
#     data_construction(data_dir)