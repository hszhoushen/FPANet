import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class draw_confusion_matrix():
    def __init__(self, data_name):
        self.data_name = data_name

        self.esc10_path = 'ESC10_4_20_confusion.npy'
        self.sun_rgbd_path = 'SUN_RGBD_1_20_confusion.npy'

        self.esc10_labels = ['Dog', 'Rooster', 'Rain', 'Sea Waves', 'Crackling Fire',
                  'Crying Baby', 'Sneezing', 'Clock Tick', 'Helicopter', 'Chainsaw']
        self.sun_rgbd_labels = ['bathroom', 'bedroom', 'classroom', 'computer_room', 'conference_room',
                             'corridor', 'dining_area', 'dining_room', 'discussion_area', 'furniture_store',
                             'home_office', 'kitchen', 'lab', 'lecture_theatre', 'library',
                             'living_room', 'office', 'rest_space', 'study_space']

        if self.data_name == 'SUN_RGBD':
            self.fig_path = 'sun_rgbd_conf_mat.jpg'
        elif self.data_name == 'ESC10':
            self.fig_path = 'esc10_conf_mat.jpg'

    def save_fig(self, heatmap):
        heatmap = heatmap.get_figure()
        heatmap.savefig(self.fig_path, dpi=300, bbox_inches='tight')

    def normalize(self, conf_mat):
        rows, columns = conf_mat.shape
        normalized = np.zeros(conf_mat.shape)

        for i in range(rows):
            total_row = np.sum(conf_mat[i, :])
            print('total_row:', total_row, type(total_row))

            for j in range(columns):
                print('conf_mat[i, j]:', conf_mat[i, j], type(conf_mat[i, j]))
                normalized[i, j] = format(100 * conf_mat[i, j] / total_row, '.1f')
                print('normalized[i,j]:', normalized[i, j])

        return normalized
    
    def draw_conf_mat(self):

        # f, ax=plt.subplots()
        fig = plt.figure(figsize=(18, 18))
        
        # esc10
        if self.data_name == 'ESC10':
            labels = self.esc10_labels
            conf_mat_path = self.esc10_path

        elif self.data_name == 'SUN_RGBD':
            labels = self.sun_rgbd_labels
            conf_mat_path = self.sun_rgbd_path

        conf_mat_lst = np.load(conf_mat_path)
        print(conf_mat_lst.shape)

        conf_mat_epoch60 = conf_mat_lst[59, :,:]
        
        conf_mat = np.array(conf_mat_epoch60)

        print('conf_mat:', conf_mat)
        conf_mat = self.normalize(conf_mat)
        print('conf_mat:', conf_mat)

        sns.set(font_scale=1.)
        plt.subplots(figsize=(16,16)) # 设置画面大小
        

        
        if self.data_name == 'ESC10':
            conf_mat_heatmap = sns.heatmap(conf_mat,
                                           annot=True,
                                           vmax=100,
                                           vmin=0,
                                           xticklabels=labels,
                                           yticklabels=labels,
                                           annot_kws={"fontsize": 18},
                                           cbar_kws={"orientation": "vertical"},
                                           cmap="YlGnBu")  # 画热力图

            plt.title('ESC10 Confusion Matrix', fontsize=24)
            plt.xlabel('Predicted label',fontsize=20)
            plt.ylabel('True label',fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xticks(rotation=45)
            plt.yticks(rotation=360)

            # ------------设置颜色条刻度字体的大小-------------------------#
            # cb = conf_mat_heatmap.figure.colorbar(conf_mat_heatmap.collections[0])    # 显示colorbar
            # cb.ax.tick_params(labelsize=28)             # 设置colorbar刻度字体大小。


            # cbar = conf_mat_heatmap.collections[0].colorbar
            # cbar.conf_mat_heatmap.tick_params(labelsize=20)

            plt.show()

        elif self.data_name == 'SUN_RGBD':
            conf_mat_heatmap = sns.heatmap(conf_mat,
                                           annot=True,
                                           vmax=100,
                                           vmin=0,
                                           xticklabels=labels,
                                           yticklabels=labels,
                                           annot_kws={"fontsize": 16},
                                           # cbar_kws={"orientation": "horizontal"},
                                           cbar_kws={"orientation": "vertical"},
                                           cmap="YlGnBu")  # 画热力图

            plt.title('SUN RGBD Confusion Matrix', fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Predicted label',fontsize=20)
            plt.ylabel('True label',fontsize=20)
            plt.xticks(rotation=75)
            plt.show()


        self.save_fig(conf_mat_heatmap)

#
draw_conf_mat = draw_confusion_matrix(data_name='SUN_RGBD')
draw_conf_mat.draw_conf_mat()

draw_conf_mat = draw_confusion_matrix(data_name='ESC10')
draw_conf_mat.draw_conf_mat()