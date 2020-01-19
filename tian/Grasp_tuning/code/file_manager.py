# https://realpython.com/working-with-files-in-python/#traversing-directories-and-processing-files
import os
import shutil

count = 0
file_id = 0
idmapping_file = open(r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\IdMapping.txt', 'w')

for dirpath, dirnames, files in os.walk(r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        char_file_name = list(file_name)
        # establish an ascending index to the file id
        if char_file_name[-1] == 'g':
            data_id = ''.join(char_file_name[3:7])
            map = str(file_id) + ',' + data_id + '\n'
            print(map)
            idmapping_file.write(map)
            file_id += 1
        # delete visualization images
        # if char_file_name[-1] == 't' or char_file_name[-5] == 'r':
        #     os.remove(os.path.join(dirpath, file_name))
        #     print(file_name, 'deleted')

        # copy original object image into the directory
        # cornell_dataset_dir = r'D:\PhD\NNTraining\data\rawDataSet'
        # destination_dir = r'C:\Users\75077\OneDrive\2019\grasp evaluation paper\dataset'
        # if ''.join(char_file_name[-5]) == 'r':
        #     print(file_name)
        #     file_id = ''.join(char_file_name[3:7])
        #     # cneg_file_name = 'pcd' + file_id + 'cneg.txt'
        #     # cpos_file_name = 'pcd' + file_id + 'cpos.txt'
        #     # cneg_file = os.path.join(cornell_dataset_dir, cneg_file_name)
        #     # cpos_file = os.path.join(cornell_dataset_dir, cpos_file_name)
        #     # print(cneg_file)
        #     # print(cpos_file)
        #     # shutil.copy(cpos_file, destination_dir)
        #     # shutil.copy(cneg_file, destination_dir)
        #     pcd_file_name = 'pcd' + file_id + '.txt'
        #     pcd_file = os.path.join(cornell_dataset_dir, pcd_file_name)
        #     # print(pcd_file)
        #     shutil.copy(pcd_file, destination_dir)

        # rename files
        # if ''.join(char_file_name[-5]) == 'r':
        #     file_id += 1
        #     rgb_o = os.path.join(dirpath, file_name)
        #     rgb_n_name = 'pcd0' + str(file_id) + 'r.png'
        #     rgb_n = os.path.join(dirpath, rgb_n_name)
        #     mask_o_name = 'pcd' + ''.join(char_file_name[3:7]) + 'mask.png'
        #     mask_n_name = 'pcd0' + str(file_id) + 'mask.png'
        #     mask_o = os.path.join(dirpath, mask_o_name)
        #     mask_n = os.path.join(dirpath, mask_n_name)
        #
        #     cneg_o_name = 'pcd' + ''.join(char_file_name[3:7]) + 'cneg.txt'
        #     cneg_n_name = 'pcd0' + str(file_id) + 'cneg.txt'
        #     cneg_o = os.path.join(dirpath, cneg_o_name)
        #     cneg_n = os.path.join(dirpath, cneg_n_name)
        #
        #     cpos_o_name = 'pcd' + ''.join(char_file_name[3:7]) + 'cpos.txt'
        #     cpos_n_name = 'pcd0' + str(file_id) + 'cpos.txt'
        #     cpos_o = os.path.join(dirpath, cpos_o_name)
        #     cpos_n = os.path.join(dirpath, cpos_n_name)
        #
        #     pcd_o_name = 'pcd' + ''.join(char_file_name[3:7]) + '.txt'
        #     pcd_n_name = 'pcd0' + str(file_id) + '.txt'
        #     pcd_o = os.path.join(dirpath, pcd_o_name)
        #     pcd_n = os.path.join(dirpath, pcd_n_name)
        #     # print(cneg_o_name)
        #     # print(cneg_n_name)
        #     # print(cpos_o_name)
        #     # print(cpos_n_name)
        #     # print(pcd_o_name)
        #     # print(pcd_n_name)
        #     os.rename(mask_o, mask_n)
        #     os.rename(rgb_o, rgb_n)
        #     os.rename(cneg_o, cneg_n)
        #     os.rename(cpos_o, cpos_n)
        #     os.rename(pcd_o, pcd_n)
idmapping_file.close()
print('job done!')
