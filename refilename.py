# -*- coding: utf-8 -*-
"""
File function: Batch modify file name
@author: sunlichun
Created on November 10, 2021
"""

import os

# Path of the file to be modified
inputdir = r"D:/A_Postgraduate/Original/true_catalog1/"

def re_filename():
  
    if not os.path.exists(inputdir):
        print("file's path not exist"+'\n'+"please enter a correct path")
    if os.path.exists(inputdir):
        file_list = os.listdir(inputdir)
        
        for i in range(0, len(file_list)):
            path = os.path.join(inputdir, file_list[i])
            filename = inputdir + os.path.basename(path)
            # string format
            new_filename = 'ldev_2dimage_bin'+str(os.path.basename(path)).replace('truth_cat4ldev2dimage_bin','') 
            os.renames(filename, inputdir+format(new_filename))
            
            print("{} file already changed to {}".format(os.path.basename(path),new_filename))

re_filename()
