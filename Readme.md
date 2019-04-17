# Segmentation of Nuclei Cells
### Instructions on how to run this:
Upon cloning this repo, first you need to run the shell script get_data.sh with the command  
bash get_data.sh  
This will download the data for you and place it into folders. Then it will automatically run  
python scripts to get the data in folders and get them prepared to work with the model and the  
dataloader written there.  
Next run the python script augment_data.py to augment the data, if you want augmentations.  
After that, you should run get_data_ids.py to get the ids for the data ids in a csv file.  
The data are ready for the next operations, such as optimization and predictions.
