#### Files specific to use case:
**.h5 file** - Face recognition model trained from colab  
**.csv file** - Names for each face  
  
*Delete the above files and replace the following:*  
- *Trained face-recognition model (h5 file)*  
- *Face names (csv file)*
- *File names in references of CSV and H5 files in **app-rec-facenet.py***  


#### Colab Notebooks:  
**embedding-trainer.ipynb** - Generate trained face-recognition model (.h5 file) from cropped faces  
**vid2vid.ipynb** - Testing script - Input: Any video -> Output: Annotated video (squares on detected faces)  
*(Training dataset and input video to be input from Google Drive)*
