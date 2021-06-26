import os
import sys
import signal
from PIL import Image
import numpy as np
import paddleocr
from paddleocr import PaddleOCR, draw_ocr 
import streamlit as st
from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)
src_dict={"English":"en","Hindi": "hi" ,"Telugu":"te", "Tamil": "ta"}

def save_uploadedfile(uploadedfile):
     with open(os.path.join(uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return "Saved File:{} ".format(uploadedfile.name)

@st.cache
def load_model(src):
    x=src_dict[src]
    return PaddleOCR(lang=x) 
    

def main():       
    image = Image.open('sai_app_header.png')
    st.set_option('deprecation.showfileUploaderEncoding',False)
    st.image(image,use_column_width=True)
    st.subheader('Select the language from sidebar') 
    st.sidebar.subheader('OCR Application \n Language Selection Menu')   
    src = st.sidebar.selectbox("",['English','Hindi','Telugu','Tamil'])
    
        
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg','JPG'])
    if image_file is not None:         
        st.subheader('Image you Uploaded...')
        st.image(image_file)
        save_uploadedfile(image_file)
        if st.button("Convert"): 
            
            with st.spinner('Extracting Text from given Image'):                                 

                ocr = load_model(src)
                result = ocr.ocr(image_file.name)
                image = Image.open(image_file).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                im_show = draw_ocr(image, boxes, txts, scores, font_path='arial-unicode-ms.ttf')
                im_show = Image.fromarray(im_show)
                st.subheader('Extracted text is ...')
                    
            st.image(im_show)
            st.write(txts)
                
    else:
        st.subheader('Image not found! Please Upload an Image.')

st.markdown("OCR Application built by -- Sai Tej")
if __name__ == '__main__':   
   
    main()
