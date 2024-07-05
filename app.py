import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['brand'].unique())

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# gpu_type of laptop
gpu_type = st.selectbox('GPU Type',df['gpu_type'].unique())

# Vam
Vram = st.selectbox('VRAM(in GB)',[0,2,4,6,8,12,16])

#cpu
cpu = st.selectbox('CPU',df['cpu'].unique())

# ssd
ssd = st.selectbox('SSD',['No','Yes'])

#storage
storage = st.selectbox('Storage(in GB)',[0,128,256,512,1024,2048])

#OS
os = st.selectbox('OS',df['OS'].unique())


if st.button('Predict Price'):
    # pass
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ssd == 'Yes':
        ssd = 1
    else:
        ssd = 0

    if gpu_type != 'dedicated':
        Vram = 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, gpu_type, Vram, ram, ssd, storage, touchscreen, os, ppi, cpu], dtype=object)

    # query.reshape(1, -1)

    query = query.reshape(1,10)
    st.title(np.exp(pipe.predict(query)[0]))

# brand	gpu_type	Vram	ram_memory	primary_storage_type	primary_storage_capacity	is_touch_screen	OS	ppi	cpu
