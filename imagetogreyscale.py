from PIL import Image
import pandas as pd
import numpy as np

def data(str):
    name=str+'.png'
    img=Image.open(name).convert('L')
    width,height=img.size
    data=list(img.getdata())
    df=pd.DataFrame(data)
    df=255-df
    sample=(df-df.min())/(df.max()-df.min())
    TEST=sample.to_numpy()
    return TEST


