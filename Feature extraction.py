import radiomics
import pandas as pd
import os
import SimpleITK as sitk

def traversalDir_FirstDir(path):
    list1 = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            list1.append(m)
    return(list1)

para_name = r'C:\Users\CT_Normalize.yaml'
pathPatient = r'C:\Users\*'     #the pathway of image
resultpath = r'C:\Users\*'      #the pathway of results

index_list = ['Plaque','PCAT']
for i in range(len(index_list)):
    index = index_list[i]
    df = pd.DataFrame()
    a =[]
    b = []
    alist=traversalDir_FirstDir(pathPatient + '\\' + 'image')
    for afile in alist:
        pathImage = traversalDir_FirstDir(afile)
        for Image in pathImage:
            m = Image.split("\\")[-2]
            n = Image.split("\\")[-1]
            a.append(m)
            b.append(n)
            print(Image)
            mask = pathPatient+'\\'+ 'roi'+'\\' + Image.split("\\")[-2]+'\\'+index+'\\'+Image.split("\\")[-1]
            print(mask)
            extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(para_name)
            image = sitk.ReadImage(Image)
            mask = sitk.ReadImage(mask)
            featureVector = extractor.execute(Image, mask,label = 1)
            df_add = pd.DataFrame.from_dict(featureVector.values()).T
            df_add.columns = featureVector.keys()
            df = pd.concat([df,df_add])
    a = pd.DataFrame(a)
    b = pd.DataFrame(b)
    df1 = df.reset_index()
    df1=df1.drop('index', axis=1) 
    result = pd.concat([b,a,df1],axis=1)
    result.to_excel(resultpath+'\\'+index+'.xlsx')