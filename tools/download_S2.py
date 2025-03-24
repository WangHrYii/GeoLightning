############################################################################
import os
import sys
import pandas as pd
import requests
import json
import subprocess
import datetime
from tqdm import tqdm
import datetime
import time

def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')

# 获取Access token
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]

# 检查文件是否已下载的函数  
def is_file_downloaded(file_id):  
    file_name = f"{output_dir}{file_id}.txt"  
    return os.path.exists(file_name)  

# 中断下载后继续下载 
# 存储未下载文件id的列表
def download_files(data_id_list,data_name_list,email,password,output_dir):  
    downloaded_false = []  
    wget_str=[]
    part1='''wget --header "Authorization: Bearer '''
    part2='''" "http://catalogue.dataspace.copernicus.eu/odata/v1/Products('''
    part3=''')/$value" -O '''
    for i in tqdm(range(len(data_id_list))):
        file_id = data_name_list[i]
        file_name = f"{output_dir}{file_id}.txt"
        if not is_file_downloaded(file_id):
            access_token = get_access_token(email, password)
            command=part1+access_token+part2+data_id_list[i]+part3+output_dir+data_name_list[i]+'.zip'
            # print(command)
            wget_str.append(command)
            try:
                print('[',datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S'),'] '+'开始下载: '+data_name_list[i])
                subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                # subprocess.run(command, shell=True, check=True)
                with open(file_name, "w") as file:  
                    file.write('file downloaded successfully.')
                print('[',datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S'),'] '+'下载成功: '+data_name_list[i])
            except Exception as e:
                print('[',datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S'),'] '+'下载失败: '+data_name_list[i])
                downloaded_false.append(file_id)


    if downloaded_false:
        print("Files that have not been downloaded：\n")    
        print(downloaded_false)
        file_false = f"{output_dir}downloadfalse.txt"
        with open(file_false,"w") as f:
            f.truncate(0)
            f.write("Files that have not been downloaded：\n")
            f.write(downloaded_false)   
        return False
    else:
        print("All files downloaded successfully.")
        return True

if __name__ == "__main__":
    #基础设置
    ############################################################################
    # 1 起始日期(包含起始)
    startDate='2024-02-01'
    endDate  ='2024-11-01'

    # 2 所需卫星数据 SENTINEL-5P, SENTINEL-1, SENTINEL-2
    satellite='SENTINEL-2'
    ## band 5
    # productType = 'L1B_RA_BD5'
    # ## OFFL,RPRO
    # Timeliness = 'RPRO'

    # 3 检索时文件名需包括的字符串  对于哨兵2 可以用来筛选区块或者产品等级
    contains_str='L2A'

    # 4 检索区域 可在该网站绘制geojson文件 https://geojson.io/#map=5.12/34.13/122.8
    roi_geojson='/mnt/data_1/Industry/Truth/geojsons/bbox_1030.geojson'
    # roi_geojson= None

    # 5 最低下载文件大小 单位MB
    min_file_size= 100

    # 6 数据保存路径
    output_dir= os.path.join('/mnt/data_1/Industry/Truth/S2/',roi_geojson.split('/')[-1].split('.')[0]+'/')
    check_path(output_dir)

    # 7 新版哥白尼数据中心账号密码 即这个网站的账号密码 https://dataspace.copernicus.eu/
    email="ghming048@gmail.com"
    password="Fighti1205&@"

    # 8 属性设置
    cloudCoverPercentage = "30.00"

    ########################################################################
    # 生成检索链接
    #基础前缀
    base_prefix="https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
    #检索条件 记得检索条件之间要加 and 
    str_in_name="contains(Name,'"+contains_str+"')"
    collection="Collection/Name eq '"+satellite+"'"
    size = "ContentLength ge "+str(min_file_size*1024*1024)

    if roi_geojson==None:
        roi=None    
    else:
        # 从geojson文件中读取坐标，以字符串'x y,...,x y'组织。
        with open(roi_geojson, 'r') as f:
            data = f.read()
        geojson_data = json.loads(data)
        coordinates=geojson_data['features'][0]['geometry']['coordinates'][0]
        coordinates_str=''
        for i in range(len(coordinates)):
            coordinates_str=coordinates_str+str(coordinates[i][0])+' '+str(coordinates[i][1])+', '
        coordinates_str=coordinates_str[:-2]
        roi="OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("+coordinates_str+"))') "

    time_range="ContentDate/Start ge "+startDate+"T00:00:00.000Z and ContentDate/Start le "+endDate+"T00:00:00.000Z"

    #检索属性
    search_lim="&$top=1000"
    expand_assets="&$expand=Assets"
    cloudCover = "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le "+cloudCoverPercentage+")"
    

    ##OrderBy(默认升序)
    #OrderBy ="&$orderby=ContentDate/Start desc"

    #最终的检索链接 
    #判断卫星产品类型
    if satellite=='SENTINEL-1':
        if roi==None:
            request_url=base_prefix+str_in_name+" and "+collection+" and "+size+" and "+time_range+search_lim+expand_assets
        else:
            request_url=base_prefix+str_in_name+" and "+collection+" and "+size+" and "+roi+" and "+time_range+search_lim+expand_assets
    elif satellite=='SENTINEL-2':
        if roi==None:
            request_url=base_prefix+str_in_name+" and "+collection+" and "+size+" and "+time_range+search_lim+expand_assets
        else:
            request_url=base_prefix+str_in_name+" and "+collection+" and "+size+" and "+roi+" and "+cloudCover+" and "+time_range+search_lim+expand_assets
    # elif satellite=='SENTINEL-5P':
    #     productTypeStr = "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '"+productType+"')"
    #     if roi==None:
    #         request_url=base_prefix+collection+" and "+productTypeStr+" and "+size+" and "+time_range+search_lim+expand_assets
    #     else:
    #         request_url=base_prefix+collection+" and "+productTypeStr+" and "+size+" and "+roi+" and "+time_range+search_lim+expand_assets

    print("检索条件：{}".format(request_url))
    #开始检索
    JSON = requests.get(request_url).json()
    df = pd.DataFrame.from_dict(JSON['value'])
    print("查询数据条数：{}".format(len(df)))
    #原始数据id列表
    data_id_list=df.Id
    data_name_list=df.Name
    data_s3_path = df.S3Path

    # 筛选RPRO数据
    # if Timeliness:
    #     print("正在筛选{}...".format(Timeliness))
    #     data_id_list_selected = []
    #     data_name_list_selected = []   
    #     for i in range(len(data_name_list)):
    #         if Timeliness in data_s3_path[i]:
    #             data_id_list_selected.append(data_id_list[i])
    #             data_name_list_selected.append(data_name_list[i])
    #     print("筛选后数据条数：{}".format(len(data_id_list_selected)))
    #     data_id_list = data_id_list_selected
    #     data_name_list = data_name_list_selected

    if len(df)==0:
        print('未查询到数据')
        sys.exit()
    try:
        if len(df)>999:
            print("查询数据条数：{}".format(len(df)))
            raise Exception("检索数量超出1000，请重新设置检索条件")  
        print("查询数据条数：{}".format(len(df)))

    except Exception as e:
        print("Program stopped:", e)


    # 快视图下载链接 
    # quickview_url=[file[0]['DownloadLink'] for file in df.Assets]
    # quickview_url_txt = open(output_dir+"quickview_url.txt", "w")
    # for id in range(len(df.Name)):
    #     quickview_url_txt.write(df.Name[id]+" "+quickview_url[id]+"\n")
    # quickview_url_txt.close()
    
    # 中断后继续下载
    is_file_all_downloaded=False
    while True:
        try:
            is_file_all_downloaded = download_files(data_id_list,data_name_list,email,password,output_dir)
            if not is_file_all_downloaded:

                print("下载未完成，将在几分钟后重试...")
                time.sleep(300)  # 等待300秒（5分钟）
            else:
                break  # 如果下载成功完成，退出循环
        except Exception as e:
            print(f"遇到错误: {e}")
            break           
    