lat = 37.0
lon = -122.0

lat_list = [82, 80, 78, 76, 75, 73, 70]
lon_list = [53, 50, 48, 46, 43, 40, 37, 35]
l = []
with open('./1KDataset/coordinates.txt', 'w') as f:
    for i in range(len(lat_list)):
        if i == len(lat_list)-1:
            break
        
        for j in range(len(lon_list)):
            if j == len(lon_list)-1:
                break
            # print(lat_list[i],lat_list[i+1], lon_list[j],lon_list[j+1])
            text = ''
            text+= f'--min_lat {lat_list[i+1]/100+lat} '
            text+= f'--max_lat {lat_list[i]/100+lat} '

            text+= f'--min_lon -{lon_list[j]/100-lon} '
            text+= f'--max_lon -{lon_list[j+1]/100-lon} '
            l.append(text)   
            f.write(text)
            f.write('\n')
            print(text)


