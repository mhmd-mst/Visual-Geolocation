from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from tkinter import *
from pathlib import Path
import shutil
from tkinter import ttk

from pathlib import Path
import shutil
import os
# from PIL import Image
from PIL import ImageTk, Image


class App(Tk):
    def __init__(self, files, driver):
        super().__init__()

        self.name = None
        self.file = None

        self.names = [n.split('@')[5] + ', ' + n.split('@')[6] for n in files]
        self.files = files
        self.src_pth = '1KDataset/night/TEST'
        self.dst_pth = '1KDataset/night/OK'

        # configure the root window
        self.title('My Awesome App')
        #         self.geometry('300x50')
        self.attributes('-zoomed', True)

        # frame
        self.frame = ttk.Frame(self)
        self.frame.pack()
        self.frame.place(anchor='center', relx=0.5, rely=0.5)

        # label
        self.label = ttk.Label(self.frame)
        self.label.pack(side="bottom", fill="both", expand="yes")

        # button frame
        self.frame_btn = ttk.Frame(self)
        self.frame_btn.pack(side=TOP, anchor="e", padx=8, pady=8)

        # button_next
        self.button_next = ttk.Button(self.frame_btn, text='\nNext\n')
        self.button_next['command'] = self.button_next_clicked
        self.button_next.pack()

        # button_ok
        self.button_ok = ttk.Button(self.frame_btn, text='\nOk\n')
        self.button_ok['command'] = self.button_ok_clicked
        self.button_ok.pack(pady=8)

        # Selenium
        self.driver = driver
        self.driver.get("https://www.instantstreetview.com/")
        self.search_field = self.driver.find_element_by_id("search")

        self.search_field.clear()

    def draw_img(self, file, name):
        with Image.open(os.path.join(self.src_pth, file)) as img:
            width, height = img.size
            self.frame.config(width=width, height=height)
            img = ImageTk.PhotoImage(img)

            # Create a Label Widget to display the text or Image
            self.label.configure(image=img)
            self.label.image = img

        self.search_field.clear()
        self.search_field.send_keys(name)

    def button_next_clicked(self):
        self.name = self.names.pop()
        print(self.name)
        self.file = self.files.pop()
        self.title(self.name)

        self.draw_img(self.file, self.name)

    def button_ok_clicked(self):
        # TODO: OK BUTTON PRESSED

        url = self.driver.current_url
        print(url)
        # @40.409192,49.866289,xxxxxxx
        lat = url.split(',')[0].split('@')[1]
        lon = url.split(',')[1]
        lat_lon = f'@{lat}@{lon}'
        org_lat_lon = f"@{self.file.split('@')[5]}@{self.file.split('@')[6]}"
        print(lat_lon)
        print(org_lat_lon)

        src_file = os.path.join(self.src_pth, self.file)
        #         self.file.replace(org_lat_lon, lat_lon)
        dst_file = os.path.join(self.dst_pth, self.file)

        Path(self.dst_pth).mkdir(exist_ok=True)
        shutil.move(src_file, dst_file)
        os.rename(dst_file, dst_file.replace(org_lat_lon, lat_lon))

        self.name = self.names.pop()
        self.file = self.files.pop()
        self.title(self.name)

        self.draw_img(self.file, self.name)


files = os.listdir('./1KDataset/night/TEST')
files_test = files[:3]

driver = webdriver.Chrome(service=Service(f"{os.getcwd()}/1KDataset/chromedriver100"))

app = App(files_test, driver)
app.mainloop()
