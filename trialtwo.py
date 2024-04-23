import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

trialtwo = tk.Tk()
trialtwo.geometry("532x622")
trialtwo.title = ("Movie Poster Generator")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master = trialtwo, height = 40, width = 512, text_color = "black", fg_color = "white")
prompt.place(x = 10, y = 10)

lmain = ctk.CTkLabel(master = trialtwo, height = 512, width = 512)
lmain.place(x = 10, y = 110)


modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision = "fp16", torch_dtype = torch.float16, use_auth_token = auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    img.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(height = 40, width = 120, text_font = ("Arial", 20), text_color = "white", fg_color = "green", command = generate)
trigger.configure(text = "Generate!")
trigger.place(x = 206, y = 60)

trialtwo.mainloop()
