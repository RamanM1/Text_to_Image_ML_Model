import tkinter as tk
from PIL import ImageTk
import customtkinter as ctk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

# Initialize the main window
main_window = tk.Tk()
main_window.geometry("532x632")
main_window.title("Image Generator")
ctk.set_appearance_mode("dark")

# Define elements
input_field = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white")
input_field.place(x=10, y=10)

image_display = ctk.CTkLabel(height=512, width=512)
image_display.place(x=10, y=110)

# Load the Stable Diffusion model
model_identifier = "CompVis/stable-diffusion-v1-4"
processing_device = "cuda"
diffusion_pipeline = StableDiffusionPipeline.from_pretrained(model_identifier, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
diffusion_pipeline.to(processing_device)

# Function to generate images
def create_image():
    with autocast(processing_device):
        generated_image = diffusion_pipeline(input_field.get(), guidance_scale=8.5)["sample"][0]
    
    generated_image.save('output_image.png')
    display_image = ImageTk.PhotoImage(generated_image)
    image_display.configure(image=display_image)
    image_display.image = display_image  # Keep a reference

generate_button = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=create_image)
generate_button.configure(text="Create Image")
generate_button.place(x=206, y=60)

# Start the application loop
main_window.mainloop()
