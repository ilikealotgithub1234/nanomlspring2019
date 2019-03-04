# conda install pillow
from PIL import Image, ImageDraw

image = Image.open("duckhunt.jpg")

print(image.format) # Output: JPEG
print(image.mode) # Output: RGB
print(image.palette) # Output: None

image.show()

draw = ImageDraw.Draw(image)
draw.line((0, 0) + image.size, fill=128)
draw.line((0, image.size[1], image.size[0], 0), fill=128)
del draw

# write to stdout
image.save("out/cross-dog.png", format="PNG")
