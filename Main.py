# My own files
import LoadModel
import Recognize
# Python libraries
from tkinter import *
from PIL import Image, ImageDraw
import os.path
import datetime

# Load the pretrained model
model = LoadModel.loadmodel('EMNISTModel.pt', 'cpu')


def read():
    """Save lats image and recognize the image"""
    global other  # top 5 perdiction
    path = 'test.jpg'
    image.save(path)
    # recognize the character in the image
    top1, top5 = Recognize.recognize(path, model)
    other = []
    mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    text_1.insert(END, mapping[top1])
    for val in top5[1:]:
        other.append(mapping[val])
    button_3['text'] = mapping[top5[1]]
    button_4['text'] = mapping[top5[2]]
    button_5['text'] = mapping[top5[3]]
    button_6['text'] = mapping[top5[4]]
    w.delete('all')  # clear canvas
    draw.rectangle((0, 0, 400, 400), fill='black')  # clear the image


def paint(event):
    """Painting"""
    x1, y1 = (event.x - 15), (event.y - 15)
    x2, y2 = (event.x + 15), (event.y + 15)
    w.create_oval(x1, y1, x2, y2, fill='black')  # draw in the canvas
    draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')  # draw the image


def here():
    """clear the old content and display the new instructions"""
    # clear old
    label_1.grid_forget()
    for label in master.grid_slaves():
        if int(label.grid_info()["row"]) <= 1 and int(label.grid_info()["column"]) <= 2:
            label.grid_forget()
    # display new
    label_2 = Label(master, width=25, height=2, text='See the output below!\nEdit it anyway you want!',
                    font=("Arial Bold", 25))
    label_2.grid(row=0, column=0, rowspan=2, columnspan=3)


def togo():
    """clear the old content and display the new instructions for path"""
    # clear old
    label_1.grid_forget()
    for label in master.grid_slaves():
        if int(label.grid_info()["row"]) <= 1 and int(label.grid_info()["column"]) <= 2:
            label.grid_forget()
    # display new
    label_3 = Label(master, width=25, height=1, text='Enter the full path below!', font=("Arial Bold", 25))
    label_3.grid(row=0, column=0, columnspan=3)
    # entry to take user input path
    entry = Entry(master, textvariable=var)
    entry.grid(row=1, column=0, columnspan=2)
    button_8 = Button(text='Save', height=1, width=10, command=save_text)
    button_8.grid(row=1, column=2)


def save_text():
    """Save the text to loacl .txt file"""
    save_path = var.get()  # the path user input
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # file name
    completename = os.path.join(save_path, file_name + ".txt")  # join
    try:
        f = open(completename, "w")  # create the file
    except FileNotFoundError:
        path_error()
        return
    tofile = text_1.get("1.0", END)  # get the content to be saved
    f.write(tofile)  # write to file
    f.close()


def path_error():
    """Display the path error message"""
    path_err = Toplevel()
    path_err.title('Path Error')
    label_8 = Label(path_err, width=30, height=5,
                    text='Please check the path again!\nDo not include the file name.\nDetails in \'Help\'',
                    font=("Arial Bold", 20))
    label_8.grid(row=0, column=0)


def change(c):
    """Change which recognized result to display"""
    text_1.delete('end-2c')
    text_1.insert(END, other[c])


def add(c):
    """insert a character"""
    text_1.insert(END, c)


def caps():
    """Change the last character between upper and lower case"""
    c = text_1.get('end-2c')
    if c.isdigit():
        error()  # error trigger
    if c.isupper():
        c = c.lower()
    else:
        c = c.upper()
    text_1.delete('end-2c')
    text_1.insert(END, c)


def delete():
    """Delete the character on demand"""
    text_1.delete('end-2c')


def error():
    """Display the error message"""
    err = Toplevel()
    err.title('Error')
    label_7 = Label(err, width=30, height=5, text='Can\'t do that on numbers \nDetails in \'Help\'',
                    font=("Arial Bold", 20))
    label_7.grid(row=0, column=0)


def tip():
    """Help window"""
    top = Toplevel()
    top.title("Help")
    tips = '1. You can write numbers and characters from english alphabet in the gray area.\n' \
           '2. You can only write one number or character at a time.\n' \
           '3. Press \'read\' button to reconize what you wrote.\n' \
           '4. The result will be presented at the bottom greenish area.\n' \
           '5. After you finished writting, choose a way you want to deal with the input at the top.\n' \
           '6. If you choose to go, do not include the file name, only input the path to the folder.\n' \
           '7. On the left, you can see some other possibilities given by the model\n' \
           '8. \'Upper/Lower\' button is used to switch between upper case and lower case\n' \

    label_6 = Label(top, width=90, height=15, text=tips,
                    font=("Arial Bold", 20), justify=LEFT)
    label_6.grid(row=0, column=0)


# Draw Window Parameters
width = 400
height = 400

# Main Window
master = Tk()
master.title("Writting Board")
master.resizable(width=FALSE, height=FALSE)
master.geometry('700x620')

var = StringVar()

# top instruction
label_1 = Label(master, width=30, height=2, text='Choose a way on left\nwhen you finish writting',
                font=("Arial Bold", 25))
label_1.grid(row=0, column=0, rowspan=2, columnspan=3)
# place where recognized result displaied
text_1 = Text(master, height=7, width=40, bg='azure', font=("Arial Bold", 15))
text_1.grid(row=10, column=0, rowspan=2, columnspan=3)
# For Here button
button_1 = Button(text='For Here', height=1, width=10, command=here)
button_1.grid(row=0, column=3, columnspan=2)
# To Go button
button_2 = Button(text='To Go', height=1, width=10, command=togo)
button_2.grid(row=1, column=3, columnspan=2)
# Second top prediction
button_3 = Button(text='', height=2, width=10, command=lambda j=0: change(j), highlightbackground='lightblue2')
button_3.grid(row=4, column=3)
# Third top prediction
button_4 = Button(text='', height=2, width=10, command=lambda j=1: change(j), highlightbackground='lightblue2')
button_4.grid(row=5, column=3)
# Forth top prediction
button_5 = Button(text='', height=2, width=10, command=lambda j=2: change(j), highlightbackground='lightblue2')
button_5.grid(row=4, column=4)
# Fifth top prediction
button_6 = Button(text='', height=2, width=10, command=lambda j=3: change(j), highlightbackground='lightblue2')
button_6.grid(row=5, column=4)
# display instructions
label_9 = Label(master, width=30, height=2, text='Function Keys', font=("Arial Bold", 15))
label_9.grid(row=6, column=3, columnspan=2)
# upper and lower switch button
button_7 = Button(text='Upper/Lower', height=2, width=10, command=caps, highlightbackground='lightblue2')
button_7.grid(row=7, column=3)
# Space button
button_10 = Button(text='Space', height=2, width=10, command=lambda j=' ': add(j), highlightbackground='lightblue2')
button_10.grid(row=7, column=4)
# comma button
button_11 = Button(text=',', height=2, width=10, command=lambda j=',': add(j), highlightbackground='lightblue2')
button_11.grid(row=8, column=3)
# period button
button_12 = Button(text='.', height=2, width=10, command=lambda j='.': add(j), highlightbackground='lightblue2')
button_12.grid(row=8, column=4)
# display other predictions
label_5 = Label(master, width=30, height=2, text='Do you mean', font=("Arial Bold", 15))
label_5.grid(row=3, column=3, columnspan=2)
# Canvas
w = Canvas(master, width=width, height=height, bg='white smoke')
w.bind("<B1-Motion>", paint)
w.grid(row=3, column=0, rowspan=7, columnspan=3)
# The image
image = Image.new('RGB', (width, height), (0, 0, 0))
draw = ImageDraw.Draw(image)
# Read button
button = Button(text='Read', height=2, width=10, command=read, highlightbackground='lightblue2')
button.grid(row=10, column=3)
# Delete button
button_9 = Button(text='Delete', height=2, width=10, command=delete, highlightbackground='lightblue2')
button_9.grid(row=10, column=4)
# Help button
button = Button(text='Help', height=2, width=20, command=tip, highlightbackground='lightblue2')
button.grid(row=11, column=3, columnspan=2)
# Display the author
label_4 = Label(master, width=20, height=3, text='By Yutao Chen\n@ Jun 2019', font=("Arial Bold", 20))
label_4.grid(row=9, column=3, columnspan=2)
# Main loop
mainloop()
