import json

if __name__ == '__main__':

    # display the image with the bounding boxes
    def show(image, boxes):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image

        # load the image
        image = Image.open(image)

        # get the context for drawing boxes
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # plot each box
        for box in boxes:
            # get coordinates
            x, y, width, height = box
            # create the shape
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            # draw the box
            ax.add_patch(rect)

        # show the plot
        plt.show()

    with open('data.json', 'r') as f:
        data = json.load(f)

    for image, boxes in data.items():
        show(image, boxes)
